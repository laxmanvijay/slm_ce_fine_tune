import re
import gc
import ast
import json
import time
import pathlib
import datetime
import torch
from collections import Counter
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from peft import PeftModel


from owlapy.class_expression import (
    OWLClass, OWLObjectIntersectionOf, OWLObjectUnionOf,
    OWLObjectComplementOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom,
    OWLObjectMinCardinality, OWLObjectMaxCardinality,
    OWLThing, OWLNothing, OWLClassExpression,
)
from owlapy.owl_property import OWLObjectProperty
from owlapy.iri import IRI


class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics_history: list[dict] = []

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs: dict | None = None, **kwargs):
        if logs:
            entry = {"step": state.global_step}
            if state.epoch is not None:
                entry["epoch"] = round(state.epoch, 4)
            entry.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
            self.metrics_history.append(entry)



IRI_PREFIX_TO_DATASET = {
    "http://www.aifb.uni-karlsruhe.de/": "aifb",
    "http://data.bgs.ac.uk/":            "bgs",
    "http://dl-learner.org/carcinogenesis#": "mutag",
}


def get_dataset_name(iri: str) -> str | None:
    for prefix, name in IRI_PREFIX_TO_DATASET.items():
        if iri.startswith(prefix):
            return name
    return None


def build_prompt(explanation: str, template: str) -> str:
    marker = '**Sentence(s)**: "'
    idx = template.rfind(marker)
    if idx == -1:
        raise ValueError("Marker '**Sentence(s)**' not found in prompt template")
    header = template[: idx + len(marker)]
    return header + explanation + '"\n\n**DL Expression**:\n'


IRI_PATTERN = re.compile(r"IRI\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)")


def extract_valid_iris(owl_dl_dict: dict[str, str]) -> set[tuple[str, str]]:
    """Return the set of (namespace, name) pairs seen in all ground-truth expressions."""
    valid: set[tuple[str, str]] = set()
    for expr in owl_dl_dict.values():
        for ns, name in IRI_PATTERN.findall(expr):
            valid.add((ns, name))
    return valid


_owl_dl_all: dict[str, str] = {}
_valid_iris_by_dataset: dict[str, set[tuple[str, str]]] = {}

for _ds in ("aifb", "bgs", "mutag"):
    with open(f"./distillation/owl_dl_{_ds}.json") as f:
        _ds_dict: dict[str, str] = json.load(f)
    _owl_dl_all.update(_ds_dict)
    _valid_iris_by_dataset[_ds] = extract_valid_iris(_ds_dict)


_EVAL_GLOBALS = {
    "OWLClass": OWLClass,
    "OWLObjectIntersectionOf": OWLObjectIntersectionOf,
    "OWLObjectUnionOf": OWLObjectUnionOf,
    "OWLObjectComplementOf": OWLObjectComplementOf,
    "OWLObjectSomeValuesFrom": OWLObjectSomeValuesFrom,
    "OWLObjectAllValuesFrom": OWLObjectAllValuesFrom,
    "OWLObjectMinCardinality": OWLObjectMinCardinality,
    "OWLObjectMaxCardinality": OWLObjectMaxCardinality,
    "OWLObjectProperty": OWLObjectProperty,
    "IRI": IRI,
    "OWLThing": OWLThing,
    "OWLNothing": OWLNothing,
}


_ALL_CONSTRUCTORS = [
    "OWLObjectSomeValuesFrom", "OWLObjectAllValuesFrom",
    "OWLObjectIntersectionOf", "OWLObjectUnionOf",
    "OWLObjectComplementOf",
    "OWLObjectMinCardinality", "OWLObjectMaxCardinality",
    "OWLClass", "OWLObjectProperty",
]


def _clean_completion(completion: str) -> str:
    completion = re.sub(r"```(?:python)?\s*", "", completion)
    completion = re.sub(r"```", "", completion)
    return completion.strip()


def _count_constructors(text: str) -> Counter:
    return Counter(c for c in _ALL_CONSTRUCTORS if c in text)


def owl_dl_syntax_reward(prompts, completions, **kwargs) -> list[float]:
    """
    Score:
      Valid OWLClassExpression  →  1.0
      Parses but wrong type     →  0.0
      SyntaxError / NameError   → -0.5  (wrong constructor name or bad syntax)
      Any other exception       → -1.0
    """
    scores = []
    for completion in completions:
        cleaned = _clean_completion(completion)

        if not any(c in cleaned for c in _ALL_CONSTRUCTORS):
            scores.append(-1.0)
            continue
        try:
            result = eval(cleaned, {"__builtins__": {}}, _EVAL_GLOBALS)
            if isinstance(result, OWLClassExpression):
                scores.append(1.0)
            else:
                scores.append(0.0)
        except (SyntaxError, NameError):
            scores.append(-0.5)
        except Exception:
            scores.append(-1.0)
    return scores


def owl_dl_ontology_conformance_reward(prompts, completions, owl_dl_target, **kwargs) -> list[float]:
    """
    Score  =  valid_count / (valid_count + invalid_count)   [0..1]
             − 0.5 * (invalid_count / total)                 [penalty]

    Completions with no IRI references at all → 0.0 (neutral; syntax reward
    already handles the case via –1.0).
    """
    scores = []
    for completion, gt_expr in zip(completions, owl_dl_target):
        best_ds = None
        best_overlap = -1
        gt_iris = set(IRI_PATTERN.findall(gt_expr))
        for ds, valid_set in _valid_iris_by_dataset.items():
            overlap = len(gt_iris & valid_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_ds = ds

        valid_set = _valid_iris_by_dataset.get(best_ds, set()) if best_ds else set()

        found_iris = IRI_PATTERN.findall(_clean_completion(completion))
        if not found_iris:
            scores.append(0.0)
            continue

        valid_count = sum(1 for pair in found_iris if tuple(pair) in valid_set)
        invalid_count = len(found_iris) - valid_count
        total = len(found_iris)

        score = (valid_count / total) - 0.5 * (invalid_count / total)
        scores.append(round(score, 4))
    return scores

def owl_dl_constructor_similarity_reward(prompts, completions, owl_dl_target, **kwargs) -> list[float]:
    """
    Score = intersection size / union size  (Jaccard on constructor multisets)
    Range: [0, 1]
    """
    scores = []
    for completion, gt_expr in zip(completions, owl_dl_target):
        gen_counts = _count_constructors(_clean_completion(completion))
        gt_counts  = _count_constructors(gt_expr)

        intersection = sum((gen_counts & gt_counts).values())
        union        = sum((gen_counts | gt_counts).values())

        scores.append(intersection / union if union > 0 else 0.0)
    return scores

def owl_dl_format_reward(prompts, completions, **kwargs) -> list[float]:
    """
    Penalises anything that is not clean OWL-DL output:
      - Prose explanations / running sentences
      - Markdown headings or bold markers
      - <think> / </think> reasoning tokens
      - Role tokens (assistant, user)
      - The prompt being echoed back
      - Very short completions (< 20 chars, essentially empty)

    Score starts at 1.0; each violation deducts 0.25.  Minimum is -1.0.
    """
    bad_patterns = [
        re.compile(r"<think>|</think>", re.IGNORECASE),
        re.compile(r"^\s*(assistant|user)\s*$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"###\s|\*\*[A-Za-z]|\bExplanation:\b", re.IGNORECASE),
        re.compile(r"\bSentence\(s\)\b", re.IGNORECASE),   # prompt echo
        re.compile(r"\bDL Expression\b", re.IGNORECASE),   # prompt echo
        re.compile(r"(?<!['\(])[A-Z][a-z]+\s+[a-z].*\.", re.MULTILINE),  # natural language sentence
    ]
    scores = []
    for completion in completions:
        cleaned = _clean_completion(completion)
        if len(cleaned) < 20:
            scores.append(-1.0)
            continue
        score = 1.0
        for pattern in bad_patterns:
            if pattern.search(cleaned):
                score -= 0.25
        scores.append(max(score, -1.0))
    return scores


nlef_prompts: dict[str, str] = {}
for ds_name in ("aifb", "bgs", "mutag"):
    with open(f"./distillation/nlef_prompt_{ds_name}.txt", "r", encoding="utf-8") as f:
        nlef_prompts[ds_name] = f.read()

with open("./distillation/explanations.json", "r") as f:
    explanations: dict[str, str] = json.load(f)

grpo_data: list[dict] = []
for iri, owl_dl_expr in _owl_dl_all.items():
    if iri not in explanations:
        continue
    ds_name = get_dataset_name(iri)
    if ds_name is None:
        continue
    prompt = build_prompt(explanations[iri], nlef_prompts[ds_name])
    grpo_data.append({
        "prompt": prompt,
        "owl_dl_target": owl_dl_expr,
    })

print(f"GRPO dataset size: {len(grpo_data)} samples")
grpo_dataset = Dataset.from_list(grpo_data)

experiment_models = [
    "Qwen/Qwen3-8B",
    # "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-0.6B",
]

experiment_quantization_config = {
    "Qwen/Qwen3-8B":  BitsAndBytesConfig(load_in_8bit=True),
    "Qwen/Qwen3-4B":  BitsAndBytesConfig(load_in_8bit=True),
    "Qwen/Qwen3-1.7B": None,
    "Qwen/Qwen3-0.6B": None,
}

for model_name in experiment_models:
    short_name = model_name.split("/")[-1]
    SFT_WEIGHTS_DIR = f"./weights/sft_lora_weights_nl_to_owl_dl_{short_name}"
    RL_OUTPUT_DIR   = f"./weights/rl_lora_weights_nl_to_owl_dl_{short_name}"
    METRICS_FILE    = pathlib.Path(f"rl_nl_to_owl_dl_run_metrics_{short_name}_2.json")

    inf_tokenizer = AutoTokenizer.from_pretrained(SFT_WEIGHTS_DIR)
    inf_base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=experiment_quantization_config.get(model_name),
        device_map="auto",
    )
    inf_model = PeftModel.from_pretrained(inf_base_model, SFT_WEIGHTS_DIR)
    inf_model.train()

    metrics_callback = MetricsCallback()

    config = GRPOConfig(
        num_generations=8,
        temperature=0.9,

        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_completion_length=512,

        # [syntax, ontology_conformance, constructor_similarity, format]
        reward_weights=[0.40, 0.30, 0.20, 0.10],

        bf16=True,
        output_dir=f"./checkpoints/grpo_nl_to_owl_dl_output_{short_name}",
        report_to="none",
        save_steps=100,
        save_total_limit=2,
    )

    trainer = GRPOTrainer(
        model=inf_model,
        processing_class=inf_tokenizer,
        args=config,
        train_dataset=grpo_dataset,
        reward_funcs=[
            owl_dl_syntax_reward,
            owl_dl_ontology_conformance_reward,
            owl_dl_constructor_similarity_reward,
            owl_dl_format_reward,
        ],
        callbacks=[metrics_callback],
    )

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    trainer.train(resume_from_checkpoint=True)
    t_end = time.perf_counter()
    peak_memory_gb   = torch.cuda.max_memory_allocated() / 1024 ** 3
    reserved_memory_gb = torch.cuda.max_memory_reserved() / 1024 ** 3

    trainer.model.save_pretrained(RL_OUTPUT_DIR)
    inf_tokenizer.save_pretrained(RL_OUTPUT_DIR)
    print(f"RL fine-tuned model saved to {RL_OUTPUT_DIR}")

    elapsed_seconds = t_end - t_start
    rl_metrics = {
        "model": model_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "time_taken_seconds": round(elapsed_seconds, 2),
        "time_taken_human": str(datetime.timedelta(seconds=int(elapsed_seconds))),
        "peak_gpu_memory_allocated_gb": round(peak_memory_gb, 3),
        "peak_gpu_memory_reserved_gb": round(reserved_memory_gb, 3),
        "num_generations": config.num_generations,
        "learning_rate": config.learning_rate,
        "reward_weights": config.reward_weights,
        "metrics_history": metrics_callback.metrics_history,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(rl_metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_FILE}: time={rl_metrics['time_taken_human']}, "
          f"peak_mem={rl_metrics['peak_gpu_memory_allocated_gb']:.3f} GB")

    del trainer, inf_model, inf_base_model, inf_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU memory cleared after {model_name}\n")
