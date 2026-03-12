import gc
import json
import time
import pathlib
import datetime
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class LossCallback(TrainerCallback):
    """Collects training loss at every logging step for later plotting."""

    def __init__(self):
        self.loss_history: list[dict] = []

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs: dict | None = None, **kwargs):
        if logs and "loss" in logs:
            self.loss_history.append({
                "step": state.global_step,
                "epoch": round(state.epoch, 4) if state.epoch is not None else None,
                "loss": logs["loss"],
            })


# ---------------------------------------------------------------------------
# IRI-prefix → dataset name mapping
# ---------------------------------------------------------------------------
IRI_PREFIX_TO_DATASET = {
    "http://www.aifb.uni-karlsruhe.de/": "aifb",
    "http://data.bgs.ac.uk/":            "bgs",
    "http://dl-learner.org/carcinogenesis#": "mutag",
}


def get_dataset_name(iri: str) -> str | None:
    """Return the dataset name for a given IRI, or None if unrecognised."""
    for prefix, name in IRI_PREFIX_TO_DATASET.items():
        if iri.startswith(prefix):
            return name
    return None


def build_prompt(explanation: str, template: str) -> str:
    """Insert the NL explanation into the NLEF prompt template.

    The template ends with a placeholder sentence between quotes after
    ``**Sentence(s)**: "``.  We replace that sentence with the actual
    explanation and close with the ``**DL Expression**:`` marker so the
    model learns to generate the OWL-DL string that follows.
    """
    marker = '**Sentence(s)**: "'
    idx = template.rfind(marker)
    if idx == -1:
        raise ValueError("Marker '**Sentence(s)**' not found in prompt template")
    header = template[: idx + len(marker)]
    return header + explanation + '"\n\n**DL Expression**:\n'


# ---------------------------------------------------------------------------
# 1. Load NLEF prompt templates (one per dataset)
# ---------------------------------------------------------------------------
nlef_prompts: dict[str, str] = {}
for ds_name in ("aifb", "bgs", "mutag"):
    with open(f"nlef_prompt_{ds_name}.txt", "r", encoding="utf-8") as f:
        nlef_prompts[ds_name] = f.read()

# ---------------------------------------------------------------------------
# 2. Load NL explanations  (IRI → natural-language explanation)
# ---------------------------------------------------------------------------
with open("explanations.json", "r") as f:
    explanations: dict[str, str] = json.load(f)

# ---------------------------------------------------------------------------
# 3. Load OWL-DL targets (merge all three dataset files)
# ---------------------------------------------------------------------------
owl_dl: dict[str, str] = {}
for ds_name in ("aifb", "bgs", "mutag"):
    with open(f"owl_dl_{ds_name}.json", "r") as f:
        owl_dl.update(json.load(f))

# ---------------------------------------------------------------------------
# 4. Build SFT dataset:  explanation  →  OWL-DL expression
#    - user   : NLEF prompt template (with ontology context) + NL explanation
#    - assistant : OWL-DL expression
# ---------------------------------------------------------------------------
sft_data: list[dict] = []
for iri, owl_dl_expr in owl_dl.items():
    if iri not in explanations:
        continue
    ds_name = get_dataset_name(iri)
    if ds_name is None:
        continue
    prompt = build_prompt(explanations[iri], nlef_prompts[ds_name])
    sft_data.append({
        "messages": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": owl_dl_expr},
        ]
    })

print(f"SFT dataset size: {len(sft_data)} samples")
dataset = Dataset.from_list(sft_data)

# ---------------------------------------------------------------------------
# 5. Model / LoRA configurations  (Qwen3 family only)
# ---------------------------------------------------------------------------
experiment_model_configs = [
    {
        "model_name": "Qwen/Qwen3-8B",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        "learning_rate": 2e-5,
    },
    {
        "model_name": "Qwen/Qwen3-4B",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        "learning_rate": 2e-5,
    },
    {
        "model_name": "Qwen/Qwen3-1.7B",
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_target_modules": "all-linear",
        "learning_rate": 1e-4,
    },
    {
        "model_name": "Qwen/Qwen3-0.6B",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target_modules": "all-linear",
        "learning_rate": 2e-4,
    },
    # SmolLM-135M — LoRA prevents catastrophic forgetting on the small SFT dataset
    {
        "model_name": "HuggingFaceTB/SmolLM-135M",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": "all-linear",
        "learning_rate": 3e-4,
    },
]

experiment_quantization_config = {
    "Qwen/Qwen3-8B": BitsAndBytesConfig(
        load_in_8bit=True,
    ),
    "Qwen/Qwen3-4B": BitsAndBytesConfig(
        load_in_8bit=True,
    ),
    "Qwen/Qwen3-1.7B": None,
    "Qwen/Qwen3-0.6B": None,
    "HuggingFaceTB/SmolLM-135M": None,
}

# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------
WEIGHTS_DIR = pathlib.Path("weights")
run_metrics: list[dict] = []

for config in experiment_model_configs:
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if "mm_token_type_ids" in getattr(tokenizer, "model_input_names", []):
        tokenizer.model_input_names.remove("mm_token_type_ids")

    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% endif %}"
            "{% if message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=experiment_quantization_config.get(config["model_name"]),
        device_map="auto",
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=config["learning_rate"],
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        output_dir="./sft_nl_to_owl_dl_output",
        report_to="none",
    )

    loss_callback = LossCallback()

    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset,
        callbacks=[loss_callback],
    )

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    sft_trainer.train()
    t_end = time.perf_counter()
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    reserved_memory_gb = torch.cuda.max_memory_reserved() / 1024 ** 3

    short_name = config["model_name"].split("/")[-1]
    save_dir = WEIGHTS_DIR / f"sft_lora_weights_nl_to_owl_dl_{short_name}"
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"SFT training complete. LoRA weights saved to {save_dir}")

    elapsed_seconds = t_end - t_start
    metrics_entry = {
        "model": config["model_name"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "time_taken_seconds": round(elapsed_seconds, 2),
        "time_taken_human": str(datetime.timedelta(seconds=int(elapsed_seconds))),
        "peak_gpu_memory_allocated_gb": round(peak_memory_gb, 3),
        "peak_gpu_memory_reserved_gb": round(reserved_memory_gb, 3),
        "lora_r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "quantization": "int8" if experiment_quantization_config.get(config["model_name"]) else "none",
        "learning_rate": config["learning_rate"],
        "loss_history": loss_callback.loss_history,
    }
    run_metrics.append(metrics_entry)
    metrics_file = pathlib.Path(f"sft_nl_to_owl_dl_run_metrics_{short_name}.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_entry, f, indent=2)
    print(f"Metrics saved to {metrics_file}: time={metrics_entry['time_taken_human']}, "
          f"peak_mem={metrics_entry['peak_gpu_memory_allocated_gb']:.3f} GB")

    del sft_trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU memory cleared after {config['model_name']}")
