import json
import pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

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
    """Insert the NL explanation into the NLEF prompt template."""
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
# 3. Load OWL-DL ground-truth targets (for reference in output)
# ---------------------------------------------------------------------------
owl_dl: dict[str, str] = {}
for ds_name in ("aifb", "bgs", "mutag"):
    with open(f"owl_dl_{ds_name}.json", "r") as f:
        owl_dl.update(json.load(f))

# ---------------------------------------------------------------------------
# 4. Build test cases
#    Pick a few representative IRIs from each dataset that exist in both
#    explanations.json and the OWL-DL files.
# ---------------------------------------------------------------------------
test_cases: list[dict] = []
for iri, owl_dl_expr in owl_dl.items():
    if iri not in explanations:
        continue
    ds_name = get_dataset_name(iri)
    if ds_name is None:
        continue
    prompt = build_prompt(explanations[iri], nlef_prompts[ds_name])
    test_cases.append({
        "iri": iri,
        "dataset": ds_name,
        "prompt": prompt,
        "ground_truth_owl_dl": owl_dl_expr,
    })

print(f"Total test cases: {len(test_cases)}")

# ---------------------------------------------------------------------------
# 5. Model configs  (Qwen3 family only)
# ---------------------------------------------------------------------------
experiment_models = ["Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"]

experiment_quantization_config = {
    "Qwen/Qwen3-8B": BitsAndBytesConfig(
        load_in_8bit=True,
    ),
    "Qwen/Qwen3-4B": BitsAndBytesConfig(
        load_in_8bit=True,
    ),
    "Qwen/Qwen3-1.7B": None,
    "Qwen/Qwen3-0.6B": None,
}

# ---------------------------------------------------------------------------
# 6. Inference loop
# ---------------------------------------------------------------------------
for model_name in experiment_models:
    short_name = model_name.split("/")[-1]
    weights_dir = f"./weights/sft_lora_weights_nl_to_owl_dl_{short_name}"
    print(f"Loading base model {model_name} with quantization config: {experiment_quantization_config.get(model_name)}")
    inf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=experiment_quantization_config.get(model_name),
        device_map="auto",
    )

    inf_model = PeftModel.from_pretrained(inf_model, weights_dir)
    inf_model.eval()

    inf_tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    print(f"Model {model_name} loaded successfully.\n")

    def generate_owl_dl(prompt: str, max_new_tokens: int = 1024) -> str:
        messages = [{"role": "user", "content": prompt}]
        tokenized = inf_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(inf_model.device)

        with torch.no_grad():
            output_ids = inf_model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=inf_tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        prompt_len = tokenized["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        return inf_tokenizer.decode(new_tokens, skip_special_tokens=True)

    output_file = f"owl_dl_predictions_{short_name}.txt"
    with open(output_file, "a") as f:
        for i, tc in enumerate(test_cases):
            print(f"\n=== Test Case {i+1} ({tc['dataset']}: {tc['iri']}) ===")

            prediction = generate_owl_dl(tc["prompt"])
            print("Generated OWL-DL:\n", prediction)

            f.write(f"=== Test Case {i+1} ({tc['dataset']}: {tc['iri']}) ===\n")
            f.write("NL Explanation:\n")
            f.write(explanations[tc["iri"]].strip() + "\n")
            f.write("Ground-truth OWL-DL:\n")
            f.write(tc["ground_truth_owl_dl"].strip() + "\n")
            f.write("Generated OWL-DL:\n")
            f.write(prediction.strip() + "\n\n")

    print(f"\nResults saved to {output_file}")

    del inf_model, inf_tokenizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU memory cleared after {model_name}\n")
