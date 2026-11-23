"""Bounded single-GPU LoRA smoke test for Nemotron 3.5 Bolt E4B.

Measures setup separately from training. Synthetic text is for startup and
execution validation only, not model quality evaluation. No remote model code.
"""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import time

START = time.monotonic()
EVENTS = []


def event(name, **fields):
    row = {"event": name, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "elapsed_seconds": round(time.monotonic() - START, 3), **fields}
    EVENTS.append(row)
    print("MLQ_BENCH " + json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16")
    parser.add_argument("--revision", default="49843eed8db1f1c616c47af75cfd36e07aec0667")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error("training duration must be 1..300 seconds")
    root = Path(os.environ["MLQ_JOB_DIR"])
    out = Path(os.environ.get("MLQ_ARTIFACT_DIR", str(root / "artifacts"))) / "startup-benchmark"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root / "model-cache")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_MODE"] = "offline"
    event("process_start", model=args.model, revision=args.revision)
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    import wandb
    event("imports_complete", packages={p: importlib.metadata.version(p) for p in
          ["torch", "transformers", "accelerate", "peft", "huggingface-hub", "safetensors", "wandb"]})
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires one allocated GPU")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(42)
    event("model_download_start")
    model_path = snapshot_download(args.model, revision=args.revision,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"], max_workers=4)
    event("model_download_complete")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = Gemma4ForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    event("model_loaded", gpu=torch.cuda.get_device_name(), gpu_allocated_bytes=torch.cuda.memory_allocated())
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    texts = [
        "Question: What is 17 multiplied by 23? Answer: 391. " * 20,
        "A training environment contains pinned dependencies. A checkpoint records model state. " * 20,
        "The purpose of this synthetic example is to measure startup and training execution. " * 20,
    ]
    batches = [tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda") for text in texts]
    run = wandb.init(project="mlq-startup-benchmark", mode="offline", dir=str(root),
        config={"model": args.model, "revision": args.revision, "seconds": args.seconds, "synthetic": True})
    steps = 0
    torch.cuda.synchronize()
    training_start = time.monotonic()
    event("training_start", trainable_parameters=sum(p.numel() for p in trainable))
    last_report = training_start
    while time.monotonic() - training_start < args.seconds:
        batch = batches[steps % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        result = model(**batch, labels=batch["input_ids"], use_cache=False)
        if not torch.isfinite(result.loss):
            raise RuntimeError("Non-finite loss")
        result.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        steps += 1
        loss = float(result.loss.detach())
        if steps == 1 or time.monotonic() - last_report >= 10:
            event("training_step", step=steps, loss=loss)
            run.log({"loss": loss, "step": steps})
            last_report = time.monotonic()
    event("training_complete", steps=steps, training_seconds=time.monotonic()-training_start,
          peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    model.save_pretrained(out / "lora-adapter", safe_serialization=True)
    run.finish()
    event("success")
    (out / "timings.json").write_text(json.dumps(EVENTS, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        event("failed", error_type=type(exc).__name__, error=str(exc))
        raise
