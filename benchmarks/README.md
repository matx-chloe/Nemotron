# MLQ startup smoke benchmark

`mlq_bolt_smoke.py` runs a bounded, single-GPU LoRA training smoke test with
NVIDIA Nemotron 3.5 Bolt E4B BF16 at a pinned Hugging Face revision. It uses
synthetic text; this is a startup/performance experiment, not a quality evaluation.

The script emits UTC and monotonic timestamps for imports, model download,
model loading, first training step, and completion. Training runs for 300 seconds
plus completion of the final step. Allocate extra wall time for image construction
and model loading. It saves a small LoRA adapter and timing report under artifacts/.

The runtime must provide torch, transformers, accelerate, peft, huggingface-hub,
safetensors, numpy, wandb, pillow and sentencepiece. MLQ supplies MLQ_JOB_DIR.
W&B logging is offline. Model code is loaded with trust_remote_code=False.

Example inside the approved MLQ image:

```bash
python benchmarks/mlq_bolt_smoke.py --seconds 300
```

The smoke test is prepared but has not yet been executed on a GPU.
