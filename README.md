
# LoRA Fine‑Tuning on Apple Silicon (MPS)

Fine-tune and run Large Language Models (LLMs) locally on Mac (M1/M2/M3) using [LoRA](https://arxiv.org/abs/2106.09685) adapters and Apple's MPS backend. This repository provides:

- 🚀 **Interactive Streamlit UI** for easy, no-code fine-tuning with real-time monitoring
- 📒 A well-documented notebook for programmatic fine-tuning
- 🧪 Minimal benchmark and testing helpers
- 📝 Ready-to-use code snippets for saving, loading, inference, and merging LoRA adapters

If you find this useful, please ⭐ star the repo!

---

## Overview

This project enables fast, memory-efficient LLM fine-tuning on Apple Silicon Macs without NVIDIA GPUs. LoRA adapters allow you to train and save only a few MB of parameters, making experimentation accessible and reproducible.


## Why Use This Repo?
- 🚀 Fast iteration on Mac without NVIDIA GPUs
- 🎨 User-friendly Streamlit interface (no coding required!)
- 💾 Minimal memory footprint via LoRA
- 🔁 Clear, reproducible steps adaptable to your own datasets and models


## Requirements
- macOS 12.3+ (recommended: 13+)
- Python 3.10+
- Apple Silicon (M1/M2/M3) with MPS
- Packages: [transformers](https://github.com/huggingface/transformers), [datasets](https://github.com/huggingface/datasets), [peft](https://github.com/huggingface/peft), [accelerate](https://github.com/huggingface/accelerate), [trl](https://github.com/huggingface/trl), [torch](https://pytorch.org/), [streamlit](https://streamlit.io/)


## Installation
Using a virtual environment is recommended (`uv` or `venv`).

```bash
# Create and activate a venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** If the base model is gated (e.g., Llama), log in:

```bash
pip install huggingface_hub
huggingface-cli login
```


## Quickstart (Streamlit UI) 🎨

The easiest way to fine-tune models is using the interactive Streamlit interface:

```bash
streamlit run app.py
```

Or use the Makefile:

```bash
make run
```

The web interface will open automatically in your browser. From there you can:

1. **Setup Tab**: Configure model, dataset, and LoRA parameters, then load and apply LoRA
2. **Training Tab**: Start training with real-time progress monitoring
3. **Testing Tab**: Test your fine-tuned model with custom prompts
4. **Export Tab**: Save LoRA adapters or merge them into a full model

**Features:**
- 🎯 Preset models (Llama 3.2 1B/3B) or custom HuggingFace models
- 📊 Preset datasets (IMDB, WikiText) or custom datasets
- ⚙️ Interactive LoRA configuration (rank, alpha, dropout)
- 📈 Real-time training metrics and visualization
- 🧪 Live testing with adjustable generation parameters
- 💾 Export options: LoRA adapters (few MB) or merged models


## Quickstart (Notebook)
1. Open `notebooks/mps-lora.ipynb` in Jupyter or VS Code.
2. Set `model_name` (default: `meta-llama/Llama-3.2-1B-Instruct`).
3. Run cells in order:
    - Device check
    - Load model/tokenizer
    - Configure LoRA
    - Prepare dataset
    - Train
    - Save
    - Test
    - Merge (optional)


## Inference Example (after training)


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct"
adapters_dir = "./llama-1b-imdb-lora-finetuned"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
model = PeftModel.from_pretrained(base, adapters_dir).eval()
tok = AutoTokenizer.from_pretrained(adapters_dir)

inputs = tok("This movie was absolutely", return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"].to(device),
    attention_mask=inputs["attention_mask"].to(device),
    max_new_tokens=64, temperature=0.7, top_p=0.9, do_sample=True
)
print(tok.decode(outputs[0], skip_special_tokens=True))
```

**Example output:**
```
This movie was absolutely fantastic! The performances were top-notch and the story kept me engaged throughout.
```


## Merge Adapters (Optional, for Deployment)
```python
merged = model.merge_and_unload()
merged_dir = "./llama-1b-imdb-lora-merged"
merged.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
```

**Note:** Merged models are larger on disk; you can’t edit just the adapters after merging.


## Tips for Apple MPS

- Use FP32 on MPS during training for stability (fp16/bf16 may cause kernel issues).
- If you hit OOM: set `per_device_train_batch_size=1`, increase `gradient_accumulation_steps`, reduce `max_length` (e.g., 512→256/384), and use lighter LoRA (`r=8`, `alpha=16`).
- The first epoch may be slower due to kernel compilation; later epochs speed up.
- Close heavy apps to free unified memory.


## Troubleshooting & FAQ

**Q: Training is slow or crashes on MPS.**
A: Use FP32, reduce batch size, and close other apps. See Tips above.

**Q: How do I use a different model or dataset?**
A: Change `model_name` and dataset loading code in the notebook/script.

**Q: Can I run this on Intel Macs or Windows?**
A: MPS is only available on Apple Silicon. For other platforms, use CUDA or CPU.

---

## Contributing
Issues and PRs are welcome. Share what model/dataset you tuned and results you observed!


## License
Apache-2.0 License


## Acknowledgments
- [Hugging Face Transformers](https://github.com/huggingface/transformers), [Datasets](https://github.com/huggingface/datasets), [Accelerate](https://github.com/huggingface/accelerate), [TRL](https://github.com/huggingface/trl)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [Streamlit](https://streamlit.io/) for the interactive UI framework
- [Meta (Llama models)](https://ai.meta.com/tools/llama/) and the open‑source community