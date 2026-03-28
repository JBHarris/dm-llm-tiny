"""
Step 4: Push the model to HuggingFace Hub.

Uploads the merged model and GGUF file to your HF repo.

Target: JBHarris/dm-llm-tiny
"""

import os
from pathlib import Path

from huggingface_hub import HfApi, login

# --- Configuration ---
HF_REPO = "JBHarris/dm-llm-tiny"
MERGED_DIR = Path("merged_model")
GGUF_DIR = Path("gguf")

MODEL_CARD = """---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- dnd
- dungeons-and-dragons
- rpg
- text-generation
- qlora
- tinyllama
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
pipeline_tag: text-generation
---

# DM-LLM-Tiny

A tiny (1.1B parameter) language model fine-tuned for **Dungeons & Dragons** content generation.

## What it does

Generates creative D&D content including:
- **NPCs** — memorable characters with backstories, motivations, and quirks
- **Quests** — hooks, outlines, and full quest arcs
- **Dialog** — in-character conversations, monologues, and banter
- **Locations** — vivid descriptions of dungeons, towns, and wilderness
- **Encounters** — combat, social, and puzzle encounters

## Usage

### With Ollama (easiest)
```bash
ollama run JBHarris/dm-llm-tiny
```

### With Transformers
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="JBHarris/dm-llm-tiny")
messages = [
    {"role": "system", "content": "You are a creative D&D dungeon master's assistant."},
    {"role": "user", "content": "Create a mysterious NPC for a tavern scene."},
]
result = pipe(messages, max_new_tokens=512)
print(result[0]["generated_text"][-1]["content"])
```

## Training

- **Base model:** TinyLlama-1.1B-Chat-v1.0
- **Method:** QLoRA (4-bit NF4 quantization + LoRA r=64)
- **Data:** ~500 synthetic D&D instruction/response pairs generated with Claude
- **Hardware:** NVIDIA RTX 4080 16GB

## Limitations

This is a 1.1B parameter model. It's creative and fun for brainstorming but will not match
the quality of larger models (7B+). Best used as a quick idea generator, not a replacement
for a human DM's judgment.

## License

Apache 2.0 (same as base model)
"""


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in your .env file")

    login(token=token)
    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")

    # Write model card
    readme_path = MERGED_DIR / "README.md"
    readme_path.write_text(MODEL_CARD, encoding="utf-8")

    # Upload merged model
    print(f"Uploading merged model to {HF_REPO}...")
    api.upload_folder(
        folder_path=str(MERGED_DIR),
        repo_id=HF_REPO,
        repo_type="model",
    )

    # Upload GGUF file
    gguf_file = GGUF_DIR / "dm-llm-tiny-Q4_K_M.gguf"
    if gguf_file.exists():
        print(f"Uploading GGUF file...")
        api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo="dm-llm-tiny-Q4_K_M.gguf",
            repo_id=HF_REPO,
            repo_type="model",
        )

    print(f"\nDone! Model available at: https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()
