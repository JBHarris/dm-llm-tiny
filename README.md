# dm-llm-tiny

A tiny (1.1B parameter) language model fine-tuned for **Dungeons & Dragons** content generation. Built by fine-tuning [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) on synthetic D&D data using QLoRA.

The model generates:
- **NPCs** — characters with backstories, motivations, and quirks
- **Quests** — hooks, outlines, and full quest arcs
- **Dialog** — in-character conversations, monologues, and banter
- **Locations** — vivid descriptions of dungeons, towns, and wilderness
- **Encounters** — combat, social, and puzzle encounters

## How It Works

The pipeline has four automated steps:

1. **Generate training data** — Uses the Claude API to create ~500 D&D instruction/response pairs across five categories (NPCs, quests, dialog, locations, encounters). Each prompt includes random variation for diversity.
2. **Fine-tune with QLoRA** — Loads TinyLlama in 4-bit quantization, attaches LoRA adapters (r=64), and trains for 3 epochs using SFTTrainer. Runs on a single consumer GPU.
3. **Merge & export** — Merges LoRA weights back into the base model, then converts to GGUF format (Q4_K_M quantization) for lightweight inference.
4. **Push to HuggingFace** — Uploads the merged model and GGUF file to HuggingFace Hub.

## Recreate It Yourself

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4080 16GB)
- An [Anthropic API key](https://console.anthropic.com/) for data generation
- A [HuggingFace account and token](https://huggingface.co/settings/tokens) with write access
- `cmake` and a C compiler (for GGUF conversion via llama.cpp)

### Setup

```bash
git clone https://github.com/JBHarris/dm-llm-tiny.git
cd dm-llm-tiny
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):

```
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
```

### Run the Full Pipeline

```bash
./run_all.sh
```

Or run each step individually:

```bash
python3 01_generate_data.py    # ~10 min (API calls with rate limiting)
python3 02_train.py            # ~15-30 min on a consumer GPU
python3 03_merge_and_export.py # ~5 min
python3 04_push_to_hub.py      # ~2 min
```

### Customize It

**Want a different domain?** Edit the prompt categories and system prompts in `01_generate_data.py`. The same pipeline works for any creative writing niche — sci-fi worldbuilding, horror scenarios, superhero backstories, etc.

**Want more quality?** Increase `NUM_EXAMPLES` in `01_generate_data.py` (1000-2000 gives noticeably better results). You can also increase `EPOCHS` in `02_train.py`, though 3 is a solid default.

**Want a bigger base model?** Swap `BASE_MODEL` in `02_train.py` and `03_merge_and_export.py` to any HuggingFace chat model. [Phi-2](https://huggingface.co/microsoft/phi-2) (2.7B) or [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) are good next steps.

## Use the Model

### With Ollama (easiest)

```bash
ollama create dm-llm-tiny -f Modelfile
ollama run dm-llm-tiny
>>> Create a mysterious NPC for my tavern scene
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

## Tech Stack

| Component | What | Why |
|-----------|------|-----|
| [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | Base model | Small enough to run anywhere, chat-formatted |
| [QLoRA](https://arxiv.org/abs/2305.14314) | Training method | 4-bit quantization + LoRA = low VRAM fine-tuning |
| [TRL](https://github.com/huggingface/trl) | Training library | SFTTrainer handles chat formatting and training loop |
| [Claude API](https://docs.anthropic.com/) | Data generation | High-quality synthetic training examples |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | GGUF conversion | Industry-standard format for local inference |
| [Ollama](https://ollama.com/) | Local inference | One-command model running for end users |

## License

Apache 2.0 (same as base model)
