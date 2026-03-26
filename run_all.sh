#!/bin/bash
# Full pipeline: generate data -> train -> merge -> push
# Run from the dm-llm-tiny directory with .env configured

set -e

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=== Step 1: Generate Training Data ==="
python3 01_generate_data.py

echo ""
echo "=== Step 2: Fine-tune Model ==="
python3 02_train.py

echo ""
echo "=== Step 3: Merge & Convert to GGUF ==="
python3 03_merge_and_export.py

echo ""
echo "=== Step 4: Push to HuggingFace ==="
python3 04_push_to_hub.py

echo ""
echo "=== All done! ==="
echo "Model: https://huggingface.co/JBHarris/dm-llm-tiny"
echo ""
echo "To test locally with Ollama:"
echo "  ollama create dm-llm-tiny -f Modelfile"
echo "  ollama run dm-llm-tiny"
