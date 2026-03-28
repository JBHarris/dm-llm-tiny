"""
Step 3: Merge LoRA weights into base model and export.

Merges the trained LoRA adapter back into the full model weights,
saves the merged model, and optionally converts to GGUF format.

Input:  output/ (LoRA adapter)
Output: merged_model/ (full merged model)
        gguf/dm-llm-tiny.Q4_K_M.gguf (quantized GGUF)
"""

import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_DIR = Path("output")
MERGED_DIR = Path("merged_model")
GGUF_DIR = Path("gguf")


def merge_model():
    """Merge LoRA adapter into base model."""
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA adapter from {LORA_DIR}")
    model = PeftModel.from_pretrained(model, str(LORA_DIR))

    print("Merging weights...")
    model = model.merge_and_unload()

    MERGED_DIR.mkdir(exist_ok=True)
    print(f"Saving merged model to {MERGED_DIR}")
    model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print("Merge complete!")


def convert_to_gguf():
    """Convert merged model to GGUF format using llama.cpp."""
    GGUF_DIR.mkdir(exist_ok=True)

    llama_cpp_path = Path("llama.cpp")
    if not llama_cpp_path.exists():
        print("Cloning llama.cpp for GGUF conversion...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"],
            check=True,
        )
        # Install conversion dependencies
        subprocess.run(
            [sys.executable, "-m", "pip3", "install", "-r", "llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"],
            check=True,
        )

    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    output_file = GGUF_DIR / "dm-llm-tiny-f16.gguf"

    print("Converting to GGUF (F16)...")
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(MERGED_DIR),
            "--outfile",
            str(output_file),
            "--outtype",
            "f16",
        ],
        check=True,
    )

    # Quantize to Q4_K_M (good balance of quality and size)
    print("Quantizing to Q4_K_M...")

    # Build llama-quantize if needed
    quantize_bin = llama_cpp_path / "build" / "bin" / "Release" / "llama-quantize"
    if not quantize_bin.exists():
        print("Building llama.cpp quantize tool...")
        build_dir = llama_cpp_path / "build"
        build_dir.mkdir(exist_ok=True)
        subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=str(build_dir), check=True)
        subprocess.run(["cmake", "--build", ".", "--config", "Release", "-t", "llama-quantize", "-j"], cwd=str(build_dir), check=True)

    quantized_file = GGUF_DIR / "dm-llm-tiny-Q4_K_M.gguf"

    print(quantize_bin)
    print(quantized_file)
    print(output_file)
    
    subprocess.run(
        [str(quantize_bin), str(output_file), str(quantized_file), "Q4_K_M"],
        check=True,
    )

    print(f"\nGGUF files:")
    print(f"  F16:     {output_file} ({output_file.stat().st_size / 1e9:.2f} GB)")
    print(f"  Q4_K_M:  {quantized_file} ({quantized_file.stat().st_size / 1e9:.2f} GB)")

    # Clean up F16 (large, only needed for quantization)
    output_file.unlink()
    print(f"  Removed F16 intermediate file.")


def main():
    merge_model()
    convert_to_gguf()

    print("\n--- Next steps ---")
    print("1. Run: python 04_push_to_hub.py")
    print("2. Or test locally: ollama create dm-llm-tiny -f Modelfile")


if __name__ == "__main__":
    main()
