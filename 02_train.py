"""
Step 2: Fine-tune TinyLlama on D&D data using QLoRA.

Loads the synthetic dataset, applies 4-bit quantization + LoRA adapters,
and trains using SFTTrainer from the TRL library.

Input:  data/dnd_training.jsonl
Output: output/ (LoRA adapter weights + training logs)
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# --- Configuration ---
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = Path("data/dnd_training.jsonl")
OUTPUT_DIR = Path("output")
MAX_SEQ_LENGTH = 1024

# QLoRA config
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training hyperparameters
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"


def load_dataset_from_jsonl(path: Path) -> Dataset:
    """Load JSONL chat dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            examples.append(obj)
    return Dataset.from_list(examples)


def format_chat(example, tokenizer):
    """Apply the chat template to messages."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    print(f"Loading base model: {BASE_MODEL}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format dataset
    print(f"Loading dataset from {DATA_FILE}")
    dataset = load_dataset_from_jsonl(DATA_FILE)
    dataset = dataset.map(lambda x: format_chat(x, tokenizer))
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    print(f"Train: {len(dataset['train'])} examples, Eval: {len(dataset['test'])} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        max_grad_norm=0.3,
        group_by_length=True,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("Training complete!")


if __name__ == "__main__":
    main()
