import os
import argparse
import json
import torch
from datasets import IterableDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def get_streaming_dataset(manifest_path, library_path, tokenizer, max_length=512):
    # 1. Load the Library into memory once (Efficient for strings)
    print(f"Loading drug metadata library from {library_path}...")
    with open(library_path, 'r') as f:
        metadata_library = json.load(f)

    # 2. Generator function to avoid flattening the manifest
    def gen():
        with open(manifest_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Rehydrate drug pairs using IDs
                meta_a = metadata_library.get(item['drug_1_id'], "")
                meta_b = metadata_library.get(item['drug_2_id'], "")
                label = item.get('label', "unknown")
                
                # Format for Qwen3 fine-tuning
                full_text = f"DRUG 1: {meta_a}\nDRUG 2: {meta_b}\nInteraction Classification: {label}<|endoftext|>"
                
                tokenized = tokenizer(
                    full_text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length
                )
                yield {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": tokenized["input_ids"].copy()
                }

    return IterableDataset.from_generator(gen)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--train_manifest", type=str)
    parser.add_argument("--library_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model with 4-bit Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Config (Targeting Qwen layers)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # 4. Prepare Streaming Dataset
    train_dataset = get_streaming_dataset(
        args.train_manifest, 
        args.library_path, 
        tokenizer
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=args.max_steps, # Required for IterableDataset
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        optim="paged_adamw_32bit",
        report_to="none",
        logging_dir="/gcs/deep_learning_dataset/logs"
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    
    # Save only the LoRA adapters to GCS
    model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))

if __name__ == "__main__":
    train()