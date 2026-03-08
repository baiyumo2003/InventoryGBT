from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import os
import torch

# === Path setup ===
path = ""
dataset_path = os.path.join(path, "inventory_gpt2_dataset_hf")
model_path = os.path.join("", "llama3_inventory_bf16_resized")  # updated to use resized model
output_path = os.path.join(path, "llama3_inventory_bf16")         # output for training results

# === Load dataset ===
dataset = load_from_disk(dataset_path)

# === Load tokenizer & resized model ===
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
)

# === Tokenization ===
def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=2048,
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch")

# === Data collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# === Training Arguments ===
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,  
    logging_steps=10,
    logging_dir=os.path.join(output_path, "logs"),
    save_steps=3000,
    save_total_limit=3,
    report_to="none",
    bf16=True,  
    gradient_checkpointing=True,
    optim="adamw_torch_fused",  
)


# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# === Start training ===
trainer.train()
