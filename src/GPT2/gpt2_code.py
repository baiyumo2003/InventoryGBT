from datasets import load_from_disk
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import os

# Load dataset
path = ""

dataset_path=os.path.join(path,"inventory_gpt2_dataset_hf")
dataset = load_from_disk(dataset_path)

# Add special tokens
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens_dict = {"additional_special_tokens": ["[OBS]", "[ACT]"]}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token

# Load model and resize embeddings
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Tokenize
def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch")

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# TrainingArguments optimized for H100
training_args = TrainingArguments(
    output_dir=os.path.join(path,"gpt2_inventory_bf16"),
    per_device_train_batch_size=32,  
    num_train_epochs=3,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    fp16=False,      # disable fp16
    bf16=True,       # use bf16 on H100
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
