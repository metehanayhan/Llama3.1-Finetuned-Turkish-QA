# train.py
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
import torch
from config import *

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    **lora_config
)

from datasets import load_dataset
dataset = load_dataset("csv", data_files="Data/dataset.csv")["train"]
dataset = dataset.rename_column("text", "text")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        **training_args
    )
)

if __name__ == "__main__":
    trainer.train()