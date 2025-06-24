# upload_to_hub.py
from unsloth import FastLanguageModel
from config import *

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# kendinize göre ayarlayın
model.push_to_hub_gguf(
    repo_id="your_username/your_model_name",
    tokenizer=tokenizer,
    quantization_method=quant_method,
    token="hf_XXX"
)