# config.py
max_seq_length = 1024
load_in_4bit = True
dtype = None 
model_name = "Meta-Llama-3.1-8B"
quant_method = "q8_0"
lora_config = {
    "r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0,
    "bias": "none",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3407,
    "use_rslora": False,
    "loftq_config": None
}
training_args = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 20,
    "max_steps": 300,      # num_train_epochs ya da bunu.. ikisinden birini kullanın.
    "num_train_epochs": 1, # tüm veriyi bir kez kullanır. Büyük verilerde süre artabilir.
    "learning_rate": 2e-5,
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": "outputs",
    "report_to": "none"
}