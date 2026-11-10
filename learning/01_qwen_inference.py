
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. 基础配置
# ==========================================
MODEL_ID = "/home/lsw/Levi/models/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"

print("=" * 60)
print("Qwen2.5-0.5B-Instruct Inference")
print("=" * 60)

# ==========================================
# 2. 加载 Tokenizer
# ==========================================
print("\n[1] Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    local_files_only=True
)

print("Vocabulary size:", len(tokenizer))
print("EOS token:", tokenizer.eos_token)
print("EOS token ID:", tokenizer.eos_token_id)

# ==========================================
# 3. 加载模型
# ==========================================
print("\n[2] Loading Model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    local_files_only=True
).to(DEVICE)

model.eval()

total_params = sum(p.numel() for p in model.parameters())

print("Model parameters:", f"{total_params:,}")
print("Model device:", next(model.parameters()).device)

# ==========================================
# 4. 构造对话输入
# ==========================================
messages = [
    {
        "role": "user",
        "content": "请用两句话解释什么是监督微调SFT。"
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("\n[3] Chat Template:")
print(prompt)

# ==========================================
# 5. Tokenization
# ==========================================
inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(DEVICE)

print("\n[4] Tokenizer Results:")
print("Input IDs shape:", inputs["input_ids"].shape)
print("Token IDs:", inputs["input_ids"][0].tolist())
print("Attention Mask:", inputs["attention_mask"][0].tolist())

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0].tolist()
)

print("Tokens:", tokens)

# ==========================================
# 6. 模型推理
# ==========================================
print("\n[5] Generating Answer...")

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

input_length = inputs["input_ids"].shape[1]
generated_ids = outputs[0][input_length:]

answer = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
)

print("\n[6] Model Answer:")
print(answer)

# ==========================================
# 7. GPU 显存统计
# ==========================================
print("\n[7] GPU Memory:")
print(
    "Allocated:",
    round(torch.cuda.memory_allocated(0) / 1024**3, 3),
    "GiB"
)
print(
    "Peak allocated:",
    round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
    "GiB"
)

print("\nInference completed successfully!")
