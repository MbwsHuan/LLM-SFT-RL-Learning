
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/lsw/Levi/models/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    local_files_only=True
).to(DEVICE)

model.eval()

messages = [
    {
        "role": "user",
        "content": "请用两句话解释什么是监督微调SFT。"
    }
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(DEVICE)

attention_mask = torch.ones_like(input_ids)

print("Input shape:", tuple(input_ids.shape))

# 前向传播，只执行一次，不调用 generate
with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False
    )

logits = outputs.logits

print("Logits shape:", tuple(logits.shape))

# 最后一个位置预测下一个 Token
next_token_logits = logits[0, -1, :]

# 转换为概率
probs = torch.softmax(next_token_logits, dim=-1)

# 取得概率最高的 5 个 Token
top_probs, top_ids = torch.topk(probs, k=5)

print("\nTop 5 next-token predictions:")

for token_id, prob in zip(top_ids, top_probs):
    token_id = token_id.item()
    prob = prob.item()

    token_text = tokenizer.decode([token_id])

    print(
        f"Token ID: {token_id:<8} "
        f"Probability: {prob:.4f} "
        f"Text: {repr(token_text)}"
    )

print("\nProbability sum:", probs.sum().item())
print(
    "Greedy next token:",
    repr(tokenizer.decode([top_ids[0].item()]))
)
