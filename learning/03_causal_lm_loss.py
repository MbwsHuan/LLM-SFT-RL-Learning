# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ============================================================
# 1. Environment
# ============================================================

MODEL_PATH = (
    "/home/lsw/Levi/models/"
    "Qwen2.5-0.5B-Instruct"
)

DEVICE = "cuda:0"

print("=" * 60)
print("Day 3: Causal Language Modeling Loss")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# ============================================================
# 2. Load Tokenizer
# ============================================================

print("\n[1] Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Vocabulary size:", len(tokenizer))
print("EOS token:", tokenizer.eos_token)


# ============================================================
# 3. Load Model
# ============================================================

print("\n[2] Loading Model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    local_files_only=True
).to(DEVICE)

model.eval()

print("Model parameters:",
      sum(p.numel() for p in model.parameters()))

print("Model device:", DEVICE)

print("Model vocab size:",
      model.config.vocab_size)


# ============================================================
# 4. Construct Training Sample
# ============================================================

print("\n[3] Constructing Training Sample...")

messages = [
    {
        "role": "user",
        "content": "什么是监督微调SFT？"
    },
    {
        "role": "assistant",
        "content": (
            "监督微调SFT是一种利用标注数据"
            "训练预训练语言模型的方法。"
            "通过学习输入指令与目标回答之间的关系，"
            "使模型更好地遵循用户指令。"
        )
    }
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
).to(DEVICE)

attention_mask = torch.ones_like(input_ids)

labels = input_ids.clone()

print("Input IDs shape:", tuple(input_ids.shape))
print("Labels shape:", tuple(labels.shape))

print("\nDecoded Training Sample:")
print(
    tokenizer.decode(
        input_ids[0],
        skip_special_tokens=False
    )
)


# ============================================================
# 5. Forward Pass
# ============================================================

print("\n[4] Forward Pass...")

with torch.inference_mode():

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False
    )

logits = outputs.logits

print("Logits shape:", tuple(logits.shape))


# ============================================================
# 6. Built-in Causal LM Loss
# ============================================================

print("\n[5] Built-in Loss...")

builtin_loss = outputs.loss

print("Built-in Loss:",
      builtin_loss.item())


# ============================================================
# 7. Manual Causal LM Shift
# ============================================================

print("\n[6] Manual Shift...")

shift_logits = logits[:, :-1, :].contiguous()

shift_labels = labels[:, 1:].contiguous()

print("Shift Logits shape:",
      tuple(shift_logits.shape))

print("Shift Labels shape:",
      tuple(shift_labels.shape))


# ============================================================
# 8. Manual Cross Entropy Loss
# ============================================================

print("\n[7] Manual Cross Entropy Loss...")

manual_loss = F.cross_entropy(
    shift_logits.view(
        -1,
        shift_logits.size(-1)
    ),
    shift_labels.view(-1)
)

print("Manual Loss:",
      manual_loss.item())


# ============================================================
# 9. Compare Two Losses
# ============================================================

print("\n[8] Loss Comparison...")

difference = abs(
    builtin_loss.item() -
    manual_loss.item()
)

print("Built-in Loss:", builtin_loss.item())

print("Manual Loss:", manual_loss.item())

print("Difference:", difference)

print(
    "Loss Match:",
    torch.allclose(
        builtin_loss,
        manual_loss,
        atol=1e-5
    )
)


# ============================================================
# 10. Perplexity
# ============================================================

print("\n[9] Perplexity...")

perplexity = torch.exp(manual_loss)

print("Perplexity:",
      perplexity.item())


# ============================================================
# 11. GPU Memory
# ============================================================

print("\n[10] GPU Memory...")

print(
    "Allocated:",
    round(
        torch.cuda.memory_allocated() / 1024**3,
        3
    ),
    "GiB"
)

print(
    "Peak allocated:",
    round(
        torch.cuda.max_memory_allocated() / 1024**3,
        3
    ),
    "GiB"
)

print("\nExperiment completed successfully!")