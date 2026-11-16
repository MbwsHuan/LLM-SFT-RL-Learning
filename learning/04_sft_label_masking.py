
# -*- coding: utf-8 -*-
"""
Day 4: Padding, Attention Mask and Label Masking

实验目标：
1. 构造两条不同长度的 SFT 样本
2. 使用 Right Padding 组成 Batch
3. 构建 Attention Mask
4. 构建 Full Sequence Labels
5. 构建 Answer-only Labels
6. 对比两种 Loss

本实验只执行前向传播，不更新模型参数。
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==================================================
# 1. 基础配置
# ==================================================

MODEL_PATH = (
    "/home/lsw/Levi/models/Qwen2.5-0.5B-Instruct"
)

DEVICE = "cuda:0"

print("=" * 60)
print("Day 4: SFT Label Masking")
print("=" * 60)


# ==================================================
# 2. 加载 Tokenizer 和模型
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

# 训练场景使用右侧 Padding
tokenizer.padding_side = "right"

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    local_files_only=True
).to(DEVICE)

model.eval()

print("\n[1] Model loaded")
print("Pad token ID:", tokenizer.pad_token_id)


# ==================================================
# 3. 构造两条不同长度的训练数据
# ==================================================

samples = [
    {
        "question": "什么是SFT？",
        "answer": "SFT是使用标注数据对预训练模型进行监督微调。"
    },
    {
        "question": "请解释LoRA为什么能够减少微调参数量。",
        "answer": (
            "LoRA冻结预训练模型原有权重，"
            "通过两个低秩矩阵学习权重增量。"
            "因此只需要更新少量新增参数，"
            "就能够降低微调过程的参数存储"
            "和优化器状态开销。"
        )
    }
]


# ==================================================
# 4. 使用 Chat Template 编码
# ==================================================

all_input_ids = []
all_prompt_lengths = []

for sample in samples:

    user_messages = [
        {
            "role": "user",
            "content": sample["question"]
        }
    ]

    full_messages = [
        {
            "role": "user",
            "content": sample["question"]
        },
        {
            "role": "assistant",
            "content": sample["answer"]
        }
    ]

    # Prompt 包含 assistant 起始标记
    prompt_ids = tokenizer.apply_chat_template(
        user_messages,
        tokenize=True,
        add_generation_prompt=True
    )

    # 完整对话包括 assistant 回答与结束标记
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False
    )

    # 验证两种编码具有相同的前缀
    assert full_ids[:len(prompt_ids)] == prompt_ids, (
        "Chat Template 前缀未对齐，"
        "不能直接使用长度构建 Label Mask"
    )

    assert len(full_ids) > len(prompt_ids)

    all_input_ids.append(full_ids)
    all_prompt_lengths.append(len(prompt_ids))

print("\n[2] Original sequence lengths")

for i, ids in enumerate(all_input_ids):
    print(
        f"Sample {i}: "
        f"Total={len(ids)}, "
        f"Prompt={all_prompt_lengths[i]}, "
        f"Answer={len(ids)-all_prompt_lengths[i]}"
    )


# ==================================================
# 5. Padding：构建统一 Batch
# ==================================================

batch = tokenizer.pad(
    {"input_ids": all_input_ids},
    padding=True,
    return_attention_mask=True,
    return_tensors="pt"
)

input_ids = batch["input_ids"].to(DEVICE)
attention_mask = batch["attention_mask"].to(DEVICE)

batch_size, max_length = input_ids.shape

print("\n[3] Padding Results")
print("Input IDs shape:", tuple(input_ids.shape))
print("Attention Mask shape:", tuple(attention_mask.shape))

for i in range(batch_size):
    print(
        f"Sample {i} Attention Mask:",
        attention_mask[i].tolist()
    )


# ==================================================
# 6. Full Sequence Labels
# ==================================================

labels_full = input_ids.clone()

# 忽略全部 Padding 位置
labels_full[attention_mask == 0] = -100


# ==================================================
# 7. Answer-only Labels
# ==================================================

labels_answer = labels_full.clone()

# System、User、Assistant Header 不计算 Loss
for i, prompt_length in enumerate(all_prompt_lengths):
    labels_answer[i, :prompt_length] = -100


# ==================================================
# 8. 检查有效 Loss Token
# ==================================================

print("\n[4] Label Masking Results")

for i in range(batch_size):

    full_count = (labels_full[i] != -100).sum().item()
    answer_count = (labels_answer[i] != -100).sum().item()
    padding_count = (attention_mask[i] == 0).sum().item()

    print(f"\nSample {i}")
    print("Full label tokens:", full_count)
    print("Answer label tokens:", answer_count)
    print("Padding tokens:", padding_count)

    # 显示实际参与 Answer-only Loss 的文本
    answer_token_ids = labels_answer[i][
        labels_answer[i] != -100
    ].tolist()

    print(
        "Target text:",
        tokenizer.decode(
            answer_token_ids,
            skip_special_tokens=False
        )
    )

# 确保所有 Padding 标签都被忽略
assert torch.all(
    labels_answer[attention_mask == 0] == -100
)

# 确保每条样本都有有效的回答目标
assert torch.all(
    (labels_answer != -100).sum(dim=1) > 0
)


# ==================================================
# 9. 模型前向传播
# ==================================================

print("\n[5] Forward Pass")

with torch.inference_mode():

    outputs_full = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels_full,
        use_cache=False
    )

    outputs_answer = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels_answer,
        use_cache=False
    )

logits = outputs_full.logits

print("Logits shape:", tuple(logits.shape))


# ==================================================
# 10. 手动计算 Answer-only Loss
# ==================================================

shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels_answer[:, 1:].contiguous()

manual_answer_loss = F.cross_entropy(
    shift_logits.reshape(
        -1,
        shift_logits.size(-1)
    ),
    shift_labels.reshape(-1),
    ignore_index=-100
)


# ==================================================
# 11. Loss 对比
# ==================================================

print("\n[6] Loss Comparison")

print("Full Sequence Loss:",
      outputs_full.loss.item())

print("Answer-only Loss:",
      outputs_answer.loss.item())

print("Manual Answer-only Loss:",
      manual_answer_loss.item())

print(
    "Loss Match:",
    torch.allclose(
        outputs_answer.loss,
        manual_answer_loss,
        atol=1e-5
    )
)


# ==================================================
# 12. 查看 GPU 显存
# ==================================================

print("\n[7] GPU Memory")

print(
    "Peak allocated:",
    round(
        torch.cuda.max_memory_allocated() / 1024**3,
        3
    ),
    "GiB"
)

print("\nDay 4 Experiment completed!")
