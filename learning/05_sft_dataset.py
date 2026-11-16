
# -*- coding: utf-8 -*-
"""
Day 5: SFT Dataset, DataLoader and Data Collator

学习目标：
1. 使用 Dataset 封装训练数据
2. 对 Qwen Chat Template 进行 Tokenization
3. 创建 Train / Validation Split
4. 使用 DataLoader 读取 Batch
5. 自定义 Data Collator 完成动态 Padding
6. 使用 Answer-only Labels 计算 Loss

本实验只计算 Loss，不更新模型参数。
"""

import random
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================================
# 1. 实验配置
# ==========================================================

MODEL_PATH = (
    "/home/lsw/Levi/models/Qwen2.5-0.5B-Instruct"
)

DEVICE = "cuda:0"
BATCH_SIZE = 2
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("Day 5: SFT Dataset and Data Collator")
print("=" * 60)


# ==========================================================
# 2. 加载 Tokenizer
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

tokenizer.padding_side = "right"

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n[1] Tokenizer")
print("Pad Token ID:", tokenizer.pad_token_id)


# ==========================================================
# 3. 构造原始训练数据
# ==========================================================

raw_samples = [
    {
        "question": "什么是监督微调SFT？",
        "answer": "SFT是利用标注数据对预训练模型进行监督微调。"
    },
    {
        "question": "什么是Tokenizer？",
        "answer": "Tokenizer负责把输入文本转换为模型能够处理的Token ID序列。"
    },
    {
        "question": "解释Logits的作用。",
        "answer": "Logits是语言模型输出的原始预测分数，可以通过Softmax转换为概率分布。"
    },
    {
        "question": "Cross Entropy Loss有什么作用？",
        "answer": "交叉熵损失用于衡量模型预测分布与真实标签之间的差异。"
    },
    {
        "question": "为什么Causal LM需要Shift？",
        "answer": "因为当前位置的Logits用于预测下一个Token，需要将预测结果与后一个位置的真实标签对齐。"
    },
    {
        "question": "什么是Label Masking？",
        "answer": "Label Masking通过将不需要训练的位置设置为-100，使这些位置不参与交叉熵损失。"
    },
    {
        "question": "LoRA为什么能够减少可训练参数？",
        "answer": "LoRA冻结原始模型权重，使用低秩矩阵表示权重增量，只更新少量新增参数。"
    },
    {
        "question": "DataLoader和Data Collator有什么区别？",
        "answer": "DataLoader负责读取和组织样本批次，Data Collator负责将批次中的变长样本补齐并整理成Tensor。"
    }
]

print("\n[2] Raw Samples")
print("Total samples:", len(raw_samples))


# ==========================================================
# 4. 自定义 SFT Dataset
# ==========================================================

class SFTDataset(Dataset):

    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        prompt_messages = [
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

        # Prompt：包含 assistant 消息起始前缀
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True
        )

        # 完整对话：Prompt + Assistant 正文 + 结束标记
        full_ids = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False
        )

        # 确认 Prompt 确实是 Full Sequence 的前缀
        assert full_ids[:len(prompt_ids)] == prompt_ids

        # 当前教学样本不能出现空回答
        assert len(full_ids) > len(prompt_ids)

        # 当前只对 Assistant 正文和结束标记计算 Loss
        labels = (
            [-100] * len(prompt_ids)
            + full_ids[len(prompt_ids):]
        )

        assert len(labels) == len(full_ids)

        return {
            "input_ids": full_ids,
            "labels": labels,
            "sample_id": idx
        }


dataset = SFTDataset(raw_samples, tokenizer)

print("\n[3] Dataset")
print("Dataset length:", len(dataset))

first_item = dataset[0]

print("First sample length:", len(first_item["input_ids"]))
print(
    "First sample answer tokens:",
    sum(x != -100 for x in first_item["labels"])
)


# ==========================================================
# 5. 固定随机种子划分 Train / Validation
# ==========================================================

indices = list(range(len(dataset)))

split_rng = random.Random(SEED)
split_rng.shuffle(indices)

train_indices = indices[:6]
val_indices = indices[6:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

assert set(train_indices).isdisjoint(val_indices)

print("\n[4] Dataset Split")
print("Train indices:", train_indices)
print("Validation indices:", val_indices)
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))


# ==========================================================
# 6. 自定义 Data Collator
# ==========================================================

class SFTDataCollator:

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):

        # features 是一个 Batch 的样本列表
        max_length = max(
            len(item["input_ids"])
            for item in features
        )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        sample_ids = []

        for item in features:

            ids = item["input_ids"]
            labels = item["labels"]

            assert len(ids) == len(labels)

            padding_length = max_length - len(ids)

            padded_ids = (
                ids
                + [self.pad_token_id] * padding_length
            )

            attention_mask = (
                [1] * len(ids)
                + [0] * padding_length
            )

            padded_labels = (
                labels
                + [-100] * padding_length
            )

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(padded_labels)
            sample_ids.append(item["sample_id"])

        batch = {
            "input_ids": torch.tensor(
                batch_input_ids, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                batch_attention_mask, dtype=torch.long
            ),
            "labels": torch.tensor(
                batch_labels, dtype=torch.long
            ),
            "sample_ids": sample_ids
        }

        # 验证 Padding 不参与 Loss
        assert torch.all(
            batch["labels"][
                batch["attention_mask"] == 0
            ] == -100
        )

        # 确保每条样本在 Shift 后仍有有效目标
        assert torch.all(
            (batch["labels"][:, 1:] != -100)
            .sum(dim=1) > 0
        )

        return batch


collator = SFTDataCollator(
    pad_token_id=tokenizer.pad_token_id
)


# ==========================================================
# 7. 创建 DataLoader
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
    num_workers=0,
    generator=torch.Generator().manual_seed(SEED)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
    num_workers=0
)

print("\n[5] DataLoader")
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ==========================================================
# 8. 查看第一个训练 Batch
# ==========================================================

batch = next(iter(train_loader))

print("\n[6] First Training Batch")

print("Sample IDs:", batch["sample_ids"])
print("Input IDs shape:", tuple(batch["input_ids"].shape))
print("Attention Mask shape:",
      tuple(batch["attention_mask"].shape))
print("Labels shape:", tuple(batch["labels"].shape))

for i in range(batch["input_ids"].size(0)):

    input_length = (
        batch["attention_mask"][i].sum().item()
    )

    valid_labels = (
        batch["labels"][i] != -100
    ).sum().item()

    padding_tokens = (
        batch["attention_mask"][i] == 0
    ).sum().item()

    print(f"\nBatch Sample {i}")
    print("Real tokens:", input_length)
    print("Answer target tokens:", valid_labels)
    print("Padding tokens:", padding_tokens)

    answer_ids = batch["labels"][i][
        batch["labels"][i] != -100
    ].tolist()

    print(
        "Answer text:",
        tokenizer.decode(
            answer_ids,
            skip_special_tokens=False
        )
    )


# ==========================================================
# 9. 加载 Qwen 模型
# ==========================================================

print("\n[7] Loading Qwen")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    local_files_only=True
).to(DEVICE)

model.eval()


# ==========================================================
# 10. 模型前向传播与 Loss
# ==========================================================

model_batch = {
    "input_ids": batch["input_ids"].to(DEVICE),
    "attention_mask": batch["attention_mask"].to(DEVICE),
    "labels": batch["labels"].to(DEVICE)
}

with torch.inference_mode():
    outputs = model(
        **model_batch,
        use_cache=False
    )

print("\n[8] Model Forward")
print("Logits shape:", tuple(outputs.logits.shape))
print("Built-in Loss:", outputs.loss.item())


# ==========================================================
# 11. 手动验证 Causal LM Loss
# ==========================================================

shift_logits = outputs.logits[:, :-1, :]
shift_labels = model_batch["labels"][:, 1:]

manual_loss = F.cross_entropy(
    shift_logits.reshape(
        -1,
        shift_logits.size(-1)
    ),
    shift_labels.reshape(-1),
    ignore_index=-100
)

effective_tokens = (
    shift_labels != -100
).sum().item()

print("\n[9] Loss Validation")
print("Manual Loss:", manual_loss.item())
print("Effective loss tokens:", effective_tokens)

print(
    "Loss Match:",
    torch.allclose(
        outputs.loss,
        manual_loss,
        atol=1e-5
    )
)


# ==========================================================
# 12. 检查 Validation DataLoader
# ==========================================================

val_batch = next(iter(val_loader))

print("\n[10] Validation Batch")
print("Sample IDs:", val_batch["sample_ids"])
print("Input IDs shape:",
      tuple(val_batch["input_ids"].shape))

print(
    "Peak GPU memory:",
    round(
        torch.cuda.max_memory_allocated() / 1024**3,
        3
    ),
    "GiB"
)

print("\nDay 5 Experiment completed!")
