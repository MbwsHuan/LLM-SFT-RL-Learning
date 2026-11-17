
import os
import math
import random
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Day 6: SFT Training Loop
# Full Fine-tuning + Gradient Accumulation
# ============================================================

SEED = 42

MODEL_PATH = os.path.expanduser(
    "~/Levi/models/Qwen2.5-0.5B-Instruct"
)

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2
EPOCHS = 1

LEARNING_RATE = 2e-5
MAX_LENGTH = 256

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("=" * 60)
print("Day 6: SFT Training Loop")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 1. Load Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n[1] Tokenizer loaded")
print("Pad token ID:", tokenizer.pad_token_id)


# ============================================================
# 2. Raw Training Samples
# ============================================================

samples = [
    {
        "question": "什么是监督微调SFT？",
        "answer": "SFT是使用标注数据对预训练模型进行监督微调，使模型学习按照用户指令生成回答。"
    },
    {
        "question": "什么是反向传播？",
        "answer": "反向传播通过链式法则计算损失函数对模型参数的梯度，为优化器更新参数提供依据。"
    },
    {
        "question": "什么是Logits？",
        "answer": "Logits是语言模型输出的原始预测分数，可以通过Softmax转换为概率分布。"
    },
    {
        "question": "什么是AdamW？",
        "answer": "AdamW是一种自适应优化器，它结合一阶和二阶动量估计，并采用解耦权重衰减。"
    },
    {
        "question": "什么是Label Masking？",
        "answer": "Label Masking通过将不需要训练的位置设置为-100，使这些位置不参与交叉熵损失。"
    },
    {
        "question": "什么是Gradient Accumulation？",
        "answer": "梯度累积通过多个小批次累积梯度，再执行一次参数更新，从而模拟更大的有效Batch Size。"
    },
    {
        "question": "什么是Learning Rate？",
        "answer": "Learning Rate控制优化器每次更新模型参数的步长，是影响训练稳定性和收敛速度的重要超参数。"
    },
    {
        "question": "什么是Validation Loss？",
        "answer": "Validation Loss是在验证数据上计算的损失，用于监控模型的泛化表现。"
    }
]


# ============================================================
# 3. Dataset
# ============================================================

class SFTDataset(Dataset):

    def __init__(self, samples, tokenizer):
        self.data = []

        for item in samples:

            messages = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": item["question"]
                },
                {
                    "role": "assistant",
                    "content": item["answer"]
                }
            ]

            prompt_messages = messages[:-1]

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False
            )["input_ids"]

            full_ids = tokenizer(
                full_text,
                add_special_tokens=False
            )["input_ids"]

            # Keep the full sequence within MAX_LENGTH.
            # The prompt must not consume the entire sequence.
            if len(full_ids) > MAX_LENGTH:
                raise ValueError(
                    "Sample exceeds MAX_LENGTH. "
                    "Increase MAX_LENGTH or shorten the sample."
                )

            if len(prompt_ids) >= len(full_ids):
                raise ValueError("Invalid prompt/answer boundary.")

            # Verify that prompt IDs are the prefix of full IDs.
            assert full_ids[:len(prompt_ids)] == prompt_ids

            labels = (
                [-100] * len(prompt_ids)
                + full_ids[len(prompt_ids):]
            )

            self.data.append({
                "input_ids": full_ids,
                "labels": labels
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = SFTDataset(samples, tokenizer)

indices = list(range(len(dataset)))
random.Random(SEED).shuffle(indices)

train_indices = indices[:6]
val_indices = indices[6:]

from torch.utils.data import Subset

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

print("\n[2] Dataset")
print("Total:", len(dataset))
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))


# ============================================================
# 4. Data Collator
# ============================================================

def collate_fn(features):

    max_len = max(
        len(item["input_ids"])
        for item in features
    )

    input_ids = []
    attention_masks = []
    labels = []

    for item in features:

        ids = item["input_ids"]
        target = item["labels"]

        padding_len = max_len - len(ids)

        input_ids.append(
            ids + [tokenizer.pad_token_id] * padding_len
        )

        attention_masks.append(
            [1] * len(ids) + [0] * padding_len
        )

        labels.append(
            target + [-100] * padding_len
        )

    return {
        "input_ids": torch.tensor(
            input_ids, dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            attention_masks, dtype=torch.long
        ),
        "labels": torch.tensor(
            labels, dtype=torch.long
        )
    }


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

print("\n[3] DataLoader")
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ============================================================
# 5. Load Model
# ============================================================

print("\n[4] Loading Model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

model.config.use_cache = False

# Reduce activation memory during full fine-tuning.
model.gradient_checkpointing_enable()

model.to(DEVICE)

print("Model loaded:", model.__class__.__name__)

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)

assert total_params == trainable_params, (
    "This experiment expects full fine-tuning."
)


# ============================================================
# 6. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
    foreach=False
)

print("\n[5] Optimizer")
print("Optimizer: AdamW")
print("Learning rate:", LEARNING_RATE)


# ============================================================
# 7. Check Initial Parameters
# ============================================================

first_param = next(
    p for p in model.parameters()
    if p.requires_grad
)

initial_param = (
    first_param.detach().view(-1)[:100].clone()
)


# ============================================================
# 8. Training Loop
# ============================================================

print("\n[6] Training")

optimizer_updates = 0
optimizer.zero_grad(set_to_none=True)

model.train()

for epoch in range(EPOCHS):

    epoch_loss = 0.0

    total_batches = len(train_loader)

    for step, batch in enumerate(train_loader):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )

        loss = outputs.loss

        # Accumulate the mean loss gradients.
        # Also handle a final incomplete accumulation window.
        window_start = (
            step // GRAD_ACCUM_STEPS
        ) * GRAD_ACCUM_STEPS

        current_window_size = min(
            GRAD_ACCUM_STEPS,
            total_batches - window_start
        )

        scaled_loss = loss / current_window_size

        # Backward
        scaled_loss.backward()

        epoch_loss += loss.item()

        # Update after the accumulation window is complete.
        should_update = (
            (step + 1) % GRAD_ACCUM_STEPS == 0
            or step + 1 == total_batches
        )

        if should_update:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_updates += 1

        print(
            f"Epoch {epoch + 1} | "
            f"Batch {step + 1}/{total_batches} | "
            f"Loss: {loss.item():.4f} | "
            f"Updates: {optimizer_updates}"
        )

        del outputs, loss, scaled_loss
        del input_ids, attention_mask, labels

    avg_train_loss = epoch_loss / total_batches

    print(
        f"\nEpoch {epoch + 1} Average Train Loss: "
        f"{avg_train_loss:.4f}"
    )


# ============================================================
# 9. Validation
# ============================================================

print("\n[7] Validation")

model.eval()

val_losses = []

with torch.inference_mode():

    for batch in val_loader:

        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            labels=batch["labels"].to(DEVICE),
            use_cache=False
        )

        val_losses.append(outputs.loss.item())

        del outputs

avg_val_loss = sum(val_losses) / len(val_losses)

print("Validation Loss:", avg_val_loss)


# ============================================================
# 10. Verify Parameter Updates
# ============================================================

final_param = (
    first_param.detach().view(-1)[:100]
)

param_diff = (
    final_param - initial_param
).abs().max().item()

print("\n[8] Parameter Update Verification")

print("Initial parameter sample:", initial_param[:5].tolist())
print("Final parameter sample:", final_param[:5].tolist())
print("Max parameter difference:", param_diff)

print("Optimizer updates:", optimizer_updates)

assert optimizer_updates == 3
assert param_diff > 0

print("Parameter update: SUCCESS")


# ============================================================
# 11. GPU Memory
# ============================================================

if torch.cuda.is_available():

    torch.cuda.synchronize()

    print("\n[9] GPU Memory")

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

# ============================================================
# 12. Save Fine-tuned Model
# ============================================================

SAVE_PATH = os.path.expanduser(
    "~/Levi/checkpoints/day6_qwen_full_ft"
)

os.makedirs(SAVE_PATH, exist_ok=True)

model.save_pretrained(
    SAVE_PATH,
    safe_serialization=True
)

tokenizer.save_pretrained(SAVE_PATH)

print("\n[10] Model Saved")
print("Save path:", SAVE_PATH)

print("\nDay 6 Experiment completed successfully!")
