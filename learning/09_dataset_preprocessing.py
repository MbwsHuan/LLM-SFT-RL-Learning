import os
import json
import math
from collections import Counter

from transformers import AutoTokenizer

# ============================================================
# Data Adjustment: 09 Dataset Preprocessing
# ============================================================

MODEL_PATH = os.path.expanduser(
    "~/Levi/models/Qwen2.5-0.5B-Instruct"
)

DATA_DIR = os.path.expanduser(
    "~/Levi/datasets/align_anything_zh/processed"
)

OUTPUT_DIR = os.path.join(DATA_DIR, "tokenized")

MAX_LENGTH = 1024

SPLITS = ["train", "validation", "test"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0

    index = math.ceil(p * len(values)) - 1
    return values[max(0, index)]


def process_split(tokenizer, split):
    input_path = os.path.join(
        DATA_DIR, f"{split}.jsonl"
    )

    output_path = os.path.join(
        OUTPUT_DIR, f"{split}_tokenized.jsonl"
    )

    print("\n" + "=" * 65)
    print(f"Processing: {split}")
    print("=" * 65)

    total_lengths = []
    answer_lengths = []

    stats = Counter()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_number, line in enumerate(fin, 1):

            item = json.loads(line)

            prompt = item["prompt"]
            response = item["response"]

            # ------------------------------------------------
            # 1. Construct Prompt
            # ------------------------------------------------

            prompt_messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # ------------------------------------------------
            # 2. Construct Full Conversation
            # ------------------------------------------------

            full_messages = [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]

            # ------------------------------------------------
            # 3. Tokenize Prompt
            # ------------------------------------------------

            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True
            )

            # ------------------------------------------------
            # 4. Tokenize Full Conversation
            # ------------------------------------------------

            input_ids = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False
            )

            prompt_length = len(prompt_ids)
            total_length = len(input_ids)

            # ------------------------------------------------
            # 5. Verify Prompt Prefix
            # ------------------------------------------------

            if input_ids[:prompt_length] != prompt_ids:
                raise ValueError(
                    f"Prompt prefix mismatch at line {line_number}"
                )

            if total_length <= prompt_length:
                raise ValueError(
                    f"Empty answer tokens at line {line_number}"
                )

            # ------------------------------------------------
            # 6. Construct Labels
            # ------------------------------------------------

            labels = (
                [-100] * prompt_length
                + input_ids[prompt_length:]
            )

            # ------------------------------------------------
            # 7. Attention Mask
            # ------------------------------------------------

            attention_mask = [1] * total_length

            # ------------------------------------------------
            # 8. Verify EOS
            # ------------------------------------------------

            # Qwen Chat Template appends a newline token
            # after the assistant EOS token.
            # Therefore, check the final assistant terminator
            # rather than checking input_ids[-1].

            has_eos = (
                len(input_ids) >= 2
                and input_ids[-2] == tokenizer.eos_token_id
            )

            if not has_eos:
                stats["missing_eos"] += 1

            # ------------------------------------------------
            # 9. Statistics
            # ------------------------------------------------

            answer_length = total_length - prompt_length

            total_lengths.append(total_length)
            answer_lengths.append(answer_length)

            stats["total"] += 1

            if total_length > MAX_LENGTH:
                stats["over_max_length"] += 1

            if total_length > 2048:
                stats["over_2048"] += 1

            if total_length > 4096:
                stats["over_4096"] += 1

            effective_tokens = sum(
                x != -100 for x in labels
            )

            if effective_tokens != answer_length:
                raise ValueError(
                    f"Label masking error at line {line_number}"
                )

            # ------------------------------------------------
            # 10. Save Tokenized Sample
            # ------------------------------------------------

            output_item = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "total_length": total_length,
                "prompt_length": prompt_length,
                "answer_length": answer_length,
                "has_eos": has_eos
            }

            fout.write(
                json.dumps(output_item) + "\n"
            )

            if line_number % 10000 == 0:
                print(
                    f"Processed {line_number} samples..."
                )

    # --------------------------------------------------------
    # 11. Final Statistics
    # --------------------------------------------------------

    print("\n[Statistics]")
    print("Total samples:", stats["total"])

    print(
        "Total tokens P50:",
        percentile(total_lengths, 0.50)
    )

    print(
        "Total tokens P95:",
        percentile(total_lengths, 0.95)
    )

    print(
        "Total tokens P99:",
        percentile(total_lengths, 0.99)
    )

    print(
        "Total tokens Max:",
        max(total_lengths)
    )

    print(
        "Answer tokens P50:",
        percentile(answer_lengths, 0.50)
    )

    print(
        "Answer tokens P95:",
        percentile(answer_lengths, 0.95)
    )

    print(
        "Answer tokens Max:",
        max(answer_lengths)
    )

    print(
        f"Over {MAX_LENGTH} tokens:",
        stats["over_max_length"]
    )

    print(
        "Over 2048 tokens:",
        stats["over_2048"]
    )

    print(
        "Over 4096 tokens:",
        stats["over_4096"]
    )

    print("Missing EOS:", stats["missing_eos"])

    print("Saved:", output_path)

    return {
        "samples": stats["total"],
        "total_length_p50": percentile(total_lengths, 0.50),
        "total_length_p95": percentile(total_lengths, 0.95),
        "total_length_p99": percentile(total_lengths, 0.99),
        "total_length_max": max(total_lengths),
        "answer_length_p50": percentile(answer_lengths, 0.50),
        "answer_length_p95": percentile(answer_lengths, 0.95),
        "answer_length_max": max(answer_lengths),
        "over_1024": stats["over_max_length"],
        "over_2048": stats["over_2048"],
        "over_4096": stats["over_4096"],
        "missing_eos": stats["missing_eos"]
    }


def main():

    print("=" * 65)
    print("Data Adjustment: Tokenization and Preprocessing")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load Tokenizer
    # --------------------------------------------------------

    print("\n[1] Loading Tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    print("Tokenizer:", tokenizer.__class__.__name__)
    print("Vocab size:", tokenizer.vocab_size)
    print("Pad token ID:", tokenizer.pad_token_id)
    print("EOS token ID:", tokenizer.eos_token_id)
    print(
        "Chat template available:",
        tokenizer.chat_template is not None
    )

    assert tokenizer.chat_template is not None

    # --------------------------------------------------------
    # 2. Process All Splits
    # --------------------------------------------------------

    reports = {}

    for split in SPLITS:
        reports[split] = process_split(
            tokenizer, split
        )

    # --------------------------------------------------------
    # 3. Save Report
    # --------------------------------------------------------

    report_path = os.path.join(
        OUTPUT_DIR,
        "preprocessing_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            reports,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 65)
    print("Preprocessing Completed!")
    print("=" * 65)

    for split, report in reports.items():
        print(
            f"{split}: {report['samples']} samples"
        )

    print("Output directory:", OUTPUT_DIR)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
