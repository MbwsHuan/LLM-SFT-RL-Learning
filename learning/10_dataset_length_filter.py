import json
import os

BASE = os.path.expanduser(
    "~/Levi/datasets/align_anything_zh/processed"
)

INPUT_DIR = os.path.join(BASE, "tokenized")
OUTPUT_DIR = os.path.join(BASE, "filtered")

MAX_LENGTH = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in ["train", "validation", "test"]:

    input_path = os.path.join(
        INPUT_DIR, f"{split}_tokenized.jsonl"
    )

    output_path = os.path.join(
        OUTPUT_DIR, f"{split}_filtered.jsonl"
    )

    total = 0
    kept = 0
    removed = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:

            item = json.loads(line)
            total += 1

            ids = item["input_ids"]
            labels = item["labels"]

            valid = (
                len(ids) == len(labels)
                and len(ids) == len(item["attention_mask"])
                and len(ids) <= MAX_LENGTH
                and item["has_eos"]
                and item["answer_length"] > 0
                and any(x != -100 for x in labels)
            )

            if valid:
                fout.write(
                    json.dumps(item, ensure_ascii=False) + "\n"
                )
                kept += 1
            else:
                removed += 1

    print(f"\n[{split}]")
    print("Original:", total)
    print("Kept:", kept)
    print("Removed:", removed)
    print("Saved:", output_path)

print("\nFiltering completed!")
