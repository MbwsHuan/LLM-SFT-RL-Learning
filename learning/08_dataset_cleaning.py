import os
import json
import random
import hashlib
from collections import Counter

# ============================================================
# Data Adjustment: 08 Dataset Cleaning and Splitting
# ============================================================

RAW_DIR = os.path.expanduser(
    "~/Levi/datasets/align_anything_zh/raw/data"
)

OUTPUT_DIR = os.path.expanduser(
    "~/Levi/datasets/align_anything_zh/processed"
)

SEED = 42
VALIDATION_RATIO = 0.02

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_jsonl(path):
    records = []
    stats = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            stats["raw"] += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_json"] += 1
                continue

            if not isinstance(item, dict):
                stats["invalid_structure"] += 1
                continue

            if "prompt" not in item or "response" not in item:
                stats["missing_fields"] += 1
                continue

            if not isinstance(item["prompt"], str):
                stats["invalid_prompt_type"] += 1
                continue

            if not isinstance(item["response"], str):
                stats["invalid_response_type"] += 1
                continue

            prompt = item["prompt"].strip()
            response = item["response"].strip()

            if not prompt:
                stats["empty_prompt"] += 1
                continue

            if not response:
                stats["empty_response"] += 1
                continue

            records.append({
                "prompt": prompt,
                "response": response
            })

            stats["valid"] += 1

    return records, stats


def pair_hash(item):
    content = (
        item["prompt"] + "\0" + item["response"]
    )
    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def deduplicate(records):
    seen = set()
    unique_records = []
    duplicate_count = 0

    for item in records:
        key = pair_hash(item)

        if key in seen:
            duplicate_count += 1
            continue

        seen.add(key)
        unique_records.append(item)

    return unique_records, duplicate_count


def remove_test_overlap(train_records, test_records):
    test_hashes = {
        pair_hash(item) for item in test_records
    }

    filtered = []
    overlap_count = 0

    for item in train_records:
        if pair_hash(item) in test_hashes:
            overlap_count += 1
            continue

        filtered.append(item)

    return filtered, overlap_count


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )


def main():
    print("=" * 65)
    print("Data Adjustment: Dataset Cleaning and Splitting")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Read original datasets
    # --------------------------------------------------------

    print("\n[1] Reading Raw Datasets")

    train_raw, train_stats = read_jsonl(
        os.path.join(RAW_DIR, "train.jsonl")
    )

    test_raw, test_stats = read_jsonl(
        os.path.join(RAW_DIR, "test.jsonl")
    )

    print("Train raw records:", train_stats["raw"])
    print("Train valid records:", len(train_raw))
    print("Test raw records:", test_stats["raw"])
    print("Test valid records:", len(test_raw))

    # --------------------------------------------------------
    # 2. Deduplicate training data
    # --------------------------------------------------------

    print("\n[2] Removing Exact Duplicates")

    train_clean, duplicate_count = deduplicate(train_raw)

    print("Before deduplication:", len(train_raw))
    print("Duplicates removed:", duplicate_count)
    print("After deduplication:", len(train_clean))

    # --------------------------------------------------------
    # 3. Check train-test leakage
    # --------------------------------------------------------

    print("\n[3] Checking Train-Test Overlap")

    train_clean, overlap_count = remove_test_overlap(
        train_clean, test_raw
    )

    print("Exact overlapping pairs removed from train:",
          overlap_count)
    print("Remaining training records:", len(train_clean))

    # Do not deduplicate or randomly resplit official test data.
    test_clean = test_raw

    # --------------------------------------------------------
    # 4. Deterministic train-validation split
    # --------------------------------------------------------

    print("\n[4] Train / Validation Split")

    rng = random.Random(SEED)

    indices = list(range(len(train_clean)))
    rng.shuffle(indices)

    validation_size = round(
        len(indices) * VALIDATION_RATIO
    )

    validation_indices = set(indices[:validation_size])

    train_records = [
        train_clean[i]
        for i in range(len(train_clean))
        if i not in validation_indices
    ]

    validation_records = [
        train_clean[i]
        for i in range(len(train_clean))
        if i in validation_indices
    ]

    print("Random seed:", SEED)
    print("Validation ratio:", VALIDATION_RATIO)
    print("Train:", len(train_records))
    print("Validation:", len(validation_records))
    print("Test:", len(test_clean))

    # --------------------------------------------------------
    # 5. Save datasets
    # --------------------------------------------------------

    print("\n[5] Saving Processed Datasets")

    save_jsonl(
        train_records,
        os.path.join(OUTPUT_DIR, "train.jsonl")
    )

    save_jsonl(
        validation_records,
        os.path.join(OUTPUT_DIR, "validation.jsonl")
    )

    save_jsonl(
        test_clean,
        os.path.join(OUTPUT_DIR, "test.jsonl")
    )

    # --------------------------------------------------------
    # 6. Save cleaning report
    # --------------------------------------------------------

    report = {
        "seed": SEED,
        "validation_ratio": VALIDATION_RATIO,
        "train_raw": train_stats["raw"],
        "train_valid_before_dedup": len(train_raw),
        "train_duplicates_removed": duplicate_count,
        "train_test_exact_overlaps_removed": overlap_count,
        "train_final": len(train_records),
        "validation_final": len(validation_records),
        "test_raw": test_stats["raw"],
        "test_final": len(test_clean),
        "train_read_statistics": dict(train_stats),
        "test_read_statistics": dict(test_stats),
    }

    report_path = os.path.join(
        OUTPUT_DIR, "cleaning_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            report, f, ensure_ascii=False, indent=2
        )

    # --------------------------------------------------------
    # 7. Final verification
    # --------------------------------------------------------

    print("\n[6] Final Verification")

    print("Train file:", len(train_records))
    print("Validation file:", len(validation_records))
    print("Test file:", len(test_clean))

    assert len(train_records) + len(validation_records) == len(train_clean)
    assert len(test_clean) == len(test_raw)

    print("\nCleaning completed successfully!")
    print("Output directory:", OUTPUT_DIR)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
