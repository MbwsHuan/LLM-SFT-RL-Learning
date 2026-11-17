
import json
import os
from collections import Counter
from transformers import AutoTokenizer


# ============================================================
# Day 7: Dataset Quality Audit
# ============================================================

DATA_DIR = os.path.expanduser(
    "~/Levi/datasets/align_anything_zh/raw/data"
)

MODEL_PATH = os.path.expanduser(
    "~/Levi/models/Qwen2.5-0.5B-Instruct"
)

REPORT_DIR = os.path.expanduser(
    "~/Levi/datasets/align_anything_zh/processed"
)

os.makedirs(REPORT_DIR, exist_ok=True)


def percentile(values, p):
    if not values:
        return 0
    values = sorted(values)
    index = round((len(values) - 1) * p / 100)
    return values[index]


def audit_file(path, tokenizer):
    print("\n" + "=" * 60)
    print("Auditing:", path)
    print("=" * 60)

    total = 0
    malformed = 0
    missing_fields = 0
    empty_prompt = 0
    empty_response = 0

    exact_pairs = set()
    prompts = set()

    prompt_lengths = []
    answer_lengths = []
    total_lengths = []

    duplicate_pairs = 0
    duplicate_prompts = 0
    template_prefix_mismatch = 0

    with open(path, "r", encoding="utf-8") as f:

        for line in f:
            total += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            if not isinstance(sample, dict):
                malformed += 1
                continue

            if "prompt" not in sample or "response" not in sample:
                missing_fields += 1
                continue

            prompt = sample["prompt"]
            response = sample["response"]

            if not isinstance(prompt, str) or not isinstance(response, str):
                missing_fields += 1
                continue

            prompt = prompt.strip()
            response = response.strip()

            if not prompt:
                empty_prompt += 1
                continue

            if not response:
                empty_response += 1
                continue

            # Exact duplicate detection
            pair = (prompt, response)

            if pair in exact_pairs:
                duplicate_pairs += 1
            else:
                exact_pairs.add(pair)

            # Duplicate prompt detection
            if prompt in prompts:
                duplicate_prompts += 1
            else:
                prompts.add(prompt)

            # Character lengths
            prompt_lengths.append(len(prompt))
            answer_lengths.append(len(response))

            # Qwen Chat Template tokenization
            prompt_messages = [
                {"role": "user", "content": prompt}
            ]

            full_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]

            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True
            )

            full_ids = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False
            )

            if full_ids[:len(prompt_ids)] != prompt_ids:
                template_prefix_mismatch += 1

            prompt_tokens = len(prompt_ids)
            total_tokens = len(full_ids)
            answer_tokens = max(0, total_tokens - prompt_tokens)

            total_lengths.append(total_tokens)
            answer_lengths_token = answer_tokens

            # Store token statistics separately
            if not hasattr(audit_file, "_answer_token_lengths"):
                audit_file._answer_token_lengths = {}

            audit_file._answer_token_lengths.setdefault(path, []).append(
                answer_lengths_token
            )

    answer_token_lengths = audit_file._answer_token_lengths.get(path, [])

    valid = len(total_lengths)

    print("\n[1] Basic Statistics")
    print("Raw records:", total)
    print("Valid records:", valid)
    print("Malformed JSON:", malformed)
    print("Missing/invalid fields:", missing_fields)
    print("Empty prompts:", empty_prompt)
    print("Empty responses:", empty_response)

    print("\n[2] Duplicate Statistics")
    print("Duplicate prompt-response pairs:", duplicate_pairs)
    print("Repeated prompts:", duplicate_prompts)

    print("\n[3] Prompt Character Length")
    print("Mean:", round(sum(prompt_lengths) / max(1, len(prompt_lengths)), 2))
    print("P50:", percentile(prompt_lengths, 50))
    print("P95:", percentile(prompt_lengths, 95))
    print("Max:", max(prompt_lengths, default=0))

    print("\n[4] Answer Character Length")
    print("Mean:", round(sum(answer_lengths) / max(1, len(answer_lengths)), 2))
    print("P50:", percentile(answer_lengths, 50))
    print("P95:", percentile(answer_lengths, 95))
    print("Max:", max(answer_lengths, default=0))

    print("\n[5] Token Length Statistics")
    print("Total tokens - P50:", percentile(total_lengths, 50))
    print("Total tokens - P95:", percentile(total_lengths, 95))
    print("Total tokens - P99:", percentile(total_lengths, 99))
    print("Total tokens - Max:", max(total_lengths, default=0))

    print("Answer tokens - P50:", percentile(answer_token_lengths, 50))
    print("Answer tokens - P95:", percentile(answer_token_lengths, 95))
    print("Answer tokens - Max:", max(answer_token_lengths, default=0))

    print("Template prefix mismatches:", template_prefix_mismatch)

    return {
        "file": os.path.basename(path),
        "raw_records": total,
        "valid_records": valid,
        "malformed": malformed,
        "missing_fields": missing_fields,
        "empty_prompt": empty_prompt,
        "empty_response": empty_response,
        "duplicate_pairs": duplicate_pairs,
        "duplicate_prompts": duplicate_prompts,
        "template_prefix_mismatch": template_prefix_mismatch,
        "total_tokens_p50": percentile(total_lengths, 50),
        "total_tokens_p95": percentile(total_lengths, 95),
        "total_tokens_p99": percentile(total_lengths, 99),
        "total_tokens_max": max(total_lengths, default=0),
        "answer_tokens_p50": percentile(answer_token_lengths, 50),
        "answer_tokens_p95": percentile(answer_token_lengths, 95),
        "answer_tokens_max": max(answer_token_lengths, default=0),
    }


def main():
    print("=" * 60)
    print("Day 7: SFT Dataset Quality Audit")
    print("=" * 60)

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )

    print("Tokenizer loaded.")
    print("Vocabulary:", tokenizer.vocab_size)

    reports = []

    for filename in ["train.jsonl", "test.jsonl"]:
        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        reports.append(audit_file(path, tokenizer))

    report_path = os.path.join(
        REPORT_DIR, "dataset_audit_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Audit completed!")
    print("Report saved to:", report_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
