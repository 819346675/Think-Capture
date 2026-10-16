#!/usr/bin/env python3
"""
Merge positive/negative qualitative data to construct GRPO Stage 2 training set.

Inputs:
  - grpo_stage2_positive_quality.jsonl  (Positive: baseline incorrect, correct after adding 3D)
  - grpo_stage2_negative_quality.jsonl  (Negative: baseline correct, incorrect after adding 3D)

Outputs:
  - grpo_stage2_combined.jsonl  (Combined + shuffled)

Usage:
  python prepare_data.py [--output OUTPUT] [--max_pos N] [--max_neg N] [--seed SEED]
"""

import json
import random
import argparse
from pathlib import Path


def load_jsonl(path: Path, min_images: int = 2) -> list:
    """Load JSONL files, filter out single-frame samples."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
            # Required fields
            required = ["question", "images", "formatted_3d", "gt_answer", "sample_type"]
            if not all(k in sample for k in required):
                continue
            # Filter single frame (at least 2 frames required for multi-view value)
            if len(sample.get("images", [])) < min_images:
                continue
            # Check image existence
            if not all(Path(p).exists() for p in sample["images"]):
                continue
            rows.append(sample)
    print(f"{path}: retained {len(rows)} valid multi-image records")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Combine GRPO Stage 2 positive and negative data")
    parser.add_argument("--positive", type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent
                                    / "data" / "grpo_aligned" / "grpo_stage2_positive_quality.jsonl"))
    parser.add_argument("--negative", type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent
                                    / "data" / "grpo_aligned" / "grpo_stage2_negative_quality.jsonl"))
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).resolve().parent / "grpo_stage2_combined.jsonl"))
    parser.add_argument("--max_pos", type=int, default=-1, help="Max positive sample limit, -1 for all")
    parser.add_argument("--max_neg", type=int, default=-1, help="Max negative sample limit, -1 for all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading positive samples: {args.positive}")
    pos_rows = load_jsonl(Path(args.positive))
    print(f"  Valid positive samples: {len(pos_rows)}")

    print(f"Loading negative samples: {args.negative}")
    neg_rows = load_jsonl(Path(args.negative))
    print(f"  Valid negative samples: {len(neg_rows)}")

    rng = random.Random(args.seed)

    if args.max_pos > 0 and args.max_pos < len(pos_rows):
        rng.shuffle(pos_rows)
        pos_rows = pos_rows[:args.max_pos]
        print(f"  Sampled positive: {len(pos_rows)}")

    if args.max_neg > 0 and args.max_neg < len(neg_rows):
        rng.shuffle(neg_rows)
        neg_rows = neg_rows[:args.max_neg]
        print(f"  Sampled negative: {len(neg_rows)}")

    combined = pos_rows + neg_rows
    rng.shuffle(combined)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in combined:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    pos_count = sum(1 for s in combined if s["sample_type"] == "positive")
    neg_count = len(combined) - pos_count
    print(f"\nWritten to: {output_path}")
    print(f"Total: {len(combined)} items (Positive {pos_count}, Negative {neg_count})")


if __name__ == "__main__":
    main()
