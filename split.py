"""Stratified split of all.jsonl into train/valid/test for MLX-LM.

MLX-LM expects exactly these names in the --data dir:
  train.jsonl  — main training set
  valid.jsonl  — validation set (loss tracked during training)
  test.jsonl   — held-out test set (used by baseline.py / infer_mlx.py)

Stratification: the proportion of each label is preserved across all splits.
Seed is fixed → split is reproducible.

Default fractions: train=0.70, valid=0.15, test=0.15.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 42
TRAIN_FRAC = 0.70
VALID_FRAC = 0.15


def main() -> int:
    src = DATA / "all.jsonl"
    if not src.exists():
        print(f"FATAL: {src} not found — run build_dataset.py first", file=sys.stderr)
        return 1

    by_label: dict[str, list[dict]] = defaultdict(list)
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = (obj.get("_meta") or {}).get("expected") or "unknown"
            by_label[label].append(obj)

    rng = random.Random(SEED)
    train: list[dict] = []
    valid: list[dict] = []
    test: list[dict] = []
    for _label, items in by_label.items():
        items_shuffled = items[:]
        rng.shuffle(items_shuffled)
        n = len(items_shuffled)
        n_train = round(n * TRAIN_FRAC)
        n_valid = max(1, round(n * VALID_FRAC))
        train.extend(items_shuffled[:n_train])
        valid.extend(items_shuffled[n_train:n_train + n_valid])
        test.extend(items_shuffled[n_train + n_valid:])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)

    for name, bucket in (("train", train), ("valid", valid), ("test", test)):
        path = DATA / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in bucket:
                # strip _meta for the actual training files
                clean = {"messages": ex["messages"]}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
        labels = Counter((ex.get("_meta") or {}).get("expected", "?") for ex in bucket)
        print(f"  {name:>5}: {len(bucket):>3}  -> {dict(labels)}  -> {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
