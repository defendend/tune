"""Build a fine-tuning dataset in OpenAI / MLX-LM JSONL format.

Reads:
  prompts/system.md  → system message
  examples.jsonl     → records {description, expected, assistant}

Writes:
  data/all.jsonl     → {"messages":[system, user, assistant]} per line

The "expected" field is preserved in `_meta` for stratified split later.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = ROOT / "prompts" / "system.md"
EXAMPLES = ROOT / "examples.jsonl"
OUT = ROOT / "data" / "all.jsonl"


def main() -> int:
    if not SYSTEM.exists():
        print(f"FATAL: {SYSTEM} not found", file=sys.stderr)
        return 1
    if not EXAMPLES.exists():
        print(f"FATAL: {EXAMPLES} not found", file=sys.stderr)
        return 1

    system = SYSTEM.read_text(encoding="utf-8").strip()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    seen_payload: dict[str, int] = {}
    out = []
    with EXAMPLES.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            description = (obj.get("description") or "").strip()
            expected = (obj.get("expected") or "").strip()
            assistant = (obj.get("assistant") or "").strip()
            if not description or not assistant or expected not in {"sufficient", "insufficient"}:
                print(f"  skip line {i}: missing fields", file=sys.stderr)
                continue
            key = description
            if key in seen_payload:
                print(f"  skip line {i}: duplicate of line {seen_payload[key]}", file=sys.stderr)
                continue
            seen_payload[key] = i
            out.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": description},
                    {"role": "assistant", "content": assistant},
                ],
                "_meta": {"expected": expected},
            })

    with OUT.open("w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    counts = {"sufficient": 0, "insufficient": 0}
    for ex in out:
        counts[ex["_meta"]["expected"]] += 1
    print(f"wrote {len(out)} examples (sufficient={counts['sufficient']}, insufficient={counts['insufficient']}) → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
