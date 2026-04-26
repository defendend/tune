"""End-to-end confidence-aware evaluation harness.

Runs the 3-layer pipeline (constraint + logprobs + redundancy) over three
test categories and prints per-category metrics + per-stage call counts +
latency p50/p95 + dollar cost.

Categories (each is a JSONL of {"description","expected","note"} records):
  - correct.jsonl   clean inputs with known label
  - edge.jsonl      manually crafted borderline cases
  - noisy.jsonl     adversarial / malformed inputs

Usage:
  export OPENAI_API_KEY=sk-...
  python3 evaluate.py
  python3 evaluate.py --no-redundancy            # disable stage 3 to compare
  python3 evaluate.py --out evaluation.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = ROOT / "prompts" / "system.md"
CORRECT = ROOT / "correct.jsonl"
EDGE = ROOT / "edge.jsonl"
NOISY = ROOT / "noisy.jsonl"


def load_correct(n: int = 8) -> list[dict]:
    """Load clean cases. correct.jsonl is in OpenAI/MLX JSONL format
    (`messages: [system, user, assistant]`). We parse the gold verdict
    from the assistant message and treat it as the expected label."""
    items: list[dict] = []
    if not CORRECT.exists():
        return items
    for line in CORRECT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        msgs = obj.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        assistant_gold = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        if "Insufficient" in assistant_gold:
            expected = "insufficient"
        elif "Sufficient" in assistant_gold:
            expected = "sufficient"
        else:
            continue
        items.append({"description": user, "expected": expected, "note": "clean test case"})
        if len(items) >= n:
            break
    return items


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def run_category(name: str, items: list[dict], system_prompt: str, *,
                 model: str, enable_redundancy: bool, client) -> dict:
    from confidence import classify_with_confidence

    print(f"\n-- {name} -- ({len(items)} items)", file=sys.stderr)
    rows = []
    for i, it in enumerate(items, 1):
        res = classify_with_confidence(
            it["description"],
            system_prompt,
            client=client,
            model=model,
            enable_redundancy=enable_redundancy,
        )
        verdict_match = (res.verdict or "").lower() == (it["expected"] or "").lower()
        row = {
            "idx": i,
            "expected": it["expected"],
            "predicted": res.verdict,
            "status": res.status,
            "verdict_match": int(verdict_match) if res.verdict else 0,
            "confidence": res.confidence,
            "calls": res.calls,
            "latency_ms": res.latency_ms,
            "cost_usd": round(res.cost_usd, 6),
            "reason": res.reason,
            "note": it.get("note", ""),
            "raw_text": (res.raw_text or "")[:300],
        }
        rows.append(row)
        print(f"  [{i:>2}] expected={it['expected']:>13} pred={str(res.verdict):>13} "
              f"status={res.status:<7} conf={res.confidence:.3f} "
              f"calls={res.calls} {res.latency_ms}ms ${res.cost_usd:.5f} :: {res.reason}",
              file=sys.stderr)

    summary = summarize(rows)
    return {"category": name, "n": len(rows), "summary": summary, "rows": rows}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    statuses = Counter(r["status"] for r in rows)
    matched = sum(r["verdict_match"] for r in rows)
    matched_among_ok = sum(r["verdict_match"] for r in rows if r["status"] == "OK")
    n_ok = statuses.get("OK", 0)
    return {
        "status_counts": dict(statuses),
        "verdict_accuracy_overall": round(matched / len(rows), 3),
        "verdict_accuracy_when_OK": round(matched_among_ok / n_ok, 3) if n_ok else None,
        "calls_total": sum(r["calls"] for r in rows),
        "calls_avg": round(sum(r["calls"] for r in rows) / len(rows), 2),
        "latency_p50_ms": int(percentile([r["latency_ms"] for r in rows], 50)),
        "latency_p95_ms": int(percentile([r["latency_ms"] for r in rows], 95)),
        "cost_total_usd": round(sum(r["cost_usd"] for r in rows), 5),
        "cost_avg_usd": round(statistics.mean(r["cost_usd"] for r in rows), 6),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--no-redundancy", action="store_true")
    p.add_argument("--correct-n", type=int, default=8)
    p.add_argument("--out", default="evaluation.json")
    args = p.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("FATAL: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    if not SYSTEM_PROMPT.exists():
        print(f"FATAL: system prompt not found at {SYSTEM_PROMPT}", file=sys.stderr)
        return 1
    system_prompt = SYSTEM_PROMPT.read_text(encoding="utf-8")

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    categories = [
        ("correct", load_correct(args.correct_n)),
        ("edge", load_jsonl(EDGE)),
        ("noisy", load_jsonl(NOISY)),
    ]

    payload = {
        "model": args.model,
        "redundancy_enabled": not args.no_redundancy,
        "categories": [],
    }

    for cname, items in categories:
        if not items:
            print(f"  [warn] category {cname} empty, skipping", file=sys.stderr)
            continue
        result = run_category(cname, items, system_prompt,
                              model=args.model,
                              enable_redundancy=not args.no_redundancy,
                              client=client)
        payload["categories"].append(result)

    all_rows = [r for c in payload["categories"] for r in c["rows"]]
    payload["overall"] = summarize(all_rows)

    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== OVERALL ===\n" + json.dumps(payload["overall"], ensure_ascii=False, indent=2))
    print(f"\nSaved -> {args.out}")

    print("\n=== PER CATEGORY ===")
    for c in payload["categories"]:
        print(f"\n[{c['category']}] n={c['n']}")
        for k, v in c["summary"].items():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
