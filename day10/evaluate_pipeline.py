"""Evaluate the two-tier pipeline (micro + LLM fallback).

Compares against an "always LLM" baseline (multi_stage v2 from day 9) on the
same set of inputs. Reports:

  - % of requests handled by the micro-model alone
  - % escalated to the LLM fallback
  - Total LLM calls vs the always-LLM baseline (savings)
  - Average / total cost
  - Latency p50 / p95
  - Accuracy: pipeline overall vs micro-only verdict on the OK slice

Input categories (same as day-7):
  - correct.jsonl  — clean, labeled by gold assistant
  - edge.jsonl     — borderline / underspecified
  - noisy.jsonl    — empty / emoji / gibberish / prompt injection
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DAY7 = ROOT.parent / "day7"
DAY9 = ROOT.parent / "day9"
CORRECT = DAY7 / "correct.jsonl"
EDGE = DAY7 / "edge.jsonl"
NOISY = DAY7 / "noisy.jsonl"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DAY9))
from micro import MicroModel  # noqa: E402
from multi_stage import classify_multi_stage  # noqa: E402
from pipeline import classify_pipeline  # noqa: E402

from openai import OpenAI


def load_correct(n: int = 8) -> list[dict]:
    items = []
    if not CORRECT.exists():
        return items
    for line in CORRECT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        msgs = obj.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        gold = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        if "Insufficient" in gold:
            expected = "insufficient"
        elif "Sufficient" in gold:
            expected = "sufficient"
        else:
            continue
        items.append({"description": user, "expected": expected})
        if len(items) >= n:
            break
    return items


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def run_pipeline(client, micro, items, *, fallback: str = "multi_stage",
                 monolithic_system_prompt: str | None = None) -> list[dict]:
    rows = []
    for i, it in enumerate(items, 1):
        res = classify_pipeline(
            it["description"], client=client, micro=micro,
            fallback=fallback, monolithic_system_prompt=monolithic_system_prompt,
        )
        match = (res.verdict or "").lower() == it["expected"].lower()
        row = {
            "idx": i, "expected": it["expected"], "predicted": res.verdict,
            "verdict_match": int(match) if res.verdict else 0,
            "status": res.status, "handled_by": res.handled_by,
            "micro_status": res.micro.status,
            "micro_verdict": res.micro.verdict,
            "micro_confidence": res.micro.confidence,
            "calls": res.calls_total,
            "llm_calls": res.llm_calls,
            "cost_usd": round(res.cost_usd_total, 6),
            "latency_ms": res.latency_ms_total,
        }
        rows.append(row)
        marker = "      " if res.handled_by == "micro" else "->LLM"
        print(f"  [{i:>2}] {marker} pred={str(res.verdict or 'None'):>13} match={row['verdict_match']} "
              f"micro_conf={res.micro.confidence:.3f} calls={res.calls_total} ${res.cost_usd_total:.6f} {res.latency_ms_total}ms",
              file=sys.stderr)
    return rows


def run_always_llm(client, items) -> list[dict]:
    """Baseline: always call multi_stage v2."""
    rows = []
    for i, it in enumerate(items, 1):
        res = classify_multi_stage(it["description"], client=client)
        match = (res.verdict or "").lower() == it["expected"].lower()
        row = {
            "idx": i, "expected": it["expected"], "predicted": res.verdict,
            "verdict_match": int(match) if res.verdict else 0,
            "status": "OK" if res.verdict else "FAIL",
            "handled_by": "llm_only",
            "calls": res.calls,
            "llm_calls": res.calls,
            "cost_usd": round(res.cost_usd_total, 6),
            "latency_ms": res.latency_ms_total,
        }
        rows.append(row)
        print(f"  [{i:>2}] always_llm pred={str(res.verdict or 'None'):>13} match={row['verdict_match']} calls={res.calls} ${res.cost_usd_total:.6f} {res.latency_ms_total}ms",
              file=sys.stderr)
    return rows


def summarize(rows: list[dict], name: str) -> dict:
    if not rows:
        return {}
    n = len(rows)
    matched = sum(r["verdict_match"] for r in rows)
    by_handler = Counter(r.get("handled_by", "") for r in rows)
    return {
        "pipeline": name,
        "n": n,
        "verdict_accuracy": round(matched / n, 3),
        "handled_by_counts": dict(by_handler),
        "micro_share": round(by_handler.get("micro", 0) / n, 3),
        "fallback_share": round(by_handler.get("llm_fallback", 0) / n, 3),
        "calls_avg": round(sum(r["calls"] for r in rows) / n, 2),
        "llm_calls_total": sum(r["llm_calls"] for r in rows),
        "cost_total_usd": round(sum(r["cost_usd"] for r in rows), 5),
        "cost_avg_usd": round(sum(r["cost_usd"] for r in rows) / n, 6),
        "latency_p50_ms": int(percentile([r["latency_ms"] for r in rows], 50)),
        "latency_p95_ms": int(percentile([r["latency_ms"] for r in rows], 95)),
    }


def slice_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    micro_rows = [r for r in rows if r["handled_by"] == "micro"]
    fb_rows = [r for r in rows if r["handled_by"] == "llm_fallback"]
    out = {}
    if micro_rows:
        m_match = sum(r["verdict_match"] for r in micro_rows)
        out["when_micro_handled"] = {
            "n": len(micro_rows),
            "accuracy": round(m_match / len(micro_rows), 3),
            "avg_confidence": round(sum(r["micro_confidence"] for r in micro_rows) / len(micro_rows), 3),
        }
    if fb_rows:
        f_match = sum(r["verdict_match"] for r in fb_rows)
        out["when_fallback_used"] = {
            "n": len(fb_rows),
            "accuracy": round(f_match / len(fb_rows), 3),
            "avg_micro_confidence": round(sum(r["micro_confidence"] for r in fb_rows) / len(fb_rows), 3),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--correct-n", type=int, default=8)
    p.add_argument("--out", default="evaluation_micro_pipeline.json")
    p.add_argument("--skip-baseline", action="store_true",
                   help="skip the always-LLM baseline (saves money/time)")
    p.add_argument("--micro-thr", type=float, default=None,
                   help="override micro confidence threshold (default 0.10; "
                        "0.50-0.70 makes micro more conservative + more accurate)")
    p.add_argument("--fallback", choices=["multi_stage", "unified"], default="multi_stage")
    args = p.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("FATAL: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    monolithic_system_prompt = None
    if args.fallback == "unified":
        sys_path = DAY7 / "prompts" / "system.md"
        monolithic_system_prompt = sys_path.read_text(encoding="utf-8")

    micro = MicroModel(client=client)
    if args.micro_thr is not None:
        micro.threshold = args.micro_thr
    micro.build_references()

    items = load_correct(args.correct_n) + load_jsonl(EDGE) + load_jsonl(NOISY)
    print(f"Total: {len(items)} items, micro_threshold={micro.threshold}, fallback={args.fallback}", file=sys.stderr)

    payload = {"micro_threshold": micro.threshold, "fallback": args.fallback}

    print("\n-- two_tier pipeline (micro + LLM fallback) --", file=sys.stderr)
    pipe_rows = run_pipeline(
        client, micro, items,
        fallback=args.fallback,
        monolithic_system_prompt=monolithic_system_prompt,
    )
    payload["pipeline"] = {
        "summary": summarize(pipe_rows, "two_tier"),
        "slice": slice_summary(pipe_rows),
        "rows": pipe_rows,
    }

    if not args.skip_baseline:
        print("\n-- always_llm (multi_stage v2 baseline) --", file=sys.stderr)
        base_rows = run_always_llm(client, items)
        payload["always_llm"] = {"summary": summarize(base_rows, "always_llm"), "rows": base_rows}

    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.out}")

    print("\n=== SUMMARY ===")
    for name in ("pipeline", "always_llm"):
        if name not in payload:
            continue
        s = payload[name]["summary"]
        print(f"\n[{s['pipeline']}]")
        for k, v in s.items():
            if k == "pipeline":
                continue
            print(f"  {k}: {v}")

    if "slice" in payload.get("pipeline", {}):
        print("\n[slice — only the two_tier pipeline]")
        for k, v in payload["pipeline"]["slice"].items():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
