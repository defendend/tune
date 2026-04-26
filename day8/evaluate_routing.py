"""End-to-end routing evaluation.

Runs three pipelines side-by-side on the same inputs and prints comparison:

  A. tier0_only   — gpt-4o-mini with the day-7 confidence pipeline
  B. router       — tier 0 first, escalate to gpt-4o on UNSURE/FAIL
  C. tier1_only   — gpt-4o always (single call + constraint check)

For each pipeline we report:
  - status counts (OK/UNSURE/FAIL)
  - verdict accuracy overall
  - verdict accuracy when status == OK
  - calls per request
  - average and total cost
  - latency p50 / p95
  - (router only) escalation rate, average tier-1 cost share

Reuses test sets from day 7 (correct.jsonl, edge.jsonl, noisy.jsonl).

Usage:
  export OPENAI_API_KEY=sk-...
  python3 evaluate_routing.py
  python3 evaluate_routing.py --skip-tier1     # skip the expensive baseline
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
SYSTEM_PROMPT = DAY7 / "prompts" / "system.md"
CORRECT = DAY7 / "correct.jsonl"
EDGE = DAY7 / "edge.jsonl"
NOISY = DAY7 / "noisy.jsonl"

sys.path.insert(0, str(ROOT))
from confidence import classify_with_confidence, constraint_check, _one_call  # noqa: E402
from router import TIER1_DEFAULT, _tier1_cost, route  # noqa: E402


def load_correct(n: int = 8) -> list[dict]:
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
        items.append({"description": user, "expected": expected, "note": "clean"})
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


# -- three pipelines ---------------------------------------------------------
def run_tier0_only(client, items, system_prompt) -> list[dict]:
    rows = []
    for i, it in enumerate(items, 1):
        cr = classify_with_confidence(it["description"], system_prompt, client=client)
        match = (cr.verdict or "").lower() == it["expected"].lower()
        row = {
            "idx": i, "expected": it["expected"], "predicted": cr.verdict,
            "status": cr.status, "verdict_match": int(match) if cr.verdict else 0,
            "calls": cr.calls, "cost_usd": round(cr.cost_usd, 6),
            "latency_ms": cr.latency_ms,
            "route": "tier0_only", "escalated": False,
            "tier0_calls": cr.calls, "tier1_calls": 0,
            "tier0_cost": cr.cost_usd, "tier1_cost": 0.0,
            "note": it.get("note", ""),
        }
        rows.append(row)
        print(f"  [{i:>2}] tier0_only      pred={str(cr.verdict):>13} status={cr.status:<7} calls={cr.calls} ${cr.cost_usd:.5f}", file=sys.stderr)
    return rows


def run_router(client, items, system_prompt) -> list[dict]:
    rows = []
    for i, it in enumerate(items, 1):
        rr = route(it["description"], system_prompt, client=client)
        match = (rr.verdict or "").lower() == it["expected"].lower()
        row = {
            "idx": i, "expected": it["expected"], "predicted": rr.verdict,
            "status": rr.status, "verdict_match": int(match) if rr.verdict else 0,
            "calls": rr.calls_total, "cost_usd": round(rr.cost_usd_total, 6),
            "latency_ms": rr.latency_ms_total,
            "route": rr.route, "escalated": rr.escalated,
            "tier0_calls": rr.tier0_calls, "tier1_calls": rr.tier1_calls,
            "tier0_cost": rr.tier0_cost, "tier1_cost": rr.tier1_cost,
            "escalation_reason": rr.escalation_reason,
            "note": it.get("note", ""),
        }
        rows.append(row)
        marker = "->T1" if rr.escalated else "    "
        print(f"  [{i:>2}] router      {marker} pred={str(rr.verdict):>13} status={rr.status:<7} calls={rr.calls_total} ${rr.cost_usd_total:.5f} {rr.route}", file=sys.stderr)
    return rows


def run_tier1_only(client, items, system_prompt, *, model: str = TIER1_DEFAULT) -> list[dict]:
    rows = []
    for i, it in enumerate(items, 1):
        t0 = time.time()
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": it["description"]}]
        try:
            choice, text, pt, ct = _one_call(client, model, msgs, with_logprobs=False, temperature=0.0)
            cost = _tier1_cost(pt, ct)
            cstr = constraint_check(text)
            verdict = cstr.extra["verdict"] if cstr.passed else None
            status = "OK" if cstr.passed else "FAIL"
        except Exception:
            text = None; verdict = None; status = "FAIL"; cost = 0.0
        elapsed_ms = int((time.time() - t0) * 1000)
        match = (verdict or "").lower() == it["expected"].lower()
        row = {
            "idx": i, "expected": it["expected"], "predicted": verdict,
            "status": status, "verdict_match": int(match) if verdict else 0,
            "calls": 1, "cost_usd": round(cost, 6),
            "latency_ms": elapsed_ms,
            "route": "tier1_only", "escalated": False,
            "tier0_calls": 0, "tier1_calls": 1,
            "tier0_cost": 0.0, "tier1_cost": cost,
            "note": it.get("note", ""),
        }
        rows.append(row)
        print(f"  [{i:>2}] tier1_only      pred={str(verdict):>13} status={status:<7} calls=1 ${cost:.5f}", file=sys.stderr)
    return rows


def summarize(rows: list[dict], pipeline_name: str) -> dict:
    if not rows:
        return {}
    n = len(rows)
    statuses = Counter(r["status"] for r in rows)
    matched = sum(r["verdict_match"] for r in rows)
    n_ok = statuses.get("OK", 0)
    matched_ok = sum(r["verdict_match"] for r in rows if r["status"] == "OK")
    escalated = sum(1 for r in rows if r.get("escalated"))
    tier0_calls = sum(r["tier0_calls"] for r in rows)
    tier1_calls = sum(r["tier1_calls"] for r in rows)
    tier0_cost = sum(r["tier0_cost"] for r in rows)
    tier1_cost = sum(r["tier1_cost"] for r in rows)
    return {
        "pipeline": pipeline_name,
        "n": n,
        "status_counts": dict(statuses),
        "verdict_accuracy_overall": round(matched / n, 3),
        "verdict_accuracy_when_OK": round(matched_ok / n_ok, 3) if n_ok else None,
        "calls_avg": round(sum(r["calls"] for r in rows) / n, 2),
        "calls_total": sum(r["calls"] for r in rows),
        "tier0_calls_total": tier0_calls,
        "tier1_calls_total": tier1_calls,
        "escalation_rate": round(escalated / n, 3),
        "cost_total_usd": round(sum(r["cost_usd"] for r in rows), 5),
        "cost_avg_usd": round(sum(r["cost_usd"] for r in rows) / n, 6),
        "tier0_cost_share": round(tier0_cost / max(tier0_cost + tier1_cost, 1e-9), 3),
        "tier1_cost_share": round(tier1_cost / max(tier0_cost + tier1_cost, 1e-9), 3),
        "latency_p50_ms": int(percentile([r["latency_ms"] for r in rows], 50)),
        "latency_p95_ms": int(percentile([r["latency_ms"] for r in rows], 95)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--correct-n", type=int, default=8)
    p.add_argument("--out", default="evaluation_routing.json")
    p.add_argument("--skip-tier1", action="store_true",
                   help="skip the tier1-only baseline (saves money/time)")
    args = p.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("FATAL: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    if not SYSTEM_PROMPT.exists():
        print(f"FATAL: prompt not found at {SYSTEM_PROMPT}", file=sys.stderr)
        return 1
    system_prompt = SYSTEM_PROMPT.read_text(encoding="utf-8")

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    items = load_correct(args.correct_n) + load_jsonl(EDGE) + load_jsonl(NOISY)
    print(f"Total items: {len(items)}", file=sys.stderr)

    pipelines: list[tuple[str, list[dict]]] = []

    print("\n-- tier0_only --", file=sys.stderr)
    pipelines.append(("tier0_only", run_tier0_only(client, items, system_prompt)))

    print("\n-- router --", file=sys.stderr)
    pipelines.append(("router", run_router(client, items, system_prompt)))

    if not args.skip_tier1:
        print("\n-- tier1_only --", file=sys.stderr)
        pipelines.append(("tier1_only", run_tier1_only(client, items, system_prompt)))

    payload = {"pipelines": []}
    for name, rows in pipelines:
        s = summarize(rows, name)
        payload["pipelines"].append({"name": name, "summary": s, "rows": rows})

    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.out}")

    print("\n=== COMPARISON ===")
    print(f"{'pipeline':<12} {'acc':>6} {'when_OK':>8} {'calls':>6} {'esc%':>6} {'$/req':>10} {'$/1k':>10} {'p50':>5} {'p95':>5}")
    print("-" * 88)
    for name, rows in pipelines:
        s = summarize(rows, name)
        print(f"{s['pipeline']:<12} "
              f"{s['verdict_accuracy_overall']:>6.3f} "
              f"{(s['verdict_accuracy_when_OK'] or 0):>8.3f} "
              f"{s['calls_avg']:>6.2f} "
              f"{s['escalation_rate']*100:>5.1f}% "
              f"${s['cost_avg_usd']:>9.6f} "
              f"${s['cost_total_usd']*1000/s['n']:>9.4f} "
              f"{s['latency_p50_ms']/1000:>4.1f}s "
              f"{s['latency_p95_ms']/1000:>4.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
