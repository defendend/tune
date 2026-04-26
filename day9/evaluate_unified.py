"""Compare three pipelines on the same inputs:

  A. monolithic       — single gpt-4o-mini call with the full system prompt
  B. multi_stage v2   — 3 short calls (extract -> verdict -> format)
  C. unified          — multi_stage v2 + self-check + tier-1 escalation

Reuses test sets from day-7 (correct.jsonl, edge.jsonl, noisy.jsonl).

  export OPENAI_API_KEY=sk-...
  python3 evaluate_unified.py
  python3 evaluate_unified.py --skip-mono       # skip baseline if already known
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
from multi_stage import DEFAULT_MODEL, _cost, classify_multi_stage  # noqa: E402
from unified import classify_unified  # noqa: E402

SECTIONS = [
    re.compile(r"-\s*Task type\s*:", re.I),
    re.compile(r"-\s*What changes\s*:", re.I),
    re.compile(r"-\s*Expected behavior\s*:", re.I),
    re.compile(r"Verdict\s*:", re.I),
]
VERDICT_LINE = re.compile(r"Verdict\s*:\s*([A-Za-z]+)")


def detect_verdict(text: str) -> str:
    m = VERDICT_LINE.search(text or "")
    if not m:
        return ""
    raw = m.group(1).strip().capitalize()
    if raw in {"Sufficient", "Insufficient"}:
        return raw
    if raw.startswith("Suff"):
        return "Sufficient"
    if raw.startswith("Insuff"):
        return "Insufficient"
    return ""


def fmt_score(text: str) -> int:
    return sum(1 for rx in SECTIONS if rx.search(text or ""))


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


def run_monolithic(client, items, system_prompt, model: str = DEFAULT_MODEL) -> list[dict]:
    rows = []
    for i, it in enumerate(items, 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": it["description"]}],
                max_tokens=512, temperature=0,
            )
            elapsed_ms = int((time.time() - t0) * 1000)
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            pt = usage.prompt_tokens if usage else 0
            ct = usage.completion_tokens if usage else 0
            cost = _cost(pt, ct)
        except Exception:
            text = ""; elapsed_ms = int((time.time() - t0) * 1000); cost = 0; pt = ct = 0

        verdict = detect_verdict(text)
        match = (verdict or "").lower() == it["expected"].lower()
        row = {
            "idx": i, "expected": it["expected"], "predicted": verdict or None,
            "verdict_match": int(match) if verdict else 0,
            "format_score": fmt_score(text),
            "status": "OK" if verdict and fmt_score(text) == 4 else "FAIL",
            "calls": 1, "cost_usd": round(cost, 6),
            "latency_ms": elapsed_ms,
            "escalated": False,
        }
        rows.append(row)
        print(f"  [{i:>2}] mono     pred={str(verdict or 'None'):>13} match={row['verdict_match']} fmt={row['format_score']}/4 ${cost:.5f} {elapsed_ms}ms", file=sys.stderr)
    return rows


def run_multi(client, items, model: str = DEFAULT_MODEL) -> list[dict]:
    rows = []
    for i, it in enumerate(items, 1):
        res = classify_multi_stage(it["description"], client=client, model=model)
        match = (res.verdict or "").lower() == it["expected"].lower()
        row = {
            "idx": i, "expected": it["expected"], "predicted": res.verdict,
            "verdict_match": int(match) if res.verdict else 0,
            "format_score": fmt_score(res.final_text),
            "status": "OK" if res.verdict else "FAIL",
            "calls": res.calls, "cost_usd": round(res.cost_usd_total, 6),
            "latency_ms": res.latency_ms_total,
            "rule": res.rule,
            "escalated": False,
        }
        rows.append(row)
        print(f"  [{i:>2}] multi    pred={str(res.verdict or 'None'):>13} match={row['verdict_match']} fmt={row['format_score']}/4 rule={res.rule} ${res.cost_usd_total:.5f} {res.latency_ms_total}ms", file=sys.stderr)
    return rows


def run_unified(client, items, monolithic_system: str) -> list[dict]:
    rows = []
    for i, it in enumerate(items, 1):
        res = classify_unified(it["description"], monolithic_system, client=client)
        match = (res.verdict or "").lower() == it["expected"].lower()
        row = {
            "idx": i, "expected": it["expected"], "predicted": res.verdict,
            "verdict_match": int(match) if res.verdict else 0,
            "format_score": fmt_score(res.final_text),
            "status": res.status,
            "calls": res.calls_total, "cost_usd": round(res.cost_usd_total, 6),
            "latency_ms": res.latency_ms_total,
            "route": res.route,
            "escalated": res.escalated,
            "selfcheck_p_yes": res.selfcheck_p_yes,
            "tier0_verdict": res.tier0_verdict,
            "tier1_verdict": res.tier1_verdict,
            "escalation_reason": res.escalation_reason,
        }
        rows.append(row)
        marker = "->T1" if res.escalated else "    "
        print(f"  [{i:>2}] unified  {marker} pred={str(res.verdict or 'None'):>13} status={res.status:<7} match={row['verdict_match']} p_yes={res.selfcheck_p_yes:.3f} calls={res.calls_total} ${res.cost_usd_total:.5f} {res.latency_ms_total}ms {res.route}", file=sys.stderr)
    return rows


def summarize(rows: list[dict], pipeline: str) -> dict:
    if not rows:
        return {}
    n = len(rows)
    statuses = Counter(r["status"] for r in rows)
    matched = sum(r["verdict_match"] for r in rows)
    parsed = sum(1 for r in rows if r["predicted"])
    fmt = sum(r["format_score"] for r in rows)
    matched_ok = sum(r["verdict_match"] for r in rows if r["status"] == "OK")
    n_ok = statuses.get("OK", 0)
    escalated = sum(1 for r in rows if r.get("escalated"))
    return {
        "pipeline": pipeline,
        "n": n,
        "status_counts": dict(statuses),
        "verdict_accuracy_overall": round(matched / n, 3),
        "verdict_accuracy_when_OK": round(matched_ok / n_ok, 3) if n_ok else None,
        "parse_rate": round(parsed / n, 3),
        "format_avg": round(fmt / (n * 4), 3),
        "escalation_rate": round(escalated / n, 3),
        "calls_avg": round(sum(r["calls"] for r in rows) / n, 2),
        "calls_total": sum(r["calls"] for r in rows),
        "cost_total_usd": round(sum(r["cost_usd"] for r in rows), 5),
        "cost_avg_usd": round(sum(r["cost_usd"] for r in rows) / n, 6),
        "latency_p50_ms": int(percentile([r["latency_ms"] for r in rows], 50)),
        "latency_p95_ms": int(percentile([r["latency_ms"] for r in rows], 95)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--correct-n", type=int, default=8)
    p.add_argument("--out", default="evaluation_unified.json")
    p.add_argument("--skip-mono", action="store_true")
    p.add_argument("--skip-multi", action="store_true")
    args = p.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("FATAL: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    if not SYSTEM_PROMPT.exists():
        print(f"FATAL: prompt missing: {SYSTEM_PROMPT}", file=sys.stderr)
        return 1
    monolithic_system = SYSTEM_PROMPT.read_text(encoding="utf-8")

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    items = load_correct(args.correct_n) + load_jsonl(EDGE) + load_jsonl(NOISY)
    print(f"Total: {len(items)} items", file=sys.stderr)

    payload = {"pipelines": []}

    if not args.skip_mono:
        print("\n-- monolithic --", file=sys.stderr)
        mono = run_monolithic(client, items, monolithic_system)
        payload["pipelines"].append({"name": "monolithic", "summary": summarize(mono, "monolithic"), "rows": mono})

    if not args.skip_multi:
        print("\n-- multi_stage v2 --", file=sys.stderr)
        multi = run_multi(client, items)
        payload["pipelines"].append({"name": "multi_stage_v2", "summary": summarize(multi, "multi_stage_v2"), "rows": multi})

    print("\n-- unified --", file=sys.stderr)
    uni = run_unified(client, items, monolithic_system)
    payload["pipelines"].append({"name": "unified", "summary": summarize(uni, "unified"), "rows": uni})

    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.out}")

    print("\n=== COMPARISON ===")
    print(f"{'pipeline':<16} {'acc':>6} {'when_OK':>8} {'parse':>7} {'fmt':>6} {'esc%':>6} {'calls':>6} {'$/req':>10} {'$/1k':>9} {'p50':>5} {'p95':>5}")
    print("-" * 100)
    for sub in payload["pipelines"]:
        s = sub["summary"]
        print(f"{s['pipeline']:<16} "
              f"{s['verdict_accuracy_overall']:>6.3f} "
              f"{(s['verdict_accuracy_when_OK'] or 0):>8.3f} "
              f"{s['parse_rate']:>7.3f} "
              f"{s['format_avg']:>6.3f} "
              f"{s['escalation_rate']*100:>5.1f}% "
              f"{s['calls_avg']:>6.2f} "
              f"${s['cost_avg_usd']:>9.6f} "
              f"${s['cost_total_usd']*1000/s['n']:>8.4f} "
              f"{s['latency_p50_ms']/1000:>4.1f}s "
              f"{s['latency_p95_ms']/1000:>4.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
