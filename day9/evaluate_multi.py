"""Compare monolithic vs multi-stage inference on the same inputs.

Pipelines:
  A. monolithic   — one big call with the full system prompt + constraint check
  B. multi_stage  — three short calls (extract -> verdict -> format)

Reuses test sets from day-7 (correct.jsonl, edge.jsonl, noisy.jsonl).

Usage:
  export OPENAI_API_KEY=sk-...
  python3 evaluate_multi.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DAY7 = ROOT.parent / "day7"
SYSTEM_PROMPT = DAY7 / "prompts" / "system.md"
CORRECT = DAY7 / "correct.jsonl"
EDGE = DAY7 / "edge.jsonl"
NOISY = DAY7 / "noisy.jsonl"

sys.path.insert(0, str(ROOT))
from multi_stage import DEFAULT_MODEL, _cost, classify_multi_stage  # noqa: E402

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
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": it["description"]},
                ],
                max_tokens=512,
                temperature=0,
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
            "calls": 1, "cost_usd": round(cost, 6),
            "prompt_tokens": pt, "completion_tokens": ct,
            "latency_ms": elapsed_ms,
            "raw_text": text[:300],
        }
        rows.append(row)
        print(f"  [{i:>2}] mono     pred={str(verdict or 'None'):>13} match={row['verdict_match']} fmt={row['format_score']}/4 in={pt} out={ct} ${cost:.5f} {elapsed_ms}ms", file=sys.stderr)
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
            "rule": res.rule,
            "calls": res.calls,
            "cost_usd": round(res.cost_usd_total, 6),
            "latency_ms": res.latency_ms_total,
            "stage_costs": [round(s.cost_usd, 6) for s in res.stages],
            "stage_latency_ms": [s.latency_ms for s in res.stages],
            "stage_prompt_tokens": [s.prompt_tokens for s in res.stages],
            "stage_completion_tokens": [s.completion_tokens for s in res.stages],
            "stage1_raw": res.stages[0].raw_output[:200],
            "stage2_raw": res.stages[1].raw_output[:100],
        }
        rows.append(row)
        s_costs = "+".join(f"{c:.5f}" for c in row["stage_costs"])
        print(f"  [{i:>2}] multi    pred={str(res.verdict or 'None'):>13} match={row['verdict_match']} fmt={row['format_score']}/4 rule={res.rule} calls={res.calls} ${res.cost_usd_total:.5f}  ({s_costs})  {res.latency_ms_total}ms", file=sys.stderr)
    return rows


def summarize(rows: list[dict], pipeline: str) -> dict:
    if not rows:
        return {}
    n = len(rows)
    matched = sum(r["verdict_match"] for r in rows)
    fmt = sum(r["format_score"] for r in rows)
    parsed_ok = sum(1 for r in rows if r["predicted"])
    return {
        "pipeline": pipeline,
        "n": n,
        "verdict_accuracy_overall": round(matched / n, 3),
        "verdict_accuracy_when_parsed": round(matched / parsed_ok, 3) if parsed_ok else None,
        "parse_rate": round(parsed_ok / n, 3),
        "format_avg": round(fmt / (n * 4), 3),
        "calls_avg": round(sum(r["calls"] for r in rows) / n, 2),
        "calls_total": sum(r["calls"] for r in rows),
        "cost_total_usd": round(sum(r["cost_usd"] for r in rows), 5),
        "cost_avg_usd": round(sum(r["cost_usd"] for r in rows) / n, 6),
        "latency_p50_ms": int(percentile([r["latency_ms"] for r in rows], 50)),
        "latency_p95_ms": int(percentile([r["latency_ms"] for r in rows], 95)),
    }


def stage_breakdown(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    s_costs = [r["stage_costs"] for r in rows if "stage_costs" in r]
    s_lat = [r["stage_latency_ms"] for r in rows if "stage_latency_ms" in r]
    s_pt = [r["stage_prompt_tokens"] for r in rows if "stage_prompt_tokens" in r]
    s_ct = [r["stage_completion_tokens"] for r in rows if "stage_completion_tokens" in r]
    if not s_costs:
        return {}
    out = {}
    for i, name in enumerate(["extract", "verdict", "format"]):
        out[name] = {
            "avg_cost_usd": round(sum(c[i] for c in s_costs) / n, 6),
            "avg_latency_ms": int(sum(l[i] for l in s_lat) / n),
            "avg_prompt_tokens": int(sum(t[i] for t in s_pt) / n),
            "avg_completion_tokens": int(sum(t[i] for t in s_ct) / n),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--correct-n", type=int, default=8)
    p.add_argument("--out", default="evaluation_multi.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("FATAL: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    if not SYSTEM_PROMPT.exists():
        print(f"FATAL: prompt missing: {SYSTEM_PROMPT}", file=sys.stderr)
        return 1
    system_prompt = SYSTEM_PROMPT.read_text(encoding="utf-8")

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    items = load_correct(args.correct_n) + load_jsonl(EDGE) + load_jsonl(NOISY)
    print(f"Total: {len(items)} items", file=sys.stderr)

    print("\n-- monolithic --", file=sys.stderr)
    mono_rows = run_monolithic(client, items, system_prompt, model=args.model)

    print("\n-- multi_stage --", file=sys.stderr)
    multi_rows = run_multi(client, items, model=args.model)

    payload = {
        "model": args.model,
        "monolithic": {"summary": summarize(mono_rows, "monolithic"), "rows": mono_rows},
        "multi_stage": {
            "summary": summarize(multi_rows, "multi_stage"),
            "stage_breakdown": stage_breakdown(multi_rows),
            "rows": multi_rows,
        },
    }

    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.out}")

    print("\n=== COMPARISON ===")
    print(f"{'pipeline':<12} {'acc':>6} {'parse':>7} {'fmt':>6} {'calls':>6} {'$/req':>10} {'$/1k':>10} {'p50':>5} {'p95':>5}")
    print("-" * 88)
    for p_name, sub in [("monolithic", payload["monolithic"]), ("multi_stage", payload["multi_stage"])]:
        s = sub["summary"]
        print(f"{p_name:<12} "
              f"{s['verdict_accuracy_overall']:>6.3f} "
              f"{s['parse_rate']:>7.3f} "
              f"{s['format_avg']:>6.3f} "
              f"{s['calls_avg']:>6.2f} "
              f"${s['cost_avg_usd']:>9.6f} "
              f"${s['cost_total_usd']*1000/s['n']:>9.4f} "
              f"{s['latency_p50_ms']/1000:>4.1f}s "
              f"{s['latency_p95_ms']/1000:>4.1f}s")

    print("\n=== multi_stage per-stage breakdown ===")
    sb = payload["multi_stage"]["stage_breakdown"]
    for name in ("extract", "verdict", "format"):
        s = sb[name]
        print(f"  {name:<8} avg_cost=${s['avg_cost_usd']:.6f}  avg_in_tokens={s['avg_prompt_tokens']}  avg_out_tokens={s['avg_completion_tokens']}  avg_latency={s['avg_latency_ms']}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
