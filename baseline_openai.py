"""Baseline run via OpenAI gpt-4o-mini WITHOUT fine-tuning.

Same interface as baseline.py but routes inference to OpenAI Chat Completions.
Used as a second reference point alongside the local Qwen baseline.

  export OPENAI_API_KEY=sk-...
  python3 baseline_openai.py --n 10
  # -> baseline_openai_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST = ROOT / "data" / "test.jsonl"
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

sys.path.insert(0, str(ROOT))
from baseline import detect_verdict, format_score, score_results  # noqa: E402


def run(model: str, examples: list[dict]) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = []
    for i, ex in enumerate(examples, 1):
        msgs = ex["messages"]
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        gold_assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
        gold_verdict = detect_verdict(gold_assistant)

        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=prompt_msgs,
            max_tokens=512,
            temperature=0,
        )
        elapsed = time.time() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        toks = usage.completion_tokens if usage else max(1, len(text) // 3)

        pred_verdict = detect_verdict(text)
        match = int(pred_verdict == gold_verdict and gold_verdict != "unknown")
        item = {
            "idx": i,
            "user": next(m["content"] for m in msgs if m["role"] == "user")[:200],
            "gold_verdict": gold_verdict,
            "gold_assistant": gold_assistant,
            "pred_text": text,
            "pred_verdict": pred_verdict,
            "verdict_match": match,
            "format_score": format_score(text),
            "elapsed_s": round(elapsed, 2),
            "approx_tok_s": round(toks / max(elapsed, 1e-3), 1),
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        }
        print(f"  [{i:>2}] gold={gold_verdict:>13}  pred={pred_verdict:>13}  match={match}  fmt={item['format_score']}/4  {elapsed:.1f}s  in={item['prompt_tokens']} out={item['completion_tokens']}", file=sys.stderr)
        out.append(item)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out", default="baseline_openai_results.json")
    args = p.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("FATAL: OPENAI_API_KEY not set", file=sys.stderr)
        return 1
    if not TEST.exists():
        print(f"FATAL: {TEST} not found", file=sys.stderr)
        return 1

    examples = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n]
    print(f"baseline OpenAI on {len(examples)} examples, model={args.model}", file=sys.stderr)
    results = run(args.model, examples)
    score = score_results(results)
    Path(args.out).write_text(
        json.dumps({"model": args.model, "n": len(results), "score": score, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nScore:", json.dumps(score, ensure_ascii=False, indent=2))
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
