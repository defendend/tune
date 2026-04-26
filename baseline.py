"""Baseline run on a local Qwen2.5-7B-Instruct-4bit (MLX) WITHOUT fine-tuning.

Loads the model, runs the first N (default 10) examples from data/test.jsonl,
saves the model output and a per-example score to baseline_results.json.

Score:
  - verdict_match: did the model produce the same verdict (Sufficient/Insufficient)
                   as the gold assistant message? (binary)
  - format_score:  how many of the 4 required sections are present
                   (Task type / What changes / Expected behavior / Verdict)
  - approx tok/s

Usage:
  python3 baseline.py                    # run + save results
  python3 baseline.py --score            # recompute summary from saved file
  python3 baseline.py --model X --n 10   # override model / count
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST = ROOT / "data" / "test.jsonl"
OUT = ROOT / "baseline_results.json"

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

VERDICT_OK = re.compile(r"\bSufficient\b", re.I)
VERDICT_BAD = re.compile(r"\bInsufficient\b", re.I)
SECTIONS = [
    re.compile(r"-\s*Task type:", re.I),
    re.compile(r"-\s*What changes:", re.I),
    re.compile(r"-\s*Expected behavior:", re.I),
    re.compile(r"Verdict:", re.I),
]


def detect_verdict(text: str) -> str:
    has_ok = bool(VERDICT_OK.search(text))
    has_bad = bool(VERDICT_BAD.search(text))
    if has_bad and not has_ok:
        return "insufficient"
    if has_ok and not has_bad:
        return "sufficient"
    if has_ok and has_bad:
        last_ok = list(VERDICT_OK.finditer(text))[-1].start()
        last_bad = list(VERDICT_BAD.finditer(text))[-1].start()
        return "sufficient" if last_ok > last_bad else "insufficient"
    return "unknown"


def format_score(text: str) -> int:
    return sum(1 for rx in SECTIONS if rx.search(text))


def run_inference(model_id: str, examples: list[dict]) -> list[dict]:
    from mlx_lm import load, generate

    print(f"loading model: {model_id}", file=sys.stderr)
    t0 = time.time()
    model, tokenizer = load(model_id)
    print(f"loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    out = []
    for i, ex in enumerate(examples, 1):
        msgs = ex["messages"]
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        gold_assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
        gold_verdict = detect_verdict(gold_assistant)

        prompt = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False)

        t0 = time.time()
        text = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)
        elapsed = time.time() - t0
        approx_toks = max(1, len(text or "") // 3)

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
            "approx_tok_s": round(approx_toks / max(elapsed, 1e-3), 1),
        }
        print(f"  [{i:>2}] gold={gold_verdict:>13}  pred={pred_verdict:>13}  match={match}  fmt={item['format_score']}/4  {elapsed:.1f}s  ~{item['approx_tok_s']:.0f} tok/s", file=sys.stderr)
        out.append(item)
    return out


def score_results(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    matches = sum(r["verdict_match"] for r in results)
    fmt = sum(r["format_score"] for r in results)
    suff = [r for r in results if r["gold_verdict"] == "sufficient"]
    insuff = [r for r in results if r["gold_verdict"] == "insufficient"]
    return {
        "n": n,
        "verdict_accuracy": round(matches / n, 3),
        "verdict_accuracy_sufficient": round(sum(r["verdict_match"] for r in suff) / max(len(suff), 1), 3),
        "verdict_accuracy_insufficient": round(sum(r["verdict_match"] for r in insuff) / max(len(insuff), 1), 3),
        "format_avg": round(fmt / (n * 4), 3),
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results) / n, 2),
        "avg_tok_s": round(sum(r["approx_tok_s"] for r in results) / n, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--score", action="store_true", help="recompute summary from existing baseline_results.json")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    if args.score:
        if not Path(args.out).exists():
            print(f"FATAL: {args.out} not found", file=sys.stderr)
            return 1
        data = json.loads(Path(args.out).read_text(encoding="utf-8"))
        print(json.dumps(score_results(data["results"]), ensure_ascii=False, indent=2))
        return 0

    if not TEST.exists():
        print(f"FATAL: {TEST} not found — run split.py first", file=sys.stderr)
        return 1

    examples = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n]
    print(f"baseline on {len(examples)} examples, model={args.model}", file=sys.stderr)
    results = run_inference(args.model, examples)
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
