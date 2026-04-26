"""Inference with MLX — with or without a LoRA adapter.

Used for after-comparison: same interface as baseline.py, but loads
an optional adapter and writes to a separate results file.

  # after fine-tune:
  python3 infer_mlx.py --adapter adapters --out finetuned_results.json
  # rescore from saved file:
  python3 infer_mlx.py --score --out finetuned_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST = ROOT / "data" / "test.jsonl"
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

sys.path.insert(0, str(ROOT))
from baseline import detect_verdict, format_score, score_results  # noqa: E402


def run(model_id: str, adapter_path: str | None, examples: list[dict]) -> list[dict]:
    from mlx_lm import load, generate

    print(f"loading model: {model_id}  adapter={adapter_path or 'none'}", file=sys.stderr)
    t0 = time.time()
    model, tokenizer = load(model_id, adapter_path=adapter_path) if adapter_path else load(model_id)
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
        print(f"  [{i:>2}] gold={gold_verdict:>13}  pred={pred_verdict:>13}  match={match}  fmt={item['format_score']}/4  {elapsed:.1f}s", file=sys.stderr)
        out.append(item)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None, help="path to LoRA adapter directory (e.g. 'adapters')")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out", default="finetuned_results.json")
    p.add_argument("--score", action="store_true")
    args = p.parse_args()

    if args.score:
        data = json.loads(Path(args.out).read_text(encoding="utf-8"))
        print(json.dumps(score_results(data["results"]), ensure_ascii=False, indent=2))
        return 0

    if not TEST.exists():
        print(f"FATAL: {TEST} not found", file=sys.stderr)
        return 1
    examples = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n]
    results = run(args.model, args.adapter, examples)
    score = score_results(results)
    Path(args.out).write_text(
        json.dumps({"model": args.model, "adapter": args.adapter, "n": len(results), "score": score, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nScore:", json.dumps(score, ensure_ascii=False, indent=2))
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
