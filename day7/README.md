# day7 — confidence-aware inference

Four orthogonal quality-control layers stacked on top of `gpt-4o-mini`.
Every result comes back with an explicit `OK / UNSURE / FAIL` decision,
a numeric confidence, the call count, the per-stage trace, and the cost
spent.

## Layers

1. **Constraint** — post-hoc check on the model's primary output:
   - all 4 required sections present (`Task type`, `What changes`,
     `Expected behavior`, `Verdict`)
   - verdict word ∈ `{Sufficient, Insufficient}`
   - if Insufficient → ≤ 3 questions
   - free (no extra API call). Anything wrong → status=FAIL.

2. **Logprob scoring** — ask for `logprobs=True, top_logprobs=5` on
   the primary call. Walk the token stream, find the first content
   token after `Verdict:`, aggregate top-k alternatives into the
   `{Sufficient, Insufficient}` distribution.
   `margin = top_prob - second_prob`. Free.

3. **Self-check** — a separate "judge" prompt that sees the description,
   the verdict the primary call produced, and its reasoning. The judge
   answers exactly **YES** or **NO**. We pull `P(YES)` from the first-token
   logprobs.
   This is a **semantic signal**, orthogonal to step 2 — it catches
   "confidently wrong" cases where logprob alone keeps saying margin=1.0
   even on garbage input. Costs +1 short call (~5 tokens out).

4. **Redundancy** — only if either margin or self-check is borderline.
   Re-run the same prompt 2 more times with `temperature=0.7`, then
   majority-vote across all 3 verdicts. Costs +2 calls.

## Decision rule

```
constraint fails                           -> FAIL
margin < LOW_THR (0.50)                    -> FAIL
margin >= HIGH_THR (0.95) AND p_yes >= 0.85 -> OK   (both signals agree)
otherwise                                  -> redundancy
   3/3 agree                               -> upgrade to OK
   2/3 agree                               -> UNSURE (return majority)
   no parseable verdicts                   -> FAIL
```

## Test categories

| File | What it tests | n |
|---|---|---|
| `correct.jsonl` | clean inputs (reuses the day-6 dataset) | 8 |
| `edge.jsonl`    | borderline / underspecified tickets | 8 |
| `noisy.jsonl`   | empty / emoji / gibberish / foreign / prompt injection | 8 |

## Run

```bash
export OPENAI_API_KEY=sk-...
pip install openai
python3 evaluate.py                                 # full 4-layer pipeline
python3 evaluate.py --no-redundancy --out a.json    # ablation: skip stage 4
```

## Results — comparing the layers

Run on a real-world dataset (71 inputs total: 55 production-like tickets
with known labels, 8 borderline, 8 adversarial).

| Pipeline | OK | UNSURE | FAIL | acc overall | acc when OK | calls/req | $/1000 | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logprob only (margin ≥ 0.95) | 70 | 0 | 1 | 0.662 | 0.671 | **1.00** | $0.50 | 3 s / 4 s |
| **+ self-check + redundancy** | 64 | 6 | 1 | 0.690 | **0.719** | 2.92 | $1.00 | 6 s / 11 s |

What changed:

- **`accuracy_when_OK`: 0.671 → 0.719 (+4.8 pp).** The 6 cases that the
  self-check pulled out of OK were genuinely hard — redundancy ran on
  them and still couldn't reach 3/3 agreement, so they correctly land in
  UNSURE for human review instead of being rubber-stamped wrong.

- **Adversarial input handling.** On `noisy.jsonl`, one previously-OK
  garbage input is now correctly flagged UNSURE. The empty input still
  correctly FAILs at the constraint stage.

- **Cost: 2x money, 3x calls.** Self-check fires on every request
  (cheap — short prompt, ~3 tokens out). Redundancy fires only on the
  uncertain ~10% (expensive — full prompt, 2 extra calls).

- **Latency: median 2x, tail 2.5x.** p50 went from 3 s to 6 s (the
  judge call adds ~3 s sequentially); p95 went from 4 s to 11 s on the
  redundancy-triggering tail. In production, route uncertain requests
  asynchronously, accept confident ones inline.

## Findings

1. **Logprob margin alone is a fake confidence signal for subjective
   judgment tasks.** gpt-4o-mini emits margin = 1.000 even on `🎉🎉🎉🎉🎉`,
   gibberish, and prompt-injection inputs. Without the self-check layer,
   confidence-gating is a no-op.

2. **Constraint check is the only deterministic safety net.** Empty /
   structurally-malformed responses always FAIL here, regardless of how
   confident the model is. Cheap insurance.

3. **Self-check (a separate judge call) adds real signal.** It moved
   accuracy_when_OK from 0.67 to 0.72 by demoting confidently-wrong
   answers to UNSURE. It's not free (+1 call/request) but the per-call
   cost is tiny because the prompt is short and the output is one token.

4. **Redundancy alone wouldn't have helped here.** With the original
   logprob-only pipeline, redundancy was dead code (margin always ≥ 0.95).
   Self-check is what makes redundancy actually fire on the cases that
   need it.

5. **gpt-4o-mini is fundamentally weak on this judgment task** — even
   with the full 4-layer stack, accuracy_overall is 0.69. Confidence
   wrappers help you reject bad answers; they don't fix the model.
   The day-6 fine-tune is the right way to push the floor.

## Files

- `confidence.py` — the 4-layer pipeline (one file, no external deps
  beyond `openai`)
- `evaluate.py`   — meta-runner; per-category metrics and totals
- `correct.jsonl` — 30 clean examples (from the day-6 dataset)
- `edge.jsonl`    — 8 borderline cases
- `noisy.jsonl`   — 8 adversarial inputs
- `prompts/system.md` — generic English ticket-quality system prompt
