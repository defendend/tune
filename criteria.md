# Evaluation criteria — "did fine-tuning help?"

We compare `baseline_results.json` (Qwen2.5-7B-Instruct-4bit, no adapter)
against `finetuned_results.json` (same model + LoRA adapter) on the same
held-out examples in `data/test.jsonl`.

## Metrics

### 1. Verdict accuracy (primary)
**verdict_accuracy** — fraction of examples where the model produced the same
final verdict (Sufficient / Insufficient) as the gold assistant. Computed in
`baseline.py::score_results`. Broken down further:
- `verdict_accuracy_sufficient` — accuracy on tickets that should pass
- `verdict_accuracy_insufficient` — accuracy on tickets that should be rejected

**Target delta:** +0.10 over baseline (e.g. 0.80 → 0.90+).
If baseline is already 1.00, the test set is too easy — expand it before
drawing any conclusions about the fine-tune.

### 2. Format adherence
**format_score** — how many of the 4 required sections are present:
1. `- Task type:`
2. `- What changes:`
3. `- Expected behavior:`
4. `Verdict:`

We average across N examples (max 4 per example). After fine-tuning the
target is `format_avg = 1.0` — the format should become deterministic.
Baseline typically scores 0.6 – 0.95 (the model adds extra prefixes /
preambles).

### 3. Style (qualitative)
Eyeball check:
- **Brevity:** "What changes" / "Expected behavior" should be one sentence
  each, not paragraphs.
- **No filler:** no "Let me analyze…", "Looking at this ticket…" preambles.
- **Question discipline:** Insufficient verdicts include only blocking
  questions, max 3.

Baseline typically has 5-7 style nits per 10 examples; after fine-tune
expect 0-2.

### 4. Speed
**avg_tok_s** — generation tokens/second. Used as a regression check; LoRA
adapters add a tiny overhead (a few percent) but should not halve throughput.

**Target:** > 25 tok/s on M3 Pro 36GB for Qwen2.5-7B-Instruct-4bit.

## Decision rule

The fine-tune is considered a win if:
- `verdict_accuracy_finetuned ≥ verdict_accuracy_baseline + 0.10`
- AND `format_avg_finetuned ≥ 0.95`
- AND no regression on `verdict_accuracy_sufficient` (we did not start
  rejecting good tickets to chase recall on bad ones)
- AND `avg_tok_s ≥ 25`

If any condition fails, tweak hyperparameters (`iters`, `learning_rate`,
`num_layers`) or clean the dataset.

## What we deliberately do NOT measure

- Response length — the format is fixed, length is meaningless.
- BLEU / ROUGE against the gold assistant — the structured format admits
  many valid rewordings of "What changes" and "Expected behavior".
- Validation loss — useful for early stopping during training, but does not
  enter the final accept/reject decision.
