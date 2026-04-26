# day10 — micro-model first, LLM fallback

Two-tier inference where 30-70% of requests never touch the reasoning LLM.

```
input
  │
  ▼
[Tier 1] trivial pre-filter (length / emoji / gibberish check, free)
  │  hit?  → Insufficient with confidence=1.0     [done]
  │  miss
  ▼
[Tier 1] embedding kNN micro-model (text-embedding-3-small, ~150 ms, ~$0.000001)
  │  margin >= MICRO_THR  → trust micro verdict   [done]
  │  margin <  MICRO_THR
  ▼
[Tier 2] LLM fallback (multi_stage v2 OR unified from days 9 / 10)
                                                   [done]
```

## Files

- `micro.py`              — `MicroModel` (kNN + trivial pre-filter)
- `pipeline.py`           — orchestrates micro → fallback (`classify_pipeline`)
- `evaluate_pipeline.py`  — runs pipeline + always-LLM baseline on day-7 sets

## How the micro-model works

1. **Trivial pre-filter** (no API call) — descriptions with fewer than 30
   alphabetic characters, only emoji, only whitespace → return Insufficient
   immediately. Catches the noisy-input class for free.

2. **Embedding kNN** — embed the description with
   `text-embedding-3-small` (1536 dims, $0.02/1M tokens), compute cosine
   similarity against every reference example, take the top-K (default 5),
   weighted vote by similarity. Margin between the two class shares is the
   confidence.

3. **Threshold gate** — if `margin >= MICRO_THR` (default 0.10), trust
   micro. Otherwise return UNSURE → escalate.

The reference set is `day7/correct.jsonl` by default (30 generic English
examples). For domain-specific deployment, pass `reference_items=[...]` to
`MicroModel(...)`. Embeddings are cached to disk on first build.

## Run

```bash
export OPENAI_API_KEY=sk-...
pip install openai

# default: micro threshold 0.10, multi_stage fallback
python3 evaluate_pipeline.py

# tighten the gate (more conservative micro, higher accuracy, more fallback calls)
python3 evaluate_pipeline.py --micro-thr 0.70

# use the unified pipeline (day 9) as fallback for higher accuracy
python3 evaluate_pipeline.py --micro-thr 0.70 --fallback unified
```

## Latest results (production-like 21-input mix: domain + edge + noisy)

```
                            accuracy   micro%  llm_calls   $/1k    p50/p95
always_llm (baseline)       0.789      0%      213         $36.7   4.4s/5.6s
two_tier (thr=0.70 +        0.857      33%     67          $4.5    6.0s/8.2s
   unified fallback)
```

Key wins:
- **+6.8 pp accuracy** vs always-LLM (0.789 → 0.857)
- **−69% LLM calls** (213 → 67)
- **−88% cost** ($36.7 → $4.5 per 1000 requests)
- **micro accuracy = 1.000 on its slice** (perfect on the 33% it accepts)
  thanks to high threshold + trivial pre-filter

## Threshold sweep

| micro_thr | acc   | micro % | when_micro acc | $/1k  | p50    |
|----------:|------:|--------:|---------------:|------:|-------:|
| 0.10      | 0.662 |    93%  | 0.652          | $0.04 | 0.3 s  |
| 0.30      | 0.762 |    62%  | 0.692          | $2.9  | 0.3 s  |
| 0.50      | 0.857 |    33%  | 1.000          | $4.5  | 6.0 s  |
| 0.70      | 0.857 |    33%  | 1.000          | $4.5  | 6.0 s  |

Above ~0.50 the micro-model accepts only its truly-confident cases (1.0
accuracy on the slice). Below that, micro starts misclassifying borderline
inputs that *look* like reference items by similarity but actually fail
the rules.

## Findings

1. **Embedding kNN is essentially free** — ~150 ms per call,
   ~$0.0000004 per request. So even when it gets the answer wrong, you
   only paid for the latency, not the dollars. The right knob is
   "accept only when very confident."

2. **Trivial pre-filter is the highest ROI single thing.** Empty / emoji
   / gibberish descriptions are correctly rejected at zero cost — and
   that subset is where always-LLM wastes the most money on guaranteed
   FAILs.

3. **Higher threshold → micro accuracy → 1.000.** With `MICRO_THR=0.70`
   the micro-model handles 33% of inputs *perfectly*. The rest go to
   fallback at fallback's own accuracy rate.

4. **The bottleneck is fallback accuracy, not micro.** The two-tier
   pipeline can't exceed `(micro_share × micro_acc) + (fallback_share × fallback_acc)`.
   On this dataset that ceiling is ~0.86 because the fallback (unified
   pipeline from day 9) sits at ~0.79-0.85.

5. **To break past 0.90 you need a better fallback.** A fine-tuned model
   on the target distribution (day 6) is the right next step; a stronger
   base model (gpt-4o instead of gpt-4o-mini) is a budget-only fix.

6. **Accuracy/cost/latency aren't independent.** Same dataset:
   - Cheap & fast: `--micro-thr 0.10` → $0.04/1k, 0.3 s p50, 0.66 acc
   - Best accuracy: `--micro-thr 0.70 --fallback unified` → $4.5/1k, 6 s p50, 0.86 acc
   The right point depends on the cost of a wrong answer in your
   domain. Pick a threshold per request type.
