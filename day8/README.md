# day8 — model routing with confidence-based fallback

Two-tier model router built on top of the day-7 confidence pipeline:

- **Tier 0** — `gpt-4o-mini` with the full 4-layer confidence pipeline
  (constraint + logprob + self-check + redundancy). Cheap and fast.
- **Tier 1** — `gpt-4o`, single inference + constraint check. ~17×
  more expensive per token, used as the escalation target.

The routing heuristic is the day-7 `OK / UNSURE / FAIL` status:

```
tier 0 OK      -> trust, no escalation
tier 0 UNSURE  -> escalate to tier 1
tier 0 FAIL    -> escalate to tier 1
tier 1 broken  -> final FAIL
```

This is the "if not confident, escalate" pattern — but the confidence
signal is the composite from day 7 (multi-layer), not a single number.

## Files

- `confidence.py`        — copy of day-7 pipeline (4 layers)
- `router.py`            — `route()` entry point + `RouteResult` dataclass
                           with full per-tier trace (calls, cost, latency,
                           which tier handled it, why escalated)
- `evaluate_routing.py`  — runs three pipelines side-by-side
                           (tier0_only / router / tier1_only) and prints
                           a comparison table

## Run

```bash
export OPENAI_API_KEY=sk-...
pip install openai
python3 evaluate_routing.py                      # full 3-way comparison
python3 evaluate_routing.py --skip-tier1         # skip the tier1-only baseline
```

Test sets are reused from `day7/`: `correct.jsonl` (8 clean), `edge.jsonl`
(8 borderline), `noisy.jsonl` (8 adversarial).

## Results — public test set (24 inputs)

```
pipeline        acc  when_OK  calls   esc%      $/req       $/1k   p50   p95
----------------------------------------------------------------------------------------
tier0_only    0.833    0.870   2.71   0.0% $ 0.000192 $   0.1917  4.1s  6.7s
router        0.833    0.870   2.88  16.7% $ 0.000435 $   0.4350  3.5s  7.3s
tier1_only    0.833    0.870   1.00   0.0% $ 0.001619 $   1.6192  1.2s  1.6s
```

On this small/easy test set all three pipelines hit the same accuracy.
The router pays ~2.3× the tier-0 cost (still ~3.7× cheaper than
tier1_only) and escalated 16.7% of cases (4 out of 24).

Interesting: router's median latency (3.5 s) is actually **lower** than
tier0_only (4.1 s). When a request hits tier 1, the strong model finishes
the call faster than tier 0 would have spent on redundancy. p95 still
spikes on the escalated path.

## What gets escalated

From the public run, the 4 escalations were:
- 2 cases where tier 0 self-check disagreed with the verdict
- 1 case where redundancy split (UNSURE)
- 1 case where the constraint check failed (no parseable verdict)

All 4 escalations resolved successfully on tier 1 (3 OK, 1 final FAIL).
The single FAIL was an adversarial input that broke the format on both
tiers — correctly escalated, correctly refused.

## Pricing model

- gpt-4o-mini:  $0.150 in / $0.600 out per 1M tokens
- gpt-4o:       $2.500 in / $10.000 out per 1M tokens (~17×)

Router average cost is bounded by:
```
cost(router) <= cost(tier0) * (1 + escalation_rate * cost_ratio)
```
For `escalation_rate = 0.17` and `cost_ratio ≈ 17`, that's roughly
~3.9× tier-0 cost, while still ~3.7× cheaper than always-tier1.

## Findings

1. **The routing pattern is essentially a money lever.** Router cost sits
   between tier-0 (cheap, less reliable) and tier-1 (~10× more expensive,
   more reliable). Good for tasks where you want best-effort accuracy on
   most inputs and willingness to pay extra for the hard ones.

2. **Median latency goes DOWN, p95 goes UP.** Counter-intuitive: when
   tier 0 is uncertain, day-7 redundancy adds 2-3 calls. Tier 1 is a
   single fast call, so escalating skips the redundancy tail. But the
   escalated requests still take the longest overall.

3. **The escalation rate is the only knob that matters for cost.**
   Tighten the day-7 thresholds (HIGH_THR ↑, SELFCHECK_HIGH ↑) and you
   escalate more. Loosen them and you save money but may accept more
   wrong answers on tier 0.

4. **On easy data, you can't tell the tiers apart.** Public test set
   shows identical accuracy across all three pipelines. Routing only
   pays off when tier 0 actually gets some answers wrong that tier 1
   gets right — i.e. on harder, production-like inputs.
