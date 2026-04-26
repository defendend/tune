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

## Results — small test set (24 inputs)

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

The 4 escalations: 2 self-check disagreements, 1 redundancy split, 1
constraint FAIL. All 4 resolved on tier 1 (3 OK + 1 final FAIL on an
adversarial input that broke the format on both tiers).

## Results — larger production-like test set (71 inputs)

```
pipeline        acc  when_OK  calls   esc%      $/req       $/1k   p50   p95
----------------------------------------------------------------------------------------
tier0_only    0.690    0.734   2.96   0.0% $ 0.001034 $   1.0344  6.6s  9.4s
router        0.690    0.710   2.97  11.3% $ 0.001843 $   1.8428  4.6s  8.7s
tier1_only    0.718    0.773   1.00   0.0% $ 0.008444 $   8.4439  1.6s  2.7s
```

This is the result that actually says something. **Routing did not
improve accuracy here.** Worse — `accuracy_when_OK` for the router
*decreased* slightly (0.734 → 0.710), because the escalated cases
weren't actually any easier for tier 1 either. The 8 escalations
(11.3%) replaced 8 tier-0 OK answers with tier-1 OK answers of similar
or worse quality.

Cost-wise: router sits at ~$1.84 / 1000, between tier-0 ($1.03) and
tier-1 ($8.44). So you pay 78% more than tier-0 for **no accuracy gain**.
Always-tier-1 buys +2.8 pp accuracy for **8.2× the cost** of tier-0 — a
better trade if accuracy is the only thing you care about.

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

1. **Routing only pays off when tier 1 is actually better than tier 0
   on the cases the router escalates.** On the production-like 71-item
   set, gpt-4o is only 2.8 pp more accurate than gpt-4o-mini on this
   task (0.72 vs 0.69). The 8 cases the router escalated were hard
   *for both models*, so swapping to tier 1 didn't help. Result:
   identical overall accuracy, slightly worse `accuracy_when_OK`,
   78% higher cost.
   *Implication:* before deploying a router, validate that tier 1
   actually outperforms tier 0 on the slice you intend to escalate
   — otherwise the router is pure cost overhead.

2. **The right routing target is failure mode, not "uncertainty".**
   gpt-4o-mini's confidence signal (day-7 composite) flags
   uncertainty, but uncertainty correlates with task difficulty —
   not with "tier-1 will fix this". A better router would route on
   *what kind* of failure (e.g. format errors → tier 1; semantic
   ambiguity → human, not tier 1).

3. **Median latency goes DOWN, p95 stays roughly the same.**
   Counter-intuitive: when tier 0 is uncertain, day-7 redundancy adds
   2-3 calls. Tier 1 is a single fast call, so escalating skips the
   redundancy tail. p50 6.6 s → 4.6 s. p95 only marginally better.

4. **gpt-4o is fast.** Single-call median 1.6 s vs 6.6 s for the
   redundancy-heavy tier-0 pipeline. If you only care about latency
   and have budget, always-tier-1 is competitive.

5. **The escalation rate is the only knob that matters for cost.**
   Tighten day-7 thresholds (HIGH_THR ↑, SELFCHECK_HIGH ↑) → escalate
   more, costs go up linearly. Loosen → save money but accept more
   wrong tier-0 answers.

6. **On easy data, you can't tell the tiers apart.** Small test set
   shows identical 0.833 accuracy across all three pipelines. You can
   only validate routing on data hard enough to differentiate the
   models.
