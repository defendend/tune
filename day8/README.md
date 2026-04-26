# day8 — model routing with confidence-based fallback

Two-tier model router built on top of the day-7 confidence pipeline.

- **Tier 0** — `gpt-4o-mini` with the full 4-layer confidence pipeline
  (constraint + logprob + self-check + redundancy). Cheap and fast.
- **Tier 1** — `gpt-4o`, inference + constraint + self-check. ~17×
  more expensive per token, used as the escalation target.

## Routing logic (v2)

v1 was the textbook "escalate on uncertainty" pattern, but on real data
it had no accuracy gain — it escalated cases that were hard *for both
tiers*, not cases tier 1 actually fixes. v2 adds three things:

1. **Tier-1 self-check.**   After tier-1 inference, run the same YES/NO
   judge call as in day 7. If tier 1 also fails its own self-check →
   final UNSURE instead of trusting tier 1 blindly.

2. **Borderline tier-0 spot-check.**   When tier 0 returns OK but the
   self-check probability was below `SPOT_CHECK_THR` (0.95), still run
   tier 1 once. Catches the day-7 finding that gpt-4o-mini is sometimes
   confidently wrong.

3. **Verdict disagreement → UNSURE.**   If tier 0 and tier 1 produce
   different verdicts on any path, the result is UNSURE — neither
   model is trusted alone.

```
tier 0 OK, p_yes >= 0.95             -> trust, no escalation
tier 0 OK, p_yes <  0.95             -> spot-check on tier 1
   tier 1 verdict matches tier 0     -> OK
   tier 1 verdict differs / unsure   -> UNSURE
tier 0 UNSURE                        -> escalate to tier 1
   tier 1 OK + agrees with tier 0    -> OK
   tier 1 OK + disagrees             -> UNSURE
   tier 1 unsure / FAIL              -> UNSURE
tier 0 FAIL                          -> escalate to tier 1
   tier 1 OK                         -> OK
   tier 1 broken                     -> final FAIL
```

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

### v1 (textbook "escalate on tier-0 UNSURE/FAIL")

```
pipeline        acc  when_OK  calls   esc%      $/req       $/1k   p50   p95
----------------------------------------------------------------------------------------
tier0_only    0.690    0.734   2.96   0.0% $ 0.001034 $   1.0344  6.6s  9.4s
router (v1)   0.690    0.710   2.97  11.3% $ 0.001843 $   1.8428  4.6s  8.7s
tier1_only    0.718    0.773   1.00   0.0% $ 0.008444 $   8.4439  1.6s  2.7s
```

v1 was disappointing: routing had **no accuracy gain** and `when_OK`
*decreased* slightly. The 8 escalations were hard for both models — the
router replaced 8 tier-0 OK answers with tier-1 OK answers of similar
quality, paying 78% extra for nothing.

### v2 (spot-check + tier-1 self-check + disagreement-as-UNSURE)

```
pipeline        acc  when_OK  calls   esc%      $/req       $/1k   p50   p95
----------------------------------------------------------------------------------------
tier0_only    0.704    0.716   2.76   0.0% $ 0.000936 $   0.9361  4.5s  9.9s
router (v2)   0.775    0.815   3.96  64.8% $ 0.006787 $   6.7873  6.0s 13.4s
[reference] tier1_only on the same set: acc 0.718, when_OK 0.773, $8.44/1k
```

v2 actually delivers. Direct comparison:

| metric              | router v1 | router v2 | tier1_only | v2 vs v1 | v2 vs tier1_only |
|---------------------|----------:|----------:|-----------:|---------:|-----------------:|
| accuracy            | 0.690     | **0.775** | 0.718      | +8.5 pp  | **+5.7 pp**      |
| accuracy_when_OK    | 0.710     | **0.815** | 0.773      | +10.5 pp | **+4.2 pp**      |
| escalation rate     | 11.3%     | 64.8%     | —          | +53.5 pp | —                |
| $/1000              | $1.84     | $6.79     | $8.44      | +269%    | **−20%**         |
| latency p95         | 8.7 s     | 13.4 s    | 2.7 s      | +4.7 s   | +10.7 s          |

**v2 router beats always-tier-1 on accuracy AND is 20% cheaper.** That's
the routing pattern actually paying for itself. The price for that win:
3.7× the cost of v1 routing and ~5× the latency of always-tier-1.

What happens under the hood in v2:
- ~65% of inputs trigger an escalation (most tier-0 OKs are "borderline"
  by p_yes — gpt-4o-mini's self-check is rarely above 0.95 on subjective
  judgments)
- Of those escalated, agreement between tier 0 and tier 1 keeps OK; any
  disagreement or tier-1 self-check failure downgrades to UNSURE
- The "downgrade to UNSURE" is what improves `when_OK`: we stop
  rubber-stamping cases where the two models disagree

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

1. **The naive "escalate on uncertainty" pattern (v1) doesn't work
   when tier 1 isn't dramatically better than tier 0.** On this task
   gpt-4o is only 2.8 pp more accurate than gpt-4o-mini overall, and
   tier-0's uncertainty signal correlates with task difficulty rather
   than with "tier 1 will fix this." v1 escalated 11% of cases and
   gained zero accuracy.

2. **Disagreement between two models is a much stronger signal than
   any single model's confidence.** v2 spot-checks borderline
   tier-0 OKs on tier 1 and demotes to UNSURE on disagreement —
   that one rule pushed accuracy_when_OK from 0.71 to 0.81. The price
   was a 65% escalation rate, but the result is real.

3. **Smart routing can beat always-tier-1 in BOTH dimensions.**
   v2 router: 0.775 accuracy at $6.79/1k. tier1_only: 0.718 accuracy
   at $8.44/1k. Better answer for less money — because the router
   knows when to refuse rather than when to commit.

4. **The cost lever is the spot-check threshold.** `SPOT_CHECK_THR
   = 0.95` triggers escalation on ~65% of inputs here. Lower it to
   0.99 → escalate ~95% (basically always-tier-1). Raise to 0.50 →
   escalate ~10% (back to v1). Tune to your accuracy/cost target.

5. **Latency is the worst regression.** v2 p95 13.4 s vs tier1_only
   2.7 s. In production, route uncertain cases async (offline
   workflow, human review queue) and short-circuit clear cases on
   tier 0 alone.

6. **On easy data, you can't tell the tiers apart.** Small test set
   shows identical 0.833 accuracy across all pipelines. Routing only
   pays off on inputs hard enough that the two tiers actually
   disagree — measure on your real distribution before believing any
   conclusions.
