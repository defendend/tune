# day9 — multi-stage inference decomposition

The same ticket-quality task, re-implemented as a 3-stage pipeline of
short focused calls instead of one large monolithic call. We compare
both side-by-side on the same inputs and measure accuracy, parse
reliability, format adherence, cost, and latency.

## Two pipelines

### A. monolithic
One `gpt-4o-mini` call with the full system prompt. The model has to
do everything at once: classify task type, extract changes, judge
against 8+ rules, decide verdict, format the response.

### B. multi-stage (3 calls)

**Stage 1 — EXTRACT** (TOON-style features). Tiny prompt (~250 tokens)
asks the model to emit ONLY a key=value list:
```
type=feature, has_ui_location=false, has_user_trigger=false,
has_concrete_values=false, has_link=true, has_link_only=false,
has_contradiction=false, len_chars=42
```

**Stage 2 — VERDICT** (enum). Input: only the features from stage 1
(NOT the original description). Tiny prompt with the rule table.
Output is two lines:
```
verdict=Insufficient
rule=L
```

**Stage 3 — FORMAT** (4-section response). Input: original description
+ verdict + rule. Tiny formatter prompt generates the final response
in the same 4-section layout as the monolithic baseline.

## Files

- `multi_stage.py`        — 3-stage pipeline, `classify_multi_stage()` entry point
- `evaluate_multi.py`     — compares monolithic vs multi-stage
- `unified.py`            — composes multi-stage + day-7 self-check + day-8 tier-1 escalation
- `evaluate_unified.py`   — three-way comparison (monolithic / multi_stage / unified)

## Run

```bash
export OPENAI_API_KEY=sk-...
pip install openai
python3 evaluate_multi.py        # 2-way: mono vs multi-stage
python3 evaluate_unified.py      # 3-way: mono vs multi-stage vs unified
```

## Results

### Small public test set (24 inputs, ~400-token monolithic prompt)

```
pipeline        acc   parse    fmt  calls      $/req       $/1k   p50   p95
----------------------------------------------------------------------------------------
monolithic    0.833   0.958  0.958   1.00 $ 0.000095 $   0.0954  1.8s  2.3s
multi_stage   0.750   1.000  1.000   3.00 $ 0.000213 $   0.2133  4.8s  5.8s
```

### Larger production-like test set (71 inputs, ~2700-token monolithic prompt)

```
pipeline           acc   parse    fmt  calls      $/req       $/1k   p50   p95
----------------------------------------------------------------------------------------
monolithic        0.704   0.986  0.975   1.00 $ 0.000498 $   0.4976  2.7s  4.0s
multi_stage v1    0.634   1.000  1.000   3.00 $ 0.000264 $   0.2638  4.1s  5.2s
multi_stage v2    0.831   1.000  1.000   3.00 $ 0.000515 $   0.5155  4.9s  5.9s   ← winner
```

v1 lost ~7 pp accuracy because:
- Stage-2 only saw extracted features, not the original text — no recovery from
  Stage-1 misses
- Stage-1 and Stage-2 prompts were too terse, no counter-examples for the rules

v2 fixed both:
- Stage 2 now sees BOTH the structured features and the original description.
  The features are a hint; the rule engine can override them when the text
  clearly says otherwise.
- Stage 1 and Stage 2 prompts include 4 worked examples each (in a deliberately
  different domain — CRM/business, not the eval domain — to avoid leaking gold
  answers into the prompt).

### Per-stage breakdown (v2, large set)

```
extract  avg_cost=$0.000202  avg_in_tokens=1185  avg_out_tokens=39  avg_latency=1480ms
verdict  avg_cost=$0.000202  avg_in_tokens=1314  avg_out_tokens=7   avg_latency=1026ms
format   avg_cost=$0.000112  avg_in_tokens=400   avg_out_tokens=86  avg_latency=2323ms
```

### Unified pipeline — multi-stage + self-check + tier-1 escalation

`unified.py` composes everything from days 7-9 into one pipeline:

1. multi_stage v2 on tier 0 (gpt-4o-mini)        — 3 calls
2. self-check on tier-0 output                   — 1 short YES/NO call (day-7 trick)
3. if `p_yes >= 0.85` → trust tier 0
   else → escalate to tier 1 monolithic (gpt-4o, 1 call) and check agreement
   - tier-1 OK + same verdict       → OK
   - tier-1 OK + different verdict  → UNSURE
   - tier-1 broken format           → FAIL

Result on the same 71-input large test set:

```
pipeline            acc   when_OK   parse    fmt   esc%   calls   $/1k   p50/p95
----------------------------------------------------------------------------------------
monolithic        0.690    0.696    0.986   0.975   0%    1.00   $0.50   2.7s/4.2s
multi_stage v2    0.775    0.775    1.000   1.000   0%    3.00   $0.52   4.8s/6.2s
unified           0.803    0.852    1.000   1.000  72%    4.72   $6.58   6.7s/8.5s   <- best
```

Notes:
- `accuracy_when_OK = 0.852` is the highest of any pipeline in this set —
  when unified says OK, it's right ~85% of the time vs ~78% for multi_stage_v2
  and ~70% for monolithic.
- Cost goes 12× over multi_stage_v2 because the self-check threshold (0.85)
  forces escalation on ~72% of cases. Lowering the threshold to 0.7 cuts
  escalation roughly in half and brings cost to ~$2.5/1k — at the price of
  accepting tier-0 verdicts more readily.
- UNSURE status (8 cases here) is what makes unified different from a
  one-shot pipeline: it explicitly hands off ambiguous cases for human
  review instead of producing a confident wrong answer.
- 100% format/parse adherence inherited from multi-stage.

## Findings

1. **Decomposition trades accuracy for reliability.** Format adherence
   and parse rate go to 100% with multi-stage (vs 96-99% for monolithic
   — single calls occasionally collapse to a refusal-like 3-token
   response on adversarial input). The cost is ~8 pp of verdict
   accuracy in both test sets.

2. **The cost story flips with prompt size.** When the monolithic
   prompt is small (~400 tokens), multi-stage is **2.2× more expensive**
   (3 calls × small prompt > 1 call × small prompt). When the monolithic
   prompt is large (~2700 tokens), multi-stage is **1.9× cheaper**
   because the rule text is sent only once (in stage 2, very compactly)
   instead of being part of every call. This is the practical lesson:
   decomposition pays for itself once your monolithic prompt grows past
   ~1000 tokens.

3. **Information loss between stages explains the accuracy drop.**
   Stage 2 only sees the features extracted in stage 1. If stage 1
   misses a UI location buried in a long sentence, the rule engine in
   stage 2 has no way to recover — it over-fires rule L
   ("feature without UI location") on borderline cases the monolithic
   call would accept.

4. **Stage 2 doesn't actually need an LLM.** Once stage 2's input is
   structured features and its output is a pure enum, it could be a
   20-line `if`/`elif` ladder. That removes one call (~$0.05/1000 +
   ~1 s latency) entirely. We kept it as an LLM call here purely to
   demonstrate the multi-stage pattern; in production, replace it with
   code.

5. **Latency stacks up.** Three sequential calls × ~1.5 s each ≈
   4-5 s end-to-end vs ~2 s for the single call. Stages 2 and 3 depend
   on prior outputs, so this is not parallelisable here. If latency
   matters more than format guarantees, monolithic wins.

6. **The trade-off summary, in plain language**:
   - **Monolithic**: faster, more accurate on subtle cases, occasional
     format glitches.
   - **Multi-stage**: bulletproof format, cheaper on big-rule-set
     prompts, but loses some accuracy because each stage only sees a
     slice of the input.
