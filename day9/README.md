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

- `multi_stage.py`     — 3-stage pipeline, `classify_multi_stage()` entry point
- `evaluate_multi.py`  — runs both pipelines on day-7's test sets and prints comparison

## Run

```bash
export OPENAI_API_KEY=sk-...
pip install openai
python3 evaluate_multi.py
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
pipeline        acc   parse    fmt  calls      $/req       $/1k   p50   p95
----------------------------------------------------------------------------------------
monolithic    0.718   0.986  0.986   1.00 $ 0.000497 $   0.4966  2.4s  3.7s
multi_stage   0.634   1.000  1.000   3.00 $ 0.000264 $   0.2638  4.1s  5.2s
```

### Per-stage breakdown (large set)

```
extract  avg_cost=$0.000092  avg_in_tokens=453  avg_out_tokens=39   avg_latency=1178ms
verdict  avg_cost=$0.000053  avg_in_tokens=320  avg_out_tokens=7    avg_latency=789ms
format   avg_cost=$0.000119  avg_in_tokens=400  avg_out_tokens=98   avg_latency=2164ms
```

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
