"""Confidence-aware inference for the ticket-quality classifier.

Three orthogonal layers stacked on top of a single OpenAI Chat Completions
call:

  1. Constraint-based check
     - the response must contain all 4 required sections
     - "Verdict:" must be one of {Sufficient, Insufficient}
     - if Insufficient → at most 3 questions
     If anything is broken → status=FAIL.

  2. Logprob-based scoring
     - we ask for `logprobs=True, top_logprobs=5` on the verdict tokens
     - we extract the model's probability for "Sufficient" vs "Insufficient"
       at the position right after "Verdict:"
     - confidence = max(p) - second(p) (margin)
     - confidence ≥ HIGH_THR → OK; LOW_THR..HIGH_THR → UNSURE; < LOW_THR → FAIL

  3. Redundancy / self-consistency (only if step 2 said UNSURE)
     - we re-run the same prompt 2 more times with temperature=0.7
     - majority vote across all 3 verdicts
     - if all 3 agree → upgrade to OK
     - if 2/3 agree → keep UNSURE, return majority verdict
     - if all 3 differ (impossible for 2 classes, but kept for the API) → FAIL

The escalation order is intentional: constraint is free (post-hoc on the
already-made call), logprobs are free (single call already gives them), and
redundancy is the only step that costs additional model calls.
"""
from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

# OpenAI gpt-4o-mini pricing as of 2024-07 (USD per 1M tokens)
PRICE_INPUT_PER_1M = 0.150
PRICE_OUTPUT_PER_1M = 0.600

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

# GPT-4o-mini emits very peaked distributions for binary verdicts (margin
# is typically 0.99+ even on noisy / adversarial inputs). Setting HIGH_THR
# below 0.95 means logprob alone always says "OK" and the redundancy layer
# never fires. We deliberately tighten the band so margin-based confidence
# acts as a real gate rather than a rubber stamp.
HIGH_THR = 0.95   # logprob margin above this → candidate OK (still self-checked)
LOW_THR = 0.50    # below this → FAIL
REDUNDANCY_N = 2  # additional calls when self-check disagrees
REDUNDANCY_T = 0.7

# Self-check thresholds (P(YES) from the judge call's first-token logprobs)
SELFCHECK_HIGH = 0.85   # judge confirms strongly → OK
SELFCHECK_LOW = 0.50    # below this → trigger redundancy

ALLOWED_VERDICTS = {"Sufficient", "Insufficient"}

# Bilingual section detectors — English or Russian section headers.
# Russian aliases let the same code score Russian responses if you swap in
# a Russian system prompt.
SECTIONS = [
    re.compile(r"-\s*(Task type|Тип задачи)\s*:", re.I),
    re.compile(r"-\s*(What changes|Что изменено)\s*:", re.I),
    re.compile(r"-\s*(Expected behavior|Ожидаемое поведение)\s*:", re.I),
    re.compile(r"(Verdict|Вердикт)\s*:", re.I),
]
# `Verdict:` followed by a word, possibly preceded by ✅/❌ in the Russian gold.
VERDICT_LINE = re.compile(r"(?:Verdict|Вердикт)\s*:\s*([✅❌]?\s*[A-Za-zА-Яа-я]+)")
QUESTION_LINE = re.compile(r"^\s*\d+\.\s+\S", re.MULTILINE)

# Map a model-emitted verdict word (any language / with emoji) to canonical
# {Sufficient, Insufficient}. Used for both response parsing and for picking
# verdict probability from the logprob distribution.
VERDICT_ALIASES = {
    "Sufficient": "Sufficient",
    "Достаточно": "Sufficient",
    "✅": "Sufficient",
    "Insufficient": "Insufficient",
    "Недостаточно": "Insufficient",
    "❌": "Insufficient",
}


@dataclass
class CheckResult:
    """Per-stage check result. Stages are appended to ConfidenceResult.stages."""
    name: str
    passed: bool
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceResult:
    verdict: Optional[str]
    status: str  # "OK" | "UNSURE" | "FAIL"
    confidence: float  # 0..1, derived margin
    reason: str
    stages: list[CheckResult] = field(default_factory=list)
    raw_text: Optional[str] = None
    raw_texts_redundancy: list[str] = field(default_factory=list)
    calls: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0


# ── helpers ───────────────────────────────────────────────────────────────────
def _normalize_verdict(token: str) -> Optional[str]:
    """Map any of {Sufficient, Insufficient, Достаточно, Недостаточно, ✅, ❌}
    to the canonical 'Sufficient' / 'Insufficient'. Tolerates leading/trailing
    whitespace and case differences."""
    if not token:
        return None
    s = token.strip()
    if not s:
        return None
    # exact match first
    if s in VERDICT_ALIASES:
        return VERDICT_ALIASES[s]
    # case-insensitive
    for alias, canon in VERDICT_ALIASES.items():
        if s.lower() == alias.lower():
            return canon
    # prefix match — common when the tokenizer splits "Suff" + "icient"
    s_low = s.lower()
    for alias, canon in VERDICT_ALIASES.items():
        if alias.lower().startswith(s_low) and len(s_low) >= 3:
            return canon
        if s_low.startswith(alias.lower()) and len(alias) >= 3:
            return canon
    return None


def _detect_verdict(text: str) -> Optional[str]:
    m = VERDICT_LINE.search(text or "")
    if not m:
        return None
    raw = m.group(1).strip()
    # if we matched an emoji + word, prefer the word
    parts = raw.split()
    for p in (parts[-1], parts[0]) if parts else (raw,):
        v = _normalize_verdict(p)
        if v:
            return v
    return None


def _format_score(text: str) -> int:
    return sum(1 for rx in SECTIONS if rx.search(text or ""))


def _count_questions(text: str) -> int:
    """Count numbered question lines under 'Questions for developer:'."""
    if not text:
        return 0
    # crude: count lines that start with "1.", "2.", "3." after the questions header
    m = re.search(r"Questions for developer:\s*(.*?)(?:Verdict:|$)", text, re.S)
    if not m:
        return 0
    block = m.group(1)
    if "—" in block or "-" == block.strip():
        return 0
    return len(QUESTION_LINE.findall(block))


def _cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000) * PRICE_INPUT_PER_1M + (completion_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M


# ── stage 1: constraint check ─────────────────────────────────────────────────
def constraint_check(text: str) -> CheckResult:
    issues: list[str] = []
    fmt = _format_score(text)
    if fmt < 4:
        issues.append(f"missing_sections({fmt}/4)")
    verdict = _detect_verdict(text)
    if verdict is None:
        issues.append("verdict_unrecognized")
    if verdict == "Insufficient":
        nq = _count_questions(text)
        if nq > 3:
            issues.append(f"too_many_questions({nq})")
    return CheckResult(
        name="constraint",
        passed=not issues,
        detail="ok" if not issues else "; ".join(issues),
        extra={"format_score": fmt, "verdict": verdict, "questions": _count_questions(text)},
    )


# ── stage 2: logprobs-based scoring ───────────────────────────────────────────
def _verdict_token_margin(choice) -> tuple[float, dict[str, float]]:
    """Find the token in the logprobs stream where the verdict word starts and
    return (margin_between_top2_classes, normalised_class_prob_map).

    Robust to the tokenizer either:
      - producing 'Verdict' + ':' + ' Sufficient' as 3 tokens
      - producing 'Verdict:' + ' Sufficient' as 2 tokens
      - producing ' Verdict' + ':' + ' Insufficient' (with leading space) etc.

    Strategy: accumulate token text; once the running buffer contains a
    'Verdict:' or 'Вердикт:' marker, the very next non-whitespace token is the
    verdict word. We then aggregate top_logprobs at that position into the
    canonical {Sufficient, Insufficient} distribution."""
    logprobs = getattr(choice, "logprobs", None)
    if not logprobs or not getattr(logprobs, "content", None):
        return 0.0, {}

    tokens = logprobs.content
    buf = ""
    marker_re = re.compile(r"(verdict|вердикт)\s*:", re.I)
    seen_marker = False

    for entry in tokens:
        token_text = entry.token or ""
        buf += token_text

        if not seen_marker:
            if marker_re.search(buf):
                seen_marker = True
            continue

        # After the marker — wait until we hit a token with non-whitespace content
        if not token_text.strip():
            continue

        # Build distribution from top_logprobs at THIS position
        probs: dict[str, float] = {}
        for alt in (entry.top_logprobs or []):
            canon = _normalize_verdict(alt.token)
            if not canon:
                continue
            probs[canon] = probs.get(canon, 0.0) + math.exp(alt.logprob)

        # Include the actually sampled token if not already represented
        sampled_canon = _normalize_verdict(token_text)
        if sampled_canon and sampled_canon not in probs:
            probs[sampled_canon] = math.exp(entry.logprob)

        if not probs:
            return 0.0, {}

        # Normalise across the two known classes only (anything else → 0).
        total = sum(probs.values()) or 1.0
        normed = {k: round(v / total, 4) for k, v in probs.items()}
        sorted_p = sorted(normed.values(), reverse=True)
        margin = sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0)
        return round(margin, 4), normed

    return 0.0, {}


# ── stage 2.5: semantic self-check via a separate judge call ─────────────────
SELFCHECK_SYSTEM = (
    "You are a strict quality reviewer. You will be given a software ticket "
    "description and another reviewer's verdict on whether a QA engineer can "
    "write at least one test case from the description. Your job is to judge "
    "whether the verdict is CORRECT.\n\n"
    "Reply with exactly one word: YES if the verdict is correct, NO if it is wrong. "
    "Do not explain. Do not add anything else."
)


def _yes_no_margin(choice) -> tuple[float, dict[str, float]]:
    """Extract P(YES) - P(NO) from the first content token's logprobs."""
    logprobs = getattr(choice, "logprobs", None)
    if not logprobs or not getattr(logprobs, "content", None):
        return 0.0, {}

    for entry in logprobs.content:
        token_text = (entry.token or "").strip().upper()
        if not token_text:
            continue
        probs: dict[str, float] = {"YES": 0.0, "NO": 0.0}
        for alt in (entry.top_logprobs or []):
            t = (alt.token or "").strip().upper()
            if t.startswith("YES") or t == "Y":
                probs["YES"] += math.exp(alt.logprob)
            elif t.startswith("NO") or t == "N":
                probs["NO"] += math.exp(alt.logprob)
        # also include the actually-sampled token
        if token_text.startswith("YES") or token_text == "Y":
            probs["YES"] = max(probs["YES"], math.exp(entry.logprob))
        elif token_text.startswith("NO") or token_text == "N":
            probs["NO"] = max(probs["NO"], math.exp(entry.logprob))

        total = probs["YES"] + probs["NO"]
        if total <= 0:
            return 0.0, {}
        normed = {k: round(v / total, 4) for k, v in probs.items()}
        return round(normed["YES"], 4), normed
    return 0.0, {}


def _self_check(client, model: str, description: str, verdict: str, reasoning: str
                ) -> tuple[float, dict[str, float], int, int, str]:
    """Returns (P(yes), prob_map, prompt_tokens, completion_tokens, raw)."""
    msgs = [
        {"role": "system", "content": SELFCHECK_SYSTEM},
        {"role": "user", "content": (
            f"Ticket description:\n{description}\n\n"
            f"Verdict given: {verdict}\n\n"
            f"Reviewer reasoning:\n{reasoning}\n\n"
            "Is this verdict correct?"
        )},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        max_tokens=3,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
    )
    choice = resp.choices[0]
    p_yes, probs = _yes_no_margin(choice)
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    return p_yes, probs, pt, ct, (choice.message.content or "").strip()


# ── core: one inference call ──────────────────────────────────────────────────
def _one_call(client, model: str, messages: list[dict], *, with_logprobs: bool, temperature: float) -> tuple[Any, str, int, int]:
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": temperature,
    }
    if with_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    text = choice.message.content or ""
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    return choice, text, pt, ct


# ── main entry point ──────────────────────────────────────────────────────────
def classify_with_confidence(
    description: str,
    system_prompt: str,
    *,
    client=None,
    model: str = DEFAULT_MODEL,
    enable_redundancy: bool = True,
    enable_selfcheck: bool = True,
) -> ConfidenceResult:
    """Run the three-layer confidence-aware classifier on a single description."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": description},
    ]

    result = ConfidenceResult(verdict=None, status="FAIL", confidence=0.0, reason="")

    t_start = time.time()

    # ── primary call (with logprobs) ─────────────────────────────────────────
    try:
        choice, text, pt, ct = _one_call(client, model, messages, with_logprobs=True, temperature=0.0)
    except Exception as e:
        result.reason = f"api_error: {e}"
        result.stages.append(CheckResult("api", False, str(e)))
        result.latency_ms = int((time.time() - t_start) * 1000)
        return result

    result.calls += 1
    result.prompt_tokens_total += pt
    result.completion_tokens_total += ct
    result.cost_usd += _cost(pt, ct)
    result.raw_text = text

    # ── stage 1: constraint check ────────────────────────────────────────────
    cstr = constraint_check(text)
    result.stages.append(cstr)
    if not cstr.passed:
        result.status = "FAIL"
        result.reason = f"constraint_failed: {cstr.detail}"
        result.latency_ms = int((time.time() - t_start) * 1000)
        return result

    primary_verdict = cstr.extra["verdict"]
    result.verdict = primary_verdict

    # ── stage 2: logprob scoring ─────────────────────────────────────────────
    margin, probs = _verdict_token_margin(choice)
    result.confidence = round(margin, 4)
    result.stages.append(CheckResult(
        name="logprob_scoring",
        passed=margin >= LOW_THR,
        detail=f"margin={margin:.3f} probs={probs}",
        extra={"margin": margin, "probs": probs},
    ))

    if margin < LOW_THR:
        # too uncertain to even bother with redundancy — escalate
        result.status = "FAIL"
        result.reason = f"low_confidence margin={margin:.3f}"
        result.latency_ms = int((time.time() - t_start) * 1000)
        return result

    # ── stage 2.5: SEMANTIC self-check (a separate judge call) ─────────────
    # We always self-check at this point — even high-margin verdicts get a
    # second-opinion pass. The model's own logit confidence is uncalibrated
    # for subjective judgment tasks; the judge call gives an orthogonal signal.
    p_yes = 1.0  # default if disabled
    selfcheck_probs: dict[str, float] = {}
    if enable_selfcheck:
        try:
            p_yes, selfcheck_probs, pts, cts, judge_raw = _self_check(
                client, model, description, primary_verdict, text)
            result.calls += 1
            result.prompt_tokens_total += pts
            result.completion_tokens_total += cts
            result.cost_usd += _cost(pts, cts)
            result.stages.append(CheckResult(
                name="self_check",
                passed=p_yes >= SELFCHECK_LOW,
                detail=f"p_yes={p_yes:.3f} probs={selfcheck_probs} judge='{judge_raw}'",
                extra={"p_yes": p_yes, "probs": selfcheck_probs, "judge_raw": judge_raw},
            ))
        except Exception as e:
            result.stages.append(CheckResult("self_check", False, f"api_error: {e}"))
            p_yes = 0.5  # neutral on failure → triggers redundancy

    # Combined decision: both signals must agree to OK.
    if margin >= HIGH_THR and p_yes >= SELFCHECK_HIGH:
        result.status = "OK"
        result.reason = f"high_confidence margin={margin:.3f} p_yes={p_yes:.3f}"
        result.latency_ms = int((time.time() - t_start) * 1000)
        return result

    if not enable_redundancy:
        # Margin or self-check is borderline — keep verdict but mark UNSURE.
        result.status = "UNSURE"
        result.reason = (f"borderline margin={margin:.3f} p_yes={p_yes:.3f} "
                         f"redundancy_disabled")
        result.latency_ms = int((time.time() - t_start) * 1000)
        return result

    # ── stage 3: redundancy ──────────────────────────────────────────────────
    extra_verdicts: list[Optional[str]] = []
    for _ in range(REDUNDANCY_N):
        try:
            _choice, text2, pt2, ct2 = _one_call(client, model, messages, with_logprobs=False, temperature=REDUNDANCY_T)
        except Exception as e:
            result.stages.append(CheckResult("redundancy_call", False, f"api_error: {e}"))
            continue
        result.calls += 1
        result.prompt_tokens_total += pt2
        result.completion_tokens_total += ct2
        result.cost_usd += _cost(pt2, ct2)
        result.raw_texts_redundancy.append(text2)
        v = constraint_check(text2).extra.get("verdict")
        extra_verdicts.append(v)

    all_verdicts = [primary_verdict, *extra_verdicts]
    counter = Counter(v for v in all_verdicts if v)
    if not counter:
        result.status = "FAIL"
        result.reason = "redundancy_all_unparseable"
        result.latency_ms = int((time.time() - t_start) * 1000)
        return result

    top, top_count = counter.most_common(1)[0]
    total_valid = sum(counter.values())
    result.verdict = top
    agreement = top_count / total_valid

    result.stages.append(CheckResult(
        name="redundancy",
        passed=agreement >= 1.0,
        detail=f"verdicts={all_verdicts} agreement={agreement:.2f}",
        extra={"verdicts": all_verdicts, "agreement": agreement, "n": total_valid},
    ))

    if agreement >= 1.0 and total_valid >= 2:
        # full agreement (e.g. 3/3) — upgrade to OK
        result.status = "OK"
        result.reason = f"redundancy_unanimous ({top_count}/{total_valid})"
    elif agreement >= 2 / 3:
        result.status = "UNSURE"
        result.reason = f"redundancy_majority ({top_count}/{total_valid})"
    else:
        result.status = "FAIL"
        result.reason = f"redundancy_split ({top_count}/{total_valid})"

    result.latency_ms = int((time.time() - t_start) * 1000)
    return result
