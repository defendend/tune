"""Unified pipeline combining day-9 multi-stage + day-7 self-check
+ day-8 tier-1 escalation.

Flow:

  1. Multi-stage v2 on tier 0 (gpt-4o-mini)
       -> 3 calls: extract -> verdict -> format
       -> produces a verdict, a rule, and a 4-section response

  2. Self-check (1 short call, tier 0)
       -> a YES/NO judge call on (description, verdict, response)
       -> p_yes from first-token logprobs

  3. Decision:
       p_yes >= SELFCHECK_HIGH (0.85)            -> OK, return tier-0 answer
       p_yes <  SELFCHECK_HIGH                   -> escalate to tier 1
            tier-1 monolithic call (gpt-4o, single inference + constraint check)
            tier-1 verdict matches tier 0        -> OK
            tier-1 verdict differs from tier 0   -> UNSURE
            tier-1 broken (constraint fail)      -> final FAIL

The result includes a `route` field for auditability, the same
RouteResult-style trace, and per-stage cost/latency.
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from multi_stage import (
    DEFAULT_MODEL as TIER0_MODEL,
    PRICE_INPUT_PER_1M as T0_IN,
    PRICE_OUTPUT_PER_1M as T0_OUT,
    _cost as tier0_call_cost,
    classify_multi_stage,
)

# Tier 1 — strong, more expensive.
TIER1_MODEL = "gpt-4o-2024-08-06"
T1_IN = 2.50
T1_OUT = 10.00


def tier1_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000) * T1_IN + (completion_tokens / 1_000_000) * T1_OUT


SELFCHECK_HIGH = 0.85
SELFCHECK_LOW = 0.50

SELFCHECK_SYSTEM = (
    "You are a strict reviewer. You will be shown a software ticket description "
    "and another reviewer's verdict on whether a QA engineer can write at least "
    "one happy-path test from the description. Judge whether the verdict is "
    "CORRECT.\n\n"
    "Reply with exactly one word: YES if the verdict is correct, NO if wrong. "
    "Do not explain. Do not add anything else."
)


# ── result dataclass ─────────────────────────────────────────────────────────
@dataclass
class UnifiedResult:
    verdict: Optional[str]
    status: str                    # "OK" | "UNSURE" | "FAIL"
    confidence: float              # p_yes from tier-0 self-check
    route: str                     # "tier0_ok" | "tier0_to_tier1" | "tier1_disagree" | "tier1_fail"
    final_text: str                # 4-section response (from whichever tier won)
    escalated: bool
    escalation_reason: str
    tier0_verdict: Optional[str]
    tier0_rule: Optional[str]
    tier0_text: str
    tier0_calls: int
    tier0_cost_usd: float
    tier0_latency_ms: int
    selfcheck_p_yes: float
    selfcheck_calls: int
    selfcheck_cost_usd: float
    selfcheck_latency_ms: int
    tier1_verdict: Optional[str] = None
    tier1_text: Optional[str] = None
    tier1_calls: int = 0
    tier1_cost_usd: float = 0.0
    tier1_latency_ms: int = 0
    calls_total: int = 0
    cost_usd_total: float = 0.0
    latency_ms_total: int = 0
    stages: list = field(default_factory=list)


# ── helpers (bilingual EN / RU) ───────────────────────────────────────────────
SECTIONS = [
    re.compile(r"-\s*(Task type|Тип задачи)\s*:", re.I),
    re.compile(r"-\s*(What changes|Что изменено)\s*:", re.I),
    re.compile(r"-\s*(Expected behavior|Ожидаемое поведение)\s*:", re.I),
    re.compile(r"(Verdict|Вердикт)\s*:", re.I),
]
VERDICT_LINE = re.compile(r"(?:Verdict|Вердикт)\s*:\s*([✅❌]?\s*[A-Za-zА-Яа-я]+)")
VERDICT_ALIASES = {
    "Sufficient": "Sufficient", "Достаточно": "Sufficient", "✅": "Sufficient",
    "Insufficient": "Insufficient", "Недостаточно": "Insufficient", "❌": "Insufficient",
}


def _normalize_verdict(token: str) -> Optional[str]:
    if not token:
        return None
    s = token.strip()
    if not s:
        return None
    for alias, canon in VERDICT_ALIASES.items():
        if s.lower() == alias.lower():
            return canon
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
    parts = raw.split()
    for p in (parts[-1], parts[0]) if parts else (raw,):
        v = _normalize_verdict(p)
        if v:
            return v
    return None


def _format_passed(text: str) -> bool:
    return all(rx.search(text or "") for rx in SECTIONS)


def _selfcheck(client: OpenAI, model: str, description: str, verdict: str,
               reasoning: str) -> tuple[float, int, int, int, str]:
    """Returns (p_yes, prompt_tokens, completion_tokens, latency_ms, raw)."""
    msgs = [
        {"role": "system", "content": SELFCHECK_SYSTEM},
        {"role": "user", "content": (
            f"Ticket description:\n{description}\n\n"
            f"Verdict given: {verdict}\n\n"
            f"Reviewer reasoning:\n{reasoning}\n\n"
            "Is this verdict correct?"
        )},
    ]
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        max_tokens=3,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    choice = resp.choices[0]
    raw = (choice.message.content or "").strip()
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0

    # Pull P(YES) from the first content token's logprobs
    p_yes = 0.0
    lp = getattr(choice, "logprobs", None)
    if lp and getattr(lp, "content", None):
        for entry in lp.content:
            tok = (entry.token or "").strip().upper()
            if not tok:
                continue
            yes = no_ = 0.0
            for alt in (entry.top_logprobs or []):
                t = (alt.token or "").strip().upper()
                if t.startswith("YES") or t == "Y":
                    yes += math.exp(alt.logprob)
                elif t.startswith("NO") or t == "N":
                    no_ += math.exp(alt.logprob)
            sampled = tok
            if sampled.startswith("YES") or sampled == "Y":
                yes = max(yes, math.exp(entry.logprob))
            elif sampled.startswith("NO") or sampled == "N":
                no_ = max(no_, math.exp(entry.logprob))
            total = yes + no_
            p_yes = (yes / total) if total > 0 else 0.5
            break
    return round(p_yes, 4), pt, ct, elapsed_ms, raw


def _tier1_monolithic(client: OpenAI, model: str, system_prompt: str,
                      description: str) -> dict:
    """Single inference + constraint check on tier-1."""
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": description},
            ],
            max_tokens=512,
            temperature=0,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        cost = tier1_cost(pt, ct)
    except Exception as e:
        return {
            "ok": False, "verdict": None, "text": None, "cost": 0.0,
            "latency_ms": int((time.time() - t0) * 1000),
            "reason": f"api_error: {e}",
        }
    if not _format_passed(text):
        return {
            "ok": False, "verdict": None, "text": text, "cost": cost,
            "latency_ms": elapsed_ms,
            "reason": "tier1_constraint_fail",
        }
    return {
        "ok": True, "verdict": _detect_verdict(text), "text": text,
        "cost": cost, "latency_ms": elapsed_ms,
        "reason": "tier1_ok",
    }


# ── public API ────────────────────────────────────────────────────────────────
def classify_unified(
    description: str,
    monolithic_system_prompt: str,
    *,
    client: Optional[OpenAI] = None,
    tier0_model: str = TIER0_MODEL,
    tier1_model: str = TIER1_MODEL,
) -> UnifiedResult:
    """Full unified pipeline: multi-stage + self-check + tier-1 fallback."""
    if client is None:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    t_start = time.time()

    # ── Step 1: multi-stage v2 on tier 0 ─────────────────────────────────────
    ms = classify_multi_stage(description, client=client, model=tier0_model)
    tier0_calls = ms.calls
    tier0_cost = ms.cost_usd_total
    tier0_latency = ms.latency_ms_total
    tier0_verdict = ms.verdict
    tier0_text = ms.final_text
    tier0_rule = ms.rule

    # If multi-stage didn't even produce a parseable verdict, escalate immediately
    needs_escalation = tier0_verdict is None
    p_yes = 0.0
    sc_pt = sc_ct = sc_ms = 0
    sc_raw = ""

    # ── Step 2: self-check on tier-0 output (if verdict parsed) ──────────────
    if tier0_verdict is not None:
        try:
            p_yes, sc_pt, sc_ct, sc_ms, sc_raw = _selfcheck(
                client, tier0_model, description, tier0_verdict, tier0_text)
        except Exception:
            p_yes = 0.5  # neutral on failure → triggers escalation
        needs_escalation = p_yes < SELFCHECK_HIGH

    sc_cost = tier0_call_cost(sc_pt, sc_ct) if sc_pt or sc_ct else 0.0

    # ── Step 3a: tier-0 OK path ──────────────────────────────────────────────
    if not needs_escalation:
        return UnifiedResult(
            verdict=tier0_verdict, status="OK", confidence=p_yes,
            route="tier0_ok", final_text=tier0_text,
            escalated=False, escalation_reason="",
            tier0_verdict=tier0_verdict, tier0_rule=tier0_rule, tier0_text=tier0_text,
            tier0_calls=tier0_calls, tier0_cost_usd=tier0_cost, tier0_latency_ms=tier0_latency,
            selfcheck_p_yes=p_yes, selfcheck_calls=1,
            selfcheck_cost_usd=sc_cost, selfcheck_latency_ms=sc_ms,
            calls_total=tier0_calls + 1,
            cost_usd_total=tier0_cost + sc_cost,
            latency_ms_total=int((time.time() - t_start) * 1000),
            stages=[{"name": s.name, "cost": s.cost_usd, "latency_ms": s.latency_ms,
                     "in_tokens": s.prompt_tokens, "out_tokens": s.completion_tokens}
                    for s in ms.stages] + [{"name": "self_check", "cost": sc_cost,
                                            "latency_ms": sc_ms, "in_tokens": sc_pt,
                                            "out_tokens": sc_ct, "p_yes": p_yes,
                                            "judge_raw": sc_raw}],
        )

    # ── Step 3b: escalate to tier-1 monolithic ───────────────────────────────
    t1 = _tier1_monolithic(client, tier1_model, monolithic_system_prompt, description)
    tier1_calls = 1
    tier1_cost = t1["cost"]
    tier1_latency = t1["latency_ms"]

    if not t1["ok"]:
        # tier-1 also broken
        return UnifiedResult(
            verdict=tier0_verdict, status="FAIL",
            confidence=p_yes,
            route="tier1_fail", final_text=tier0_text,
            escalated=True,
            escalation_reason=f"tier0_p_yes={p_yes:.3f}; {t1['reason']}",
            tier0_verdict=tier0_verdict, tier0_rule=tier0_rule, tier0_text=tier0_text,
            tier0_calls=tier0_calls, tier0_cost_usd=tier0_cost, tier0_latency_ms=tier0_latency,
            selfcheck_p_yes=p_yes, selfcheck_calls=1 if tier0_verdict is not None else 0,
            selfcheck_cost_usd=sc_cost, selfcheck_latency_ms=sc_ms,
            tier1_verdict=None, tier1_text=t1["text"],
            tier1_calls=tier1_calls, tier1_cost_usd=tier1_cost,
            tier1_latency_ms=tier1_latency,
            calls_total=tier0_calls + (1 if tier0_verdict is not None else 0) + tier1_calls,
            cost_usd_total=tier0_cost + sc_cost + tier1_cost,
            latency_ms_total=int((time.time() - t_start) * 1000),
        )

    # tier-1 produced a verdict — compare with tier-0
    if tier0_verdict is None or t1["verdict"] == tier0_verdict:
        # Either tier-0 had no verdict (use tier-1 outright), or they agree
        return UnifiedResult(
            verdict=t1["verdict"], status="OK", confidence=max(p_yes, 0.95),
            route="tier0_to_tier1", final_text=t1["text"],
            escalated=True,
            escalation_reason=f"tier0_p_yes={p_yes:.3f}; tier1_OK and {'replaced' if tier0_verdict is None else 'agreed'}",
            tier0_verdict=tier0_verdict, tier0_rule=tier0_rule, tier0_text=tier0_text,
            tier0_calls=tier0_calls, tier0_cost_usd=tier0_cost, tier0_latency_ms=tier0_latency,
            selfcheck_p_yes=p_yes, selfcheck_calls=1 if tier0_verdict is not None else 0,
            selfcheck_cost_usd=sc_cost, selfcheck_latency_ms=sc_ms,
            tier1_verdict=t1["verdict"], tier1_text=t1["text"],
            tier1_calls=tier1_calls, tier1_cost_usd=tier1_cost,
            tier1_latency_ms=tier1_latency,
            calls_total=tier0_calls + (1 if tier0_verdict is not None else 0) + tier1_calls,
            cost_usd_total=tier0_cost + sc_cost + tier1_cost,
            latency_ms_total=int((time.time() - t_start) * 1000),
        )

    # tier-0 and tier-1 disagree → UNSURE, return tier-1 verdict (more expensive model wins)
    return UnifiedResult(
        verdict=t1["verdict"], status="UNSURE", confidence=p_yes,
        route="tier1_disagree", final_text=t1["text"],
        escalated=True,
        escalation_reason=(
            f"tier0_v={tier0_verdict} vs tier1_v={t1['verdict']} (disagreement); "
            f"tier0_p_yes={p_yes:.3f}"
        ),
        tier0_verdict=tier0_verdict, tier0_rule=tier0_rule, tier0_text=tier0_text,
        tier0_calls=tier0_calls, tier0_cost_usd=tier0_cost, tier0_latency_ms=tier0_latency,
        selfcheck_p_yes=p_yes, selfcheck_calls=1,
        selfcheck_cost_usd=sc_cost, selfcheck_latency_ms=sc_ms,
        tier1_verdict=t1["verdict"], tier1_text=t1["text"],
        tier1_calls=tier1_calls, tier1_cost_usd=tier1_cost,
        tier1_latency_ms=tier1_latency,
        calls_total=tier0_calls + 1 + tier1_calls,
        cost_usd_total=tier0_cost + sc_cost + tier1_cost,
        latency_ms_total=int((time.time() - t_start) * 1000),
    )
