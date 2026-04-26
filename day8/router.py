"""Two-tier model router with confidence-based escalation (v2).

Improvements over v1:

  • Tier-1 self-check.   After tier-1 inference, run a short YES/NO judge
    call (the same trick as day-7). If tier 1 also fails its own self-check
    → final UNSURE instead of falsely-confident OK.

  • Borderline tier-0 spot-check.   When tier 0 returns OK but the
    self-check probability was below SPOT_CHECK_THR, we still run tier 1
    once. If tier 0 and tier 1 disagree on the verdict → downgrade to
    UNSURE. This catches the day-7 finding that gpt-4o-mini is sometimes
    confidently wrong.

  • Verdict disagreement handling.   On any path where tier 0 has a
    verdict and tier 1 produces a different verdict, the result is
    UNSURE — neither model is trusted.

Strategy:

  Tier 0 — gpt-4o-mini, day-7 confidence pipeline (constraint + logprob +
            self-check + redundancy).
  Tier 1 — gpt-4o, single inference + constraint + self-check.

Routing decisions:

  tier 0 OK, p_yes >= SPOT_CHECK_THR        → trust, no escalation       (cheap)
  tier 0 OK, p_yes <  SPOT_CHECK_THR        → spot-check on tier 1       (medium)
       tier 1 verdict matches tier 0        → OK
       tier 1 verdict differs / unsure      → UNSURE
  tier 0 UNSURE                             → escalate to tier 1         (expensive)
       tier 1 OK + agrees with tier 0       → OK
       tier 1 OK + disagrees with tier 0    → UNSURE
       tier 1 unsure / FAIL                 → UNSURE
  tier 0 FAIL                               → escalate to tier 1
       tier 1 OK                            → OK
       tier 1 broken                        → final FAIL
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from confidence import (
    DEFAULT_MODEL as TIER0_DEFAULT,
    SELFCHECK_HIGH,
    SELFCHECK_LOW,
    classify_with_confidence,
    constraint_check,
    _cost,  # noqa: F401  (re-exported for callers)
    _one_call,
    _self_check,
)

# Tier 1 — strong, more expensive model. Pricing as of 2024-08 ($/1M tokens).
TIER1_DEFAULT = "gpt-4o-2024-08-06"
TIER1_PRICE_INPUT_PER_1M = 2.50
TIER1_PRICE_OUTPUT_PER_1M = 10.00

# Spot-check threshold: tier-0 OKs with p_yes (from tier-0 self-check) below
# this still trigger a tier-1 verification call.
SPOT_CHECK_THR = 0.95


def _tier1_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000) * TIER1_PRICE_INPUT_PER_1M \
         + (completion_tokens / 1_000_000) * TIER1_PRICE_OUTPUT_PER_1M


@dataclass
class RouteResult:
    verdict: Optional[str]
    status: str                # "OK" | "UNSURE" | "FAIL"
    confidence: float          # final confidence (from whichever signal won)
    route: str                 # see _ROUTE_LABELS below
    escalated: bool            # was tier 1 invoked at all?
    escalation_reason: str
    tier0_status: str
    tier1_status: Optional[str]
    calls_total: int
    cost_usd_total: float
    latency_ms_total: int
    tier0_calls: int
    tier1_calls: int
    tier0_cost: float
    tier1_cost: float
    tier0_latency_ms: int
    tier1_latency_ms: int
    tier0_text: Optional[str]
    tier1_text: Optional[str]
    tier0_reason: str
    tier1_p_yes: Optional[float]    # tier-1 self-check probability (None if not run)
    tier0_p_yes: Optional[float]    # tier-0 self-check probability (from day-7)
    stages: list = field(default_factory=list)


def _extract_tier0_p_yes(stages: list) -> Optional[float]:
    """Pull the p_yes value from the day-7 self_check stage if it ran."""
    for s in stages:
        if getattr(s, "name", None) == "self_check":
            extra = getattr(s, "extra", {}) or {}
            if "p_yes" in extra:
                return float(extra["p_yes"])
    return None


def _tier1_inference_with_selfcheck(
    client, model: str, system_prompt: str, description: str,
) -> dict:
    """Single inference + constraint + self-check on the strong model.

    Returns a dict with verdict, status (OK / UNSURE / FAIL), text,
    cost, latency, calls, p_yes (from the judge call), and a reason.
    """
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": description},
    ]
    t0 = time.time()
    try:
        choice, text, pt, ct = _one_call(client, model, msgs, with_logprobs=False, temperature=0.0)
    except Exception as e:
        return {
            "ok": False, "verdict": None, "status": "FAIL", "text": None,
            "calls": 1, "cost": 0.0,
            "latency_ms": int((time.time() - t0) * 1000),
            "p_yes": None, "reason": f"api_error: {e}",
        }
    cost = _tier1_cost(pt, ct)

    cstr = constraint_check(text)
    if not cstr.passed:
        return {
            "ok": False, "verdict": None, "status": "FAIL", "text": text,
            "calls": 1, "cost": cost,
            "latency_ms": int((time.time() - t0) * 1000),
            "p_yes": None, "reason": f"constraint_failed: {cstr.detail}",
        }

    verdict = cstr.extra["verdict"]

    # Tier-1 self-check (a YES/NO judge call on tier-1's own answer)
    p_yes = 1.0
    judge_raw = ""
    try:
        p_yes, _probs, jpt, jct, judge_raw = _self_check(
            client, model, description, verdict, text)
        cost += _tier1_cost(jpt, jct)
    except Exception as e:
        # If self-check fails, treat as inconclusive
        p_yes = 0.5
        judge_raw = f"selfcheck_error: {e}"

    elapsed_ms = int((time.time() - t0) * 1000)

    if p_yes >= SELFCHECK_HIGH:
        status = "OK"
    elif p_yes >= SELFCHECK_LOW:
        status = "UNSURE"
    else:
        # tier-1 itself is unsure → don't trust, but keep the verdict for trace
        status = "UNSURE"

    return {
        "ok": status == "OK",
        "verdict": verdict,
        "status": status,
        "text": text,
        "calls": 2,         # inference + self-check
        "cost": cost,
        "latency_ms": elapsed_ms,
        "p_yes": p_yes,
        "reason": (f"tier1_{status.lower()} p_yes={p_yes:.3f} "
                   f"judge='{judge_raw}'"),
    }


def _wrap(
    cr, t1: Optional[dict], *, route_label: str, escalated: bool,
    escalation_reason: str, status: str, verdict: Optional[str],
    confidence: float, t_start: float,
) -> RouteResult:
    tier0_p_yes = _extract_tier0_p_yes(cr.stages)
    return RouteResult(
        verdict=verdict, status=status, confidence=confidence,
        route=route_label, escalated=escalated,
        escalation_reason=escalation_reason,
        tier0_status=cr.status,
        tier1_status=(t1["status"] if t1 else None),
        calls_total=cr.calls + (t1["calls"] if t1 else 0),
        cost_usd_total=cr.cost_usd + (t1["cost"] if t1 else 0.0),
        latency_ms_total=int((time.time() - t_start) * 1000),
        tier0_calls=cr.calls,
        tier1_calls=(t1["calls"] if t1 else 0),
        tier0_cost=cr.cost_usd,
        tier1_cost=(t1["cost"] if t1 else 0.0),
        tier0_latency_ms=cr.latency_ms,
        tier1_latency_ms=(t1["latency_ms"] if t1 else 0),
        tier0_text=cr.raw_text,
        tier1_text=(t1["text"] if t1 else None),
        tier0_reason=cr.reason,
        tier1_p_yes=(t1["p_yes"] if t1 else None),
        tier0_p_yes=tier0_p_yes,
        stages=cr.stages,
    )


def route(
    description: str,
    system_prompt: str,
    *,
    client=None,
    tier0_model: str = TIER0_DEFAULT,
    tier1_model: str = TIER1_DEFAULT,
    enable_tier0_redundancy: bool = True,
    enable_tier0_selfcheck: bool = True,
    enable_spot_check: bool = True,
) -> RouteResult:
    """Run two-tier routed inference (v2). Returns a RouteResult with full trace."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    t_start = time.time()

    # ── tier 0 ─────────────────────────────────────────────────────────────
    cr = classify_with_confidence(
        description, system_prompt,
        client=client,
        model=tier0_model,
        enable_redundancy=enable_tier0_redundancy,
        enable_selfcheck=enable_tier0_selfcheck,
    )

    tier0_p_yes = _extract_tier0_p_yes(cr.stages)

    # ── tier 0 OK: maybe spot-check ────────────────────────────────────────
    if cr.status == "OK":
        # If the tier-0 self-check probability is borderline, verify on tier 1.
        needs_spot_check = (
            enable_spot_check
            and tier0_p_yes is not None
            and tier0_p_yes < SPOT_CHECK_THR
        )
        if not needs_spot_check:
            return _wrap(
                cr, None,
                route_label="tier0_ok",
                escalated=False, escalation_reason="",
                status="OK", verdict=cr.verdict,
                confidence=cr.confidence, t_start=t_start,
            )

        # Spot-check on tier 1
        t1 = _tier1_inference_with_selfcheck(client, tier1_model, system_prompt, description)
        if t1["status"] == "FAIL":
            # tier 1 broke format on a case tier 0 was confident about — downgrade
            return _wrap(
                cr, t1,
                route_label="tier0_ok_spotcheck_fail",
                escalated=True,
                escalation_reason=f"spot_check tier0_p_yes={tier0_p_yes:.3f} tier1_constraint_fail",
                status="UNSURE", verdict=cr.verdict,
                confidence=cr.confidence, t_start=t_start,
            )

        if t1["verdict"] == cr.verdict:
            # both agree → keep OK
            return _wrap(
                cr, t1,
                route_label="tier0_ok_spotcheck_agree",
                escalated=True,
                escalation_reason=f"spot_check tier0_p_yes={tier0_p_yes:.3f}",
                status="OK", verdict=cr.verdict,
                confidence=max(cr.confidence, t1["p_yes"] or 0.0),
                t_start=t_start,
            )

        # disagreement → UNSURE
        return _wrap(
            cr, t1,
            route_label="tier0_ok_spotcheck_disagree",
            escalated=True,
            escalation_reason=(
                f"spot_check tier0_p_yes={tier0_p_yes:.3f} "
                f"tier0_v={cr.verdict} tier1_v={t1['verdict']}"
            ),
            status="UNSURE", verdict=t1["verdict"],
            confidence=t1["p_yes"] or 0.0, t_start=t_start,
        )

    # ── tier 0 UNSURE / FAIL: escalate to tier 1 ───────────────────────────
    t1 = _tier1_inference_with_selfcheck(client, tier1_model, system_prompt, description)

    if t1["status"] == "FAIL":
        # Tier 1 also broken — final FAIL
        return _wrap(
            cr, t1,
            route_label="tier1_fallback_fail",
            escalated=True,
            escalation_reason=f"tier0_{cr.status.lower()}: {cr.reason}; tier1_failed",
            status="FAIL", verdict=cr.verdict,
            confidence=cr.confidence, t_start=t_start,
        )

    if t1["status"] == "UNSURE":
        # Tier 1 produced a verdict but is itself unsure → UNSURE
        return _wrap(
            cr, t1,
            route_label="tier0_to_tier1_unsure",
            escalated=True,
            escalation_reason=f"tier0_{cr.status.lower()}; tier1_self_check_unsure",
            status="UNSURE", verdict=t1["verdict"],
            confidence=t1["p_yes"] or 0.0, t_start=t_start,
        )

    # Tier 1 OK — check agreement with tier 0 (if tier 0 had a verdict)
    if cr.verdict and t1["verdict"] != cr.verdict:
        return _wrap(
            cr, t1,
            route_label="tier0_to_tier1_disagree",
            escalated=True,
            escalation_reason=(
                f"tier0_{cr.status.lower()} v={cr.verdict}; "
                f"tier1_OK v={t1['verdict']} (disagreement)"
            ),
            status="UNSURE", verdict=t1["verdict"],
            confidence=t1["p_yes"] or 0.0, t_start=t_start,
        )

    # Tier 1 OK and agrees (or tier 0 had no verdict) — accept tier 1
    return _wrap(
        cr, t1,
        route_label="tier0_to_tier1",
        escalated=True,
        escalation_reason=f"tier0_{cr.status.lower()}: {cr.reason}",
        status="OK", verdict=t1["verdict"],
        confidence=t1["p_yes"] or 1.0, t_start=t_start,
    )
