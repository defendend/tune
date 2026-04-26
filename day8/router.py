"""Two-tier model router with confidence-based escalation.

Strategy:
  Tier 0 — cheap & fast (gpt-4o-mini). Run the day-7 confidence pipeline
            (constraint + logprob + self-check + redundancy). If it returns
            status="OK", we trust the answer and stop.
  Tier 1 — strong & expensive (gpt-4o). Only invoked when tier-0 produced
            UNSURE or FAIL. We run a single inference + constraint check on
            tier 1 (no redundancy — it's already the heavy hitter).

The routing heuristic is the composite confidence from day 7:
  • OK at tier 0      → no escalation, use tier-0 answer
  • UNSURE at tier 0  → escalate (model didn't agree with itself)
  • FAIL at tier 0    → escalate (constraint broken or low logit margin)
  • FAIL at tier 1    → final FAIL (tier 1 also broke format)

The result includes a `route` field that records which tier handled it and
why escalation happened, so we can audit "what stayed cheap vs. what got
expensive".
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from confidence import (
    DEFAULT_MODEL as TIER0_DEFAULT,
    PRICE_INPUT_PER_1M,
    PRICE_OUTPUT_PER_1M,
    classify_with_confidence,
    constraint_check,
    _detect_verdict,
    _cost,
    _one_call,
)

# Tier 1 — strong, more expensive model. Pricing as of 2024-08 ($/1M tokens).
TIER1_DEFAULT = "gpt-4o-2024-08-06"
TIER1_PRICE_INPUT_PER_1M = 2.50
TIER1_PRICE_OUTPUT_PER_1M = 10.00


def _tier1_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000) * TIER1_PRICE_INPUT_PER_1M \
         + (completion_tokens / 1_000_000) * TIER1_PRICE_OUTPUT_PER_1M


@dataclass
class RouteResult:
    verdict: Optional[str]
    status: str                # "OK" | "UNSURE" | "FAIL"
    confidence: float          # final confidence (from whichever tier won)
    route: str                 # "tier0_ok" | "tier0_to_tier1" | "tier1_fallback_fail"
    escalated: bool
    escalation_reason: str     # why tier 1 was invoked (or "" if not)
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
    stages: list = field(default_factory=list)


def _tier1_inference(client, model: str, system_prompt: str, description: str) -> dict:
    """Single inference + constraint check on the strong model. No redundancy."""
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
            "calls": 1, "cost": 0.0, "latency_ms": int((time.time() - t0) * 1000),
            "reason": f"api_error: {e}",
        }
    elapsed_ms = int((time.time() - t0) * 1000)
    cost = _tier1_cost(pt, ct)
    cstr = constraint_check(text)
    if not cstr.passed:
        return {
            "ok": False, "verdict": None, "status": "FAIL", "text": text,
            "calls": 1, "cost": cost, "latency_ms": elapsed_ms,
            "reason": f"constraint_failed: {cstr.detail}",
        }
    return {
        "ok": True, "verdict": cstr.extra["verdict"], "status": "OK",
        "text": text, "calls": 1, "cost": cost, "latency_ms": elapsed_ms,
        "reason": "tier1_ok",
    }


def route(
    description: str,
    system_prompt: str,
    *,
    client=None,
    tier0_model: str = TIER0_DEFAULT,
    tier1_model: str = TIER1_DEFAULT,
    enable_tier0_redundancy: bool = True,
    enable_tier0_selfcheck: bool = True,
) -> RouteResult:
    """Run two-tier routed inference. Returns a RouteResult with full trace."""
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

    if cr.status == "OK":
        # Trust tier 0 — no escalation
        return RouteResult(
            verdict=cr.verdict, status="OK", confidence=cr.confidence,
            route="tier0_ok", escalated=False, escalation_reason="",
            tier0_status=cr.status, tier1_status=None,
            calls_total=cr.calls, cost_usd_total=cr.cost_usd,
            latency_ms_total=int((time.time() - t_start) * 1000),
            tier0_calls=cr.calls, tier1_calls=0,
            tier0_cost=cr.cost_usd, tier1_cost=0.0,
            tier0_latency_ms=cr.latency_ms, tier1_latency_ms=0,
            tier0_text=cr.raw_text, tier1_text=None,
            tier0_reason=cr.reason,
            stages=cr.stages,
        )

    # ── escalate to tier 1 ────────────────────────────────────────────────
    t1 = _tier1_inference(client, tier1_model, system_prompt, description)
    final_status = "OK" if t1["ok"] else "FAIL"
    route_label = "tier0_to_tier1" if t1["ok"] else "tier1_fallback_fail"
    return RouteResult(
        verdict=t1["verdict"] if t1["ok"] else cr.verdict,
        status=final_status,
        confidence=1.0 if t1["ok"] else cr.confidence,
        route=route_label,
        escalated=True,
        escalation_reason=f"tier0_{cr.status.lower()}: {cr.reason}",
        tier0_status=cr.status, tier1_status=t1["status"],
        calls_total=cr.calls + t1["calls"],
        cost_usd_total=cr.cost_usd + t1["cost"],
        latency_ms_total=int((time.time() - t_start) * 1000),
        tier0_calls=cr.calls, tier1_calls=t1["calls"],
        tier0_cost=cr.cost_usd, tier1_cost=t1["cost"],
        tier0_latency_ms=cr.latency_ms, tier1_latency_ms=t1["latency_ms"],
        tier0_text=cr.raw_text, tier1_text=t1["text"],
        tier0_reason=cr.reason,
        stages=cr.stages,
    )
