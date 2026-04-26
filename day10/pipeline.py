"""Two-tier pipeline: micro-model first, LLM fallback on UNSURE.

  1. Micro-model (embedding kNN, see micro.py)
       → if status="OK"     → trust the verdict, no LLM call
       → if status="UNSURE"  → escalate to fallback

  2. LLM fallback — multi-stage v2 from day-9 (3 short calls)
       → produces a full structured response

Per-request trace records which path was taken, the micro-model verdict,
the fallback verdict (if invoked), call counts, latency, and cost.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
DAY9 = ROOT.parent / "day9"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DAY9))

from micro import MICRO_THR, MicroModel, MicroResult  # noqa: E402
from multi_stage import classify_multi_stage  # noqa: E402
from unified import classify_unified  # noqa: E402


@dataclass
class PipelineResult:
    verdict: Optional[str]           # final verdict
    status: str                       # OK | UNSURE | FAIL
    handled_by: str                   # "micro" | "llm_fallback"
    micro: MicroResult                # always present
    llm_text: Optional[str] = None    # multi-stage's final 4-section output (if escalated)
    llm_rule: Optional[str] = None
    llm_calls: int = 0
    llm_cost_usd: float = 0.0
    llm_latency_ms: int = 0
    calls_total: int = 0              # micro embed call + llm calls
    cost_usd_total: float = 0.0
    latency_ms_total: int = 0


def classify_pipeline(
    description: str,
    *,
    client: Optional[OpenAI] = None,
    micro: Optional[MicroModel] = None,
    fallback: str = "multi_stage",
    monolithic_system_prompt: Optional[str] = None,
) -> PipelineResult:
    """Two-tier inference.

    fallback ∈ {"multi_stage", "unified"}.
      multi_stage — cheaper (3 calls, gpt-4o-mini), accuracy ~0.78
      unified     — multi_stage + self-check + tier-1 escalation, accuracy 0.80,
                    requires `monolithic_system_prompt` (used only on tier-1 fallback inside)
    """
    if client is None:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if micro is None:
        micro = MicroModel(client=client)
        micro.build_references()

    t_start = time.time()

    # ── Tier 1: micro-model ──────────────────────────────────────────────
    m = micro.classify(description)

    if m.status == "OK":
        return PipelineResult(
            verdict=m.verdict, status="OK", handled_by="micro",
            micro=m,
            calls_total=m.embedding_calls,
            cost_usd_total=m.cost_usd,
            latency_ms_total=int((time.time() - t_start) * 1000),
        )

    # ── Tier 2: LLM fallback ─────────────────────────────────────────────
    if fallback == "unified":
        if not monolithic_system_prompt:
            raise ValueError("unified fallback requires monolithic_system_prompt")
        u = classify_unified(description, monolithic_system_prompt, client=client)
        return PipelineResult(
            verdict=u.verdict, status=u.status,
            handled_by="llm_fallback",
            micro=m,
            llm_text=u.final_text, llm_rule=None,
            llm_calls=u.calls_total - 0,  # micro embedding is separate
            llm_cost_usd=u.cost_usd_total,
            llm_latency_ms=u.latency_ms_total,
            calls_total=m.embedding_calls + u.calls_total,
            cost_usd_total=m.cost_usd + u.cost_usd_total,
            latency_ms_total=int((time.time() - t_start) * 1000),
        )

    ms = classify_multi_stage(description, client=client)
    return PipelineResult(
        verdict=ms.verdict, status=("OK" if ms.verdict else "FAIL"),
        handled_by="llm_fallback",
        micro=m,
        llm_text=ms.final_text, llm_rule=ms.rule,
        llm_calls=ms.calls, llm_cost_usd=ms.cost_usd_total,
        llm_latency_ms=ms.latency_ms_total,
        calls_total=m.embedding_calls + ms.calls,
        cost_usd_total=m.cost_usd + ms.cost_usd_total,
        latency_ms_total=int((time.time() - t_start) * 1000),
    )
