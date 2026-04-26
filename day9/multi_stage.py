"""Multi-stage decomposition of the ticket-quality classifier.

Three short focused calls instead of one large monolithic prompt:

  Stage 1 — EXTRACT  (TOON-style structured features)
    Input  : raw ticket description
    Output : compact key=value features
             type=feature|bugfix|tech-refactor|api-contract|config-flag
             has_ui_location=true|false
             has_user_trigger=true|false
             has_concrete_values=true|false
             has_link=true|false
             has_link_only=true|false
             has_contradiction=true|false
             len_chars=<int>

  Stage 2 — VERDICT  (enum + rule)
    Input  : the features from stage 1 (NOT the full description)
    Output : verdict=Sufficient|Insufficient
             rule=A|B|C|D|E|F|G|H|L|M|none

  Stage 3 — FORMAT  (final structured response)
    Input  : original description + verdict + rule
    Output : the same 4-section layout as the monolithic baseline

Each stage gets a tiny system prompt (~200 tokens) instead of the
~1500-token monolithic rules prompt. Total tokens across 3 calls is
typically less than the single monolithic call because the rules text
is sent only once (in stage 2, very compactly).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

# Pricing (USD per 1M tokens) for gpt-4o-mini.
PRICE_INPUT_PER_1M = 0.150
PRICE_OUTPUT_PER_1M = 0.600


def _cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000) * PRICE_INPUT_PER_1M \
         + (completion_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M


# ── Stage 1 — extract features ────────────────────────────────────────────────
STAGE1_SYSTEM = """You are a feature extractor. Read the ticket description and emit ONLY
a key=value list, comma-separated, on a single line. Do NOT add prose.

Emit these keys exactly, in this order:
  type=<feature|bugfix|tech-refactor|api-contract|config-flag>
  has_ui_location=<true|false>      ← does the ticket name a screen/module?
  has_user_trigger=<true|false>     ← does it say WHEN the user sees/does it?
  has_concrete_values=<true|false>  ← any numbers, exact strings, enum values?
  has_link=<true|false>             ← contains http(s) link?
  has_link_only=<true|false>        ← is the description essentially just a link?
  has_contradiction=<true|false>    ← internal contradiction in requirements?
  len_chars=<integer>               ← length of description in characters

Rules of thumb:
  - "tech-refactor" → delete/remove/rename/migrate code, no UX change
  - "api-contract"  → new endpoint, header, body field, parameter
  - "config-flag"   → flag/experiment/config-driven behavior
  - "bugfix"        → fix existing behavior (was X, now Y)
  - "feature"       → otherwise

Output exactly one line, no markdown, no explanation."""


# ── Stage 2 — verdict ────────────────────────────────────────────────────────
STAGE2_SYSTEM = """You are a verdict engine. You receive a feature list extracted from a
ticket. Apply these rules in order and emit ONLY:
  verdict=<Sufficient|Insufficient>
  rule=<A|B|C|D|E|F|G|H|L|M|none>

Rules (apply in order, first match wins):

  E (contradiction): has_contradiction=true                          → Insufficient
  G (link only):     has_link_only=true                              → Insufficient
  F (too short):     len_chars<60 AND type≠tech-refactor             → Insufficient
  H (vague+link):    has_link=true AND has_concrete_values=false
                       AND type=feature                              → Insufficient
  L (feature):       type=feature
                       AND (has_ui_location=false OR has_user_trigger=false)
                                                                     → Insufficient
  M (config-flag):   type=config-flag AND has_concrete_values=false  → Insufficient

  A (default ok):    everything else                                 → Sufficient

For tech-refactor and api-contract, "code compiles / contract documented"
counts as sufficient — do NOT require has_ui_location or has_user_trigger.

Output exactly two lines, no prose, no markdown:
  verdict=...
  rule=..."""


# ── Stage 3 — format ──────────────────────────────────────────────────────────
STAGE3_SYSTEM = """You format the final ticket-quality verdict as 4 sections.

Use the EXACT format below. No preamble, no extra text.

  - Task type: <feature|bugfix|tech-refactor|api-contract|config-flag>
  - What changes: <one sentence; mark gaps as "gap: ..." but never reject for them>
  - Expected behavior: <one sentence describing the happy path>
  - Questions for developer: <"-" if Sufficient; OR up to 3 numbered questions if Insufficient>
  Verdict: <Sufficient|Insufficient>

If Verdict is Insufficient, the questions must be the BLOCKING ones implied
by the rule that triggered (E/F/G/H/L/M).
If Verdict is Sufficient, write "Questions for developer: -" exactly."""


# ── result type ──────────────────────────────────────────────────────────────
@dataclass
class StageRecord:
    name: str
    raw_output: str
    parsed: dict
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int


@dataclass
class MultiStageResult:
    verdict: Optional[str]
    rule: Optional[str]
    final_text: str
    parse_ok: bool
    calls: int
    cost_usd_total: float
    latency_ms_total: int
    stages: list[StageRecord] = field(default_factory=list)


# ── helpers ───────────────────────────────────────────────────────────────────
_KV = re.compile(r"([a-z_]+)\s*=\s*([^,\n]+)")


def _parse_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _KV.finditer(text or ""):
        out[m.group(1).strip()] = m.group(2).strip().rstrip(",")
    return out


def _call(client: OpenAI, model: str, system: str, user: str, *,
          max_tokens: int = 200, temperature: float = 0.0) -> tuple[str, int, int, int]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    text = resp.choices[0].message.content or ""
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    return text, pt, ct, elapsed_ms


# ── public API ────────────────────────────────────────────────────────────────
def classify_multi_stage(
    description: str,
    *,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_MODEL,
) -> MultiStageResult:
    """Three-stage decomposition: extract → verdict → format."""
    if client is None:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    t_start = time.time()
    stages: list[StageRecord] = []
    total_cost = 0.0

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    s1_text, s1_pt, s1_ct, s1_ms = _call(
        client, model, STAGE1_SYSTEM, description, max_tokens=120)
    s1_cost = _cost(s1_pt, s1_ct)
    s1_kv = _parse_kv(s1_text)
    stages.append(StageRecord(
        name="extract", raw_output=s1_text, parsed=s1_kv,
        prompt_tokens=s1_pt, completion_tokens=s1_ct,
        cost_usd=s1_cost, latency_ms=s1_ms))
    total_cost += s1_cost

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    # We feed only the canonical features back to the verdict engine, in the
    # SAME compact format. This is what makes stage 2 cheap.
    s1_canonical = ", ".join(f"{k}={v}" for k, v in s1_kv.items()) or s1_text.strip()
    s2_text, s2_pt, s2_ct, s2_ms = _call(
        client, model, STAGE2_SYSTEM, s1_canonical, max_tokens=30)
    s2_cost = _cost(s2_pt, s2_ct)
    s2_kv = _parse_kv(s2_text)
    stages.append(StageRecord(
        name="verdict", raw_output=s2_text, parsed=s2_kv,
        prompt_tokens=s2_pt, completion_tokens=s2_ct,
        cost_usd=s2_cost, latency_ms=s2_ms))
    total_cost += s2_cost

    verdict_word = (s2_kv.get("verdict") or "").strip().capitalize()
    rule_word = (s2_kv.get("rule") or "none").strip().upper()
    if verdict_word not in {"Sufficient", "Insufficient"}:
        verdict_word = None  # type: ignore[assignment]

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    s3_user = (
        f"Original description:\n{description}\n\n"
        f"Decision from previous stages:\n"
        f"  verdict={verdict_word}\n"
        f"  rule={rule_word}\n"
        f"  features={s1_canonical}\n\n"
        "Produce the 4-section formatted response."
    )
    s3_text, s3_pt, s3_ct, s3_ms = _call(
        client, model, STAGE3_SYSTEM, s3_user, max_tokens=400)
    s3_cost = _cost(s3_pt, s3_ct)
    stages.append(StageRecord(
        name="format", raw_output=s3_text, parsed={},
        prompt_tokens=s3_pt, completion_tokens=s3_ct,
        cost_usd=s3_cost, latency_ms=s3_ms))
    total_cost += s3_cost

    parse_ok = verdict_word is not None and bool(stages[2].raw_output.strip())

    return MultiStageResult(
        verdict=verdict_word,
        rule=rule_word,
        final_text=s3_text,
        parse_ok=parse_ok,
        calls=3,
        cost_usd_total=total_cost,
        latency_ms_total=int((time.time() - t_start) * 1000),
        stages=stages,
    )
