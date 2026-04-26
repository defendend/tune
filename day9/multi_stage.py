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
  has_ui_location=<true|false>      ← does the ticket name a screen / module / component?
  has_user_trigger=<true|false>     ← does it say WHEN the user sees it OR what user action triggers it?
  has_concrete_values=<true|false>  ← any numbers, exact strings, enum values, field examples?
  has_link=<true|false>             ← contains http(s) link OR markdown [text](url)?
  has_link_only=<true|false>        ← description is essentially JUST a link, < 80 chars of text outside
  has_contradiction=<true|false>    ← internal contradiction (e.g. "1x per session" vs "1x per install")
  len_chars=<integer>               ← length of description in characters

How to classify `type`:
  - tech-refactor → delete/remove/rename/migrate code, "выпилить", no UX change
  - api-contract  → new endpoint, header, body field, request/response, parameter
  - config-flag   → flag/experiment/config-driven behavior ("flag", "experiment", `feature_*`)
  - bugfix        → fix existing behavior (was X, now Y), "поправить", "починить"
  - feature       → otherwise

How to set `has_ui_location` (true if ANY of these):
  - names a specific screen ("home screen", "checkout", "cart", "feed", "Поиск", "Главный")
  - names a UI component ("pager", "banner", "header", "card", "shortcut", "сейф", "карточка фида")
  - names a module / section / scenario

How to set `has_user_trigger` (true if ANY of these):
  - explicit user action: tap, swipe, scroll, click, open, return, нажатие, свайп, возврат, открытие
  - explicit moment: "on app open", "после открытия", "при загрузке", "при выборе", "after login"
  - example trigger: "пролистывание пейджера" → true (pager + swipe trigger)
  - example trigger: "открываем экран" → true
  - example trigger: "при появлении" → true

How to set `has_concrete_values` (true if ANY of these):
  - numbers (font sizes, padding, retries, timeouts, status codes)
  - exact strings (header names, deeplink schemes, JSON field names)
  - enum values (`enabled: true/false`, "type=card|apple_pay")
  - localization keys (`empty_cart.title_v2`)
  - method + path (POST /orders, GET /api/v1/cart)

How to set `has_link_only`:
  - true ONLY if description is essentially nothing-but-a-link
  - false if there's a Figma/design link AND a meaningful sentence describing the change

EXAMPLES (different domain, for illustration only — DO NOT match patterns
to memorize, use the rules above to extract features for the actual input):

Description: "## CRM dashboard pinning\\nLet sales managers pin up to 3 customer records to the top of the dashboard list. Pin button appears on hover over a row in the dashboard."
Output: type=feature, has_ui_location=true, has_user_trigger=true, has_concrete_values=true, has_link=false, has_link_only=false, has_contradiction=false, len_chars=170

Description: "## Slack channel notifications\\nSet up new notifications."
Output: type=feature, has_ui_location=false, has_user_trigger=false, has_concrete_values=false, has_link=false, has_link_only=false, has_contradiction=false, len_chars=55

Description: "## Drop legacy ReportGeneratorV1\\nRemove ReportGeneratorV1, ReportLegacyExporter, and their unit tests after migration to ReportGeneratorV2 is complete."
Output: type=tech-refactor, has_ui_location=false, has_user_trigger=false, has_concrete_values=true, has_link=false, has_link_only=false, has_contradiction=false, len_chars=153

Description: "## Toggle analytics opt-out\\nFlag `analytics_opt_out`, field `enabled`. true → all analytics requests skipped, false → normal behaviour. Default false."
Output: type=config-flag, has_ui_location=false, has_user_trigger=false, has_concrete_values=true, has_link=false, has_link_only=false, has_contradiction=false, len_chars=147

Output exactly one line for the actual user input, no markdown, no explanation."""


# ── Stage 2 — verdict ────────────────────────────────────────────────────────
STAGE2_SYSTEM = """You are a verdict engine for a QA-readability check. You receive both
the original ticket description AND a pre-extracted feature summary. Decide whether
a QA engineer can write at least one happy-path test case from this description.

Output EXACTLY two lines, nothing else:
  verdict=<Sufficient|Insufficient>
  rule=<A|E|F|G|H|L|M>

Rules (apply in order, first match wins). Use the description as the source of truth;
the feature list is a hint, not gospel — override it if the text clearly says otherwise.

  E (contradiction):
    The description contains internally inconsistent requirements.
    Example: "show 1 time per session" together with "show 1 time per install" → Insufficient

  G (link only):
    Description is essentially just a URL with no description of what changes.
    Example: "see https://internal.example.com/thread/12345" → Insufficient
    NOT this rule if there is a Figma/design link PLUS a sentence describing the change.

  F (short and abstract, NOT tech-refactor):
    Generic verb + object with no concrete values, places, or scenarios.
    Examples → Insufficient: "Update icon", "Fix bug Y", "Support new promo type"
    DOES NOT apply to tech-refactor (which is sufficient with just "delete X" + class name).

  H (verb + link without parameters, feature only):
    Short action verb (increase / decrease / change / update / move) + a design link
    BUT no concrete values/numbers/states.
    Examples → Insufficient: "Increase header padding [Figma]", "Change banner color [Figma]"
    NOT this rule if the description has any concrete number, value, or enum.

  L (feature without integration scenario):
    type=feature AND the description does NOT make clear BOTH:
      (a) WHERE in the UI (screen/module/component), AND
      (b) WHEN/HOW the user encounters or triggers it.
    If EITHER (a) or (b) is clearly stated, do NOT fire L.
    Counter-examples (Sufficient, no L):
      - "Haptic on pager swipe" → has place (pager) AND trigger (swipe) ✓
      - "Force location refresh on home open" → place (home) AND trigger (open) ✓
      - "Pre-select method via deeplink app://payments?type=card" → trigger (deeplink) ✓
    L applies to:
      - "Support new promo type X [Figma]" → no where, no when ✗
      - "Add deeplink push_settings, opens with stack preserved" → trigger source unclear ✗
      - "New card layout in feed" → where vague, no trigger ✗

  M (config-flag without example value):
    type=config-flag AND no example value or effect mapping is shown.
    Insufficient: "Flag card_style, field style. See design."
    Sufficient:   "Flag lazy_init, enabled: bool. true → lazy, false → eager. Default false."

  A (default Sufficient):
    None of the above. tech-refactor and api-contract default to A as soon as they
    document the class/endpoint — do NOT require UI location or trigger for them.

EXAMPLES (illustrative, different domain — apply the rules to the actual
input rather than pattern-matching against these):

Description: "## Slack channel notifications\\nSet up new notifications."
Features: type=feature, has_ui_location=false, has_user_trigger=false, has_concrete_values=false, ...
Output:
verdict=Insufficient
rule=F

Description: "## CRM dashboard pinning\\nLet sales managers pin up to 3 customer records to the top of the dashboard list. Pin button appears on hover over a row in the dashboard."
Features: type=feature, has_ui_location=true, has_user_trigger=true, has_concrete_values=true, ...
Output:
verdict=Sufficient
rule=A

Description: "## Drop legacy ReportGeneratorV1\\nRemove ReportGeneratorV1, ReportLegacyExporter, and their unit tests."
Features: type=tech-refactor, has_ui_location=false, has_user_trigger=false, has_concrete_values=true, ...
Output:
verdict=Sufficient
rule=A

Description: "## New 'urgent' SLA tier\\nWe support a new SLA tier called 'urgent'. [Spec](...)"
Features: type=feature, has_ui_location=false, has_user_trigger=false, has_concrete_values=false, has_link=true, ...
Output:
verdict=Insufficient
rule=L

Description: "## Tighten table row spacing\\n[Design](...)"
Features: type=feature, has_ui_location=true, has_user_trigger=false, has_concrete_values=false, has_link=true, ...
Output:
verdict=Insufficient
rule=H

Description: "## Flag rate_limit\\nFlag `rate_limit`, field `max_per_min`. See playbook."
Features: type=config-flag, has_ui_location=false, has_user_trigger=false, has_concrete_values=false, ...
Output:
verdict=Insufficient
rule=M

Output exactly two lines, no prose, no markdown."""


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
    # v2: feed the verdict engine BOTH the structured features AND the original
    # description, so it can override mechanical rule-firing when the text
    # contradicts a missed feature. Cost: ~+150 tokens per call on average.
    s1_canonical = ", ".join(f"{k}={v}" for k, v in s1_kv.items()) or s1_text.strip()
    s2_user = (
        f"Description:\n{description}\n\n"
        f"Pre-extracted features:\n{s1_canonical}\n\n"
        "Decide verdict and rule per the system instructions."
    )
    s2_text, s2_pt, s2_ct, s2_ms = _call(
        client, model, STAGE2_SYSTEM, s2_user, max_tokens=30)
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
