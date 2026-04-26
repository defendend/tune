"""Micro-model for ticket-quality classification.

Uses OpenAI `text-embedding-3-small` (1536 dims, $0.02/1M tokens) to embed
every ticket, then a kNN majority vote against a fixed reference set of
labeled examples (the day-7 `correct.jsonl`). Returns:

  - verdict   : "Sufficient" | "Insufficient" | None
  - confidence: 0..1 margin between top-2 vote shares (binary task → either
                vote_share or |2 * vote_share - 1| depending on tiebreak)
  - status    : "OK"  if confidence >= MICRO_THR, else "UNSURE"

Reference set is loaded once and cached on disk so we only call the
embeddings endpoint for unseen items in production.

The micro-model has NO knowledge of the rules — it works purely by
similarity to known-labeled examples. That's the entire point: it should
handle the obvious cases without invoking a reasoning LLM.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE_PER_1M = 0.02  # USD / 1M tokens

# Decision threshold on the cosine-similarity margin between the top
# (sufficient vs insufficient) and second cluster. Anything below the
# threshold is "UNSURE" → escalate to the LLM fallback.
MICRO_THR = 0.10
TOP_K = 5

# Pre-filter: any description shorter than this OR consisting solely of
# whitespace/emoji is too trivial to even embed — return Insufficient
# directly with confidence=1.0.
MIN_MEANINGFUL_CHARS = 30

# kNN reference set is the day-7 public correct.jsonl. The references are
# in OpenAI/MLX JSONL format (`messages: [system, user, assistant]`); we
# parse the gold verdict from the assistant message.
ROOT = Path(__file__).resolve().parent
REFERENCE_FILE = ROOT.parent / "day7" / "correct.jsonl"
CACHE_FILE = ROOT / "embeddings_cache.json"


@dataclass
class MicroResult:
    verdict: Optional[str]      # "Sufficient" | "Insufficient" | None
    status: str                  # "OK" | "UNSURE"
    confidence: float            # margin in [0, 1]
    neighbours: list             # debugging: the top-k matched references
    embedding_calls: int
    cost_usd: float
    latency_ms: int


# ── helpers ───────────────────────────────────────────────────────────────────
def _embed_cost(tokens: int) -> float:
    return tokens / 1_000_000 * EMBED_PRICE_PER_1M


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _parse_label(assistant_text: str) -> Optional[str]:
    """Pull the verdict word out of the gold assistant message.
    Bilingual: handles both English (Verdict: Sufficient) and Russian
    (Вердикт: ✅ Достаточно)."""
    if not assistant_text:
        return None
    if "Insufficient" in assistant_text or "Недостаточно" in assistant_text or "❌" in assistant_text:
        return "Insufficient"
    if "Sufficient" in assistant_text or "Достаточно" in assistant_text or "✅" in assistant_text:
        return "Sufficient"
    return None


def _load_references() -> list[dict]:
    """Returns list of {description, label} dicts."""
    items = []
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"reference file {REFERENCE_FILE} not found")
    for line in REFERENCE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        msgs = obj.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        gold = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        label = _parse_label(gold)
        if user and label:
            items.append({"description": user.strip(), "label": label})
    return items


# ── reference embedding cache ────────────────────────────────────────────────
class MicroModel:
    """Encapsulates the embedding-based kNN classifier."""

    def __init__(self, client: Optional[OpenAI] = None,
                 *, k: int = TOP_K, threshold: float = MICRO_THR,
                 model: str = EMBED_MODEL,
                 reference_items: Optional[list[dict]] = None,
                 cache_file: Optional[Path] = None):
        """If `reference_items` is given, use those instead of loading from
        REFERENCE_FILE. Each item must be {description, label}. Useful for
        domain-specific reference sets (e.g. a held-out train split).
        `cache_file` lets you keep separate caches for different ref sets."""
        self.client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.k = k
        self.threshold = threshold
        self.model = model
        self.refs: list[dict] = []  # {description, label, embedding}
        self.refs_built = False
        self._injected_refs = reference_items
        self._cache_file = cache_file or CACHE_FILE

    # ── reference set lifecycle ──────────────────────────────────────────
    def _load_cache(self) -> Optional[list[dict]]:
        if not self._cache_file.exists():
            return None
        try:
            data = json.loads(self._cache_file.read_text(encoding="utf-8"))
            if data.get("model") != self.model:
                return None
            return data.get("refs") or None
        except Exception:
            return None

    def _save_cache(self) -> None:
        self._cache_file.write_text(
            json.dumps({"model": self.model, "refs": self.refs}, ensure_ascii=False),
            encoding="utf-8",
        )

    def build_references(self, force_rebuild: bool = False) -> None:
        """Embed the reference set (cached on disk).
        If `reference_items` was passed to __init__, use those; otherwise
        load from the default REFERENCE_FILE."""
        if not force_rebuild:
            cached = self._load_cache()
            if cached:
                self.refs = cached
                self.refs_built = True
                return
        items = list(self._injected_refs) if self._injected_refs is not None else _load_references()
        if not items:
            raise ValueError("no reference items available")
        texts = [it["description"] for it in items]
        resp = self.client.embeddings.create(model=self.model, input=texts)
        for it, emb in zip(items, resp.data):
            it["embedding"] = emb.embedding
        self.refs = items
        self.refs_built = True
        self._save_cache()
        print(f"[micro] embedded {len(items)} references → {self._cache_file.name}", flush=True)

    # ── inference ─────────────────────────────────────────────────────────
    def _trivial_filter(self, description: str) -> Optional[MicroResult]:
        """Cheap pre-filter: catch obviously-Insufficient inputs (empty,
        whitespace, single emoji string, gibberish < N alpha chars) without
        spending an embedding call."""
        text = (description or "").strip()
        # strip leading "## something" header to look at the body
        body = re.sub(r"^##.*?\n+", "", text).strip()
        # count alpha chars (any script) — emoji/digits don't count
        alpha = sum(1 for ch in body if ch.isalpha())
        if alpha < MIN_MEANINGFUL_CHARS:
            return MicroResult(
                verdict="Insufficient", status="OK", confidence=1.0,
                neighbours=[{"reason": f"trivial_filter alpha={alpha}<{MIN_MEANINGFUL_CHARS}"}],
                embedding_calls=0, cost_usd=0.0, latency_ms=0,
            )
        return None

    def classify(self, description: str) -> MicroResult:
        if not self.refs_built:
            self.build_references()

        # Pre-filter trivial inputs (no embedding call needed)
        triv = self._trivial_filter(description)
        if triv is not None:
            return triv

        t0 = time.time()
        try:
            resp = self.client.embeddings.create(model=self.model, input=[description])
            query_emb = resp.data[0].embedding
            tokens = resp.usage.total_tokens if resp.usage else max(1, len(description) // 4)
        except Exception as e:
            return MicroResult(
                verdict=None, status="UNSURE", confidence=0.0,
                neighbours=[], embedding_calls=1, cost_usd=0.0,
                latency_ms=int((time.time() - t0) * 1000),
            )

        # Score every reference; sort by cosine similarity desc.
        scored = []
        for ref in self.refs:
            sim = _cosine(query_emb, ref["embedding"])
            scored.append((sim, ref["label"], ref["description"]))
        scored.sort(key=lambda x: -x[0])
        top = scored[: self.k]

        # Vote, weighted by similarity
        votes: dict[str, float] = {"Sufficient": 0.0, "Insufficient": 0.0}
        for sim, label, _desc in top:
            votes[label] += max(sim, 0.0)
        total = sum(votes.values()) or 1.0
        share = {k: v / total for k, v in votes.items()}
        verdict = max(share, key=share.get)
        margin = abs(share["Sufficient"] - share["Insufficient"])
        status = "OK" if margin >= self.threshold else "UNSURE"

        return MicroResult(
            verdict=verdict,
            status=status,
            confidence=round(margin, 4),
            neighbours=[
                {"sim": round(s, 4), "label": l, "preview": d[:80]}
                for s, l, d in top
            ],
            embedding_calls=1,
            cost_usd=_embed_cost(tokens),
            latency_ms=int((time.time() - t0) * 1000),
        )


# ── module-level convenience ─────────────────────────────────────────────────
_DEFAULT: Optional[MicroModel] = None


def classify_micro(description: str, *, client: Optional[OpenAI] = None) -> MicroResult:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MicroModel(client=client)
        _DEFAULT.build_references()
    return _DEFAULT.classify(description)
