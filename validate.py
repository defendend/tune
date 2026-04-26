"""JSONL dataset validator for fine-tuning.

Checks:
  - each line is a valid JSON object
  - has a `messages` key — list of >= 2 items
  - roles are in order: optional `system` -> `user` -> `assistant` (one or more turns)
  - all `content` fields are non-empty strings
  - user content length within [MIN_USER_CHARS, MAX_USER_CHARS]
  - assistant content length within [MIN_ASSISTANT_CHARS, MAX_ASSISTANT_CHARS]
  - duplicates by `user.content` are reported as warnings
  - full duplicates (same system+user+assistant) are reported as errors
  - prints a summary: per-label counts, length stats, approx token count

Usage:
  python3 validate.py data/all.jsonl
  python3 validate.py data/train.jsonl data/valid.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

MAX_USER_CHARS = 8000
MAX_ASSISTANT_CHARS = 4000
MIN_USER_CHARS = 5
MIN_ASSISTANT_CHARS = 20
ALLOWED_ROLES = {"system", "user", "assistant"}


def validate_file(path: Path) -> tuple[int, int]:
    errors = 0
    warnings = 0
    seen_full: dict[str, int] = {}
    seen_user: dict[str, list[int]] = {}
    expected_counts: Counter[str] = Counter()
    user_lens: list[int] = []
    asst_lens: list[int] = []
    sys_lens: list[int] = []

    if not path.exists():
        print(f"  [!] file not found: {path}")
        return 1, 0

    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [E] line {i}: invalid JSON: {e}")
                errors += 1
                continue

            if not isinstance(obj, dict):
                print(f"  [E] line {i}: top-level must be object")
                errors += 1
                continue

            msgs = obj.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 2:
                print(f"  [E] line {i}: 'messages' must be list of >= 2 items")
                errors += 1
                continue

            roles = [m.get("role") for m in msgs]
            for j, m in enumerate(msgs):
                if not isinstance(m, dict):
                    print(f"  [E] line {i}: messages[{j}] not an object")
                    errors += 1
                    continue
                role = m.get("role")
                content = m.get("content")
                if role not in ALLOWED_ROLES:
                    print(f"  [E] line {i}: messages[{j}] bad role: {role!r}")
                    errors += 1
                if not isinstance(content, str):
                    print(f"  [E] line {i}: messages[{j}] content must be string")
                    errors += 1
                    continue
                if not content.strip():
                    print(f"  [E] line {i}: messages[{j}] empty content")
                    errors += 1
                if role == "user":
                    user_lens.append(len(content))
                    if len(content) > MAX_USER_CHARS:
                        print(f"  [E] line {i}: user content > {MAX_USER_CHARS} chars")
                        errors += 1
                    if len(content) < MIN_USER_CHARS:
                        print(f"  [E] line {i}: user content < {MIN_USER_CHARS} chars")
                        errors += 1
                elif role == "assistant":
                    asst_lens.append(len(content))
                    if len(content) > MAX_ASSISTANT_CHARS:
                        print(f"  [E] line {i}: assistant content > {MAX_ASSISTANT_CHARS} chars")
                        errors += 1
                    if len(content) < MIN_ASSISTANT_CHARS:
                        print(f"  [E] line {i}: assistant content < {MIN_ASSISTANT_CHARS} chars")
                        errors += 1
                elif role == "system":
                    sys_lens.append(len(content))

            non_system = [r for r in roles if r != "system"]
            if not non_system or non_system[0] != "user":
                print(f"  [E] line {i}: first non-system role must be 'user' (got {non_system[:1]})")
                errors += 1
            if "assistant" not in roles:
                print(f"  [E] line {i}: no assistant message")
                errors += 1
            elif roles[-1] != "assistant":
                print(f"  [E] line {i}: last role must be 'assistant' (got {roles[-1]})")
                errors += 1

            full_key = hashlib.sha256(
                json.dumps([{"role": m.get("role"), "content": m.get("content", "")} for m in msgs],
                           ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if full_key in seen_full:
                print(f"  [E] line {i}: full duplicate of line {seen_full[full_key]}")
                errors += 1
            else:
                seen_full[full_key] = i

            user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            user_key = hashlib.sha256(user_msg.encode("utf-8")).hexdigest()
            seen_user.setdefault(user_key, []).append(i)

            meta = obj.get("_meta") or {}
            exp = meta.get("expected") or "?"
            expected_counts[exp] += 1

    for _k, lines in seen_user.items():
        if len(lines) > 1:
            print(f"  [W] duplicate user content on lines {lines}")
            warnings += 1

    n = len(seen_full)
    print(f"\n  total: {n}")
    print(f"  expected: {dict(expected_counts)}")
    if user_lens:
        print(f"  user chars  : min={min(user_lens)}  avg={sum(user_lens)//len(user_lens)}  max={max(user_lens)}")
    if asst_lens:
        print(f"  asst chars  : min={min(asst_lens)}  avg={sum(asst_lens)//len(asst_lens)}  max={max(asst_lens)}")
    if sys_lens:
        print(f"  system chars: min={min(sys_lens)}  avg={sum(sys_lens)//len(sys_lens)}  max={max(sys_lens)}")
    total_chars = sum(user_lens + asst_lens + sys_lens)
    print(f"  ~tokens (all roles): {max(1, total_chars // 3)}")

    return errors, warnings


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"usage: {argv[0]} <file.jsonl> [more.jsonl ...]")
        return 2

    total_err = total_warn = 0
    for arg in argv[1:]:
        path = Path(arg)
        print(f"\n-- {path} --")
        e, w = validate_file(path)
        total_err += e
        total_warn += w
        print(f"  errors: {e}, warnings: {w}")

    print(f"\nSUMMARY: errors={total_err}, warnings={total_warn}")
    return 1 if total_err else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
