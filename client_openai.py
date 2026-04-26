"""OpenAI fine-tuning client.

Validates local files, uploads them via /v1/files (purpose=fine-tune),
creates a fine-tuning job via /v1/fine_tuning/jobs, polls status until
'succeeded' / 'failed' / 'cancelled'.

Usage:
  export OPENAI_API_KEY=sk-...
  python3 client_openai.py --model gpt-4o-mini-2024-07-18
  python3 client_openai.py --dry-run                        # validate only
  python3 client_openai.py --resume ftjob-...               # poll an existing job

Requires: pip install openai
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN = DATA / "train.jsonl"
VALID = DATA / "valid.jsonl"

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
POLL_INTERVAL_S = 30
JOB_LOG = ROOT / "openai_job.json"


def lazy_openai():
    try:
        import openai  # noqa: F401
    except ImportError:
        print("FATAL: openai SDK not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("FATAL: OPENAI_API_KEY not set in env", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)


def quick_validate(path: Path) -> int:
    """Quick check that the file is in OpenAI fine-tune format (messages array,
    valid roles, non-empty content). Returns the number of errors (0 = ok)."""
    if not path.exists():
        print(f"  [E] {path} not found", file=sys.stderr)
        return 1
    errs = 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [E] {path}:{i} bad JSON: {e}", file=sys.stderr)
                errs += 1
                continue
            msgs = obj.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 2:
                print(f"  [E] {path}:{i} bad messages", file=sys.stderr)
                errs += 1
                continue
            for j, m in enumerate(msgs):
                if m.get("role") not in {"system", "user", "assistant"}:
                    print(f"  [E] {path}:{i}.{j} bad role {m.get('role')}", file=sys.stderr)
                    errs += 1
                if not isinstance(m.get("content"), str) or not m["content"].strip():
                    print(f"  [E] {path}:{i}.{j} empty content", file=sys.stderr)
                    errs += 1
    print(f"  {path.name}: {n} examples, {errs} errors", file=sys.stderr)
    return errs


def upload_file(client, path: Path) -> str:
    print(f"\n-> uploading {path.name} ...", file=sys.stderr)
    with path.open("rb") as f:
        resp = client.files.create(file=f, purpose="fine-tune")
    print(f"  file_id={resp.id} status={resp.status} bytes={resp.bytes}", file=sys.stderr)
    return resp.id


def create_job(client, model: str, train_id: str, valid_id: str | None) -> str:
    print(f"\n-> creating fine-tuning job (model={model}) ...", file=sys.stderr)
    kwargs: dict = {"training_file": train_id, "model": model}
    if valid_id:
        kwargs["validation_file"] = valid_id
    job = client.fine_tuning.jobs.create(**kwargs)
    print(f"  job_id={job.id} status={job.status}", file=sys.stderr)
    return job.id


def poll(client, job_id: str) -> dict:
    print(f"\n-> polling job {job_id} (every {POLL_INTERVAL_S}s) ...", file=sys.stderr)
    last_status = None
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        if job.status != last_status:
            print(
                f"  [{time.strftime('%H:%M:%S')}] status={job.status} "
                f"trained_tokens={getattr(job, 'trained_tokens', None)} "
                f"fine_tuned_model={job.fine_tuned_model}",
                file=sys.stderr,
            )
            last_status = job.status
        if job.status in ("succeeded", "failed", "cancelled"):
            return job.model_dump() if hasattr(job, "model_dump") else dict(job)
        time.sleep(POLL_INTERVAL_S)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dry-run", action="store_true", help="local validation only, no upload")
    p.add_argument("--no-valid", action="store_true", help="do not upload valid.jsonl")
    p.add_argument("--resume", default=None, help="poll an existing job by id")
    args = p.parse_args()

    if args.resume:
        client = lazy_openai()
        result = poll(client, args.resume)
        JOB_LOG.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\nFinal: status={result.get('status')}  ft_model={result.get('fine_tuned_model')}")
        return 0 if result.get("status") == "succeeded" else 1

    print("-> validating local files ...", file=sys.stderr)
    errs = quick_validate(TRAIN)
    if not args.no_valid:
        errs += quick_validate(VALID)
    if errs:
        print(f"FATAL: {errs} validation errors -- fix before upload", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\nDRY RUN ok -- formats look valid for OpenAI.")
        return 0

    client = lazy_openai()
    train_id = upload_file(client, TRAIN)
    valid_id = None if args.no_valid else upload_file(client, VALID)
    job_id = create_job(client, args.model, train_id, valid_id)
    result = poll(client, job_id)
    JOB_LOG.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nFinal: status={result.get('status')}  ft_model={result.get('fine_tuned_model')}")
    print(f"\nFull job dump -> {JOB_LOG}")
    return 0 if result.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
