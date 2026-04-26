# tune

End-to-end fine-tuning pipeline for a structured-output classifier.

The toy task: a model reads a software ticket description and outputs a
structured analysis plus a binary verdict (`Sufficient` / `Insufficient`)
saying whether a QA engineer can write a test from the description.

The same pipeline works for any classification + structured-output task —
swap `examples.jsonl` and `prompts/system.md`.

**Stack:** Apple MLX + LoRA on `mlx-community/Qwen2.5-7B-Instruct-4bit`,
local on Apple Silicon, no CUDA. Optional OpenAI fine-tune client included.

## Layout

```
.
├── prompts/
│   └── system.md             # system prompt with the rules
├── examples.jsonl            # source records: {description, expected, assistant}
├── data/                     # generated train/valid/test (gitignored)
├── build_dataset.py          # examples.jsonl -> data/all.jsonl
├── validate.py               # JSONL validator: format / dups / lengths
├── split.py                  # stratified 70/15/15 split
├── baseline.py               # local Qwen baseline -> baseline_results.json
├── baseline_openai.py        # gpt-4o-mini baseline -> baseline_openai_results.json
├── infer_mlx.py              # local Qwen + LoRA adapter -> finetuned_results.json
├── train_config.yaml         # MLX-LM LoRA hyperparameters
├── train_mlx.sh              # one-line wrapper for mlx_lm.lora
├── client_openai.py          # upload + create job + poll for OpenAI fine-tuning
├── criteria.md               # accept/reject criteria for the fine-tune
├── requirements.txt
└── README.md
```

## Workflow

```bash
# 0. Install
pip install -r requirements.txt

# 1. Build dataset
python3 build_dataset.py
python3 validate.py data/all.jsonl

# 2. Stratified 70/15/15 split
python3 split.py
python3 validate.py data/train.jsonl data/valid.jsonl data/test.jsonl

# 3. Baselines (no fine-tune)
python3 baseline.py --n 10                           # local Qwen
export OPENAI_API_KEY=sk-...
python3 baseline_openai.py --n 10                    # gpt-4o-mini

# 4. Train LoRA adapter locally
./train_mlx.sh
# -> adapters/*.safetensors

# 5. Re-run with the adapter and compare
python3 infer_mlx.py --adapter adapters --n 10
diff <(jq .score baseline_results.json) <(jq .score finetuned_results.json)

# 6. (optional) Fine-tune via OpenAI Files + Jobs API
python3 client_openai.py --dry-run                    # local validation only
python3 client_openai.py --model gpt-4o-mini-2024-07-18
python3 client_openai.py --resume ftjob-...           # poll an existing job
```

## Notes

- **System prompt is large** (~1.5 KB / ~400 tokens). Re-tokenized on every
  call, so prompt processing dominates inference time on local 7B-4bit
  (effective end-to-end ~10 tok/s on M3 Pro). Pure generation is ~30-40 tok/s.
- **Dataset is small** (30 examples in `examples.jsonl`). Enough to show the
  format pattern; for serious generalization scale to ×5-10.
- **MLX file naming** — MLX-LM expects exactly `train.jsonl` and `valid.jsonl`
  in `--data <dir>`. Do not rename.
- **OpenAI client artifact** — OpenAI requires a hosted fine-tune; we keep
  this client as the formal artifact even though the actual fine-tune is run
  locally.

## License

MIT.
