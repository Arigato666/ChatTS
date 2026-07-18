# TelecomTS smoke test

`telecomts_smoke.jsonl` contains two derived TelecomTS windows: one real
jamming sample and one context-matched normal sample. Both use File traffic,
Zone A, stationary operation, and no congestion. The fixture keeps the raw KPI
values, evaluation labels, and one Q&A per category. Long descriptions,
reasoning traces, and troubleshooting tickets are intentionally excluded.

The source dataset is
[`AliMaatouk/TelecomTS`](https://huggingface.co/datasets/AliMaatouk/TelecomTS)
and is distributed under the MIT license. The fixture is only intended for
integration and inference smoke tests, not benchmark reporting.

## Server commands

From the ChatTS repository root, validate the two records without loading the
model:

```bash
python -m demo.telecomts_hf \
  --source-file demo/telecomts_smoke.jsonl \
  --task anomaly \
  --max-samples 2 \
  --dry-run
```

Run ChatTS-8B inference:

```bash
python -m demo.telecomts_hf \
  --model-path /root/autodl-tmp/ChatTS/ckpt \
  --source-file demo/telecomts_smoke.jsonl \
  --task anomaly \
  --max-samples 2 \
  --output reproduction_outputs/telecomts_smoke_results.jsonl
```

Run a TelecomTS time-series Q&A instead:

```bash
python -m demo.telecomts_hf \
  --model-path /root/autodl-tmp/ChatTS/ckpt \
  --source-file demo/telecomts_smoke.jsonl \
  --task qna \
  --qna-category timeseries \
  --max-samples 2
```

Rebuild the fixture from a full local TelecomTS download:

```bash
python scripts/build_telecomts_smoke_fixture.py \
  --data-root /path/to/TelecomTS
```
