# ruGPT3XL-8k SFT Training with Unsloth Studio

Fine-tune [evilfreelancer/ruGPT3XL-8k](https://huggingface.co/evilfreelancer/ruGPT3XL-8k) (1.3B GPT-2-style Russian LM, 8192 context) for bilingual tool-calling, agentic loops and reasoning using [Unsloth Studio](https://github.com/unslothai/unsloth).

## Quick start

### Single GPU (via Studio UI)

```bash
# 1. Start the container (GPU required)
docker compose up -d

# 2. Prepare the training dataset (~150k records)
docker exec unsloth python -u /workspace/work/prepare_dataset.py

# 3. Prepare the eval dataset (~7k records)
docker exec unsloth python -u /workspace/work/prepare_eval_dataset.py

# 4. Launch training (visible in Studio UI)
docker exec unsloth python -u /workspace/work/train_rugpt3xl.py
```

Studio UI is available at `http://<host>:8000` (default credentials: `unsloth` / `12345678`).
Jupyter Lab is at `http://<host>:8888`.

### Multi-GPU DDP (recommended, 4x RTX 5060 Ti)

```bash
docker exec unsloth torchrun --nproc_per_node=4 /workspace/work/train_rugpt3xl_ddp.py
```

Each GPU trains its own copy of the model on a separate data batch.
Effective batch size = 4 GPUs * 1 batch * 4 gradient_accumulation = 16.

### Multi-GPU pipeline (device_map="balanced")

```bash
docker exec unsloth python -u /workspace/work/train_rugpt3xl_multigpu.py
```

Splits 24 layers across all GPUs. Useful when a model doesn't fit on a single GPU. Slower than DDP because layers are processed sequentially.

### Monitor training

```bash
docker exec unsloth tensorboard --logdir /workspace/work/runs --bind_all
```

TensorBoard will be at `http://<host>:6006`.

## Memory-efficient attention

The model code (`modeling_rugpt3xl.py`) has built-in memory-efficient attention
that enables 8k context training on 16GB GPUs.

- **Dense layers** (odd indices) - receive `attention_mask=None` from
  `RuGPT3XLModel.forward`, which triggers `is_causal=True` in SDPA.
  This enables Flash Attention with O(n) memory complexity.
- **Sparse layers** (even indices) - receive the block-sparse causal mask
  and use the `EFFICIENT_ATTENTION` SDPA backend. This avoids the O(n^2)
  Math backend fallback.
- The dense causal mask is never materialised - only the sparse mask is built,
  saving ~2GB at 8k context.

This is transparent to all training scripts. No runtime monkey-patching needed.

## Dataset

Both train and eval datasets follow the same bilingual (RU+EN) SFT mix, stored as JSONL in ShareGPT `conversations` format.

### Task proportions

| Category | Share | Purpose |
|---|---|---|
| Tool/function calling (single-step) | 45% | Core skill: generate clean tool-call JSON |
| Agentic multi-turn / multi-step | 25% | Plan, call, observe, iterate, finish |
| Reasoning (no tools) | 20% | Short disciplined CoT for a 1.3B model |
| No-tool / refusal / direct answer | 10% | Know when NOT to call a tool |

### Language split

55% English, 45% Russian. English is heavier because tool schemas, function names and JSON keys are predominantly English. Russian is reinforced through dedicated RU tool-calling and RU reasoning sources.

### Training set (150k records)

Produced by `work/prepare_dataset.py` (seed=42). Output: `sft_mix_150k.jsonl`.

**English sources (82.5k)**

| Source | Records | Type |
|---|---|---|
| [Mustafaege/qwen3.5-toolcalling-v2](https://huggingface.co/datasets/Mustafaege/qwen3.5-toolcalling-v2) | 40,000 | Tool-calling SFT, OpenAI-style messages |
| [interstellarninja/hermes_reasoning_tool_use](https://huggingface.co/datasets/interstellarninja/hermes_reasoning_tool_use) | 20,000 | Reasoning + tool use, relevance scenarios |
| [Featherlabs/featherlabs_agentic_v1](https://huggingface.co/datasets/Featherlabs/featherlabs_agentic_v1) | 10,000 | Agentic workflows, multi-step tool use |
| [arcee-ai/reasoning-sharegpt](https://huggingface.co/datasets/arcee-ai/reasoning-sharegpt) | 12,500 | Reasoning-only, no tool calls |

**Russian sources (67.5k)**

| Source | Records | Type |
|---|---|---|
| [belyakoff/xlam-ru-tool-calling](https://huggingface.co/datasets/belyakoff/xlam-ru-tool-calling) | 30,000 | RU function-calling (query/tools/answers) |
| [HelioAI/Helio1-Reasoning-50K-RU](https://huggingface.co/datasets/HelioAI/Helio1-Reasoning-50K-RU) | 20,000 | RU reasoning (prompt/chosen format) |
| [ZeroAgency/ru-thinking-reasoning-r1-v2](https://huggingface.co/datasets/ZeroAgency/ru-thinking-reasoning-r1-v2) | 10,000 | RU thinking/reasoning conversations |
| [ZeroAgency/ru-big-russian-dataset](https://huggingface.co/datasets/ZeroAgency/ru-big-russian-dataset) | 7,500 | General RU, filtered by `has_reasoning` |

### Eval set (~7k records)

Produced by `work/prepare_eval_dataset.py` (seed=123, sampled from the tail of each shuffled dataset to minimize overlap). Output: `sft_eval_7k.jsonl`. Same sources and proportions at ~5% scale.

### Format

Each record is a JSON line:

```json
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Roles are normalized to `system`, `user`, `assistant`, `tool`. All text is truncated to ~30k characters (~7.5k tokens) to stay within the 8k context window.

### Post-processing

- Role normalization: `human` -> `user`, `gpt` -> `assistant`, `function_call` -> `assistant`, `function_response`/`observation` -> `tool`
- Deduplication by first 200 characters of each user message
- Final shuffle before saving

## Training configuration

### DDP (4x GPU, recommended)

Launched via `work/train_rugpt3xl_ddp.py` with `torchrun --nproc_per_node=4`.

| Parameter | Value |
|---|---|
| Model | `/workspace/models/ruGPT3XL-8k` (local mount) |
| Method | QLoRA (4-bit NF4) |
| `trust_remote_code` | `true` |
| `max_seq_length` | 8192 |
| LoRA rank / alpha | 16 / 16 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj` |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 4 (effective batch = 4 * 1 * 4 = 16) |
| Learning rate | 2e-4, cosine schedule |
| Warmup | 100 steps |
| Optimizer | `adamw_8bit` |
| Gradient checkpointing | `use_reentrant=False` |
| Eval every | 100 steps |
| Save every | 500 steps |
| TensorBoard | enabled (`/workspace/work/runs/rugpt3xl-8k-ddp/logs`) |
| Peak GPU memory | ~15.8GB / 16GB per GPU |

### Single GPU (via Studio API)

Launched via `work/train_rugpt3xl.py` which calls the Unsloth Studio REST API.

| Parameter | Value |
|---|---|
| `max_seq_length` | 4096 |
| Batch size | 2 |
| Gradient accumulation | 8 (effective batch = 16) |
| Other parameters | Same as DDP |

## Project structure

```
docker-compose.yaml              - Container setup, GPU, volumes
work/
  prepare_dataset.py             - Build 150k training JSONL
  prepare_eval_dataset.py        - Build ~7k eval JSONL
  train_rugpt3xl.py              - Launch training via Studio API (single GPU)
  train_rugpt3xl_ddp.py          - DDP training on 4 GPUs via torchrun
  train_rugpt3xl_multigpu.py     - Pipeline-parallel training via device_map
  measure_memory.py              - GPU memory profiling script
  test_single_gpu_optimized.py   - Attention optimization test
unsloth/
  unsloth/models/_utils.py       - Patch: graceful skip for custom model types in compile loop
  studio/backend/
    assets/configs/model_defaults/other/
      evilfreelancer_ruGPT3XL-8k.yaml  - Model defaults for Studio
    utils/models/
      model_config.py            - MODEL_NAME_MAPPING with ruGPT3XL-8k entries
unsloth_data/
  studio/                        - Persisted Studio state (auth DB, runs, cache)
  huggingface/                   - HF model/dataset cache
```

## Unsloth patches

Two minimal patches are bind-mounted into the container over the installed package files:

1. **`unsloth/models/_utils.py`** - wraps the `_unsloth_compile_transformers` call with `try/except ModuleNotFoundError` so that custom `trust_remote_code` model types (which don't exist under `transformers.models.*`) are gracefully skipped instead of crashing the compile loop.

2. **Studio config** - a YAML defaults file and a `MODEL_NAME_MAPPING` entry are added so that Studio recognizes `ruGPT3XL-8k` by its HuggingFace ID or local path and applies the correct defaults (including `trust_remote_code: true`).
