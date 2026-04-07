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

### Multi-GPU pipeline (device_map="balanced", recommended)

```bash
docker exec unsloth python -u /workspace/work/train_rugpt3xl_multigpu.py
```

Splits 24 layers across all GPUs. GBS = 128 (1 batch * 128 gradient accumulation).
Automatically patches the tokenizer with GigaChat3 special tokens on first run.

### Multi-GPU DDP (4x RTX 5060 Ti)

```bash
docker exec unsloth torchrun --nproc_per_node=4 /workspace/work/train_rugpt3xl_ddp.py
```

Each GPU trains its own copy of the model on a separate data batch.
GBS = 128 (4 GPUs * 1 batch * 32 gradient accumulation).

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

## Tokenizer patching (GigaChat3 compatibility)

Before training, both scripts automatically patch the `ruGPT3XL-8k` tokenizer to be compatible with vLLM's `gigachat3_tool_parser`. This is handled by `work/tokenizer_setup.py` and is fully idempotent (skips if already done).

### Special tokens added

| Token | ID | Purpose |
|---|---|---|
| `<\|role_sep\|>` | 50257 | Separates role name from message content |
| `<\|message_sep\|>` | 50258 | End-of-message boundary |
| `<\|function_call\|>` | 50259 | Marks the start of a tool/function call JSON |
| `<\|file\|>` | 50260 | File content start marker |
| `<\|/file\|>` | 50261 | File content end marker |

These match the special tokens from [ai-sage/GigaChat3.1-10B-A1.8B](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B). The original `ruGPT3XL-8k` checkpoint has `vocab_size=50264` in its weights, so tokens 50257-50261 fit into already-allocated embedding slots without requiring a resize.

### Chat template

The chat template (`work/chat_template_gigachat3.jinja`) is adapted from GigaChat3. It renders conversations using a multi-role format with tool-calling support:

```
<s>role_name<|role_sep|>
content<|message_sep|>

```

Key features:
- Renders tool schemas as TypeScript type definitions (matching GigaChat3 conventions)
- Includes a DEVSYSTEM block with role descriptions and behavioral instructions
- Supports `<think>` blocks for chain-of-thought reasoning
- Compatible with vLLM's `gigachat3_tool_parser` for inference-time tool parsing

The patching process:
1. `ensure_gigachat3_tokenizer(model_dir)` is called at the very start of training
2. Checks if all 5 special tokens already resolve to valid IDs (not `<unk>`)
3. Checks if the chat template contains `<|role_sep|>`
4. If anything is missing, adds tokens and/or applies the template, then saves in-place
5. Fixes the `extra_special_tokens` vs `additional_special_tokens` format issue across transformers versions

### Embedding handling

Since the new token IDs (50257-50261) fall within the existing weight matrix (50264 rows), no `resize_token_embeddings` is needed. The training scripts only resize UP, never down. To ensure the new token embeddings are properly learned, `modules_to_save=["embed_tokens", "lm_head"]` is passed to LoRA so these layers are trained in full precision alongside the adapters.

## Loss masking (assistant-only training)

Training loss is computed only on assistant response tokens. Everything else (system prompts, user messages, template boilerplate, DEVSYSTEM instructions) is masked with `label=-100`.

This is implemented in `work/masking.py` via `AssistantOnlyCollator`:

1. For each sample, the collator searches for the token sequence `assistant<|role_sep|>\n` (response start marker)
2. Tokens between this marker and the next `<|message_sep|>` are labeled with their actual IDs
3. All other tokens get `label=-100` (ignored by CrossEntropyLoss)
4. Works correctly for multi-turn conversations with multiple assistant responses

Samples that contain no assistant responses are filtered out before training to avoid NaN loss.

Typical masking ratio is ~75% masked / ~25% trained (varies by sample).

## Training configuration

### Multi-GPU pipeline (device_map="balanced")

Launched via `work/train_rugpt3xl_multigpu.py`. Splits 24 model layers across all GPUs.

| Parameter | Value |
|---|---|
| Model | `ruGPT3XL-8k` (1.3B params, local mount) |
| Method | QLoRA (4-bit NF4) |
| `max_seq_length` | 8192 |
| LoRA rank / alpha | 64 / 128 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj` |
| `modules_to_save` | `embed_tokens`, `lm_head` |
| Trainable params | ~262M / 1.09B (24%) |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 128 (GBS = 128) |
| Learning rate | 5e-5, cosine schedule |
| Warmup | 10% of total steps |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Optimizer | `adamw_8bit` |
| Precision | bf16 |
| Gradient checkpointing | `use_reentrant=False` |
| Eval strategy | every 100 steps + on start |
| Eval subset | 500 random samples (from 7k) |
| Save strategy | every 100 steps, keep best 5 |
| Best model metric | `eval_loss` (lower is better) |
| Early stopping | patience = 5 evals |
| Loss masking | assistant-only via `AssistantOnlyCollator` |
| TensorBoard | `/workspace/work/runs/rugpt3xl-8k-multigpu/logs` |

### DDP (4x GPU)

Launched via `work/train_rugpt3xl_ddp.py` with `torchrun --nproc_per_node=4`. Each GPU trains its own copy.

| Parameter | Value |
|---|---|
| Per-device batch size | 1 |
| Gradient accumulation | 32 (GBS = 4 * 1 * 32 = 128) |
| All other params | Same as pipeline mode above |

### Single GPU (via Studio API)

Launched via `work/train_rugpt3xl.py` which calls the Unsloth Studio REST API.

| Parameter | Value |
|---|---|
| `max_seq_length` | 4096 |
| Batch size | 2 |
| Gradient accumulation | 8 (effective batch = 16) |
| LoRA rank / alpha | 16 / 16 (original, not updated) |

## Project structure

```
docker-compose.yaml                - Container setup, GPU, volumes
work/
  train_rugpt3xl_multigpu.py       - Pipeline-parallel training (device_map="balanced")
  train_rugpt3xl_ddp.py            - DDP training on 4 GPUs via torchrun
  train_rugpt3xl.py                - Training via Studio API (single GPU)
  masking.py                       - AssistantOnlyCollator + masking logic
  tokenizer_setup.py               - Idempotent GigaChat3 tokenizer patching
  patch_tokenizer.py               - Manual tokenizer patching CLI wrapper
  chat_template_gigachat3.jinja    - GigaChat3-adapted Jinja chat template
  prepare_dataset.py               - Build 150k training JSONL
  prepare_eval_dataset.py          - Build ~7k eval JSONL
  plot_training_log.py             - Parse training logs and plot metrics
  measure_memory.py                - GPU memory profiling script
  test_single_gpu_optimized.py     - Attention optimization test
unsloth/
  unsloth/models/_utils.py         - Patch: graceful skip for custom model types
  studio/backend/
    assets/configs/model_defaults/other/
      evilfreelancer_ruGPT3XL-8k.yaml  - Model defaults for Studio
    utils/models/
      model_config.py              - MODEL_NAME_MAPPING with ruGPT3XL-8k entries
unsloth_data/
  studio/                          - Persisted Studio state (auth DB, runs, cache)
  huggingface/                     - HF model/dataset cache
```

## Unsloth patches

Two minimal patches are bind-mounted into the container over the installed package files:

1. **`unsloth/models/_utils.py`** - wraps the `_unsloth_compile_transformers` call with `try/except ModuleNotFoundError` so that custom `trust_remote_code` model types (which don't exist under `transformers.models.*`) are gracefully skipped instead of crashing the compile loop.

2. **Studio config** - a YAML defaults file and a `MODEL_NAME_MAPPING` entry are added so that Studio recognizes `ruGPT3XL-8k` by its HuggingFace ID or local path and applies the correct defaults (including `trust_remote_code: true`).
