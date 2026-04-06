"""
Prepare a 150k bilingual (RU+EN) SFT dataset for ruGPT3XL-8k tool-calling training.

Mix:  45% tool-calling | 25% agentic multi-turn | 20% reasoning | 10% no-tool/refusal
Lang: 55% EN | 45% RU

Saves result as /workspace/work/sft_mix_150k.jsonl in ShareGPT conversations format
compatible with Unsloth Studio.

Usage:
    python /workspace/work/prepare_dataset.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

random.seed(42)
OUTPUT_PATH = Path("/workspace/work/sft_mix_150k.jsonl")
MAX_TOKENS_APPROX = 7500  # rough char limit ~ 8k tokens

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def truncate(text: str, max_chars: int = MAX_TOKENS_APPROX * 4) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def wrap_conversations(convos: list[dict]) -> dict:
    """Return a single record in ShareGPT format."""
    return {"conversations": convos}


def sharegpt_to_chatml(messages: list[dict]) -> list[dict]:
    """Convert human/gpt role names to system/user/assistant."""
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
        "tool": "tool",
        "function_call": "assistant",
        "function_response": "tool",
        "observation": "tool",
    }
    out = []
    for m in messages:
        role = role_map.get(m.get("from", m.get("role", "")), m.get("role", "user"))
        content = m.get("value", m.get("content", ""))
        if content:
            out.append({"role": role, "content": truncate(str(content))})
    return out


def sample_hf(repo: str, n: int, split: str = "train", **kwargs) -> list:
    """Load HF dataset and return up to n shuffled rows."""
    print(f"  Loading {repo} (n={n}) ...")
    try:
        ds = load_dataset(repo, split=split, trust_remote_code=True, **kwargs)
    except Exception as e:
        print(f"  WARNING: failed to load {repo}: {e}")
        return []
    ds = ds.shuffle(seed=42)
    return list(ds.select(range(min(n, len(ds)))))


# -------------------------------------------------------------------------
# Per-source converters
# -------------------------------------------------------------------------

def convert_mustafaege(rows: list) -> list[dict]:
    """Mustafaege/qwen3.5-toolcalling-v2 - messages column, OpenAI format."""
    results = []
    for r in rows:
        msgs = r.get("messages") or r.get("conversations") or []
        convos = sharegpt_to_chatml(msgs)
        if convos:
            results.append(wrap_conversations(convos))
    return results


def convert_hermes(rows: list) -> list[dict]:
    """interstellarninja/hermes_reasoning_tool_use - conversations column, ShareGPT."""
    results = []
    for r in rows:
        msgs = r.get("conversations") or r.get("messages") or []
        convos = sharegpt_to_chatml(msgs)
        if convos:
            results.append(wrap_conversations(convos))
    return results


def convert_featherlabs(rows: list) -> list[dict]:
    """Featherlabs/featherlabs_agentic_v1 - messages/conversations."""
    results = []
    for r in rows:
        msgs = r.get("messages") or r.get("conversations") or []
        convos = sharegpt_to_chatml(msgs)
        if convos:
            results.append(wrap_conversations(convos))
    return results


def convert_xlam_ru(rows: list) -> list[dict]:
    """belyakoff/xlam-ru-tool-calling."""
    results = []
    for r in rows:
        msgs = r.get("messages") or r.get("conversations") or []
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except json.JSONDecodeError:
                continue
        convos = sharegpt_to_chatml(msgs)
        if convos:
            results.append(wrap_conversations(convos))
    return results


def convert_helio_reasoning(rows: list) -> list[dict]:
    """HelioAI/Helio1-Reasoning-50K-RU - conversations or messages."""
    results = []
    for r in rows:
        msgs = r.get("conversations") or r.get("messages") or []
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except json.JSONDecodeError:
                continue
        convos = sharegpt_to_chatml(msgs)
        if convos:
            results.append(wrap_conversations(convos))
    return results


def convert_zero_agency_reasoning(rows: list) -> list[dict]:
    """ZeroAgency/ru-thinking-reasoning-r1-v2."""
    results = []
    for r in rows:
        msgs = r.get("conversations") or r.get("messages") or []
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except json.JSONDecodeError:
                continue
        convos = sharegpt_to_chatml(msgs)
        if convos:
            results.append(wrap_conversations(convos))
    return results


def convert_text_pair(rows: list, user_col: str, assistant_col: str) -> list[dict]:
    """Generic converter for instruction/response style datasets."""
    results = []
    for r in rows:
        user_text = r.get(user_col, "")
        assistant_text = r.get(assistant_col, "")
        if user_text and assistant_text:
            results.append(wrap_conversations([
                {"role": "user", "content": truncate(str(user_text))},
                {"role": "assistant", "content": truncate(str(assistant_text))},
            ]))
    return results


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------

def main():
    all_records: list[dict] = []

    # === ENGLISH (82.5k total) ===
    print("\n=== English sources ===")

    # 40k tool-calling
    rows = sample_hf("Mustafaege/qwen3.5-toolcalling-v2", 40_000)
    all_records.extend(convert_mustafaege(rows))
    print(f"  -> {len(all_records)} total so far")

    # 20k reasoning + tool use
    rows = sample_hf("interstellarninja/hermes_reasoning_tool_use", 20_000)
    all_records.extend(convert_hermes(rows))
    print(f"  -> {len(all_records)} total so far")

    # 10k agentic
    rows = sample_hf("Featherlabs/featherlabs_agentic_v1", 10_000)
    all_records.extend(convert_featherlabs(rows))
    print(f"  -> {len(all_records)} total so far")

    # 12.5k reasoning-only (arcee-ai/reasoning-sharegpt or similar)
    rows = sample_hf("arcee-ai/reasoning-sharegpt", 12_500)
    if rows:
        all_records.extend(convert_hermes(rows))
    else:
        # Fallback: use more from hermes
        rows = sample_hf("interstellarninja/hermes_reasoning_tool_use", 12_500)
        all_records.extend(convert_hermes(rows))
    print(f"  -> {len(all_records)} total so far")

    # === RUSSIAN (67.5k total) ===
    print("\n=== Russian sources ===")

    # 30k tool-calling RU
    rows = sample_hf("belyakoff/xlam-ru-tool-calling", 30_000)
    all_records.extend(convert_xlam_ru(rows))
    print(f"  -> {len(all_records)} total so far")

    # 20k reasoning RU
    rows = sample_hf("HelioAI/Helio1-Reasoning-50K-RU", 20_000)
    all_records.extend(convert_helio_reasoning(rows))
    print(f"  -> {len(all_records)} total so far")

    # 10k thinking/reasoning RU
    rows = sample_hf("ZeroAgency/ru-thinking-reasoning-r1-v2", 10_000)
    all_records.extend(convert_zero_agency_reasoning(rows))
    print(f"  -> {len(all_records)} total so far")

    # 7.5k general RU (filtered)
    rows = sample_hf("ZeroAgency/ru-big-russian-dataset", 15_000)
    if rows:
        # Filter for rows with reasoning
        filtered = [r for r in rows if r.get("has_reasoning")]
        if len(filtered) < 7_500:
            filtered = rows[:7_500]
        else:
            filtered = filtered[:7_500]
        all_records.extend(convert_hermes(filtered))
    print(f"  -> {len(all_records)} total so far")

    # === Shuffle and save ===
    print(f"\nTotal records before dedup: {len(all_records)}")

    # Basic dedup by first user message
    seen = set()
    deduped = []
    for rec in all_records:
        convos = rec.get("conversations", [])
        user_msgs = [m["content"][:200] for m in convos if m["role"] == "user"]
        key = "||".join(user_msgs)
        if key not in seen:
            seen.add(key)
            deduped.append(rec)
    all_records = deduped
    print(f"After dedup: {len(all_records)}")

    random.shuffle(all_records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_records)} records to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
