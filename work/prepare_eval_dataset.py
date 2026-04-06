"""
Prepare a ~7.5k bilingual (RU+EN) eval dataset for ruGPT3XL-8k.

Same sources and proportions as the 150k training set, but:
  - ~5% of the training size (7.5k total)
  - different random seed (seed=123) to avoid overlap
  - no dedup against train set (HF shuffle with different seed is sufficient
    for datasets of this size)

Saves result as /workspace/work/sft_eval_7k.jsonl in ShareGPT conversations format.

Usage:
    python /workspace/work/prepare_eval_dataset.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

random.seed(123)
EVAL_SEED = 123
OUTPUT_PATH = Path("/workspace/work/sft_eval_7k.jsonl")
MAX_TOKENS_APPROX = 7500


def truncate(text: str, max_chars: int = MAX_TOKENS_APPROX * 4) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def wrap_conversations(convos: list[dict]) -> dict:
    return {"conversations": convos}


def sharegpt_to_chatml(messages: list[dict]) -> list[dict]:
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
    print(f"  Loading {repo} (n={n}) ...")
    try:
        ds = load_dataset(repo, split=split, trust_remote_code=True, **kwargs)
    except Exception as e:
        print(f"  WARNING: failed to load {repo}: {e}")
        return []
    ds = ds.shuffle(seed=EVAL_SEED)
    # Take from the tail of the shuffled dataset to minimize overlap with
    # the training set (which takes from the head with seed=42).
    start = max(0, len(ds) - n)
    rows = list(ds.select(range(start, len(ds))))
    print(f"    got {len(rows)} rows (dataset size: {len(ds)})")
    return rows


def convert_messages(rows: list) -> list[dict]:
    """Generic converter for datasets with messages/conversations/conversation column."""
    results = []
    for r in rows:
        msgs = (
            r.get("messages")
            or r.get("conversations")
            or r.get("conversation")
            or []
        )
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except json.JSONDecodeError:
                continue
        convos = sharegpt_to_chatml(msgs)
        if convos:
            results.append(wrap_conversations(convos))
    return results


def convert_xlam_ru(rows: list) -> list[dict]:
    """belyakoff/xlam-ru-tool-calling: query/answers/tools/reasoning columns."""
    results = []
    for r in rows:
        query = r.get("query", "")
        answers = r.get("answers", "")
        tools = r.get("tools", "")
        reasoning = r.get("reasoning", "")
        if not query or not answers:
            continue
        convos = []
        if tools:
            convos.append({
                "role": "system",
                "content": truncate(f"Available tools: {tools}"),
            })
        convos.append({"role": "user", "content": truncate(str(query))})
        assistant_text = ""
        if reasoning:
            assistant_text += f"<think>{truncate(str(reasoning), 2000)}</think>\n"
        assistant_text += str(answers)
        convos.append({"role": "assistant", "content": truncate(assistant_text)})
        results.append(wrap_conversations(convos))
    return results


def main():
    all_records: list[dict] = []

    # Proportions mirror training set at ~5% scale
    # EN: 55% of 7500 = 4125
    # RU: 45% of 7500 = 3375

    print("\n=== English eval sources ===")

    # 2000 tool-calling
    rows = sample_hf("Mustafaege/qwen3.5-toolcalling-v2", 2000)
    all_records.extend(convert_messages(rows))
    print(f"  -> {len(all_records)} total so far")

    # 1000 reasoning + tool use
    rows = sample_hf("interstellarninja/hermes_reasoning_tool_use", 1000)
    all_records.extend(convert_messages(rows))
    print(f"  -> {len(all_records)} total so far")

    # 500 agentic
    rows = sample_hf("Featherlabs/featherlabs_agentic_v1", 500)
    all_records.extend(convert_messages(rows))
    print(f"  -> {len(all_records)} total so far")

    # 625 reasoning-only
    rows = sample_hf("arcee-ai/reasoning-sharegpt", 625)
    if rows:
        all_records.extend(convert_messages(rows))
    else:
        rows = sample_hf("interstellarninja/hermes_reasoning_tool_use", 625)
        all_records.extend(convert_messages(rows))
    print(f"  -> {len(all_records)} total so far")

    print("\n=== Russian eval sources ===")

    # 1500 tool-calling RU (query/answers/tools format)
    rows = sample_hf("belyakoff/xlam-ru-tool-calling", 1500)
    all_records.extend(convert_xlam_ru(rows))
    print(f"  -> {len(all_records)} total so far")

    # 1000 reasoning RU (prompt/chosen format, use streaming due to parse issues)
    print("  Loading HelioAI/Helio1-Reasoning-50K-RU (n=1000) ...")
    try:
        ds = load_dataset(
            "HelioAI/Helio1-Reasoning-50K-RU",
            split="train",
            streaming=True,
        )
        helio_records = []
        for i, row in enumerate(ds):
            if len(helio_records) >= 1000:
                break
            if i < 7000:
                continue
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", "")
            if prompt and chosen:
                helio_records.append(wrap_conversations([
                    {"role": "user", "content": truncate(str(prompt))},
                    {"role": "assistant", "content": truncate(str(chosen))},
                ]))
        print(f"    got {len(helio_records)} rows (streaming)")
        all_records.extend(helio_records)
    except Exception as e:
        print(f"  WARNING: HelioAI failed: {e}")
    print(f"  -> {len(all_records)} total so far")

    # 500 thinking/reasoning RU (column: "conversation")
    rows = sample_hf("ZeroAgency/ru-thinking-reasoning-r1-v2", 500)
    all_records.extend(convert_messages(rows))
    print(f"  -> {len(all_records)} total so far")

    # 375 general RU (column: "conversation", filter by quality)
    rows = sample_hf("ZeroAgency/ru-big-russian-dataset", 750)
    if rows:
        filtered = [r for r in rows if r.get("has_reasoning")]
        if len(filtered) < 375:
            filtered = rows[:375]
        else:
            filtered = filtered[:375]
        all_records.extend(convert_messages(filtered))
    print(f"  -> {len(all_records)} total so far")

    # Dedup
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

    random.shuffle(all_records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_records)} eval records to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
