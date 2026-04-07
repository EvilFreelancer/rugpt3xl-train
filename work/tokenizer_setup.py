"""
Auto-patch ruGPT3XL-8k tokenizer with GigaChat3-style special tokens
and chat template for vLLM gigachat3_tool_parser compatibility.

Tokens added (matching ai-sage/GigaChat3.1-10B-A1.8B):
  <|role_sep|>       - separates role name from content
  <|message_sep|>    - end of message boundary
  <|function_call|>  - marks start of tool/function call JSON
  <|file|>           - file content start
  <|/file|>          - file content end

Called automatically at training startup. Skips if already patched.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer


GIGACHAT3_SPECIAL_TOKENS = [
    "<|role_sep|>",
    "<|message_sep|>",
    "<|function_call|>",
    "<|file|>",
    "<|/file|>",
]

CHAT_TEMPLATE_FILENAME = "chat_template_gigachat3.jinja"


def _fix_tokenizer_config_json(model_dir: Path, verbose: bool = True) -> None:
    """
    Some transformers versions write extra_special_tokens as a list,
    but newer versions expect a dict. Replace it with the standard
    additional_special_tokens list to avoid AttributeError on load.
    """
    tc_path = model_dir / "tokenizer_config.json"
    if not tc_path.exists():
        return
    tc = json.loads(tc_path.read_text(encoding="utf-8"))
    rewrite = False

    extra = tc.pop("extra_special_tokens", None)
    if extra is not None:
        rewrite = True
        if isinstance(extra, list):
            existing = tc.get("additional_special_tokens", [])
            merged = list(dict.fromkeys(existing + extra))
            tc["additional_special_tokens"] = merged
        if verbose:
            print("  Fixed extra_special_tokens -> additional_special_tokens")

    if rewrite:
        tc_path.write_text(
            json.dumps(tc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _find_chat_template(search_dirs: list[Path]) -> Path | None:
    for d in search_dirs:
        candidate = d / CHAT_TEMPLATE_FILENAME
        if candidate.exists():
            return candidate
    return None


def ensure_gigachat3_tokenizer(
    model_dir: str | Path,
    verbose: bool = True,
) -> bool:
    """
    Ensure the tokenizer at model_dir has GigaChat3 special tokens and
    chat template. Patches files in-place on first run, skips if already done.

    Returns True if any changes were made.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    unk_id = tokenizer.unk_token_id
    tokens_needed = [
        t for t in GIGACHAT3_SPECIAL_TOKENS
        if tokenizer.convert_tokens_to_ids(t) == unk_id
    ]
    template_needed = "<|role_sep|>" not in (tokenizer.chat_template or "")

    if not tokens_needed and not template_needed:
        if verbose:
            print("  Tokenizer already has GigaChat3 tokens and chat template, skipping.")
        return False

    changed = False

    if tokens_needed:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens_needed}
        )
        if verbose:
            print(f"  Added {num_added} GigaChat3 special tokens: {tokens_needed}")
        changed = True

    if template_needed:
        search_dirs = [
            Path(__file__).resolve().parent,
            model_dir,
            Path("/workspace/work"),
        ]
        template_file = _find_chat_template(search_dirs)
        if template_file:
            chat_template = template_file.read_text(encoding="utf-8").strip()
            tokenizer.chat_template = chat_template
            if verbose:
                print(f"  Chat template loaded from {template_file}")
            changed = True
        elif verbose:
            print(f"  WARNING: {CHAT_TEMPLATE_FILENAME} not found, template NOT set.")

    if changed:
        tokenizer.save_pretrained(str(model_dir))
        if verbose:
            print(f"  Tokenizer saved to {model_dir}")

        _fix_tokenizer_config_json(model_dir, verbose)

        config_path = model_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            old_vocab = config.get("vocab_size", 0)
            new_vocab = len(tokenizer)
            if old_vocab != new_vocab:
                config["vocab_size"] = new_vocab
                config_path.write_text(
                    json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                if verbose:
                    print(f"  config.json: vocab_size {old_vocab} -> {new_vocab}")

    if verbose:
        for tok in GIGACHAT3_SPECIAL_TOKENS:
            tid = tokenizer.convert_tokens_to_ids(tok)
            print(f"    {tok:25s} -> id {tid}")

    return changed
