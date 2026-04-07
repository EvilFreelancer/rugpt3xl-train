#!/usr/bin/env python3
"""
Standalone script to patch ruGPT3XL-8k tokenizer with GigaChat3 tokens.
Same logic runs automatically at training startup via tokenizer_setup.py.

Usage:
  python patch_tokenizer.py [/path/to/ruGPT3XL-8k]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokenizer_setup import ensure_gigachat3_tokenizer


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/pasha/train/ruGPT3XL-8k"
    print(f"Patching tokenizer at {model_dir} ...")
    changed = ensure_gigachat3_tokenizer(model_dir)
    if changed:
        print("\nTokenizer patched successfully.")
    else:
        print("\nNo changes needed.")


if __name__ == "__main__":
    main()
