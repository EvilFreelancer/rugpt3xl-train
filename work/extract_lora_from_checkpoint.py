#!/usr/bin/env python3
"""
Extract LoRA adapter weights from FSDP checkpoint.

Usage:
    python extract_lora_from_checkpoint.py <checkpoint_path> <output_adapter_path>

This script loads an FSDP checkpoint and extracts only the LoRA adapter weights.
Designed to run independently from the training process.
"""

import os
import sys
import json
import inspect
import torch

# Must match training config
#MODEL_DIR = "/workspace/models/ruGPT3XL-8k"
MODEL_DIR = "evilfreelancer/ruGPT3XL-8k"
LORA_R = 64
LORA_ALPHA = 128


def map_fsdp_key_to_peft_key(fsdp_key: str) -> str:
    """Convert FSDP checkpoint keys to PEFT model keys."""
    peft_key = fsdp_key[6:] if fsdp_key.startswith("model.") else fsdp_key

    # LoRA weights in PEFT include the adapter name segment.
    peft_key = peft_key.replace(".lora_A.weight", ".lora_A.default.weight")
    peft_key = peft_key.replace(".lora_B.weight", ".lora_B.default.weight")

    # Modules listed in modules_to_save are wrapped under modules_to_save.default.
    peft_key = peft_key.replace(
        ".embed_tokens.weight",
        ".embed_tokens.modules_to_save.default.weight",
    )
    peft_key = peft_key.replace(
        ".lm_head.weight",
        ".lm_head.modules_to_save.default.weight",
    )
    return peft_key


def extract_lora_adapter(checkpoint_path: str, output_path: str):
    """Extract LoRA adapter from FSDP checkpoint."""
    print(f"[Extract] Loading from: {checkpoint_path}")
    print(f"[Extract] Saving to: {output_path}")

    # Check if already exists
    adapter_file = os.path.join(output_path, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        print(f"[Extract] Adapter already exists, skipping")
        return

    # Verify checkpoint exists
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        print(f"[Extract] Error: No trainer_state.json found")
        sys.exit(1)

    # Load trainer state to get step
    with open(trainer_state_path) as f:
        trainer_state = json.load(f)
    global_step = trainer_state.get("global_step", 0)
    print(f"[Extract] Checkpoint step: {global_step}")

    # Import here to avoid slow startup if checkpoint not ready
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    # Load tokenizer
    print("[Extract] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # Load base model on CPU first (safer for extraction)
    print("[Extract] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # Load on CPU to avoid GPU memory issues
    )

    # Resize embeddings if needed
    new_vocab = len(tokenizer)
    old_vocab = model.config.vocab_size
    if new_vocab > old_vocab:
        model.resize_token_embeddings(new_vocab)
        print(f"[Extract] Resized embeddings: {old_vocab} -> {new_vocab}")

    # Apply LoRA config (same as training)
    print("[Extract] Applying LoRA...")
    lora_config_kwargs = dict(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    if "ensure_weight_tying" in inspect.signature(LoraConfig).parameters:
        lora_config_kwargs["ensure_weight_tying"] = True

    lora_config = LoraConfig(**lora_config_kwargs)
    model = get_peft_model(model, lora_config)

    # Load checkpoint weights
    print("[Extract] Loading checkpoint weights...")
    fsdp_dir = os.path.join(checkpoint_path, "pytorch_model_fsdp_0")

    if os.path.exists(fsdp_dir):
        # Load FSDP sharded checkpoint without initializing torch.distributed.
        from torch.distributed.checkpoint import FileSystemReader, load as dcp_load

        # Read metadata
        metadata_path = os.path.join(fsdp_dir, "__0_0.distcp")
        if os.path.exists(metadata_path):
            print(f"[Extract] Found FSDP checkpoint in {fsdp_dir}")
            reader = FileSystemReader(fsdp_dir)
            metadata = reader.read_metadata()
            model_state = model.state_dict()

            checkpoint_state = {}
            ckpt_to_peft = {}
            skipped_keys = 0

            for ckpt_key, tensor_meta in metadata.state_dict_metadata.items():
                peft_key = map_fsdp_key_to_peft_key(ckpt_key)
                if peft_key not in model_state:
                    skipped_keys += 1
                    continue

                checkpoint_state[ckpt_key] = torch.empty(
                    tensor_meta.size,
                    dtype=tensor_meta.properties.dtype,
                    device="cpu",
                )
                ckpt_to_peft[ckpt_key] = peft_key

            if not checkpoint_state:
                raise RuntimeError("No matching LoRA tensors found in FSDP checkpoint")

            dcp_load(checkpoint_state, storage_reader=reader, no_dist=True)

            mapped_state_dict = {}
            for ckpt_key, tensor in checkpoint_state.items():
                peft_key = ckpt_to_peft[ckpt_key]
                target = model_state[peft_key]
                if tensor.shape != target.shape:
                    print(
                        f"[Extract] Warning: shape mismatch for {peft_key}, "
                        f"checkpoint={tuple(tensor.shape)} model={tuple(target.shape)}"
                    )
                    skipped_keys += 1
                    continue
                if tensor.dtype != target.dtype:
                    tensor = tensor.to(dtype=target.dtype)

                mapped_state_dict[peft_key] = tensor

            incompatible = model.load_state_dict(mapped_state_dict, strict=False)
            print(
                f"[Extract] Loaded {len(mapped_state_dict)} tensors from FSDP "
                f"(skipped {skipped_keys})"
            )
            if incompatible.unexpected_keys:
                print(f"[Extract] Warning: unexpected keys: {len(incompatible.unexpected_keys)}")
        else:
            print(f"[Extract] Warning: No FSDP metadata found, trying regular files...")
            # Try loading individual files
            for filename in os.listdir(fsdp_dir):
                if filename.endswith('.bin') or filename.endswith('.safetensors'):
                    filepath = os.path.join(fsdp_dir, filename)
                    print(f"[Extract] Loading {filename}...")
                    if filename.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(filepath)
                    else:
                        state_dict = torch.load(filepath, map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
    else:
        # Try loading pytorch_model.bin directly
        model_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            print(f"[Extract] Loading {model_file}...")
            state_dict = torch.load(model_file, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[Extract] Warning: No model weights found in checkpoint")

    # Save adapter
    print("[Extract] Saving LoRA adapter...")
    os.makedirs(output_path, exist_ok=True)

    # Unwrap to PeftModel and save
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        print(f"[Extract] Saved to {output_path}")

        # Verify
        adapter_file = os.path.join(output_path, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            size_mb = os.path.getsize(adapter_file) / (1024 * 1024)
            print(f"[Extract] Success! Adapter size: {size_mb:.1f} MB")
        else:
            print(f"[Extract] Warning: adapter file not created")
    else:
        print(f"[Extract] Error: Model has no save_pretrained method")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_lora_from_checkpoint.py <checkpoint_path> <output_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        extract_lora_adapter(checkpoint_path, output_path)
    except Exception as e:
        print(f"[Extract] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
