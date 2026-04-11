#!/usr/bin/env python3
"""
Extract LoRA adapter weights from FSDP checkpoint.

Usage:
    python extract_lora.py /path/to/checkpoint-XXX /path/to/output_adapter

This script loads an FSDP checkpoint with PEFT model and extracts
only the LoRA adapter weights (not the full FSDP sharded state).
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType


def extract_lora_adapter(checkpoint_path: str, output_path: str):
    """Extract LoRA adapter from FSDP checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Output will be saved to: {output_path}")

    model_dir = "/workspace/models/ruGPT3XL-8k"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Resize embeddings if needed
    new_vocab = len(tokenizer)
    old_vocab = model.config.vocab_size
    if new_vocab > old_vocab:
        model.resize_token_embeddings(new_vocab)
        print(f"Resized embeddings: {old_vocab} -> {new_vocab}")

    # Apply LoRA config (same as training)
    print("Applying LoRA configuration...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Load checkpoint
    print("Loading checkpoint...")
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        print(f"Error: No trainer_state.json found in {checkpoint_path}")
        sys.exit(1)

    # Load FSDP state dict
    fsdp_model_dir = os.path.join(checkpoint_path, "pytorch_model_fsdp_0")
    if os.path.exists(fsdp_model_dir):
        # Load using torch.load
        state_dict_files = sorted([f for f in os.listdir(fsdp_model_dir) if f.endswith('.bin') or f.endswith('.safetensors')])
        print(f"Found {len(state_dict_files)} state dict files")

        # Try loading the checkpoint
        try:
            from accelerate import Accelerator
            accelerator = Accelerator()
            print("Using Accelerate to load FSDP checkpoint...")
            # This requires FSDP setup, so instead we use simpler approach
        except:
            pass

    # Alternative: directly load with trainer
    print("Loading with Trainer...")
    from trl import SFTTrainer, SFTConfig
    from masking import AssistantOnlyCollator, get_marker_ids

    resp_ids, end_ids = get_marker_ids(tokenizer)
    collator = AssistantOnlyCollator(
        tokenizer=tokenizer,
        response_marker_ids=resp_ids,
        message_end_ids=end_ids,
        max_seq_length=8192,
    )

    sft_config = SFTConfig(
        output_dir=output_path,
        max_length=8192,
        per_device_train_batch_size=1,
        bf16=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": ["RuGPT3XLDecoderLayer"],
            "backward_prefetch": "backward_pre",
            "state_dict_type": "SHARDED_STATE_DICT",
        },
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=[],
        data_collator=collator,
        args=sft_config,
    )

    # Load checkpoint
    trainer.train(resume_from_checkpoint=checkpoint_path)

    # Now save just the adapter
    print("Extracting LoRA adapter...")
    model = trainer.model

    # Unwrap FSDP
    while hasattr(model, "module"):
        model = model.module

    # Save adapter
    if isinstance(model, PeftModel):
        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        print(f"LoRA adapter saved to: {output_path}")
    else:
        print(f"Error: Model is not PeftModel, got {type(model)}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_lora.py <checkpoint_path> <output_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2]

    extract_lora_adapter(checkpoint_path, output_path)
