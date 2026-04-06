"""
Memory profiler for ruGPT3XL-8k training on RTX 5060 Ti (16GB).

Measures GPU memory at each stage of model loading and a training step
to determine optimal parameters for 8k context training.

Usage (inside container):
    python /workspace/work/measure_memory.py
"""

import gc
import os
import sys
import time

import torch

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def gpu_mem_mb(device=0):
    """Return (allocated_MB, reserved_MB, free_MB, total_MB)."""
    alloc = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2
    free = total - reserved
    return alloc, reserved, free, total


def report(label, device=0):
    alloc, reserved, free, total = gpu_mem_mb(device)
    print(f"[{label}]")
    print(f"  allocated: {alloc:>8.1f} MB")
    print(f"  reserved:  {reserved:>8.1f} MB")
    print(f"  free:      {free:>8.1f} MB")
    print(f"  total:     {total:>8.1f} MB")
    print()
    return alloc


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print()

    # Stage 1: baseline
    flush()
    report("Baseline (empty GPU)")

    # Stage 2: load model in 4-bit
    print("=" * 60)
    print("Loading model in 4-bit quantization...")
    print("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/workspace/models/ruGPT3XL-8k",
        trust_remote_code=True,
    )
    report("After tokenizer load")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/workspace/models/ruGPT3XL-8k",
            max_seq_length=8192,
            load_in_4bit=True,
            dtype=None,
            trust_remote_code=True,
        )
        report("After model load (4-bit via Unsloth)")
    except Exception as e:
        print(f"Unsloth load failed: {e}")
        print("Falling back to bitsandbytes 4-bit load...")

        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "/workspace/models/ruGPT3XL-8k",
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        report("After model load (4-bit via bitsandbytes)")

    # Stage 3: apply LoRA
    print("=" * 60)
    print("Applying LoRA adapters (r=16)...")
    print("=" * 60)

    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj",
            ],
        )
    except Exception:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total_params:,} "
          f"({100 * trainable / total_params:.2f}%)")
    report("After LoRA applied")

    # Stage 4: enable gradient checkpointing
    print("=" * 60)
    print("Enabling gradient checkpointing...")
    print("=" * 60)
    model.gradient_checkpointing_enable()
    model.train()
    report("After gradient checkpointing enabled")

    # Stage 5: test forward+backward with different seq lengths
    seq_lengths = [512, 1024, 2048, 4096, 6144, 8192]

    for seq_len in seq_lengths:
        print("=" * 60)
        print(f"Testing forward+backward with seq_len={seq_len}, batch_size=1")
        print("=" * 60)

        flush()
        report(f"Before forward (seq={seq_len})")

        input_ids = torch.randint(0, 50264, (1, seq_len), device=device)
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)

        try:
            torch.cuda.reset_peak_memory_stats()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            peak_fwd = torch.cuda.max_memory_allocated() / 1024**2
            report(f"After forward (seq={seq_len})")
            print(f"  peak during forward: {peak_fwd:.1f} MB")
            print(f"  loss: {loss.item():.4f}")

            torch.cuda.reset_peak_memory_stats()
            loss.backward()
            peak_bwd = torch.cuda.max_memory_allocated() / 1024**2
            report(f"After backward (seq={seq_len})")
            print(f"  peak during backward: {peak_bwd:.1f} MB")

            model.zero_grad(set_to_none=True)
            del outputs, loss, input_ids, labels, attention_mask
            flush()
            report(f"After cleanup (seq={seq_len})")

            print(f"  >>> seq_len={seq_len}: peak_fwd={peak_fwd:.0f}MB, "
                  f"peak_bwd={peak_bwd:.0f}MB")
            print()

        except torch.cuda.OutOfMemoryError:
            print(f"  >>> OOM at seq_len={seq_len}!")
            del input_ids, labels, attention_mask
            flush()
            report(f"After OOM cleanup (seq={seq_len})")
            break

        except Exception as e:
            print(f"  >>> Error at seq_len={seq_len}: {e}")
            del input_ids, labels, attention_mask
            flush()
            break

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    alloc, reserved, free, total = gpu_mem_mb()
    print(f"GPU total VRAM: {total:.0f} MB")
    print(f"Currently allocated: {alloc:.0f} MB")
    print(f"Currently reserved: {reserved:.0f} MB")
    print()
    print("Use these measurements to configure training parameters.")
    print("If 8192 causes OOM, consider reducing max_seq_length or")
    print("using DeepSpeed ZeRO-3 across all 4 GPUs.")


if __name__ == "__main__":
    main()
