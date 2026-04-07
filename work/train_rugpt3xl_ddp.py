"""
ruGPT3XL-8k DDP training across 4x RTX 5060 Ti (16GB each).

Each GPU trains on its own data batch with a full model copy.
Memory-efficient attention is built into the model code (modeling_rugpt3xl.py):
dense layers use Flash Attention via is_causal=True, sparse layers use
EFFICIENT_ATTENTION backend. No runtime patching needed.

Usage (inside container):
    torchrun --nproc_per_node=4 /workspace/work/train_rugpt3xl_ddp.py

Monitor:
    tensorboard --logdir /workspace/work/runs/rugpt3xl-8k-ddp/logs
"""

import os
import sys

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

MODEL_DIR = "/workspace/models/ruGPT3XL-8k"
MAX_SEQ_LENGTH = 8192
OUTPUT_DIR = "/workspace/work/runs/rugpt3xl-8k-ddp"


def format_conversations(examples, tokenizer):
    """Apply chat template to convert ShareGPT conversations to text."""
    texts = []
    for convos in examples["conversations"]:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in convos
            if m.get("role") in ("system", "user", "assistant") and m.get("content")
        ]
        if messages:
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            ))
        else:
            texts.append("")
    return {"text": texts}


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        print("=" * 70)
        print("ruGPT3XL-8k DDP Training")
        print("=" * 70)
        print(f"GPUs: {world_size}x {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
        print()

    # ── Step 0: Ensure tokenizer has GigaChat3 tokens (rank 0 only) ──
    if rank == 0:
        print("[0/5] Checking tokenizer ...")
        from tokenizer_setup import ensure_gigachat3_tokenizer
        ensure_gigachat3_tokenizer(MODEL_DIR)
    if world_size > 1:
        torch.distributed.barrier()

    # ── Step 1: Load model ──
    if rank == 0:
        print("\n[1/5] Loading model ...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
        trust_remote_code=True,
    )

    new_vocab = len(tokenizer)
    old_vocab = model.config.vocab_size
    if new_vocab > old_vocab:
        model.resize_token_embeddings(new_vocab)
        if rank == 0:
            print(f"  Resized embeddings: {old_vocab} -> {new_vocab}")
    elif new_vocab < old_vocab and rank == 0:
        print(f"  Tokenizer vocab ({new_vocab}) < model vocab ({old_vocab}), "
              f"keeping model size (extra slots are harmless)")

    # ── Step 2: Apply LoRA ──
    if rank == 0:
        print("[2/5] Applying LoRA (r=64, alpha=128) ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total_params:,} "
              f"({100 * trainable / total_params:.2f}%)")

    # ── Step 3: Load and format datasets ──
    if rank == 0:
        print("[3/5] Loading and formatting datasets ...")
    from datasets import load_dataset

    train_ds = load_dataset(
        "json",
        data_files="/workspace/work/sft_mix_150k.jsonl",
        split="train",
    )
    eval_ds = load_dataset(
        "json",
        data_files="/workspace/work/sft_eval_7k.jsonl",
        split="train",
    )

    train_ds = train_ds.map(
        format_conversations, fn_kwargs={"tokenizer": tokenizer},
        batched=True, batch_size=1000, num_proc=4,
        remove_columns=train_ds.column_names, desc="Formatting train",
    )
    eval_ds = eval_ds.map(
        format_conversations, fn_kwargs={"tokenizer": tokenizer},
        batched=True, batch_size=1000, num_proc=4,
        remove_columns=eval_ds.column_names, desc="Formatting eval",
    )

    train_ds = train_ds.filter(lambda x: bool(x["text"]))
    eval_ds = eval_ds.filter(lambda x: bool(x["text"]))

    EVAL_SUBSET_SIZE = 500
    if len(eval_ds) > EVAL_SUBSET_SIZE:
        eval_ds = eval_ds.shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))

    if rank == 0:
        print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)} (subsampled)")

    # ── Step 3b: Verify masking on a sample ──
    from masking import (
        AssistantOnlyCollator, get_marker_ids, print_masking_diagnostic,
    )
    resp_ids, end_ids = get_marker_ids(tokenizer)
    if rank == 0:
        print(f"\n  Response marker IDs: {resp_ids}")
        print(f"  Message end IDs:     {end_ids}")
        print_masking_diagnostic(tokenizer, train_ds[0]["text"], resp_ids, end_ids)

    # ── Step 4: Configure trainer ──
    if rank == 0:
        print("\n[4/5] Configuring SFTTrainer ...")
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported
    from transformers import EarlyStoppingCallback

    collator = AssistantOnlyCollator(
        tokenizer=tokenizer,
        response_marker_ids=resp_ids,
        message_end_ids=end_ids,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # GBS = world_size * per_device_train_batch_size * gradient_accumulation_steps
    # GBS = 4 * 1 * 32 = 128
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,

        num_train_epochs=3,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_8bit",

        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        eval_strategy="steps",
        eval_steps=100,
        eval_on_start=True,
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=1,
        logging_dir=f"{OUTPUT_DIR}/logs",

        seed=42,
        report_to="tensorboard",

        dataset_text_field="text",
        packing=False,

        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    if rank == 0:
        print("  Trainer ready!")
        alloc = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU{rank} memory: {alloc:.0f}MB allocated")

    # ── Step 5: Train! ──
    if rank == 0:
        print("\n[5/5] Starting training ...")
        print("=" * 70)

    trainer_stats = trainer.train()

    if rank == 0:
        print()
        print("=" * 70)
        print("Training complete!")
        print(f"  Total time: {trainer_stats.metrics['train_runtime']:.0f}s")
        print(f"  Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
        print(f"  Final loss: {trainer_stats.metrics['train_loss']:.4f}")

        save_path = f"{OUTPUT_DIR}/final_lora"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"\n  LoRA adapter saved to: {save_path}")


if __name__ == "__main__":
    main()
