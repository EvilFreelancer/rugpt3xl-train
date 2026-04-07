"""
ruGPT3XL-8k multi-GPU training with Unsloth + device_map="balanced".

Splits 24 layers across 4x RTX 5060 Ti (16GB each), giving ~64GB effective VRAM.
Memory-efficient attention is built into the model code (modeling_rugpt3xl.py):
dense layers use Flash Attention via is_causal=True, sparse layers use
EFFICIENT_ATTENTION backend. No runtime patching needed.

Usage (inside container):
    python /workspace/work/train_rugpt3xl_multigpu.py

Monitor:
    tensorboard --logdir /workspace/work/runs/rugpt3xl-8k-multigpu/logs
"""

import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

MODEL_DIR = "/workspace/models/ruGPT3XL-8k"
MAX_SEQ_LENGTH = 8192
OUTPUT_DIR = "/workspace/work/runs/rugpt3xl-8k-multigpu"


def gpu_report():
    """Print memory usage for all GPUs."""
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        total = torch.cuda.get_device_properties(i).total_memory / 1024**2
        print(f"  GPU{i}: alloc={alloc:.0f}MB  reserved={reserved:.0f}MB  "
              f"free={total - reserved:.0f}MB / {total:.0f}MB")


def format_conversations(examples, tokenizer):
    """Apply chat template to convert ShareGPT conversations to text."""
    texts = []
    for convos in examples["conversations"]:
        messages = []
        for msg in convos:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("system", "user", "assistant"):
                messages.append({"role": role, "content": content})
        if messages:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            texts.append(text)
        else:
            texts.append("")
    return {"text": texts}


def main():
    print("=" * 70)
    print("ruGPT3XL-8k Multi-GPU Training")
    print("=" * 70)
    print(f"GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print()

    # ── Step 0: Ensure tokenizer has GigaChat3 tokens ──
    print("[0/6] Checking tokenizer ...")
    from tokenizer_setup import ensure_gigachat3_tokenizer
    ensure_gigachat3_tokenizer(MODEL_DIR)

    # ── Step 1: Load model across all GPUs ──
    print("\n[1/6] Loading model with device_map='balanced' ...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
        trust_remote_code=True,
        device_map="balanced",
    )
    print("  Model loaded!")

    new_vocab = len(tokenizer)
    old_vocab = model.config.vocab_size
    if new_vocab != old_vocab:
        model.resize_token_embeddings(new_vocab)
        print(f"  Resized embeddings: {old_vocab} -> {new_vocab}")
    gpu_report()

    # ── Step 2: Apply LoRA ──
    print("\n[2/6] Applying LoRA (r=64, alpha=128) ...")
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} "
          f"({100 * trainable / total_params:.2f}%)")

    # ── Step 3: Load and format datasets ──
    print("\n[3/6] Loading datasets ...")
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
    print(f"  Train: {len(train_ds)} samples, Eval: {len(eval_ds)} samples")

    print("\n[4/6] Formatting datasets (applying chat template) ...")
    train_ds = train_ds.map(
        format_conversations,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=train_ds.column_names,
        desc="Formatting train",
    )
    eval_ds = eval_ds.map(
        format_conversations,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=eval_ds.column_names,
        desc="Formatting eval",
    )

    empty_train = sum(1 for x in train_ds if not x["text"])
    empty_eval = sum(1 for x in eval_ds if not x["text"])
    if empty_train:
        train_ds = train_ds.filter(lambda x: bool(x["text"]))
    if empty_eval:
        eval_ds = eval_ds.filter(lambda x: bool(x["text"]))

    print(f"  Train: {len(train_ds)} samples (removed {empty_train} empty)")
    print(f"  Eval:  {len(eval_ds)} samples (removed {empty_eval} empty)")
    print(f"  Sample text (first 200 chars): {train_ds[0]['text'][:200]}...")

    EVAL_SUBSET_SIZE = 500
    if len(eval_ds) > EVAL_SUBSET_SIZE:
        eval_ds = eval_ds.shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))
        print(f"  Eval subsampled to {EVAL_SUBSET_SIZE} for faster evaluation")

    # ── Step 4b: Verify masking on a sample ──
    from masking import (
        AssistantOnlyCollator, get_marker_ids, print_masking_diagnostic,
    )
    resp_ids, end_ids = get_marker_ids(tokenizer)
    print(f"\n  Response marker IDs: {resp_ids}")
    print(f"  Message end IDs:     {end_ids}")
    print_masking_diagnostic(tokenizer, train_ds[0]["text"], resp_ids, end_ids)

    # ── Step 5: Configure trainer ──
    print("\n[5/6] Configuring SFTTrainer ...")
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported
    from transformers import EarlyStoppingCallback

    collator = AssistantOnlyCollator(
        tokenizer=tokenizer,
        response_marker_ids=resp_ids,
        message_end_ids=end_ids,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # GBS = per_device_train_batch_size * gradient_accumulation_steps = 2 * 64 = 128
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,

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

        logging_steps=10,
        logging_dir=f"{OUTPUT_DIR}/logs",

        seed=42,
        report_to="tensorboard",

        dataset_text_field="text",
        packing=False,
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

    print("  Trainer ready!")
    gpu_report()

    # ── Step 6: Train! ──
    print("\n[6/6] Starting training ...")
    print("=" * 70)
    trainer_stats = trainer.train()

    print()
    print("=" * 70)
    print("Training complete!")
    print(f"  Total time: {trainer_stats.metrics['train_runtime']:.0f}s")
    print(f"  Final loss: {trainer_stats.metrics['train_loss']:.4f}")
    print()
    gpu_report()

    # Save final LoRA adapter
    save_path = f"{OUTPUT_DIR}/final_lora"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nLoRA adapter saved to: {save_path}")


if __name__ == "__main__":
    main()
