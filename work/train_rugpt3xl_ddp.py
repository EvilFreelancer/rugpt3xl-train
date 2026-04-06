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

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

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

    # ── Step 1: Load model ──
    if rank == 0:
        print("[1/5] Loading model ...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/workspace/models/ruGPT3XL-8k",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
        trust_remote_code=True,
    )

    # ── Step 2: Apply LoRA ──
    if rank == 0:
        print("[2/5] Applying LoRA (r=16) ...")
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

    if rank == 0:
        print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # ── Step 4: Configure trainer ──
    if rank == 0:
        print("[4/5] Configuring SFTTrainer ...")
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported

    # With 4 GPUs DDP: effective_batch = 4 * 1 * 4 = 16 (same as single GPU with accum=16)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,

        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.001,
        optim="adamw_8bit",

        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,

        logging_steps=10,
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
        args=sft_config,
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
