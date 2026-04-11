"""
ruGPT3XL-8k FSDP training - true 4-GPU data parallelism.

Unlike device_map="balanced" (single stream, layers split sequentially across GPUs),
FSDP shards parameters/gradients/optimizer across all GPUs and runs real data-parallel
training. This gives ~4x throughput compared to pipeline parallelism.

Key difference from the multigpu script: no Unsloth (incompatible with FSDP),
no 4-bit quantization (FSDP shards memory instead), bf16 throughout.

Memory per GPU (FSDP FULL_SHARD, bf16, 4x RTX 5060 Ti 16GB):
  Sharded params:    ~650 MB  (1.3B * 2 / 4)
  Sharded optimizer: ~2.6 GB  (Adam fp32 states / 4)
  Sharded gradients: ~650 MB  (1.3B * 2 / 4)
  Activations:       ~10 GB   (gradient checkpointing, 8k ctx)
  Total:             ~14 GB

Usage (inside container):
    torchrun --nproc_per_node=4 /workspace/work/train_rugpt3xl_fsdp.py

Monitor:
    tensorboard --logdir /workspace/work/runs/rugpt3xl-8k-fsdp/logs
    Training metrics are written to TensorBoard (logging_dir) and printed on rank 0
    (ConsoleLogCallback with flush for pipe/tee).
"""

import glob
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "1"
os.environ["NCCL_TIMEOUT"] = "3600"  # 60 min for slow checkpoint operations
os.environ["FSDP_STATE_DICT_TYPE"] = "SHARDED_STATE_DICT"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # More reliable but slower NCCL
# Note: TORCH_DISTRIBUTED_DEBUG disabled to reduce log spam from FSDP

# Suppress verbose FSDP warnings
import warnings
warnings.filterwarnings("ignore", message=".*FSDP firing post-backward hooks.*")
warnings.filterwarnings("ignore", message=".*FSDP.state_dict_type.*")
warnings.filterwarnings("ignore", message=".*ShardedTensor.*")
warnings.filterwarnings("ignore", message=".*Please use DTensor.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from transformers import TrainerCallback


def assistant_only_loss(outputs, labels, num_items_in_batch=None):
    """
    Custom loss function for assistant-only training.
    Computes cross-entropy loss only on non-masked tokens (labels != -100).
    This ensures train and eval losses are computed consistently.
    """
    logits = outputs.logits
    # Shift for causal LM (predict next token)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Compute mean loss on non-masked tokens only
    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=-100,
        reduction="mean",
    )

    return loss

MODEL_DIR = "/workspace/models/ruGPT3XL-8k"
MAX_SEQ_LENGTH = 8192
OUTPUT_DIR = "/workspace/work/runs/rugpt3xl-8k-fsdp"

LORA_R = 64
LORA_ALPHA = 128
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
GBS = 128
PER_DEVICE_BATCH = 1
EVAL_SUBSET_SIZE = 500


class ConsoleLogCallback(TrainerCallback):
    """Print the same metrics as TensorBoard to stdout on rank 0 (flush for tee/pipes)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        out = dict(logs)
        out.pop("total_flos", None)
        print(out, flush=True)


class SaveAdapterCallback(TrainerCallback):
    """Save LoRA adapter after evaluation using external script."""

    def __init__(self):
        self.last_extracted_step = 0

    def on_evaluate(self, args, state, control, **kwargs):
        """Extract LoRA adapter after evaluation on rank 0."""
        if not state.is_world_process_zero:
            return control

        # Only extract every 50 steps (after eval)
        if state.global_step - self.last_extracted_step < 50:
            return control
        self.last_extracted_step = state.global_step

        # Run extraction script in background (non-blocking)
        import subprocess
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        adapter_path = os.path.join(checkpoint_path, "adapter_model")

        # Check if adapter already exists
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            print(f"[SaveAdapterCallback] Adapter already exists at step {state.global_step}, skipping")
            return control

        # Build and run extraction script
        extract_script = os.path.join(os.path.dirname(__file__), "extract_lora_from_checkpoint.py")
        if os.path.exists(extract_script):
            cmd = [
                "python", extract_script,
                checkpoint_path,
                adapter_path,
                ">/dev/null", "2>&1", "&"
            ]
            print(f"[SaveAdapterCallback] Starting LoRA extraction for step {state.global_step}", flush=True)
            subprocess.Popen(
                " ".join(cmd),
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            print(f"[SaveAdapterCallback] Warning: extract_lora_from_checkpoint.py not found", flush=True)

        return control


def get_valid_checkpoints(output_dir: str) -> list[str]:
    """
    Ignore partially written FSDP checkpoints.
    A usable trainer checkpoint must contain trainer_state.json.
    """
    candidates = sorted(glob.glob(f"{output_dir}/checkpoint-*"))
    valid = []
    for ckpt in candidates:
        if os.path.exists(os.path.join(ckpt, "trainer_state.json")):
            valid.append(ckpt)
    return valid


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
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0

    if world_size > 1:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(rank)

    if is_main:
        print("=" * 70)
        print("ruGPT3XL-8k FSDP Training")
        print("=" * 70)
        print(f"GPUs: {world_size}x {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
        print()

    # ── Step 0: Patch tokenizer (rank 0 only) ──
    if is_main:
        print("[0/6] Checking tokenizer ...")
        from tokenizer_setup import ensure_gigachat3_tokenizer
        ensure_gigachat3_tokenizer(MODEL_DIR)
    if world_size > 1:
        torch.distributed.barrier()

    # ── Step 1: Load model in bf16 (FSDP handles memory, no quantization) ──
    if is_main:
        print("\n[1/6] Loading model (bf16, no quantization) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    new_vocab = len(tokenizer)
    old_vocab = model.config.vocab_size
    if new_vocab > old_vocab:
        model.resize_token_embeddings(new_vocab)
        if is_main:
            print(f"  Resized embeddings: {old_vocab} -> {new_vocab}")
    elif new_vocab < old_vocab and is_main:
        print(f"  Tokenizer vocab ({new_vocab}) < model vocab ({old_vocab}), "
              f"keeping model size")

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded: {total_params:,} params in bf16")

    # ── Step 2: Apply LoRA ──
    if is_main:
        print(f"\n[2/6] Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}) ...")
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    if is_main:
        model.print_trainable_parameters()

    # ── Step 3: Load datasets ──
    if is_main:
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
    if is_main:
        print(f"  Train: {len(train_ds)} samples, Eval: {len(eval_ds)} samples")

    # ── Step 4: Format + filter ──
    if is_main:
        print("\n[4/6] Formatting datasets ...")
    train_ds = train_ds.map(
        format_conversations,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=train_ds.column_names,
        desc="Formatting train" if is_main else None,
    )
    eval_ds = eval_ds.map(
        format_conversations,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=eval_ds.column_names,
        desc="Formatting eval" if is_main else None,
    )

    train_ds = train_ds.filter(lambda x: bool(x["text"]))
    eval_ds = eval_ds.filter(lambda x: bool(x["text"]))

    if len(eval_ds) > EVAL_SUBSET_SIZE:
        eval_ds = eval_ds.shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))
        if is_main:
            print(f"  Eval subsampled to {EVAL_SUBSET_SIZE}")

    from masking import (
        AssistantOnlyCollator, get_marker_ids, print_masking_diagnostic,
        build_assistant_mask,
    )
    resp_ids, end_ids = get_marker_ids(tokenizer)
    if is_main:
        print(f"  Response marker IDs: {resp_ids}")
        print(f"  Message end IDs:     {end_ids}")
        print_masking_diagnostic(tokenizer, train_ds[0]["text"], resp_ids, end_ids)

    def _has_assistant_tokens(example):
        ids = tokenizer.encode(example["text"], add_special_tokens=False)
        labels = build_assistant_mask(ids, resp_ids, end_ids)
        return any(l != -100 for l in labels)

    train_before, eval_before = len(train_ds), len(eval_ds)
    train_ds = train_ds.filter(_has_assistant_tokens, num_proc=4,
                               desc="Filter train" if is_main else None)
    eval_ds = eval_ds.filter(_has_assistant_tokens, num_proc=4,
                              desc="Filter eval" if is_main else None)
    if is_main:
        print(f"  Filtered: train {train_before}->{len(train_ds)}, "
              f"eval {eval_before}->{len(eval_ds)}")

    # ── Step 5: Configure trainer with FSDP ──
    if is_main:
        print("\n[5/6] Configuring SFTTrainer (FSDP) ...")
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback

    collator = AssistantOnlyCollator(
        tokenizer=tokenizer,
        response_marker_ids=resp_ids,
        message_end_ids=end_ids,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    grad_accum = max(1, GBS // (world_size * PER_DEVICE_BATCH))
    if is_main:
        effective_gbs = world_size * PER_DEVICE_BATCH * grad_accum
        print(f"  GBS={effective_gbs} = {world_size} GPUs x {PER_DEVICE_BATCH} batch "
              f"x {grad_accum} grad_accum")

    fsdp_config = {
        "fsdp_transformer_layer_cls_to_wrap": ["RuGPT3XLDecoderLayer"],
        "backward_prefetch": "backward_pre",
        "forward_prefetch": False,
        "use_orig_params": True,
        "sync_module_states": True,
        "cpu_ram_efficient_loading": True,
        "activation_checkpointing": True,
        "state_dict_type": "SHARDED_STATE_DICT",
    }

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_SEQ_LENGTH,

        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,

        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",

        bf16=True,
        gradient_checkpointing=False,

        fsdp="full_shard auto_wrap",
        fsdp_config=fsdp_config,

        eval_strategy="steps",
        eval_steps=50,
        eval_on_start=True,
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=1,
        logging_first_step=True,
        logging_dir=f"{OUTPUT_DIR}/logs",

        seed=42,
        report_to=["tensorboard"],
        disable_tqdm=False,

        dataset_text_field="text",
        packing=False,

        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        args=sft_config,
        compute_loss_func=assistant_only_loss,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            ConsoleLogCallback(),
            SaveAdapterCallback(),  # Extracts LoRA after eval
        ],
    )

    if is_main:
        print("  Trainer ready!")

    # ── Step 6: Train (auto-resume from latest checkpoint if available) ──
    ckpts = get_valid_checkpoints(OUTPUT_DIR)
    resume_from = ckpts[-1] if ckpts else None
    if is_main:
        print(f"\n[6/6] Starting training ...")
        if resume_from:
            print(f"  Resuming from {resume_from}")
        print("=" * 70)
    trainer.train(resume_from_checkpoint=resume_from)

    # Final save - all ranks must participate for FSDP
    save_path = f"{OUTPUT_DIR}/final_lora"

    # Unwrap FSDP and save only LoRA adapter on rank 0
    if world_size > 1:
        torch.distributed.barrier()

    if is_main:
        from peft import PeftModel

        # Get unwrapped model
        unwrapped_model = trainer.model
        while hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module

        if isinstance(unwrapped_model, PeftModel):
            unwrapped_model.save_pretrained(save_path, safe_serialization=True)
            tokenizer.save_pretrained(save_path)
            print(f"\nLoRA adapter saved to: {save_path}")
        else:
            # Fallback to trainer's save
            trainer.save_model(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
