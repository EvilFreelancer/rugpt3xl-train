"""
Launch ruGPT3XL-8k training via Unsloth Studio API.

Progress is visible in Studio UI at http://localhost:8000

Usage:
    python /workspace/work/train_rugpt3xl.py

Prerequisites:
    1. Studio running (docker compose up)
    2. Dataset prepared: /workspace/work/sft_mix_150k.jsonl
       (run prepare_dataset.py first)
"""

import json
import sys
import time

import requests

STUDIO_URL = "http://127.0.0.1:8000"
STUDIO_USER = "unsloth"
STUDIO_PASS = "12345678"

TRAIN_CONFIG = {
    "model_name": "/workspace/models/ruGPT3XL-8k",
    "training_type": "LoRA/QLoRA",
    "trust_remote_code": True,
    "load_in_4bit": True,
    "max_seq_length": 8192,

    # Dataset
    "local_datasets": ["/workspace/work/sft_mix_150k.jsonl"],
    "local_eval_datasets": ["/workspace/work/sft_eval_7k.jsonl"],
    "format_type": "sharegpt",
    "eval_steps": 100,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj",
    ],

    # Training
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": "2e-4",
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "weight_decay": 0.001,
    "optim": "adamw_8bit",
    "random_seed": 42,
    "packing": False,
    "train_on_completions": True,
    "gradient_checkpointing": "unsloth",
    "save_steps": 500,

    # Logging
    "enable_tensorboard": True,
    "tensorboard_dir": "/workspace/work/runs",
}


def get_session() -> requests.Session:
    """Authenticate and return a session with auth headers."""
    s = requests.Session()
    r = s.post(
        f"{STUDIO_URL}/api/auth/login",
        json={"username": STUDIO_USER, "password": STUDIO_PASS},
        timeout=10,
    )
    if not r.ok:
        print(f"ERROR: login failed: {r.status_code} {r.text}", file=sys.stderr)
        sys.exit(1)
    token = r.json()["access_token"]
    s.headers.update({"Authorization": f"Bearer {token}"})
    print(f"Authenticated as '{STUDIO_USER}'.")
    return s


def wait_for_studio():
    """Wait until Studio is responsive."""
    print("Waiting for Studio API ...")
    for _ in range(60):
        try:
            r = requests.get(f"{STUDIO_URL}/api/health", timeout=2)
            if r.ok:
                print("Studio is ready.")
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    print("ERROR: Studio did not become ready in 120s", file=sys.stderr)
    sys.exit(1)


def start_training(session: requests.Session):
    """POST training request to Studio and stream progress."""
    r = session.post(
        f"{STUDIO_URL}/api/train/start",
        json=TRAIN_CONFIG,
        timeout=30,
    )
    if not r.ok:
        print(f"ERROR: {r.status_code} {r.text}", file=sys.stderr)
        sys.exit(1)

    data = r.json()
    job_id = data.get("job_id", "unknown")
    print(f"Training started! job_id={job_id}")
    print(f"Monitor in Studio UI: {STUDIO_URL}")
    return job_id


def follow_progress(session: requests.Session, job_id: str):
    """Stream SSE progress events from Studio."""
    print("\n--- Training progress ---")
    try:
        with session.get(
            f"{STUDIO_URL}/api/train/progress",
            stream=True,
            timeout=None,
        ) as resp:
            current_event = ""
            for line in resp.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload:
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if current_event in ("progress", ""):
                    step = event.get("step", "?")
                    total = event.get("total_steps", "?")
                    loss = event.get("loss")
                    lr = event.get("learning_rate")
                    pct = event.get("progress_percent") or 0
                    lr_str = f"{lr:.2e}" if lr is not None else "n/a"
                    loss_str = f"{loss:.4f}" if loss is not None else "n/a"
                    print(
                        f"  step {step}/{total}  loss={loss_str}  lr={lr_str}  ({pct:.1f}%)",
                        flush=True,
                    )
                elif current_event == "complete":
                    print("\nTraining complete!")
                    output = event.get("output_dir", "?")
                    print(f"Model saved to: {output}")
                    return
                elif current_event == "error":
                    msg = event.get("message", event.get("error", "unknown"))
                    print(f"\nERROR: {msg}", file=sys.stderr)
                    return
    except KeyboardInterrupt:
        print("\nInterrupted. Training continues in background.")
        print(f"Check progress in Studio UI: {STUDIO_URL}")


def main():
    wait_for_studio()
    session = get_session()
    job_id = start_training(session)
    follow_progress(session, job_id)


if __name__ == "__main__":
    main()
