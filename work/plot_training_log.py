#!/usr/bin/env python3
"""Parse Hugging Face Trainer style logs and plot training metrics."""

from __future__ import annotations

import ast
import re
from pathlib import Path


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)


def extract_metric_dict(line: str) -> dict | None:
    line = _strip_ansi(line).replace("\r", "")
    for marker in ("{'loss':", "{'eval_loss':"):
        i = line.rfind(marker)
        if i == -1:
            continue
        frag = line[i:].strip()
        try:
            return ast.literal_eval(frag)
        except (SyntaxError, ValueError):
            continue
    return None


def parse_log(path: Path) -> tuple[list[dict], list[dict]]:
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            d = extract_metric_dict(line)
            if d is None:
                continue
            if "eval_loss" in d:
                eval_rows.append(d)
            elif "loss" in d and "learning_rate" in d:
                train_rows.append(d)
    return train_rows, eval_rows


def main() -> None:
    root = Path(__file__).resolve().parent
    log_path = root / "training_multigpu.log"
    out_dir = root / "training_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    train, ev = parse_log(log_path)
    if not train:
        raise SystemExit(f"No training metrics found in {log_path}")

    import matplotlib.pyplot as plt

    ep_tr = [r["epoch"] for r in train]
    loss_tr = [r["loss"] for r in train]
    gnorm = [r["grad_norm"] for r in train]
    lr = [r["learning_rate"] for r in train]

    fig_loss, ax_loss = plt.subplots(figsize=(11, 5.5))
    ax_loss.plot(ep_tr, loss_tr, color="#1f77b4", linewidth=1.8, label="train_loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train_loss", color="#1f77b4")
    ax_loss.tick_params(axis="y", labelcolor="#1f77b4")
    if ev:
        ep_ev = [r["epoch"] for r in ev]
        loss_ev = [r["eval_loss"] for r in ev]
        ax_r = ax_loss.twinx()
        ax_r.scatter(
            ep_ev,
            loss_ev,
            color="#d62728",
            s=80,
            zorder=5,
            label="eval_loss",
            marker="o",
            edgecolors="white",
            linewidths=1.2,
        )
        ax_r.plot(ep_ev, loss_ev, color="#d62728", linewidth=1, alpha=0.5, linestyle="--")
        ax_r.set_ylabel("eval_loss", color="#d62728")
        ax_r.tick_params(axis="y", labelcolor="#d62728")
        lines_l, lab_l = ax_loss.get_legend_handles_labels()
        lines_r, lab_r = ax_r.get_legend_handles_labels()
        ax_loss.legend(lines_l + lines_r, lab_l + lab_r, loc="upper right")
    else:
        ax_loss.legend(loc="upper right")
    ax_loss.set_title(
        "train_loss and eval_loss (ruGPT3XL-8k multi-GPU); "
        "right axis = eval (scales differ)"
    )
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()
    fig_loss.savefig(out_dir / "loss_train_eval.png", dpi=150)
    plt.close(fig_loss)

    fig2, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(ep_tr, gnorm, color="#2ca02c", linewidth=1.5)
    axes[0].set_ylabel("grad_norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Gradient norm")

    axes[1].plot(ep_tr, lr, color="#ff7f0e", linewidth=1.5)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("learning_rate")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Learning rate")

    fig2.suptitle("Training dynamics (from training_multigpu.log)", y=1.02)
    fig2.tight_layout()
    fig2.savefig(out_dir / "grad_norm_and_lr.png", dpi=150)
    plt.close(fig2)

    if ev:
        fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4))
        ep_ev = [r["epoch"] for r in ev]
        axes3[0].bar(range(len(ev)), [r["eval_runtime"] / 3600 for r in ev], color="#9467bd")
        axes3[0].set_xticks(range(len(ev)))
        axes3[0].set_xticklabels([f"e={e:.2f}" for e in ep_ev])
        axes3[0].set_ylabel("hours")
        axes3[0].set_title("eval_runtime")

        axes3[1].plot(ep_ev, [r["eval_samples_per_second"] for r in ev], "o-", color="#8c564b")
        axes3[1].set_xlabel("epoch")
        axes3[1].set_ylabel("samples/s")
        axes3[1].set_title("eval_samples_per_second")
        axes3[1].grid(True, alpha=0.3)

        axes3[2].plot(ep_ev, [r["eval_steps_per_second"] for r in ev], "o-", color="#e377c2")
        axes3[2].set_xlabel("epoch")
        axes3[2].set_ylabel("steps/s")
        axes3[2].set_title("eval_steps_per_second")
        axes3[2].grid(True, alpha=0.3)

        fig3.suptitle("Evaluation throughput (eval loop)", y=1.05)
        fig3.tight_layout()
        fig3.savefig(out_dir / "eval_throughput.png", dpi=150)
        plt.close(fig3)

    # Combined overview: loss (+ eval), grad_norm, learning_rate
    fig4, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)

    ax0.plot(ep_tr, loss_tr, label="train_loss", color="#1f77b4")
    ax0.set_ylabel("train_loss", color="#1f77b4")
    ax0.tick_params(axis="y", labelcolor="#1f77b4")
    if ev:
        ax0r = ax0.twinx()
        ax0r.scatter(ep_ev, loss_ev, color="#d62728", s=70, zorder=5, label="eval_loss")
        ax0r.set_ylabel("eval_loss", color="#d62728")
        ax0r.tick_params(axis="y", labelcolor="#d62728")
        l0, lab0 = ax0.get_legend_handles_labels()
        l1, lab1 = ax0r.get_legend_handles_labels()
        ax0.legend(l0 + l1, lab0 + lab1, loc="upper right")
    else:
        ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    ax1.plot(ep_tr, gnorm, color="#2ca02c", label="grad_norm")
    ax1.set_ylabel("grad_norm")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep_tr, lr, color="#ff7f0e", label="learning_rate")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("learning_rate")
    ax2.grid(True, alpha=0.3)

    fig4.suptitle("ruGPT3XL-8k multi-GPU - all logged training metrics", y=1.01)
    fig4.tight_layout()
    fig4.savefig(out_dir / "overview_all_metrics.png", dpi=150)
    plt.close(fig4)

    print(f"Wrote plots to {out_dir}/")
    print(f"  train log points: {len(train)}, eval log points: {len(ev)}")


if __name__ == "__main__":
    main()
