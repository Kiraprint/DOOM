#!/usr/bin/env python3
"""Parse Sample Factory train logs and plot episode reward over environment steps."""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

TRAIN_DIR = "/home/kir/Code/DOOM/train_dir"
MAX_STEPS_DEFAULT = 50_000_000  # 50M for doom_battle_appo runs
SMOOTH_WINDOW = 20              # running-mean window for smoothing


def parse_event_files(summary_dir: str) -> tuple[list[int], list[float]]:
    """Read all event files under *summary_dir* and return (steps, reward)."""
    ea = event_accumulator.EventAccumulator(
        summary_dir,
        size_guidance={"scalars": 0},  # keep all entries
    )
    ea.Reload()

    # Sample Factory logs reward under tags like "episode_reward" or "reward"
    tags = [t for t in ea.Tags()["scalars"] if "reward" in t.lower()]
    if not tags:
        raise ValueError(f"No reward tag found in {summary_dir}. Tags: {ea.Tags()['scalars']}")
    tag = tags[0]  # usually just one

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    rewards = [e.value for e in events]
    return steps, rewards


def running_mean(data: list[float], window: int) -> list[float]:
    """Simple centred running mean."""
    arr = np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid").tolist()


def main():
    runs = sorted(os.listdir(TRAIN_DIR))
    if not runs:
        print("No run directories found in", TRAIN_DIR)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for run_name in runs:
        summary_path = os.path.join(TRAIN_DIR, run_name, ".summary", "0")
        if not os.path.isdir(summary_path):
            continue

        try:
            steps, rewards = parse_event_files(summary_path)
        except Exception as e:
            print(f"Skipping {run_name}: {e}")
            continue

        # For default doom_battle_appo runs, cap at 50M steps
        max_steps = MAX_STEPS_DEFAULT if "doom_battle_appo" in run_name else None
        if max_steps:
            mask = np.array(steps) <= max_steps
            steps = list(np.array(steps)[mask])
            rewards = list(np.array(rewards)[mask])

        # Smooth
        if len(rewards) > SMOOTH_WINDOW:
            smooth = running_mean(rewards, SMOOTH_WINDOW)
            steps = steps[:len(smooth)]
            rewards = smooth

        label = run_name
        ax.plot(steps, rewards, label=label, linewidth=1.5)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Avg Episode Reward")
    ax.set_title("Training Progress — Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis with M suffix
    from matplotlib.ticker import FuncFormatter
    def fmt(x, _):
        if x >= 1e6:
            return f"{x/1e6:.1f}M"
        if x >= 1e3:
            return f"{x/1e3:.0f}K"
        return str(int(x))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))

    out_path = os.path.join(TRAIN_DIR, "train_progress.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
