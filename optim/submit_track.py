#!/usr/bin/env python3
"""Submit one experiment track of the optimizer study to SLURM.

Each track is a set of arms that differ in ONE thing from the reference arm
(`adapt` optimizer, wd 0, lr 4.05e-4), so a comparison is single-variable by
construction. Tracks are meant to run in parallel from separate git worktrees.

    python3 scripts/submit_track.py wd      # decay curve + coupled-L2 control
    python3 scripts/submit_track.py muon    # Muon with the fallback-lr fix
    python3 scripts/submit_track.py long    # 50M -> 250M confirmation

Idempotent: a cell whose train_dir/<experiment>/config.json already exists is
skipped, so re-running after a partial drain is safe.
"""

import json
import os
import subprocess
import sys

LR = 4.05e-4          # the released HPO point (gru_250m_baseline)
# AdamW is usually tuned at a larger lr than Adam, so testing it only at Adam's
# lr would leave the obvious objection open: that the arm lost on step size, not
# on the decay mechanism. This arm doubles the lr and changes nothing else.
LR_OVERRIDE = {"optim_adamw_lr2x_wd2e3": 8.1e-4}
MUON_LR = 0.02        # orthogonalised-update lr; fallback group stays at LR
WD = 1.96e-3          # the HPO point's weight_decay, inert before the patch
SEEDS = (41, 42, 43)

# (experiment, optimizer, muon_lr, weight_decay, env frames)
TRACKS = {
    "wd": [
        ("optim_adamw_wd5e4", "adamw", None, 5e-4, 50_000_000),
        ("optim_adamw_wd8e3", "adamw", None, 8e-3, 50_000_000),
        ("optim_adamw_lr2x_wd2e3", "adamw", None, WD, 50_000_000),
        ("optim_l2_wd2e4", "adam", None, 2e-4, 50_000_000),
    ],
    "muon": [
        ("optim_muon_lr5e3", "muon", 5e-3, 0.0, 50_000_000),
        ("optim_muon_lr2e2_fb", "muon", MUON_LR, 0.0, 50_000_000),
        ("optim_muon_lr2e2_wd", "muon", MUON_LR, WD, 50_000_000),
    ],
    "long": [
        ("long_adam_wd0", "adam", None, 0.0, 250_000_000),
        ("long_adamw_wd2e3", "adamw", None, WD, 250_000_000),
    ],
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def queued_jobs():
    """Names currently queued or running for this user (empty if squeue fails)."""
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                         capture_output=True, text=True)
    return set(out.stdout.split())
VENV = "/nfs/home/kfedorin/DOOM/.venv/bin/python"


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in TRACKS:
        raise SystemExit(f"usage: {sys.argv[0]} {{{'|'.join(TRACKS)}}} [--dry-run]")
    track = sys.argv[1]
    dry = "--dry-run" in sys.argv
    train_dir = os.path.join(REPO, "train_dir")   # each worktree keeps its own runs

    submitted = []
    for exp, optim, muon_lr, wd, frames in TRACKS[track]:
        for seed in SEEDS:
            name = f"{exp}_s{seed - 40}"
            if os.path.exists(os.path.join(train_dir, name, "config.json")):
                print(f"skip {name}: already has config.json")
                continue
            # config.json only appears once a cell starts, so a cell that is
            # queued but not yet running would be submitted twice by a re-run.
            # The job name is the identity, so ask the scheduler as well.
            if name in queued_jobs():
                print(f"skip {name}: already queued")
                continue
            cmd = [
                "sbatch", "-p", "gpu", "--qos=normal", "--gres=gpu:1",
                "--cpus-per-task=10", "--mem=20G", "-t", "06:00:00",
                "-J", name, f"--output={REPO}/logs/{name}.out",
                "--wrap",
                (f"cd {REPO} && {VENV} -m models.train --rnn_type gru "
                 f"--optimizer {optim} --weight_decay {wd} "
                 f"--learning_rate {LR_OVERRIDE.get(exp, LR)} --exploration_loss_coeff 0.00202 "
                 f"--train_for_env_steps {frames} --seed {seed} "
                 f"--experiment {name}"
                 + (f" --muon_lr {muon_lr}" if muon_lr is not None else "")),
            ]
            if dry:
                print("DRY:", " ".join(cmd))
                submitted.append(name)
                continue
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode != 0:
                print(f"FAILED {name}: {out.stderr.strip()[:200]}")
                continue
            print(f"{out.stdout.strip()}  {name}")
            submitted.append(name)

    print(f"\ntrack {track}: {len(submitted)} cell(s) submitted/skipped-dry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
