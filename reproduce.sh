#!/usr/bin/env bash
# reproduce.sh — Full reproduction of DOOM RNN architecture comparison
#
# Reproduces all experiments, results collection, and plots.
# Usage:
#   bash reproduce.sh                        # Run everything (days)
#   bash reproduce.sh --quick                # Short 1M-step smoke test per arch
#   bash reproduce.sh --collect              # Only parse logs + generate plots
#   bash reproduce.sh --list                 # Show available experiments and their status
#   bash reproduce.sh --hpo-transformer      # Run Transformer HPO (15 trials, ~7h)
#   bash reproduce.sh --hpo-perceiver        # Run Perceiver IO HPO (10 trials, ~8h)
#   bash reproduce.sh --ablation             # Run ablation experiments (~1h)
#   bash reproduce.sh --hpo                  # Show Mamba-2 HPO info (already completed)
#
# Requirements: Python 3.10+, CUDA GPU with >=16GB VRAM
# Installs deps automatically if missing.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Config ───────────────────────────────────────────────────────────────────
NUM_WORKERS=8
NUM_ENVS_PER_WORKER=8
BATCH_SIZE=4096
ROLLOUT=64
RECURRENCE=32
TRAIN_STEPS=50000000
TRAIN_DIR="train_dir"
PLOTS_DIR="train_dir"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="reproduce_${TIMESTAMP}.log"

SEED_BASE=42  # seed2 always uses this, seed3 uses this+1, etc.

# SLURM config (auto-detected if on cluster)
SLURM_GPU_PARTITION="gpu"
SLURM_GPU_NODES="nike,kali,mars,laplas,turing,midas"
SLURM_GPU_MEMORY="24G"  # TITAN RTX on nike/kali; 11-12GB on others
SLURM_GPU_TIME="72:00:00"  # 3 days max
SLURM_CPUS_PER_TASK=10
SLURM_MEM="20G"

# ─── Helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
err() { log "ERROR: $*"; exit 1; }
section() { log ""; log "════════════════════════════════════════════════"; log "  $*"; log "════════════════════════════════════════════════"; }

check_env() {
    log "Checking environment..."
    python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null || err "PyTorch not installed. Run: pip install -r requirements.txt"
    python3 -c "import sample_factory; print(f'Sample-Factory {sample_factory.__version__}')" 2>/dev/null || log "WARN: sample_factory version unknown"
    python3 -c "import torch; assert torch.cuda.is_available(), \"No CUDA GPU found\"" 2>/dev/null || err "CUDA GPU required"
    python3 -c "
import torch
d = torch.cuda.get_device_properties(0)
mem = d.total_memory / 1024**3
print(f'GPU: {d.name} ({mem:.0f} GB)')
assert mem >= 12, f'Need >=12GB VRAM, have {mem:.0f}GB'
" 2>/dev/null || log "WARN: GPU may have insufficient VRAM"
    log "Environment OK"
}

run_experiment() {
    local exp_name=$1; shift
    local exp_dir="$TRAIN_DIR/$exp_name"
    local target_steps="${TRAIN_STEPS:-50000000}"
    
    # Detect target from --train_for_env_steps if passed
    for arg in "$@"; do
        if [[ "$arg" =~ ^[0-9]+$ ]] && [ "$arg" -gt 1000000 ] 2>/dev/null; then
            target_steps=$arg
        fi
    done
    # Also check if last positional arg looks like step count
    local last_arg="${!#}"
    if [[ "$last_arg" =~ ^[0-9]+$ ]] && [ "$last_arg" -gt 1000000 ] 2>/dev/null; then
        target_steps=$last_arg
    fi
    
    # Check for completion: look for target or near-target frame count
    if [ -f "$exp_dir/sf_log.txt" ]; then
        local max_frames=$(grep -oP 'Total num frames: \K\d+' "$exp_dir/sf_log.txt" | tail -1)
        local min_complete=$(( target_steps * 90 / 100 ))
        if [ -n "$max_frames" ] && [ "$max_frames" -ge "$min_complete" ] 2>/dev/null; then
            log "  SKIP $exp_name — already at ${max_frames} frames (target ${target_steps})"
            return 0
        fi
    fi
    
    log "  RUN $exp_name"
    uv run python3 -m models.train "$@" --experiment "$exp_name" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ! -f "$exp_dir/sf_log.txt" ]; then
        log "  WARN: $exp_name — no log found, may have failed"
        return 1
    fi
    log "  DONE $exp_name"
}

# Resolve seed number to actual seed value and experiment suffix.
# seed=1 → no --seed flag (Sample-Factory default), suffix _seed1
# seed=2 → --seed $SEED_BASE (42), suffix _seed2
# seed=3 → --seed $((SEED_BASE+1)) (43), suffix _seed3
# seed=4 → --seed $((SEED_BASE+2)) (44), suffix _seed4
seed_args() {
    local seed=$1
    if [ "$seed" -eq 1 ]; then
        echo ""
    else
        local seed_val=$((SEED_BASE + seed - 2))
        echo "--seed $seed_val"
    fi
}

seed_suffix() {
    echo "_seed${1}"
}

collect_results() {
    section "Collecting results"
    
    uv run python3 <<'PYEOF'
import re, json, os, sys, statistics

train_dir = "train_dir"

# All experiment names. The parser checks both `_seed1` suffix and bare name.
def exp_names(base):
    return [f"{base}_seed1", base]

experiments_raw = []
# GRU baseline (3 seeds)
experiments_raw += ["gru_baseline_50m", "gru_baseline_50m_seed2", "gru_baseline_50m_seed3"]
# GRU + Mamba-2 HPs (3 seeds)
experiments_raw += ["gru_optimized_50m_seed1", "gru_optimized_50m_seed2", "gru_optimized_50m_seed3"]
# Mamba-2 HPO best (4 seeds)
experiments_raw += ["mamba2_hpo_best_50m_seed1", "mamba2_hpo_best_50m_seed2",
                    "mamba2_hpo_best_50m_seed3", "mamba2_hpo_best_50m_seed4"]
# Mamba-1 (1+ seeds)
experiments_raw += ["mamba1_50m_seed1", "mamba1_50m_seed2", "mamba1_50m_seed3"]
# Transformer (1 seed)
experiments_raw += ["transformer_50m_seed1", "transformer_50m"]
# Perceiver IO (1 seed)
experiments_raw += ["perceiver_50m_seed1", "perceiver_50m"]
# GRU 250M
experiments_raw += ["doom_battle_appo_gru_250m"]
    
# GRU optimized 50 seeds
for i in range(1, 51):
    experiments_raw.append(f"gru_optimized_50m_seed{i}")

# 50-seed runs (GRU baseline and Mamba-2)
for i in range(1, 51):
    experiments_raw.append(f"gru_baseline_50m_seed{i}")
    experiments_raw.append(f"mamba2_hpo_best_50m_seed{i}")

# 250M 10-seed runs
for i in range(1, 11):
    experiments_raw.append(f"gru_250m_seed{i}")
    experiments_raw.append(f"mamba2_250m_seed{i}")

# Window ablation: 3 arch × 3 windows × 3 seeds
for arch in ['gru_baseline', 'gru_optimized', 'mamba2_hpo_best']:
    for w in [32, 64, 128]:
        for s in range(1, 4):
            experiments_raw.append(f"{arch}_50m_w{w}_seed{s}")

# Deduplicate while preserving order
experiments = []
for e in experiments_raw:
    if e not in experiments:
        experiments.append(e)

results = []
for exp in experiments:
    path = os.path.join(train_dir, exp, "sf_log.txt")
    if not os.path.exists(path):
        print(f"  {exp:40s} NO LOG")
        continue
    with open(path) as f:
        text = f.read()
    rewards = [float(r) for r in re.findall(r"Avg episode reward: \[\(0, '([0-9.]+)'\)\]", text)]
    frames_list = [int(f) for f in re.findall(r"Total num frames: (\d+)", text)]
    fps_list = [float(f) for f in re.findall(r'300 sec: ([0-9.]+)\)', text)]
    
    if not rewards:
        print(f"  {exp:40s} NO REWARD DATA")
        continue
    
    final = rewards[-1]
    best = max(rewards)
    last20 = rewards[-max(len(rewards)//5, 10):]
    mean_r = sum(last20) / len(last20)
    std_r = (sum((r - mean_r)**2 for r in last20) / len(last20))**0.5
    mean_fps = sum(fps_list[-10:]) / len(fps_list[-10:]) if fps_list else 0
    
    results.append({
        "experiment": exp,
        "best": round(best, 2),
        "final": round(final, 2),
        "last20_mean": round(mean_r, 2),
        "last20_std": round(std_r, 2),
        "fps": round(mean_fps, 0),
        "n_rewards": len(rewards),
        "frames": max(frames_list) if frames_list else 0,
    })

# Print summary table
print()
print(f"{'Experiment':40s} {'Best':>6s} {'Final':>6s} {'Mean±Std':>12s} {'FPS':>7s}")
print("-" * 75)
for r in sorted(results, key=lambda x: -x['best']):
    print(f"{r['experiment']:40s} {r['best']:6.2f} {r['final']:6.2f} {r['last20_mean']:5.2f}±{r['last20_std']:.2f} {r['fps']:7.0f}")

# Across-seed stats
print()
groups = {}
for r in results:
    base = re.sub(r'_seed\d+$', '', r['experiment'])
    groups.setdefault(base, []).append(r)

for base, runs in sorted(groups.items()):
    if len(runs) >= 2:
        bests = [r['best'] for r in runs]
        finals = [r['final'] for r in runs]
        print(f"{base:40s} {len(runs)} seeds: best={statistics.mean(bests):.2f}±{statistics.stdev(bests):.2f}, final={statistics.mean(finals):.2f}±{statistics.stdev(finals):.2f}")
    else:
        r = runs[0]
        print(f"{base:40s} 1 seed:   best={r['best']:.2f}, final={r['final']:.2f}")

# Save JSON
with open(os.path.join(train_dir, "results_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results_summary.json ({len(results)} experiments)")
PYEOF
}

generate_plots() {
    section "Generating reward plots (Russian labels, all seeds)"
    uv run python3 <<'PYEOF'
import re, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

train_dir = "train_dir"
plt.rcParams['font.size'] = 11

def load(exp):
    # Try exact name, then fallback for naming compatibility
    for name in [exp, exp.replace('_seed1', ''), exp + '_seed1']:
        path = os.path.join(train_dir, name, 'sf_log.txt')
        if os.path.exists(path):
            with open(path) as f:
                t = f.read()
            r = [float(x) for x in re.findall(r"Avg episode reward: \[\(0, '([0-9.]+)'\)\]", t)]
            f = [int(x) for x in re.findall(r"Total num frames: (\d+)", t)]
            n = min(len(r), len(f))
            return r[:n], f[:n]
    return [], []

def smooth(r, k=5):
    if len(r) < k: return np.array(r)
    return np.convolve(r, np.ones(k)/k, mode='valid')

def fmt_x(ax):
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M'))
    ax.set_xticks([0, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000])

def annotate_best(ax, rewards, frames, color):
    best_i = rewards.index(max(rewards))
    ax.scatter(frames[best_i], rewards[best_i], color=color, s=40, marker='*', edgecolors='black', linewidth=0.5, zorder=5)

# ─── Figure 1: All 50M architectures (all seeds) ───
all_exps = [
    ('GRU baseline seed1', 'gru_baseline_50m', '#1f77b4'),
    ('GRU baseline seed2', 'gru_baseline_50m_seed2', '#aec7e8'),
    ('GRU baseline seed3', 'gru_baseline_50m_seed3', '#9ec9e0'),
    ('GRU + Mamba-2 HPs seed1', 'gru_optimized_50m_seed1', '#2ca02c'),
    ('GRU + Mamba-2 HPs seed2', 'gru_optimized_50m_seed2', '#98df8a'),
    ('GRU + Mamba-2 HPs seed3', 'gru_optimized_50m_seed3', '#3a7a3a'),
    ('Mamba-2 HPO best seed1', 'mamba2_hpo_best_50m_seed1', '#ff7f0e'),
    ('Mamba-2 HPO best seed2', 'mamba2_hpo_best_50m_seed2', '#ffbb78'),
    ('Mamba-2 HPO best seed3', 'mamba2_hpo_best_50m_seed3', '#d95f02'),
    ('Mamba-2 HPO best seed4', 'mamba2_hpo_best_50m_seed4', '#ffeda0'),
    ('Mamba-1', 'mamba1_50m_seed1', '#9467bd'),
    ('Transformer', 'transformer_50m_seed1', '#d62728'),
    ('Perceiver IO', 'perceiver_50m_seed1', '#8c564b'),
]
fig, ax = plt.subplots(figsize=(14, 7))
for label, exp, color in all_exps:
    rewards, frames = load(exp)
    if not rewards: continue
    rs = smooth(rewards)
    fs = frames[len(frames)-len(rs):]
    ax.plot(fs, rs, label=label, color=color, linewidth=1.2, alpha=0.8)
    annotate_best(ax, rewards, frames, color)
fmt_x(ax)
ax.set_xlabel(u'\u0428\u0430\u0433\u0438 \u0441\u0440\u0435\u0434\u044b', fontsize=12)
ax.set_ylabel(u'\u0421\u0440\u0435\u0434\u043d\u044f\u044f \u043d\u0430\u0433\u0440\u0430\u0434\u0430 \u0437\u0430 \u044d\u043f\u0438\u0437\u043e\u0434', fontsize=12)
ax.set_title(u'Doom Benchmark: \u0441\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u0435 \u0430\u0440\u0445\u0438\u0442\u0435\u043a\u0442\u0443\u0440 (50M \u0448\u0430\u0433\u043e\u0432)', fontsize=14)
ax.legend(fontsize=7, loc='lower right', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 50_000_000)
plt.tight_layout()
plt.savefig(os.path.join(train_dir, 'reward_comparison.png'), dpi=150, bbox_inches='tight')
print("  Saved reward_comparison.png")

# ─── Figure 2: pilot_comparison (one line per arch, best seed) ───
pilot_exps = [
    ('GRU baseline',    'gru_baseline_50m', '#1f77b4'),
    ('GRU + Mamba-2 HPs','gru_optimized_50m_seed3', '#2ca02c'),
    ('Mamba-2 best',    'mamba2_hpo_best_50m_seed3', '#ff7f0e'),
    ('Mamba-1',         'mamba1_50m_seed1', '#9467bd'),
    ('Transformer',     'transformer_50m_seed1', '#d62728'),
    ('Perceiver IO',    'perceiver_50m_seed1', '#8c564b'),
]
fig2, ax2 = plt.subplots(figsize=(14, 7))
for label, exp, color in pilot_exps:
    rewards, frames = load(exp)
    if not rewards: continue
    rs = smooth(rewards)
    fs = frames[len(frames)-len(rs):]
    ax2.plot(fs, rs, label=label, color=color, linewidth=2)
    best_i = rewards.index(max(rewards))
    ax2.scatter(frames[best_i], rewards[best_i], color=color, s=60, marker='*', edgecolors='black', linewidth=0.5, zorder=5)
    ax2.annotate(f'{max(rewards):.1f}', (frames[best_i], rewards[best_i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9, color=color)
fmt_x(ax2)
ax2.set_xlabel(u'\u0428\u0430\u0433\u0438 \u0441\u0440\u0435\u0434\u044b', fontsize=12)
ax2.set_ylabel(u'\u0421\u0440\u0435\u0434\u043d\u044f\u044f \u043d\u0430\u0433\u0440\u0430\u0434\u0430 \u0437\u0430 \u044d\u043f\u0438\u0437\u043e\u0434', fontsize=12)
ax2.set_title(u'Doom Benchmark: \u0441\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u0435 \u0430\u0440\u0445\u0438\u0442\u0435\u043a\u0442\u0443\u0440 (50M \u0448\u0430\u0433\u043e\u0432)', fontsize=14)
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 50_000_000)
plt.tight_layout()
plt.savefig(os.path.join(train_dir, 'pilot_comparison.png'), dpi=150, bbox_inches='tight')
print("  Saved pilot_comparison.png")

# ─── Figure 3: GRU 250M ───
rewards, frames = load('doom_battle_appo_gru_250m')
if rewards:
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    rs = smooth(rewards, 25)
    fs = frames[len(frames)-len(rs):]
    ax3.plot(fs, rs, color='#1f77b4', linewidth=1.5)
    ax3.axvline(x=150_000_000, color='red', linestyle='--', alpha=0.5, label=u'\u041f\u043b\u0430\u0442\u043e ~150M')
    best_i = rewards.index(max(rewards))
    ax3.scatter(frames[best_i], rewards[best_i], color='red', s=60, marker='*', zorder=5)
    ax3.set_xlabel(u'\u0428\u0430\u0433\u0438 \u0441\u0440\u0435\u0434\u044b', fontsize=12)
    ax3.set_ylabel(u'\u0421\u0440\u0435\u0434\u043d\u044f\u044f \u043d\u0430\u0433\u0440\u0430\u0434\u0430 \u0437\u0430 \u044d\u043f\u0438\u0437\u043e\u0434', fontsize=12)
    ax3.set_title(u'GRU 250M \u2014 \u0434\u043e\u043b\u0433\u0438\u0439 \u043f\u0440\u043e\u0433\u043e\u043d', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, 'gru_250m_curve.png'), dpi=150, bbox_inches='tight')
    print("  Saved gru_250m_curve.png")

# ─── Figure 4: Mamba-2 across 4 seeds ───
m2_seeds = [
    ('seed1', 'mamba2_hpo_best_50m_seed1', '#ff7f0e'),
    ('seed2', 'mamba2_hpo_best_50m_seed2', '#ffbb78'),
    ('seed3', 'mamba2_hpo_best_50m_seed3', '#d95f02'),
    ('seed4', 'mamba2_hpo_best_50m_seed4', '#ffeda0'),
]
fig4, ax4 = plt.subplots(figsize=(14, 7))
for label, exp, color in m2_seeds:
    rewards, frames = load(exp)
    if not rewards: continue
    rs = smooth(rewards)
    fs = frames[len(frames)-len(rs):]
    ax4.plot(fs, rs, label=label, color=color, linewidth=1.5)
    annotate_best(ax4, rewards, frames, color)
fmt_x(ax4)
ax4.set_xlabel(u'\u0428\u0430\u0433\u0438 \u0441\u0440\u0435\u0434\u044b', fontsize=12)
ax4.set_ylabel(u'\u0421\u0440\u0435\u0434\u043d\u044f\u044f \u043d\u0430\u0433\u0440\u0430\u0434\u0430 \u0437\u0430 \u044d\u043f\u0438\u0437\u043e\u0434', fontsize=12)
ax4.set_title(u'Mamba-2: \u0440\u0430\u0437\u0431\u0440\u043e\u0441 \u043c\u0435\u0436\u0434\u0443 \u0441\u0438\u0434\u0430\u043c\u0438 (4 \u0441\u0438\u0434\u0430)', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(train_dir, 'mamba2_seed_variance.png'), dpi=150, bbox_inches='tight')
print("  Saved mamba2_seed_variance.png")

# ─── Figure 5: GRU 3-seed comparison ───
gru_3s = [
    ('GRU baseline seed1', 'gru_baseline_50m', '#1f77b4'),
    ('GRU baseline seed2', 'gru_baseline_50m_seed2', '#aec7e8'),
    ('GRU baseline seed3', 'gru_baseline_50m_seed3', '#9ec9e0'),
    ('GRU + Mamba-2 HPs seed1', 'gru_optimized_50m_seed1', '#2ca02c'),
    ('GRU + Mamba-2 HPs seed2', 'gru_optimized_50m_seed2', '#98df8a'),
    ('GRU + Mamba-2 HPs seed3', 'gru_optimized_50m_seed3', '#3a7a3a'),
]
fig5, ax5 = plt.subplots(figsize=(14, 7))
for label, exp, color in gru_3s:
    rewards, frames = load(exp)
    if not rewards: continue
    rs = smooth(rewards)
    fs = frames[len(frames)-len(rs):]
    ax5.plot(fs, rs, label=label, color=color, linewidth=1.5)
    annotate_best(ax5, rewards, frames, color)
fmt_x(ax5)
ax5.set_xlabel(u'\u0428\u0430\u0433\u0438 \u0441\u0440\u0435\u0434\u044b', fontsize=12)
ax5.set_ylabel(u'\u0421\u0440\u0435\u0434\u043d\u044f\u044f \u043d\u0430\u0433\u0440\u0430\u0434\u0430 \u0437\u0430 \u044d\u043f\u0438\u0437\u043e\u0434', fontsize=12)
ax5.set_title(u'GRU: \u0431\u0430\u0437\u043e\u0432\u044b\u0439 \u043f\u0440\u043e\u0442\u0438\u0432 \u043e\u043f\u0442\u0438\u043c\u0438\u0437\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u043e\u0433\u043e (3 \u0441\u0438\u0434\u0430)', fontsize=14)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(train_dir, 'gru_3seed_comparison.png'), dpi=150, bbox_inches='tight')
print("  Saved gru_3seed_comparison.png")

# ─── Figure 6: Boxplot of final rewards across seeds ───
def get_final_rewards(exp_list, label_list):
    """Return list of lists: final rewards for each experiment group."""
    groups = []
    for exps, _ in zip(exp_list, label_list):
        grp = []
        for e in exps:
            r, _ = load(e)
            if r:
                grp.append(max(r[-20:]))
        if grp:
            groups.append(grp)
    return groups

box_groups = [
    [f'gru_baseline_50m_seed{i}' for i in range(1, 51)],
    [f'gru_optimized_50m_seed{i}' for i in range(1, 51)],
    [f'mamba2_hpo_best_50m_seed{i}' for i in range(1, 51)],
]
box_labels = ['GRU baseline', 'GRU + HPs', 'Mamba-2 HPO']
box_data = get_final_rewards(box_groups, box_labels)

if any(box_data):
    fig6, ax6 = plt.subplots(figsize=(10, 7))
    bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    # Overlay individual points
    for i, data in enumerate(box_data):
        jitter = np.random.normal(0, 0.04, size=len(data))
        ax6.scatter(np.ones(len(data)) * (i + 1) + jitter, data, alpha=0.3, s=15, color='black')
    ax6.set_ylabel(u'\u0424\u0438\u043d\u0430\u043b\u044c\u043d\u0430\u044f \u043d\u0430\u0433\u0440\u0430\u0434\u0430 (mean last 20)', fontsize=12)
    ax6.set_title(u'Doom Benchmark: \u0440\u0430\u0437\u0431\u0440\u043e\u0441 \u043f\u043e \u0441\u0438\u0434\u0430\u043c (50M \u0448\u0430\u0433\u043e\u0432)', fontsize=14)
    ax6.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, 'boxplot_across_seeds.png'), dpi=150, bbox_inches='tight')
    print("  Saved boxplot_across_seeds.png")
    
    # T-test
    from scipy import stats as st
    print()
    print("  T-test results (final reward, last20 mean):")
    for i in range(len(box_data)):
        for j in range(i+1, len(box_data)):
            if len(box_data[i]) >= 2 and len(box_data[j]) >= 2:
                t, p = st.ttest_ind(box_data[i], box_data[j])
                print(f"    {box_labels[i]} vs {box_labels[j]}: t={t:.3f}, p={p:.4f} {'(significant)' if p < 0.05 else '(not significant)'}")

# ─── Figure 7: Window ablation (reward vs window size) ───
win_sizes = [32, 64, 128]
win_archs = [
    ('GRU baseline', 'gru_baseline_50m', '#1f77b4', 'o-'),
    ('GRU + HPs',    'gru_optimized_50m', '#2ca02c', 's-'),
    ('Mamba-2 HPO',  'mamba2_hpo_best_50m', '#ff7f0e', 'D-'),
]
fig7, ax7 = plt.subplots(figsize=(10, 7))
for label, base, color, style in win_archs:
    means, stds = [], []
    for w in win_sizes:
        vals = []
        for s in range(1, 4):
            r, _ = load(f"{base}_w{w}_seed{s}")
            if r:
                vals.append(max(r[-20:]))
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    if means:
        ax7.errorbar(win_sizes, means, yerr=stds, label=label, color=color, fmt=style,
                     linewidth=2, capsize=5, capthick=2)
ax7.set_xlabel(u'\u0420\u0430\u0437\u043c\u0435\u0440 \u043e\u043a\u043d\u0430 (\u043a\u0430\u0434\u0440\u043e\u0432)', fontsize=12)
ax7.set_ylabel(u'\u0424\u0438\u043d\u0430\u043b\u044c\u043d\u0430\u044f \u043d\u0430\u0433\u0440\u0430\u0434\u0430 (mean last 20)', fontsize=12)
ax7.set_title(u'Doom Benchmark: \u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e\u0441\u0442\u044c \u043d\u0430\u0433\u0440\u0430\u0434\u044b \u043e\u0442 \u043e\u043a\u043d\u0430', fontsize=14)
ax7.set_xticks(win_sizes)
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(train_dir, 'window_ablation.png'), dpi=150, bbox_inches='tight')
print("  Saved window_ablation.png")

# ─── Figure 8: GRU 250M vs Mamba-2 250M (multi-seed) ───
fig8, ax8 = plt.subplots(figsize=(14, 7))
# GRU 250M (existing single seed)
r, f = load('doom_battle_appo_gru_250m')
if r:
    rs = smooth(r, 25)
    fs = f[len(f)-len(rs):]
    ax8.plot(fs, rs, label='GRU 250M (seed1)', color='#1f77b4', linewidth=1.5)
    annotate_best(ax8, r, f, '#1f77b4')
# Mamba-2 250M seeds
m2_250_colors = ['#ff7f0e', '#ffbb78', '#d95f02', '#ffeda0']
for s in range(1, 5):
    r, f = load(f'mamba2_250m_seed{s}')
    if r:
        rs = smooth(r, 25)
        fs = f[len(f)-len(rs):]
        ax8.plot(fs, rs, label=f'Mamba-2 250M (seed{s})', color=m2_250_colors[s-1], linewidth=1.2)
        annotate_best(ax8, r, f, m2_250_colors[s-1])
# GRU 250M additional seeds
gru_250_colors = ['#1f77b4', '#aec7e8', '#9ec9e0', '#6baed6']
for s in range(1, 5):
    r, f = load(f'gru_250m_seed{s}')
    if r:
        rs = smooth(r, 25)
        fs = f[len(f)-len(rs):]
        ax8.plot(fs, rs, label=f'GRU 250M (seed{s})', color=gru_250_colors[s-1], linewidth=1.2, linestyle='--')
        annotate_best(ax8, r, f, gru_250_colors[s-1])
ncol = (sum(1 for s in range(1,5) if load(f'mamba2_250m_seed{s}')[0]) + 
        sum(1 for s in range(1,5) if load(f'gru_250m_seed{s}')[0]) + 1)
ax8.set_xlabel(u'\u0428\u0430\u0433\u0438 \u0441\u0440\u0435\u0434\u044b', fontsize=12)
ax8.set_ylabel(u'\u0421\u0440\u0435\u0434\u043d\u044f\u044f \u043d\u0430\u0433\u0440\u0430\u0434\u0430 \u0437\u0430 \u044d\u043f\u0438\u0437\u043e\u0434', fontsize=12)
ax8.set_title(u'GRU vs Mamba-2: 250M \u0448\u0430\u0433\u043e\u0432', fontsize=14)
# Dynamic x-axis based on max frames
max_frames = 0
for s in range(1, 5):
    _, f = load(f'mamba2_250m_seed{s}')
    if f and f[-1] > max_frames: max_frames = f[-1]
    _, f = load(f'gru_250m_seed{s}')
    if f and f[-1] > max_frames: max_frames = f[-1]
if max_frames > 0:
    step = max_frames // 5
    step = max(step, 50_000_000)
    ax8.set_xticks(range(0, max_frames + step, step))
    ax8.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M'))
ax8.legend(fontsize=8, loc='lower right', ncol=2)
ax8.grid(True, alpha=0.3)
ax8.set_xlim(0, max_frames if max_frames > 0 else 250_000_000)
plt.tight_layout()
plt.savefig(os.path.join(train_dir, 'gru_vs_mamba2_250m.png'), dpi=150, bbox_inches='tight')
print("  Saved gru_vs_mamba2_250m.png")

# ─── Summary table ───
print()
print("=" * 80)
print("  SUMMARY TABLE: Doom Benchmark — Architecture Comparison")
print("=" * 80)
summary_groups = [
    ("GRU baseline 50M",    [f"gru_baseline_50m_seed{i}" for i in range(1, 51)]),
    ("GRU + HPs 50M",       [f"gru_optimized_50m_seed{i}" for i in range(1, 51)]),
    ("Mamba-2 HPO 50M",     [f"mamba2_hpo_best_50m_seed{i}" for i in range(1, 51)]),
    ("GRU 250M",            [f"gru_250m_seed{i}" for i in range(1, 11)] + ["doom_battle_appo_gru_250m"]),
    ("Mamba-2 250M",        [f"mamba2_250m_seed{i}" for i in range(1, 11)]),
]
print(f"{'Group':30s} {'n_seeds':>8s} {'Best(mean)':>12s} {'Best(std)':>10s} {'Final(mean)':>12s} {'Final(std)':>10s}")
print("-" * 82)
for label, exps in summary_groups:
    finals, bests = [], []
    for e in exps:
        r, _ = load(e)
        if r:
            finals.append(max(r[-20:]))
            bests.append(max(r))
    if finals:
        n = len(finals)
        print(f"{label:30s} {n:8d} {np.mean(bests):12.2f} {np.std(bests):10.2f} {np.mean(finals):12.2f} {np.std(finals):10.2f}")
print("=" * 82)
print()
PYEOF
}

# ─── Experiment Definitions ────────────────────────────────────────────────────

run_gru_baseline() {
    section "GRU Baseline (default Adam, lr=1e-4)"
    local seed=$1; shift
    local sfx=$(seed_suffix "$seed")
    local sa=$(seed_args "$seed")
    local exp_name="gru_baseline_50m${sfx}"
    # shellcheck disable=SC2086
    run_experiment "$exp_name" \
        --optimizer adam --learning_rate 1e-4 --exploration_loss_coeff 0.001 \
        $sa "$@"
}

run_gru_optimized() {
    section "GRU + Mamba-2 Optimized HPs (adamw, lr=4e-4)"
    local seed=$1; shift
    local sfx=$(seed_suffix "$seed")
    local sa=$(seed_args "$seed")
    local exp_name="gru_optimized_50m${sfx}"
    # shellcheck disable=SC2086
    run_experiment "$exp_name" \
        --optimizer adamw --learning_rate 4.05e-4 --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 $sa "$@"
}

run_mamba2() {
    section "Mamba-2 HPO Best Config"
    local seed=$1; shift
    local sfx=$(seed_suffix "$seed")
    local sa=$(seed_args "$seed")
    local exp_name="mamba2_hpo_best_50m${sfx}"
    # shellcheck disable=SC2086
    run_experiment "$exp_name" \
        --rnn_type mamba2 \
        --mamba_d_model 512 --mamba_d_state 128 --mamba_headdim 128 --mamba_expand 1 \
        --optimizer adamw --learning_rate 4.05e-4 --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 $sa "$@"
}

run_mamba1() {
    section "Mamba-1"
    local seed=$1; shift
    local sfx=$(seed_suffix "$seed")
    local sa=$(seed_args "$seed")
    local exp_name="mamba1_50m${sfx}"
    # shellcheck disable=SC2086
    run_experiment "$exp_name" \
        --rnn_type mamba1 \
        --mamba1_d_model 512 --mamba1_d_state 16 --mamba1_d_conv 4 --mamba1_expand 2 \
        --optimizer adamw --learning_rate 4.05e-4 $sa "$@" 
}

run_transformer() {
    section "Transformer"
    local exp_name="transformer_50m_seed1"
    run_experiment "$exp_name" \
        --rnn_type transformer \
        --transformer_d_model 512 --transformer_nhead 8 --transformer_window_size 64 \
        --optimizer adamw --learning_rate 4.05e-4
}

run_perceiver() {
    section "Perceiver IO"
    local exp_name="perceiver_50m_seed1"
    run_experiment "$exp_name" \
        --rnn_type perceiver \
        --perceiver_d_model 512 --perceiver_num_latents 32 \
        --perceiver_d_latents 512 --perceiver_num_blocks 2 \
        --optimizer adamw --learning_rate 4.05e-4
}

run_gru_250m() {
    section "GRU 250M (long run)"
    run_experiment "doom_battle_appo_gru_250m" \
        --optimizer adam --learning_rate 1e-4 --exploration_loss_coeff 0.001 \
        --train_for_env_steps 250000000
}

# ─── Mass Seed Runs ────────────────────────────────────────────────────────────

run_gru_50m_seed() {
    local seed=$1
    local sfx=$(seed_suffix "$seed")
    local sa=$(seed_args "$seed")
    local exp_name="gru_baseline_50m${sfx}"
    # shellcheck disable=SC2086
    run_experiment "$exp_name" \
        --optimizer adam --learning_rate 1e-4 --exploration_loss_coeff 0.001 \
        $sa
}

run_gru_50_seeds() {
    section "GRU Baseline: 50 seeds @ 50M"
    for seed in $(seq 1 50); do
        run_gru_50m_seed "$seed"
    done
}

run_mamba2_50m_seed() {
    local seed=$1
    local sfx=$(seed_suffix "$seed")
    local sa=$(seed_args "$seed")
    local exp_name="mamba2_hpo_best_50m${sfx}"
    # shellcheck disable=SC2086
    run_experiment "$exp_name" \
        --rnn_type mamba2 \
        --mamba_d_model 512 --mamba_d_state 128 --mamba_headdim 128 --mamba_expand 1 \
        --optimizer adamw --learning_rate 4.05e-4 --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 $sa
}

run_mamba2_50_seeds() {
    section "Mamba-2 HPO Best: 50 seeds @ 50M"
    for seed in $(seq 1 50); do
        run_mamba2_50m_seed "$seed"
    done
}

run_gru_250m_seed() {
    local seed=$1
    local sfx=$(seed_suffix "$seed")
    local sa=$(seed_args "$seed")
    # shellcheck disable=SC2086
    run_experiment "gru_250m${sfx}" \
        --optimizer adam --learning_rate 1e-4 --exploration_loss_coeff 0.001 \
        --train_for_env_steps 250000000 $sa
}

run_gru_250m_10_seeds() {
    section "GRU 250M: 10 seeds"
    for seed in $(seq 1 10); do
        run_gru_250m_seed "$seed"
    done
}

run_mamba2_250m_seed() {
    local seed=$1
    local sfx=$(seed_suffix "$seed")
    local sa=$(seed_args "$seed")
    # shellcheck disable=SC2086
    run_experiment "mamba2_250m${sfx}" \
        --rnn_type mamba2 \
        --mamba_d_model 512 --mamba_d_state 128 --mamba_headdim 128 --mamba_expand 1 \
        --optimizer adamw --learning_rate 4.05e-4 --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 \
        --train_for_env_steps 250000000 $sa
}

run_mamba2_250m_10_seeds() {
    section "Mamba-2 250M: 10 seeds"
    for seed in $(seq 1 10); do
        run_mamba2_250m_seed "$seed"
    done
}

# ─── Window Ablation (3 arch × 3 windows × 3 seeds = 27 exp) ──────────────

run_window_experiment() {
    local rnn_type=$1        # gru / gru_optimized / mamba2
    local window=$2          # 32 / 64 / 128
    local seed=$3; shift 3
    local sfx=$(seed_suffix "$seed")
    local sa=$(seed_args "$seed")
    
    # Map arch name to config
    local exp_name exp_extra
    case "$rnn_type" in
        gru)
            exp_name="gru_baseline_50m_w${window}${sfx}"
            exp_extra="--optimizer adam --learning_rate 1e-4 --exploration_loss_coeff 0.001"
            ;;
        gru_optimized)
            exp_name="gru_optimized_50m_w${window}${sfx}"
            exp_extra="--optimizer adamw --learning_rate 4.05e-4 --exploration_loss_coeff 0.00202 --weight_decay 0.00196"
            ;;
        mamba2)
            exp_name="mamba2_hpo_best_50m_w${window}${sfx}"
            exp_extra="--rnn_type mamba2 --mamba_d_model 512 --mamba_d_state 128 --mamba_headdim 128 --mamba_expand 1 --optimizer adamw --learning_rate 4.05e-4 --exploration_loss_coeff 0.00202 --weight_decay 0.00196"
            ;;
    esac
    # shellcheck disable=SC2086
    run_experiment "$exp_name" \
        --rollout "$window" --recurrence "$window" \
        $exp_extra $sa "$@"
}

run_window_ablation() {
    section "Window ablation: GRU base / GRU HPO / Mamba-2 @ 32/64/128, 3 seeds each"
    # Interleave by architecture (faster context switching)
    local windows="32 64 128"
    local seeds="1 2 3"
    
    for arch in gru gru_optimized mamba2; do
        section "  Window ablation: $arch"
        for w in $windows; do
            section "    Window=$w"
            for s in $seeds; do
                run_window_experiment "$arch" "$w" "$s"
            done
        done
    done
}

# ─── Multi-seed batch: GRU optimized (20 seeds) ───────────────────────────

run_gru_optimized_50m_seed() {
    local seed=$1
    local sfx=$(seed_suffix "$seed")
    local sa=$(seed_args "$seed")
    local exp_name="gru_optimized_50m${sfx}"
    # shellcheck disable=SC2086
    run_experiment "$exp_name" \
        --optimizer adamw --learning_rate 4.05e-4 --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 $sa
}

run_gru_optimized_50_seeds() {
    section "GRU Optimized: 50 seeds @ 50M"
    for seed in $(seq 1 50); do
        run_gru_optimized_50m_seed "$seed"
    done
}

run_gru_hpo() {
    section "GRU HPO (15 trials, ~7h)"
    log "Running GRU HPO..."
    uv run python3 -m _vizdoom.gru_hpo
}

# ─── Slurm Submission ──────────────────────────────────────────────────────────

is_on_cluster() {
    [[ -n "$SLURM_JOB_ID" ]] || command -v srun &>/dev/null
}

submit_slurm_job() {
    local job_name=$1
    local script=$2
    local node_constraint=${3:-""}
    local gres=${4:-"gpu:1"}
    
    local sbatch_script=$(mktemp /tmp/sbatch_XXXXXX.sh)
    cat > "$sbatch_script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_GPU_PARTITION}
#SBATCH --gres=${gres}
#SBATCH --cpus-per-task=${SLURM_CPUS_PER_TASK}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_GPU_TIME}
#SBATCH --output=${TRAIN_DIR}/slurm_%j.out
#SBATCH --error=${TRAIN_DIR}/slurm_%j.err
#SBATCH --chdir=
EOF

    if [ -n "$node_constraint" ]; then
        echo "#SBATCH --constraint=${node_constraint}" >> "$sbatch_script"
    fi
    
    cat >> "$sbatch_script" <<EOF

source ~/.bashrc
uv sync
bash ${script}
EOF
    
    sbatch "$sbatch_script"
    rm -f "$sbatch_script"
    log "Submitted: $job_name"
}

submit_all_experiments() {
    section "Submitting Slurm Jobs"
    
    # Job 1: GRU 50 seeds @ 50M
    submit_slurm_job "gru_50seeds_50m" "reproduce.sh --gru-50-seeds"
    
    # Job 2: Mamba-2 50 seeds @ 50M
    submit_slurm_job "mamba2_50seeds_50m" "reproduce.sh --mamba2-50-seeds"
    
    # Job 3: GRU HPO
    submit_slurm_job "gru_hpo" "reproduce.sh --hpo-gru"
    
    # Job 4: GRU 250M 10 seeds (needs more time)
    submit_slurm_job "gru_250m_10seeds" "reproduce.sh --gru-250m-10seeds"
    
    # Job 5: Mamba-2 250M 10 seeds (needs more time)
    submit_slurm_job "mamba2_250m_10seeds" "reproduce.sh --mamba2-250m-10seeds"
    
    log "All jobs submitted. Check: squeue -u $(whoami)"
}

run_hpo() {
    section "Mamba-2 HPO (85 trials, ~12h)"
    log "  Mamba-2 HPO already completed. Best config in use for run_mamba2()."
    log "  See also: --hpo-transformer, --hpo-perceiver, --ablation"
}

run_hpo_transformer() {
    section "Transformer HPO (15 trials, ~7h)"
    log "Running Transformer HPO..."
    uv run python3 -m _vizdoom.transformer_hpo
}

run_hpo_perceiver() {
    section "Perceiver IO HPO (10 trials, ~8h)"
    log "Running Perceiver IO HPO..."
    uv run python3 -m _vizdoom.perceiver_hpo
}

run_ablation() {
    section "Ablation experiments (~1h at --quick)"
    log "Running all ablation experiments..."
    uv run python3 -m _vizdoom.ablation --quick
}

# ─── Quick Smoke Test ─────────────────────────────────────────────────────────

run_quick() {
    section "QUICK MODE: 1M steps per architecture"
    local qs=$((TRAIN_STEPS / 50))  # 1M steps
    
    log "GRU baseline (1M)"
    uv run python3 -m models.train \
        --optimizer adam --learning_rate 1e-4 --exploration_loss_coeff 0.001 \
        --train_for_env_steps $qs --experiment quick_gru_baseline 2>&1 | tee -a "$LOG_FILE"
    
    log "GRU + Mamba-2 HPs (1M)"
    uv run python3 -m models.train \
        --optimizer adamw --learning_rate 4.05e-4 --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 \
        --train_for_env_steps $qs --experiment quick_gru_optimized 2>&1 | tee -a "$LOG_FILE"
    
    log "Mamba-2 (1M)"
    uv run python3 -m models.train --rnn_type mamba2 \
        --mamba_d_model 512 --mamba_d_state 128 --mamba_headdim 128 --mamba_expand 1 \
        --optimizer adamw --learning_rate 4.05e-4 --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 \
        --train_for_env_steps $qs --experiment quick_mamba2 2>&1 | tee -a "$LOG_FILE"
    
    log "Mamba-1 (1M)"
    uv run python3 -m models.train --rnn_type mamba1 \
        --mamba1_d_model 512 --mamba1_d_state 16 --mamba1_d_conv 4 --mamba1_expand 2 \
        --optimizer adamw --learning_rate 4.05e-4 \
        --train_for_env_steps $qs --experiment quick_mamba1 2>&1 | tee -a "$LOG_FILE"
    
    log "Transformer (1M)"
    uv run python3 -m models.train --rnn_type transformer \
        --transformer_d_model 512 --transformer_nhead 8 --transformer_window_size 64 \
        --optimizer adamw --learning_rate 4.05e-4 \
        --train_for_env_steps $qs --experiment quick_transformer 2>&1 | tee -a "$LOG_FILE"
    
    log "Perceiver IO (1M)"
    uv run python3 -m models.train --rnn_type perceiver \
        --perceiver_d_model 512 --perceiver_num_latents 32 \
        --perceiver_d_latents 512 --perceiver_num_blocks 2 \
        --optimizer adamw --learning_rate 4.05e-4 \
        --train_for_env_steps $qs --experiment quick_perceiver 2>&1 | tee -a "$LOG_FILE"
    
    section "Quick tests done — collecting results"
    collect_results
    generate_plots
    log "Quick mode complete. Results in $TRAIN_DIR/"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

list_experiments() {
    section "Available experiments in $TRAIN_DIR/"
    for d in "$TRAIN_DIR"/*/; do
        local name=$(basename "$d")
        local log="$d/sf_log.txt"
        if [ -f "$log" ]; then
            local frames=$(grep -oP 'Total num frames: \K\d+' "$log" | tail -1)
            local reward=$(grep -oP "Avg episode reward: \[\(0, '([0-9.]+)'\)\]" "$log" | tail -1 | tr -d "'")
            printf "  %-45s %10s frames  reward=%s\n" "$name" "${frames:-0}" "${reward:-?}"
        else
            printf "  %-45s (no log)\n" "$name"
        fi
    done
}

main() {
    local mode="${1:-full}"
    
    case "$mode" in
        --list|-l)
            list_experiments
            exit 0
            ;;
        --collect|-c)
            check_env
            collect_results
            generate_plots
            log "Collection complete."
            exit 0
            ;;
        --quick|-q)
            check_env
            run_quick
            exit 0
            ;;
        --hpo)
            check_env
            run_hpo
            exit 0
            ;;
        --hpo-transformer)
            check_env
            run_hpo_transformer
            exit 0
            ;;
        --hpo-perceiver)
            check_env
            run_hpo_perceiver
            exit 0
            ;;
        --ablation)
            check_env
            run_ablation
            exit 0
            ;;
        --help|-h)
            echo "Usage: bash reproduce.sh [OPTION]"
            echo ""
            echo "Options:"
            echo "  (no option)    Run ALL 50M experiments (13 runs, ~12h)"
            echo "  --quick, -q    Run 1M-step smoke tests (~30 min)"
            echo "  --collect, -c  Only parse existing logs and generate plots"
            echo "  --list, -l     List existing experiments and their status"
            echo "  --hpo          Show Mamba-2 HPO info (already completed)"
            echo "  --hpo-transformer  Run Transformer HPO (15 trials, ~7h)"
            echo "  --hpo-perceiver    Run Perceiver IO HPO (10 trials, ~8h)"
            echo "  --hpo-gru          Run GRU HPO (15 trials, ~7h)"
            echo "  --ablation         Run ablation experiments (~1h)"
            echo "  --gru-50-seeds        Run GRU baseline 50 seeds @ 50M"
            echo "  --gru-optimized-50-seeds Run GRU+HPO 50 seeds @ 50M"
            echo "  --mamba2-50-seeds     Run Mamba-2 50 seeds @ 50M"
            echo "  --gru-250m-10seeds    Run GRU 250M 10 seeds"
            echo "  --mamba2-250m-10seeds Run Mamba-2 250M 10 seeds"
            echo "  --window-ablation     Run window ablation (27 exp @ 50M, ~32h)"
            echo "  --slurm-submit        Submit all new experiments via Slurm"
            echo "  --help, -h            Show this help"
            exit 0
            ;;
    esac
    
    # New experiment modes
    case "$mode" in
        --hpo-gru)
            check_env
            run_gru_hpo
            exit 0
            ;;
        --gru-50-seeds)
            check_env
            run_gru_50_seeds
            exit 0
            ;;
        --gru-optimized-50-seeds)
            check_env
            run_gru_optimized_50_seeds
            exit 0
            ;;
        --mamba2-50-seeds)
            check_env
            run_mamba2_50_seeds
            exit 0
            ;;
        --window-ablation)
            check_env
            run_window_ablation
            collect_results
            generate_plots
            exit 0
            ;;
        --gru-250m-10seeds)
            check_env
            run_gru_250m_10_seeds
            exit 0
            ;;
        --mamba2-250m-10seeds)
            check_env
            run_mamba2_250m_10_seeds
            exit 0
            ;;
        --slurm-submit)
            submit_all_experiments
            exit 0
            ;;
        --gru-50m-seed[0-9]*)
            seed=${mode#--gru-50m-seed}
            check_env
            run_gru_50m_seed "$seed"
            exit 0
            ;;
        --mamba2-50m-seed[0-9]*)
            seed=${mode#--mamba2-50m-seed}
            check_env
            run_mamba2_50m_seed "$seed"
            exit 0
            ;;
        --gru-optimized-50m-seed[0-9]*)
            seed=${mode#--gru-optimized-50m-seed}
            check_env
            run_gru_optimized_50m_seed "$seed"
            exit 0
            ;;
        --gru-250m-seed[0-9]*)
            seed=${mode#--gru-250m-seed}
            check_env
            run_gru_250m_seed "$seed"
            exit 0
            ;;
        --mamba2-250m-seed[0-9]*)
            seed=${mode#--mamba2-250m-seed}
            check_env
            run_mamba2_250m_seed "$seed"
            exit 0
            ;;
        --window-[0-9]*)
            rest=${mode#--window-}
            arch=${rest%-seed*}
            w=${arch##*_}
            seed=${rest##*-seed}
            check_env
            run_window_experiment "$arch" "$w" "$seed"
            exit 0
            ;;
    esac
    
    check_env
    
    # ─── Full reproduction ───────────────────────────────────────────────────
    # Ordered by expected compute time (fastest first)
    
    run_gru_baseline 1     # ~38 min
    run_gru_baseline 2     # ~38 min
    run_gru_baseline 3     # ~38 min
    
    run_gru_optimized 1    # ~38 min
    run_gru_optimized 2    # ~38 min
    run_gru_optimized 3    # ~38 min
    
    run_mamba2 1           # ~38 min
    run_mamba2 2           # ~38 min
    run_mamba2 3           # ~38 min
    
    run_mamba1 1           # ~38 min
    run_mamba1 2           # ~38 min
    run_mamba1 3           # ~38 min
    
    run_transformer        # ~38 min
    
    run_perceiver          # ~50 min

    # Mamba-2 seed4 — extra seed
    run_mamba2 4           # ~38 min
    
    # GRU 250M — optional, ~2h extra
    # run_gru_250m
    
    # ─── Collect ──────────────────────────────────────────────────────────────
    collect_results
    generate_plots
    
    section "Full reproduction complete!"
    log "Results:     $TRAIN_DIR/results_summary.json"
    log "Plots:       $TRAIN_DIR/reward_comparison.png"
    log "            $TRAIN_DIR/pilot_comparison.png"
    log "            $TRAIN_DIR/gru_3seed_comparison.png"
    log "            $TRAIN_DIR/mamba2_seed_variance.png"
    log "            $TRAIN_DIR/gru_250m_curve.png"
    log "Log:         $LOG_FILE"
    log ""
    log "To view quick summary: bash reproduce.sh --list"
}

main "$@"
