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
    
    if [ -f "$exp_dir/sf_log.txt" ] && grep -q "Total num frames: 50000000\|Total num frames: 499" "$exp_dir/sf_log.txt" 2>/dev/null; then
        log "  SKIP $exp_name — already completed"
        return 0
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
    # GRU 250M
    experiments_raw += ["doom_battle_appo_gru_250m"]
    
    # 50-seed runs (GRU baseline and Mamba-2)
    for i in range(1, 51):
        experiments_raw.append(f"gru_baseline_50m_seed{i}")
        experiments_raw.append(f"mamba2_hpo_best_50m_seed{i}")
    
    # 250M 10-seed runs
    for i in range(1, 11):
        experiments_raw.append(f"gru_250m_seed{i}")
        experiments_raw.append(f"mamba2_250m_seed{i}")

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
    local exp_name="doom_battle_appo_gru_250m"
    if [ -f "$TRAIN_DIR/$exp_name/sf_log.txt" ] && grep -q "Total num frames: 249" "$TRAIN_DIR/$exp_name/sf_log.txt" 2>/dev/null; then
        log "  SKIP $exp_name — already completed"
        return 0
    fi
    log "  RUN $exp_name (250M steps ≈ 80 min)"
    uv run python3 -m models.train \
        --optimizer adam --learning_rate 1e-4 --exploration_loss_coeff 0.001 \
        --train_for_env_steps 250000000 \
        --experiment "$exp_name" 2>&1 | tee -a "$LOG_FILE"
    log "  DONE $exp_name"
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
    local exp_name="gru_250m${sfx}"
    if [ -f "$TRAIN_DIR/$exp_name/sf_log.txt" ] && grep -q "Total num frames: 249" "$TRAIN_DIR/$exp_name/sf_log.txt" 2>/dev/null; then
        log "  SKIP $exp_name — already completed"
        return 0
    fi
    # shellcheck disable=SC2086
    run_experiment "$exp_name" \
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
    local exp_name="mamba2_250m${sfx}"
    if [ -f "$TRAIN_DIR/$exp_name/sf_log.txt" ] && grep -q "Total num frames: 249" "$TRAIN_DIR/$exp_name/sf_log.txt" 2>/dev/null; then
        log "  SKIP $exp_name — already completed"
        return 0
    fi
    # shellcheck disable=SC2086
    run_experiment "$exp_name" \
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
            echo "  --gru-50-seeds     Run GRU baseline 50 seeds @ 50M"
            echo "  --mamba2-50-seeds  Run Mamba-2 50 seeds @ 50M"
            echo "  --gru-250m-10seeds Run GRU 250M 10 seeds"
            echo "  --mamba2-250m-10seeds Run Mamba-2 250M 10 seeds"
            echo "  --slurm-submit     Submit all new experiments via Slurm"
            echo "  --help, -h         Show this help"
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
        --mamba2-50-seeds)
            check_env
            run_mamba2_50_seeds
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
