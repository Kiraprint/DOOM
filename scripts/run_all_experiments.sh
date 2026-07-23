#!/usr/bin/env bash
# Launch all pending experiments for architecture comparison.
# Each experiment takes ~40-60 min on RTX 5090.
# Run from repo root: bash scripts/run_all_experiments.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Already completed:
#   Cell 1: gru_baseline_50m (existing)       — best 12.43
#   Cell 1: gru_baseline_50m_seed2             — best 14.37
#   Cell 2: gru_optimized_50m_seed1            — best 17.44
#   Cell 3: mamba2_hpo_best_50m_seed1          — running (~14.87)

SEEDS="${1:-1 2 3}"

# --- Cell 2: GRU + optimized HPs (extra seeds) ---
for SEED in $SEEDS; do
    if [ "$SEED" = "1" ]; then
        echo "[SKIP] gru_optimized_50m_seed1 already done"
        continue
    fi
    echo "[LAUNCH] gru_optimized_50m_seed${SEED}"
    uv run python -m models.train \
        --rnn_type gru \
        --optimizer adamw \
        --learning_rate 4.05e-4 \
        --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 \
        --experiment "gru_optimized_50m_seed${SEED}" &
done

# --- Cell 3: Mamba-2 best config (extra seeds) ---
for SEED in $SEEDS; do
    if [ "$SEED" = "1" ]; then
        echo "[SKIP] mamba2_hpo_best_50m_seed1 already running"
        continue
    fi
    echo "[LAUNCH] mamba2_hpo_best_50m_seed${SEED}"
    uv run python -m models.train \
        --rnn_type mamba2 \
        --mamba_d_model 512 \
        --mamba_d_state 128 \
        --mamba_headdim 128 \
        --mamba_expand 1 \
        --optimizer adamw \
        --learning_rate 4.05e-4 \
        --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 \
        --experiment "mamba2_hpo_best_50m_seed${SEED}" &
done

# --- Cell 4: Mamba-1 ---
for SEED in $SEEDS; do
    echo "[LAUNCH] mamba1_50m_seed${SEED}"
    uv run python -m models.train \
        --rnn_type mamba1 \
        --mamba1_d_model 512 \
        --mamba1_d_state 16 \
        --mamba1_d_conv 4 \
        --mamba1_expand 2 \
        --optimizer adamw \
        --learning_rate 4.05e-4 \
        --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 \
        --experiment "mamba1_50m_seed${SEED}" &
done

# --- Cell 5: Transformer ---
for SEED in $SEEDS; do
    echo "[LAUNCH] transformer_50m_seed${SEED}"
    uv run python -m models.train \
        --rnn_type transformer \
        --transformer_d_model 512 \
        --transformer_nhead 8 \
        --transformer_window_size 64 \
        --transformer_dim_feedforward 2048 \
        --transformer_dropout 0.1 \
        --optimizer adamw \
        --learning_rate 4.05e-4 \
        --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 \
        --experiment "transformer_50m_seed${SEED}" &
done

# --- Cell 6: Perceiver IO ---
for SEED in $SEEDS; do
    echo "[LAUNCH] perceiver_50m_seed${SEED}"
    uv run python -m models.train \
        --rnn_type perceiver \
        --perceiver_d_model 512 \
        --perceiver_num_latents 32 \
        --perceiver_d_latents 512 \
        --perceiver_num_blocks 2 \
        --perceiver_num_heads 8 \
        --perceiver_dropout 0.1 \
        --optimizer adamw \
        --learning_rate 4.05e-4 \
        --exploration_loss_coeff 0.00202 \
        --weight_decay 0.00196 \
        --experiment "perceiver_50m_seed${SEED}" &
done

echo ""
echo "All experiments launched. Use 'ps aux | grep models.train' to monitor."
echo "Run 'marimo edit notebooks/experiments/results_comparison.py' to track progress."
