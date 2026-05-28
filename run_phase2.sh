#!/usr/bin/env bash
# run_phase2.sh — Chain all Phase 2 experiments sequentially on single GPU
# Usage:
#   bash run_phase2.sh                    # Full chain (~5 days)
#   bash run_phase2.sh --250m-only        # Only Mamba-2 250M (4 seeds, ~20h)
#   bash run_phase2.sh --window-only      # Only window ablation (27 exp, ~32h)
#   bash run_phase2.sh --multiseed-only   # Only multi-seed 20+ (60 exp, ~72h)
#   bash run_phase2.sh --collect          # Collect + plots from existing data
#   bash run_phase2.sh --status           # Show progress of all experiments
#
# Designed to be started in tmux and left running for days.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG="phase2_chain.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
section() { log ""; log "═══════════════════════════════════════"; log "  $*"; log "═══════════════════════════════════════"; }

# ─── Block 1: Mamba-2 250M (4 seeds) ───
run_mamba2_250m_chain() {
    section "Block 1: Mamba-2 250M — 4 seeds"
    for s in 1 2 3 4; do
        log "Mamba-2 250M seed${s}"
        bash reproduce.sh --mamba2-250m-seed${s}
        # Verify completion
        if grep -q "Total num frames: 24[0-9]" train_dir/mamba2_250m_seed${s}/sf_log.txt 2>/dev/null; then
            log "  ✓ seed${s} completed"
        else
            log "  ✗ seed${s} may have failed — check train_dir/mamba2_250m_seed${s}/"
        fi
    done
    section "Block 1 complete"
}

# ─── Block 2: Window ablation (27 exp) ───
run_window_chain() {
    section "Block 2: Window ablation — 27 exp"
    bash reproduce.sh --window-ablation
    section "Block 2 complete"
}

# ─── Block 3: Multi-seed runs (60 exp) ───
run_multiseed_chain() {
    section "Block 3a: GRU baseline seeds 5-24"
    for s in $(seq 5 24); do
        log "GRU baseline seed${s} (${s}/24)"
        bash reproduce.sh --gru-50m-seed${s}
    done

    section "Block 3b: GRU+HPO seeds 4-23"
    for s in $(seq 4 23); do
        log "GRU+HPO seed${s} (${s}/23)"
        bash reproduce.sh --gru-optimized-50m-seed${s}
    done

    section "Block 3c: Mamba-2 seeds 5-24"
    for s in $(seq 5 24); do
        log "Mamba-2 seed${s} (${s}/24)"
        bash reproduce.sh --mamba2-50m-seed${s}
    done

    section "Block 3 complete"
}

# ─── Collect + Plots ───
run_collect() {
    section "Collecting results + generating plots"
    bash reproduce.sh --collect
    log "Final results in train_dir/results_summary.json"
    log "Plots in train_dir/*.png"
}

# ─── Status ───
run_status() {
    section "Phase 2 experiment status"
    bash reproduce.sh --list
    
    echo ""
    echo "=== Window ablation ==="
    for arch in gru_baseline gru_optimized mamba2_hpo_best; do
        for w in 32 64 128; do
            for s in 1 2 3; do
                d="train_dir/${arch}_50m_w${w}_seed${s}"
                if [ -f "$d/sf_log.txt" ]; then
                    frames=$(grep -oP 'Total num frames: \K\d+' "$d/sf_log.txt" | tail -1)
                    reward=$(grep -oP "Avg episode reward: \[\(0, '([0-9.]+)'\)\]" "$d/sf_log.txt" | tail -1 | tr -d "'")
                    printf "  %-45s %10s frames  reward=%s\n" "${arch}_w${w}_s${s}" "${frames:-0}" "${reward:-?}"
                else
                    printf "  %-45s (not started)\n" "${arch}_w${w}_s${s}"
                fi
            done
        done
    done
    
    echo ""
    echo "=== Multi-seed (last completed) ==="
    for prefix in gru_baseline_50m_seed gru_optimized_50m_seed mamba2_hpo_best_50m_seed; do
        last=""
        for s in $(seq 50 -1 1); do
            if [ -f "train_dir/${prefix}${s}/sf_log.txt" ]; then
                last=$s
                break
            fi
        done
        echo "  $prefix: last completed seed = ${last:-none}"
    done
}

# ─── Main ───
main() {
    local mode="${1:-full}"
    
    case "$mode" in
        --250m-only)
            run_mamba2_250m_chain
            run_collect
            ;;
        --window-only)
            run_window_chain
            run_collect
            ;;
        --multiseed-only)
            run_multiseed_chain
            run_collect
            ;;
        --collect)
            run_collect
            ;;
        --status)
            run_status
            ;;
        --help|-h)
            echo "Usage: bash run_phase2.sh [OPTION]"
            echo ""
            echo "  (no option)    Full chain (all 3 blocks, ~5 days)"
            echo "  --250m-only    Only Mamba-2 250M (4 seeds)"
            echo "  --window-only  Only window ablation (27 exp)"
            echo "  --multiseed-only Only multi-seed (60 exp)"
            echo "  --collect      Collect results + plots"
            echo "  --status       Show progress"
            echo "  --help         Show this help"
            exit 0
            ;;
        *)
            # Default: full chain
            run_mamba2_250m_chain
            run_window_chain
            run_multiseed_chain
            run_collect
            section "Phase 2 complete!"
            log "All experiments done. Results in train_dir/"
            ;;
    esac
}

main "$@"
