#!/bin/bash
#
# GPU Management Script for HPO Training
# Manages llama-server process to free GPU memory during training
#
# Usage: ./scripts/manage_gpu.sh {stop-llm|start-llm|check-gpu}
#

set -euo pipefail

# Configuration
LLAMA_SERVER_PORT=8080
LLAMA_SERVER_URL="http://localhost:${LLAMA_SERVER_PORT}"
GPU_DEVICE=0  # RTX 5090 device ID
VRAM_TOTAL_GB=24  # Total VRAM in GB
MIN_FREE_GB=8     # Minimum free VRAM required for training

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Find llama-server process IDs
find_llama_server_pids() {
    pgrep -f "llama-server" 2>/dev/null || true
}

# Check if llama-server is responding
is_server_responding() {
    curl -s --max-time 2 "${LLAMA_SERVER_URL}/health" > /dev/null 2>&1
    return $?
}

# Stop llama-server
stop_llm() {
    log_info "Stopping llama-server..."
    
    local pids
    pids=$(find_llama_server_pids)
    
    if [[ -z "$pids" ]]; then
        log_warn "No llama-server process found"
        return 0
    fi
    
    log_info "Found llama-server PIDs: $pids"
    
    # Send SIGTERM first for graceful shutdown
    kill $pids 2>/dev/null || true
    
    # Wait for process to terminate (max 10 seconds)
    local waited=0
    while [[ $waited -lt 10 ]]; do
        sleep 1
        local remaining
        remaining=$(find_llama_server_pids)
        
        if [[ -z "$remaining" ]]; then
            log_info "llama-server stopped gracefully"
            return 0
        fi
        
        waited=$((waited + 1))
    done
    
    # Force kill if still running
    log_warn "Graceful shutdown timed out, forcing kill..."
    pids=$(find_llama_server_pids)
    if [[ -n "$pids" ]]; then
        kill -9 $pids 2>/dev/null || true
        sleep 1
    fi
    
    # Final verification
    local final_pids
    final_pids=$(find_llama_server_pids)
    
    if [[ -n "$final_pids" ]]; then
        log_error "Failed to stop llama-server (PIDs: $final_pids)"
        return 1
    fi
    
    log_info "llama-server stopped successfully"
    return 0
}

# Start llama-server
start_llm() {
    log_info "Starting llama-server..."
    
    # Check if already running
    local pids
    pids=$(find_llama_server_pids)
    if [[ -n "$pids" ]]; then
        log_warn "llama-server is already running (PID: $pids)"
        return 0
    fi
    
    # Check if model path exists (common locations)
    local model_path=""
    if [[ -f "$HOME/models/Qwen3.6-35B-TQ3_4S.gguf" ]]; then
        model_path="$HOME/models/Qwen3.6-35B-TQ3_4S.gguf"
    elif [[ -f "/opt/llama.cpp/models/Qwen3.6-35B-TQ3_4S.gguf" ]]; then
        model_path="/opt/llama.cpp/models/Qwen3.6-35B-TQ3_4S.gguf"
    fi
    
    if [[ -z "$model_path" ]]; then
        log_error "Model file not found at expected locations"
        log_error "Please set MODEL_PATH environment variable or place model at:"
        log_error "  - $HOME/models/Qwen3.6-35B-TQ3_4S.gguf"
        log_error "  - /opt/llama.cpp/models/Qwen3.6-35B-TQ3_4S.gguf"
        return 1
    fi
    
    # Start llama-server
    log_info "Starting llama-server with model: $model_path"
    
    # Start in background with nohup
    nohup llama-server \
        -m "$model_path" \
        --host 0.0.0.0 \
        --port "$LLAMA_SERVER_PORT" \
        --ctx-size 256000 \
        --gpu-layers 35 \
        > /tmp/llama-server.log 2>&1 &
    
    local server_pid=$!
    log_info "llama-server started with PID: $server_pid"
    
    # Wait for server to be ready
    log_info "Waiting for server to be ready..."
    local waited=0
    while [[ $waited -lt 30 ]]; do
        if is_server_responding; then
            log_info "llama-server is ready at ${LLAMA_SERVER_URL}"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    
    log_error "llama-server failed to start within timeout"
    log_error "Check logs at /tmp/llama-server.log"
    return 1
}

# Check GPU memory availability
check_gpu() {
    log_info "Checking GPU memory availability..."
    
    # Try nvidia-smi
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Is NVIDIA driver installed?"
        return 1
    fi
    
    # Get GPU info
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits -i $GPU_DEVICE 2>/dev/null)
    
    if [[ $? -ne 0 ]]; then
        log_error "Failed to query GPU information"
        return 1
    fi
    
    IFS=',' read -r gpu_name mem_total mem_used mem_free <<< "$gpu_info"
    
    # Print GPU status
    echo ""
    log_info "GPU Status:"
    echo "  Name:     $gpu_name"
    echo "  Total:    ${mem_total} MB"
    echo "  Used:     ${mem_used} MB"
    echo "  Free:     ${mem_free} MB"
    
    # Calculate utilization percentage
    local utilization=0
    if [[ $mem_total -gt 0 ]]; then
        utilization=$((mem_used * 100 / mem_total))
    fi
    echo "  Usage:    ${utilization}%"
    
    # Check if llama-server is running
    local pids
    pids=$(find_llama_server_pids)
    if [[ -n "$pids" ]]; then
        echo "  Warning:  llama-server is RUNNING (PID: $pids)"
        log_warn "Stop llama-server before HPO training"
    fi
    
    echo ""
    
    # Check if enough memory is available
    local free_gb=$((mem_free / 1024))
    
    if [[ $free_gb -ge $MIN_FREE_GB ]]; then
        log_info "Sufficient GPU memory available (${free_gb}GB >= ${MIN_FREE_GB}GB required)"
        return 0
    else
        log_error "Insufficient GPU memory (${free_gb}GB < ${MIN_FREE_GB}GB required)"
        log_warn "Stop llama-server to free up memory"
        return 1
    fi
}

# Print usage
usage() {
    echo "GPU Management Script for HPO Training"
    echo ""
    echo "Usage: $0 {stop-llm|start-llm|check-gpu}"
    echo ""
    echo "Commands:"
    echo "  stop-llm    Stop llama-server to free GPU memory"
    echo "  start-llm   Start llama-server after HPO completes"
    echo "  check-gpu   Check GPU memory availability"
    echo ""
    echo "Examples:"
    echo "  # Before HPO training:"
    echo "  $0 stop-llm && $0 check-gpu"
    echo ""
    echo "  # After HPO training:"
    echo "  $0 start-llm"
}

# Main
if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

case "$1" in
    stop-llm)
        stop_llm
        ;;
    start-llm)
        start_llm
        ;;
    check-gpu)
        check_gpu
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
