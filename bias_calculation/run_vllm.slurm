#!/bin/bash
#SBATCH --job-name=vllm_main
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=32G
#SBATCH --time=05:45:00

set -euo pipefail

module load Python/3.11.5

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${HOME}/.venv}"

if [ ! -f "${REPO_ROOT}/main.py" ]; then
    echo "Could not find main.py in ${REPO_ROOT}" >&2
    echo "Submit this job from the repository root:" >&2
    echo "  cd /path/to/csslab-llm-bias && sbatch run_vllm.slurm" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
cd "${REPO_ROOT}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/hfkey" ]; then
    export HF_TOKEN="$(cat "${HOME}/hfkey")"
fi

export HF_HOME="${HF_HOME:-/tmp/${USER}/huggingface}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$HF_HOME" logs

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
QUANTIZATION="${QUANTIZATION:-}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
MAX_TOKENS="${MAX_TOKENS:-5}"
TEMPERATURE="${TEMPERATURE:-0.0}"
N_REPEATS="${N_REPEATS:-30}"

export MODEL_ID
export TENSOR_PARALLEL_SIZE
export MAX_MODEL_LEN
export GPU_MEMORY_UTILIZATION
export QUANTIZATION
export LOAD_FORMAT
export MAX_TOKENS
export TEMPERATURE
export N_REPEATS

echo "Running on host: $(hostname)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Submit dir: ${REPO_ROOT}"
echo "Venv: ${VENV_DIR}"
echo "Model: ${MODEL_ID}"
echo "Quantization: ${QUANTIZATION:-none}"
echo "Load format: ${LOAD_FORMAT:-auto}"
echo "N repeats: ${N_REPEATS}"

srun python3 bias_calculation/bias_calculation.py
