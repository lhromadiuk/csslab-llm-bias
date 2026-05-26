#!/bin/bash
#SBATCH --job-name=llama_vllm_test
#SBATCH --output=../logs/mwe/slurm-%j.out
#SBATCH --error=../logs/mwe/slurm-%j.out
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00

set -euo pipefail

module load Python/3.11.5

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${HOME}/.venv}"

source "${VENV_DIR}/bin/activate"
cd "${REPO_ROOT}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/hfkey" ]; then
    export HF_TOKEN="$(cat "${HOME}/hfkey")"
fi

export HF_HOME="${HF_HOME:-/tmp/${USER}/huggingface}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$HF_HOME" logs/mwe

# Override model at submission time if needed:
#   cd mwe && sbatch run_llama_vllm.slurm
#   cd mwe && sbatch --export=ALL,MODEL_ID=meta-llama/Llama-3.1-8B-Instruct run_llama_vllm.slurm
export MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
export MAX_TOKENS="${MAX_TOKENS:-16}"

echo "Running on host: $(hostname)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Submit dir: ${SUBMIT_DIR}"
echo "Venv: ${VENV_DIR}"
echo "Model: ${MODEL_ID}"
echo "Purpose: vLLM setup smoke test"

srun python3 -m mwe.main_vllm
