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
#SBATCH --time=00:15:00

set -euo pipefail

module load Python/3.11.5

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${HOME}/css-lab-project/.venv"

if [ ! -f "${REPO_ROOT}/main.py" ]; then
    echo "Could not find main.py in ${REPO_ROOT}" >&2
    echo "Submit this job from the repository root:" >&2
    echo "  cd /path/to/csslab-llm-bias && sbatch run_vllm.slurm" >&2
    exit 1
fi

# Lösche das alte VENV_DIR oder das alte "source"-Kommando und schreibe stattdessen:
source /rwthfs/rz/cluster/home/xgy78490/css-lab-project/.venv/bin/activate
cd "${REPO_ROOT}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/hfkey" ]; then
    export HF_TOKEN="$(cat "${HOME}/hfkey")"
fi

export HF_HOME="${HF_HOME:-/tmp/${USER}/huggingface}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$HF_HOME" logs

MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
PROMPTS_FILE="${PROMPTS_FILE:-prompts.txt}"
PROMPT_MODE="${PROMPT_MODE:-lines}"
OUTPUT_FILE="${OUTPUT_FILE:-logs/generations.jsonl}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_TOKENS="${MAX_TOKENS:-80}"
TEMPERATURE="${TEMPERATURE:-0.0}"

SHUFFLE_ARGS=()
if [ "${SHUFFLE_PROMPTS:-0}" = "1" ]; then
    SHUFFLE_ARGS+=(--shuffle)
fi
if [ -n "${SEED:-}" ]; then
    SHUFFLE_ARGS+=(--seed "$SEED")
fi

echo "Running on host: $(hostname)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Submit dir: ${REPO_ROOT}"
echo "Venv: ${VENV_DIR}"
echo "Model: ${MODEL_ID}"
echo "Prompts file: ${PROMPTS_FILE}"
echo "Prompt mode: ${PROMPT_MODE}"
echo "Shuffle prompts: ${SHUFFLE_PROMPTS:-0}"

srun python3 main.py \
    --model-id "$MODEL_ID" \
    --prompts-file "$PROMPTS_FILE" \
    --prompt-mode "$PROMPT_MODE" \
    --output-file "$OUTPUT_FILE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    "${SHUFFLE_ARGS[@]}"
