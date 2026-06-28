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
if [ -z "${VENV_DIR:-}" ]; then
    VENV_DIR="${REPO_ROOT}/../../.venv"
fi
if [ ! -d "${VENV_DIR}" ]; then
    VENV_DIR="${HOME}/css-lab-project/.venv"
fi

if [ ! -f "${REPO_ROOT}/bias_calc_wording.py" ]; then
    echo "Could not find bias_calc_wording.py in ${REPO_ROOT}" >&2
    echo "Submit this job from the bias_calc_wording folder:" >&2
    echo "  cd /path/to/csslab-llm-bias/bias_calc_wording.py && sbatch run_vllm_wording.slurm" >&2
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

MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
PROMPTS_FILE="${PROMPTS_FILE:-prompts_wording_variations.txt}"
PROMPT_MODE="${PROMPT_MODE:-lines}"
OUTPUT_FILE="${OUTPUT_FILE:-results/bias_results.csv}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_TOKENS="${MAX_TOKENS:-80}"
TEMPERATURE="${TEMPERATURE:-0.0}"
QUANTIZATION="${QUANTIZATION:-}"
LOAD_FORMAT="${LOAD_FORMAT:-}"

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

srun python3 bias_calc_wording.py \
    --model-id "$MODEL_ID" \
    --prompts-file "$PROMPTS_FILE" \
    --prompt-mode "$PROMPT_MODE" \
    --output-file "$OUTPUT_FILE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --quantization "$QUANTIZATION" \
    --load-format "$LOAD_FORMAT" \
    "${SHUFFLE_ARGS[@]}"
