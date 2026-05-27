#!/bin/bash
#SBATCH --job-name=list_experiment_core
#SBATCH --output=logs/list_experiment/slurm-%j.out
#SBATCH --error=logs/list_experiment/slurm-%j.err
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00

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

mkdir -p "$HF_HOME" logs/list_experiment

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-list}"
PROMPTS_FILE="${PROMPTS_FILE:-sensitive_statements.txt}"
CORE_STATEMENTS_FILE="${CORE_STATEMENTS_FILE:-core_statements.txt}"
CORE_SET_ID="${CORE_SET_ID:-core_set_default}"
LIST_TRIAL_KIND="${LIST_TRIAL_KIND:-sensitive_treatment}"
REPLICATES_PER_SENSITIVE="${REPLICATES_PER_SENSITIVE:-5}"
INCLUDE_CORE_CONTROL="${INCLUDE_CORE_CONTROL:-0}"
CORE_CONTROL_REPLICATES="${CORE_CONTROL_REPLICATES:-4}"
SHUFFLE_SENSITIVE="${SHUFFLE_SENSITIVE:-0}"
SHUFFLE_TRIALS="${SHUFFLE_TRIALS:-0}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
PROMPT_MODE="${PROMPT_MODE:-lines}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
ERRORS_FILE="${ERRORS_FILE:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_TOKENS="${MAX_TOKENS:-80}"
TEMPERATURE="${TEMPERATURE:-0.2}"

RUN_ARGS=()
if [ -n "${RUN_ID:-}" ]; then
    RUN_ARGS+=(--run-id "$RUN_ID")
fi
if [ -n "$OUTPUT_FILE" ]; then
    RUN_ARGS+=(--output-file "$OUTPUT_FILE")
fi
if [ -n "$ERRORS_FILE" ]; then
    RUN_ARGS+=(--errors-file "$ERRORS_FILE")
fi
if [ -n "$SYSTEM_PROMPT" ]; then
    RUN_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
fi

SHUFFLE_ARGS=()
if [ "${SHUFFLE_PROMPTS:-0}" = "1" ]; then
    SHUFFLE_ARGS+=(--shuffle)
fi
if [ -n "${SEED:-}" ]; then
    SHUFFLE_ARGS+=(--seed "$SEED")
fi

LIST_ARGS=()
if [ "$EXPERIMENT_MODE" = "list" ]; then
    LIST_ARGS+=(--core-statements-file "$CORE_STATEMENTS_FILE")
    LIST_ARGS+=(--core-set-id "$CORE_SET_ID")
    LIST_ARGS+=(--list-trial-kind "$LIST_TRIAL_KIND")
    LIST_ARGS+=(--replicates-per-sensitive "$REPLICATES_PER_SENSITIVE")
    LIST_ARGS+=(--core-control-replicates "$CORE_CONTROL_REPLICATES")
    if [ "$INCLUDE_CORE_CONTROL" = "1" ]; then
        LIST_ARGS+=(--include-core-control)
    fi
    if [ "$SHUFFLE_SENSITIVE" = "1" ]; then
        LIST_ARGS+=(--shuffle-sensitive)
    fi
    if [ "$SHUFFLE_TRIALS" = "1" ]; then
        LIST_ARGS+=(--shuffle-trials)
    fi
fi

echo "Running on host: $(hostname)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Submit dir: ${REPO_ROOT}"
echo "Venv: ${VENV_DIR}"
echo "Model: ${MODEL_ID}"
echo "Experiment mode: ${EXPERIMENT_MODE}"
echo "Prompts file: ${PROMPTS_FILE}"
echo "Core statements file: ${CORE_STATEMENTS_FILE}"
echo "Core set id: ${CORE_SET_ID}"
echo "List trial kind: ${LIST_TRIAL_KIND}"
echo "Replicates per sensitive: ${REPLICATES_PER_SENSITIVE}"
echo "Include core control: ${INCLUDE_CORE_CONTROL}"
echo "Core control replicates: ${CORE_CONTROL_REPLICATES}"
echo "Prompt mode: ${PROMPT_MODE}"
echo "System prompt: ${SYSTEM_PROMPT:+set}"
echo "Shuffle prompts: ${SHUFFLE_PROMPTS:-0}"
echo "Shuffle trials: ${SHUFFLE_TRIALS}"
echo "Run id: ${RUN_ID:-auto}"
echo "Output file: ${OUTPUT_FILE:-auto from run id}"
echo "Errors file: ${ERRORS_FILE:-auto from run id}"

srun python3 main.py \
    --model-id "$MODEL_ID" \
    --experiment-mode "$EXPERIMENT_MODE" \
    --prompts-file "$PROMPTS_FILE" \
    --prompt-mode "$PROMPT_MODE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    "${RUN_ARGS[@]}" \
    "${LIST_ARGS[@]}" \
    "${SHUFFLE_ARGS[@]}"
