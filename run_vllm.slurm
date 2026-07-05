#!/bin/bash
#SBATCH --job-name=vllm_main
#SBATCH --output=logs/list_experiment/slurm-%j.out
#SBATCH --error=logs/list_experiment/slurm-%j.err
#SBATCH --partition=anywhere
#SBATCH --nodelist=aventinus
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=19:45:00

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module load Python/3.11.5
else
    echo "Environment modules are unavailable; using the configured virtualenv Python."
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_PARENT="$(cd "${REPO_ROOT}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_PARENT}/.venv}"

if [ ! -f "${REPO_ROOT}/main.py" ]; then
    echo "Could not find main.py in ${REPO_ROOT}" >&2
    echo "Submit this job from the repository root:" >&2
    echo "  cd /path/to/csslab-llm-bias && sbatch run_vllm.slurm" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
cd "${REPO_ROOT}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "${REPO_ROOT}/hfkey" ]; then
    export HF_TOKEN="$(cat "${REPO_ROOT}/hfkey")"
fi

export HF_HOME="${HF_HOME:-/tmp/${USER}/huggingface}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

mkdir -p "$HF_HOME" logs/list_experiment

# Compact launcher interface
# MODE: list (default) or simple
# USE_PERSONAS: 0 (default) or 1
MODE="${MODE:-simple}"
USE_PERSONAS="${USE_PERSONAS:-0}"

if [ "$MODE" != "list" ] && [ "$MODE" != "simple" ]; then
    echo "Invalid MODE: $MODE (allowed: list, simple)" >&2
    exit 1
fi

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
RUN_ID="${RUN_ID:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

# Inputs shared across modes 
if [ "$MODE" = "simple" ]; then
    PROMPTS_FILE="${PROMPTS_FILE:-prompts.txt}"
else
    PROMPTS_FILE="${PROMPTS_FILE:-experiments/list_experiment/data/sensitive_statements.txt}"
fi
PROMPT_MODE="${PROMPT_MODE:-lines}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

# List-mode specific inputs
CORE_STATEMENTS_FILE="${CORE_STATEMENTS_FILE:-experiments/list_experiment/data/core_statements.txt}"
CORE_SET_ID="${CORE_SET_ID:-core_set_default}"
LIST_TRIAL_KIND="${LIST_TRIAL_KIND:-sensitive_treatment}"
PERSONAS_FILE="${PERSONAS_FILE:-experiments/list_experiment/data/personas.txt}"
PERSONA_PROFILE_COLUMN="${PERSONA_PROFILE_COLUMN:-core}"
PERSONA_SYSTEM_PROMPT="${PERSONA_SYSTEM_PROMPT:-}"
REPLICATES_PER_SENSITIVE="${REPLICATES_PER_SENSITIVE:-5}"
CORE_CONTROL_REPLICATES="${CORE_CONTROL_REPLICATES:-4}"
SHUFFLE_SENSITIVE="${SHUFFLE_SENSITIVE:-0}"
SHUFFLE_TRIALS="${SHUFFLE_TRIALS:-0}"

# Shared generation/model settings
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
QUANTIZATION="${QUANTIZATION:-}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
MAX_TOKENS="${MAX_TOKENS:-5}"
TEMPERATURE="${TEMPERATURE:-0.2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"

COMMON_ARGS=()
if [ -n "$RUN_ID" ]; then
    COMMON_ARGS+=(--run-id "$RUN_ID")
fi
if [ -n "$OUTPUT_FILE" ]; then
    COMMON_ARGS+=(--output-file "$OUTPUT_FILE")
fi
if [ -n "$QUANTIZATION" ]; then
    COMMON_ARGS+=(--quantization "$QUANTIZATION")
fi
if [ -n "$LOAD_FORMAT" ]; then
    COMMON_ARGS+=(--load-format "$LOAD_FORMAT")
fi
if [ "$ENFORCE_EAGER" = "1" ]; then
    COMMON_ARGS+=(--enforce-eager)
fi
if [ -n "$MAX_NUM_SEQS" ]; then
    COMMON_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [ -n "${SEED:-}" ]; then
    COMMON_ARGS+=(--seed "$SEED")
fi
if [ "$DISABLE_THINKING" = "1" ]; then
    COMMON_ARGS+=(--disable-thinking)
fi

TASK_ARGS=()
TASK=""

if [ "$MODE" = "simple" ]; then
    TASK="plain"
    TASK_ARGS+=(--prompts-file "$PROMPTS_FILE")
    TASK_ARGS+=(--prompt-mode "$PROMPT_MODE")
    if [ "${SHUFFLE_PROMPTS:-0}" = "1" ]; then
        TASK_ARGS+=(--shuffle)
    fi
    if [ "$USE_PERSONAS" = "1" ]; then
        TASK_ARGS+=(--use-personas)
        TASK_ARGS+=(--personas-file "$PERSONAS_FILE")
        TASK_ARGS+=(--persona-profile-column "$PERSONA_PROFILE_COLUMN")
        if [ -n "$PERSONA_SYSTEM_PROMPT" ]; then
            TASK_ARGS+=(--persona-system-prompt "$PERSONA_SYSTEM_PROMPT")
        fi
    elif [ -n "$SYSTEM_PROMPT" ]; then
        TASK_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
    fi
else
    TASK="list"
    TASK_ARGS+=(--prompts-file "$PROMPTS_FILE")
    TASK_ARGS+=(--core-statements-file "$CORE_STATEMENTS_FILE")
    TASK_ARGS+=(--core-set-id "$CORE_SET_ID")
    TASK_ARGS+=(--list-trial-kind "$LIST_TRIAL_KIND")
    TASK_ARGS+=(--replicates-per-sensitive "$REPLICATES_PER_SENSITIVE")
    TASK_ARGS+=(--core-control-replicates "$CORE_CONTROL_REPLICATES")

    if [ "$USE_PERSONAS" = "1" ]; then
        TASK_ARGS+=(--assignment-mode "persona_random")
        TASK_ARGS+=(--personas-file "$PERSONAS_FILE")
        TASK_ARGS+=(--persona-profile-column "$PERSONA_PROFILE_COLUMN")
        if [ -n "$PERSONA_SYSTEM_PROMPT" ]; then
            TASK_ARGS+=(--persona-system-prompt "$PERSONA_SYSTEM_PROMPT")
        fi
    else
        TASK_ARGS+=(--assignment-mode "balanced")
        if [ -n "$SYSTEM_PROMPT" ]; then
            TASK_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
        fi
    fi

    if [ "$SHUFFLE_SENSITIVE" = "1" ]; then
        TASK_ARGS+=(--shuffle-sensitive)
    fi
    if [ "$SHUFFLE_TRIALS" = "1" ]; then
        TASK_ARGS+=(--shuffle-trials)
    fi
fi

echo "Running on host: $(hostname)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Submit dir: ${REPO_ROOT}"
echo "Venv: ${VENV_DIR}"
echo "Mode: ${MODE}"
echo "Task: ${TASK}"
echo "Use personas: ${USE_PERSONAS}"
echo "Model: ${MODEL_ID}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Enforce eager: ${ENFORCE_EAGER}"
echo "Max num seqs: ${MAX_NUM_SEQS:-auto}"
echo "Use FlashInfer sampler: ${VLLM_USE_FLASHINFER_SAMPLER}"
echo "Quantization: ${QUANTIZATION:-none}"
echo "Load format: ${LOAD_FORMAT:-auto}"
echo "Prompts file: ${PROMPTS_FILE}"
if [ "$TASK" = "list" ]; then
    echo "Core statements file: ${CORE_STATEMENTS_FILE}"
    echo "List trial kind: ${LIST_TRIAL_KIND}"
    echo "Replicates per sensitive: ${REPLICATES_PER_SENSITIVE}"
    echo "Core control replicates: ${CORE_CONTROL_REPLICATES}"
fi
echo "Run id: ${RUN_ID:-auto}"
echo "Output file: ${OUTPUT_FILE:-auto from run id}"
echo "Batch size: ${BATCH_SIZE}"
echo "Disable thinking: ${DISABLE_THINKING}"

srun python3 main.py "$TASK" \
    --model-id "$MODEL_ID" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --batch-size "$BATCH_SIZE" \
    "${COMMON_ARGS[@]}" \
    "${TASK_ARGS[@]}"
