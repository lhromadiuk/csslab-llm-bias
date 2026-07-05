# csslab-llm-bias

This repository stores code and results for the Computational Social Science Lab @ RWTH Aachen University.

## Repository Structure

- `data/`: local survey/persona data. Large/raw files are not meant to be published.
- `experiments/list_experiment/`: list-experiment trial construction and data files.
- `experiments/plain_prompts/`: ordinary prompt-file generation task.
- `analysis/`: analysis scripts and Jupyter notebooks.
-  `experiments/bias_calculation/`: direct-response bias calculation code.
- `main.py`: task entry point.
- `run_vllm.slurm`: main Slurm launcher.
- `vllm_runner.py`: shared vLLM loading/generation helpers.

## Setup

Create or update the Python environment:

```bash
./setup_env.sh
```

The Slurm script expects the virtual environment in the parent directory by
default:

```text
../.venv
```

You can override this with:

```bash
VENV_DIR=/path/to/.venv sbatch ...
```

## Running Jobs

Submit from the repository root:

```bash
sbatch --export=ALL,... run_vllm.slurm
```

The current `run_vllm.slurm` header requests:

```bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=19:45:00
```

For 2-GPU tensor parallel runs, override at submit time and set
`TENSOR_PARALLEL_SIZE=2`.

## List Experiment

The list experiment uses:

- 4 core statements from `experiments/list_experiment/data/core_statements.txt`
- sensitive statements from `experiments/list_experiment/data/sensitive_statements.txt`
- optional wording-alteration statements from
  `experiments/list_experiment/data/prompts_wording_alterations_binary.txt`

Control rows contain 4 core statements. Treatment rows contain the 4 core
statements plus 1 sensitive statement.

The list-experiment estimate is:

```text
pi_LE,i = mean(Y_treatment,i) - mean(Y_control)
```

Bias relative to ALLBUS is:

```text
Bias_LE,i = pi_LE,i - pi_ALLBUS,i
```

### Balanced No-Persona Runs

Run both control and treatment for the original 14 sensitive statements:

```bash
sbatch --export=ALL,RUN_ID=qwen_original_no_persona,MODE=list,USE_PERSONAS=0,LIST_TRIAL_KIND=both,CORE_CONTROL_REPLICATES=24,REPLICATES_PER_SENSITIVE=35,MODEL_ID=Qwen/Qwen3-8B,TEMPERATURE=0.2,MAX_TOKENS=5,DISABLE_THINKING=1,SEED=42 run_vllm.slurm
```

For Llama/Mistral, omit `DISABLE_THINKING=1`.

### Persona Runs

Persona CSVs can contain multiple persona text columns. The compact file used
for faster runs is:

```text
data/ZA9089_JSON_first3.csv
```

Run original list experiment with personas:

```bash
sbatch --export=ALL,RUN_ID=qwen_persona_original,MODE=list,USE_PERSONAS=1,PERSONAS_FILE=data/ZA9089_JSON_first3.csv,PERSONA_PROFILE_COLUMN=top-2,MODEL_ID=Qwen/Qwen3-8B,TEMPERATURE=0.2,MAX_TOKENS=5,DISABLE_THINKING=1,SEED=42 run_vllm.slurm
```

With personas, each respondent receives:

- 1 core-control list
- 1 treatment list for each sensitive statement

For 14 original statements this is 15 rows per persona. For 42 wording
alterations this is 43 rows per persona.

### Wording-Alteration Runs

Use the wording file as `PROMPTS_FILE`:

```bash
sbatch --export=ALL,RUN_ID=qwen_persona_wording,MODE=list,USE_PERSONAS=1,PERSONAS_FILE=data/ZA9089_JSON_first3.csv,PERSONA_PROFILE_COLUMN=top-2,PROMPTS_FILE=experiments/list_experiment/data/prompts_wording_alterations_binary.txt,MODEL_ID=Qwen/Qwen3-8B,TEMPERATURE=0.2,MAX_TOKENS=5,DISABLE_THINKING=1,SEED=42 run_vllm.slurm
```

The wording file is organized in groups of three per ALLBUS item:

- V1: everybody-does-it logic
- V2: pure justification
- V3: moral appeal

So `sens_1..sens_3` map to the first ALLBUS item, `sens_4..sens_6`
map to the second, etc.

## Model/Generation Options

Common environment variables for `run_vllm.slurm`:

- `MODEL_ID`: Hugging Face model id.
- `TEMPERATURE`: sampling temperature.
- `MAX_TOKENS`: max generated tokens.
- `BATCH_SIZE`: number of prompts sent to vLLM per batch.
- `TENSOR_PARALLEL_SIZE`: tensor parallel size.
- `GPU_MEMORY_UTILIZATION`: vLLM GPU memory target.
- `QUANTIZATION`: e.g. `bitsandbytes`.
- `LOAD_FORMAT`: e.g. `bitsandbytes`.
- `ENFORCE_EAGER=1`: pass `--enforce-eager` to vLLM.
- `MAX_NUM_SEQS`: pass `--max-num-seqs` to vLLM.
- `DISABLE_THINKING=1`: disables Qwen3 thinking via chat template kwargs.

Example quantized Llama-70B persona run on one GPU:

```bash
sbatch --export=ALL,RUN_ID=llama70_persona_original_quantized,MODE=list,USE_PERSONAS=1,PERSONAS_FILE=data/ZA9089_JSON_first3.csv,PERSONA_PROFILE_COLUMN=top-2,MODEL_ID=meta-llama/Llama-3.1-70B-Instruct,QUANTIZATION=bitsandbytes,LOAD_FORMAT=bitsandbytes,TENSOR_PARALLEL_SIZE=1,GPU_MEMORY_UTILIZATION=0.85,MAX_MODEL_LEN=1024,TEMPERATURE=0.2,MAX_TOKENS=5,BATCH_SIZE=32,MAX_NUM_SEQS=32,ENFORCE_EAGER=1,SEED=42 run_vllm.slurm
```

## Outputs

Generation files are written as JSONL:

```text
logs/list_experiment/<run_id>.generations.jsonl
```

Slurm logs are written to:

```text
logs/list_experiment/slurm-<job_id>.out
logs/list_experiment/slurm-<job_id>.err
```

Important JSONL fields:

- `trial_kind`: `core_control` or `sensitive_treatment`
- `sensitive_id`: `null` for controls, `sens_N` for treatments
- `item_count`: 4 for controls, 5 for treatments
- `order_labels`: compact order labels
- `persona_id`: present for persona runs
- `persona_profile_column`: e.g. `core` or `top-2`
- `answer`: raw model answer

For persona runs, full prompts are not repeated in the log to keep files
smaller.

## Analysis

Main notebooks:

- `analysis/list_experiment_analysis.ipynb`: original 14-item list experiment.
- `analysis/combined_wording_list_experiment_analysis.ipynb`: list experiment
  with wording alterations.
- `experiments/cwording_only_bias/Evaluation_results.ipynb`: directional and SPB +
  with wording alterations.
