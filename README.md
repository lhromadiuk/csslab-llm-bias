# csslab-llm-bias

This repository stores code and results for the Computational Social Science Lab @ RWTH Aachen University.

## Repository structure

- `data/`: Raw data files are not published openly in this repository, please check the corresponding README file.
- `experiments/`: project tasks with task-specific code and inputs.
- `experiments/list_experiment/`: list experiment task, including trial construction and statement files.
- `experiments/plain_prompts/`: plain prompt-file generation task.
- `logs/`: logs generated during experiments.
- `mwe/`: minimal working examples for model execution.
- `main.py`: project entrypoint for selecting and running tasks.
- `prompt_loader.py`: compatibility imports for older scripts.
- `prompts.txt`: example prompt file.
- `requirements.txt`: Python dependencies needed to run the project code.
- `setup_env.sh`: setup script that creates or updates a Python virtual environment in the parent folder.
- `run_vllm.slurm`: Slurm job script for running the list experiment task.
- `vllm_runner.py`: reusable vLLM helpers used by `main.py` and the MWE.
- `mwe/main_vllm.py`: minimal smoke test for checking that vLLM loads and generates.
- `mwe/test_vllm.slurm`: Slurm job script for the vLLM smoke test.

## Running vLLM

Create or update the environment whenever `requirements.txt` changes:

```bash
./setup_env.sh
```

Submit the default test job from mwe folder:

```bash
cd mwe
sbatch test_vllm.slurm
```

The reusable vLLM functions can be imported from project code:

```python
from vllm_runner import load_model, run_prompt

llm = load_model(model_id="meta-llama/Llama-3.2-1B-Instruct")
answer = run_prompt(llm, "Explain vLLM briefly.")
```

## Project Tasks

`main.py` is the project entrypoint. Prefer the task subcommands for new runs:
`list` runs the list experiment, and `plain` runs ordinary prompt files.

```bash
python3 main.py list --list-trial-kind sensitive_treatment --replicates-per-sensitive 5
python3 main.py plain --prompts-file prompts.txt
```

## List Experiment Mode

This repository supports a list experiment setup with:

- exactly 4 core statements from `experiments/list_experiment/data/core_statements.txt`
- 1 sensitive statement at a time from `experiments/list_experiment/data/sensitive_statements.txt`
- core-only control trials with just the 4 core statements
- balanced order-effect blocks
- `replicates_per_sensitive` total trials per sensitive statement, which must be a multiple of 5

For example, `--replicates-per-sensitive 5` runs one full block per sensitive
statement: the sensitive item appears exactly once in each position 1-5. A value
of `10` runs two balanced blocks per sensitive statement.

Core-only controls use a separate block size because they contain 4 statements:
`--core-control-replicates 4` runs one block where each core statement appears
once in each position 1-4.

For analysis, parse each model `answer` as a number. The basic estimate for a
sensitive item is:

```text
mean(answer for that sensitive_treatment item)
- mean(answer for core_control trials)
```

Run core-only controls via Slurm:

```bash
sbatch --export=ALL,LIST_TRIAL_KIND=core_control,CORE_CONTROL_REPLICATES=4,SEED=42 run_vllm.slurm
```

Run sensitive-treatment trials via Slurm:

```bash
sbatch --export=ALL,LIST_TRIAL_KIND=sensitive_treatment,REPLICATES_PER_SENSITIVE=5,SEED=42 run_vllm.slurm
```

Run both controls and treatments via Slurm:

```bash
sbatch --export=ALL,LIST_TRIAL_KIND=both,CORE_CONTROL_REPLICATES=4,REPLICATES_PER_SENSITIVE=5,SEED=42 run_vllm.slurm
```

### Persona Random Mode

Use `persona_random` when personas are the respondents. Put one persona prompt
per line in:

```text
experiments/list_experiment/data/personas.txt
```

In this mode each persona gets:

- one random core-control ordering
- one random ordering for every sensitive statement

So with 14 sensitive statements, each persona produces 15 rows total. With
5,000 personas, that is 75,000 rows. The old `LIST_TRIAL_KIND`,
`REPLICATES_PER_SENSITIVE`, and `CORE_CONTROL_REPLICATES` settings are for
`balanced` mode; `persona_random` always creates the full per-persona bundle.

Run persona-random mode via Slurm:

```bash
sbatch --export=ALL,ASSIGNMENT_MODE=persona_random,PERSONAS_FILE=experiments/list_experiment/data/personas.txt,SEED=42 run_vllm.slurm
```

Default generation parameters are:

- `--temperature 0.2`
- `--max-tokens 80`

Results are written to:

- `logs/list_experiment/<run_id>.generations.jsonl`

Slurm stdout and stderr logs are written separately:

- `logs/list_experiment/slurm-<job_id>.out`
- `logs/list_experiment/slurm-<job_id>.err`

### Output fields

Successful generation rows include:

- `run_id`
- `created_at_utc`
- `trial_id`
- `trial_index_in_run`
- `mode`
- `model_id`
- `seed`
- `generation_params`
- `messages`: exact chat messages sent to vLLM
- `answer`

List-experiment rows also include:

- `core_set_id`
- `trial_kind`: `core_control` or `sensitive_treatment`
- `item_count`: `4` for controls, `5` for treatments
- `sensitive_id`: sensitive item id, or `null` for controls
- `sensitive_text`: sensitive item text, or `null` for controls
- `replicate_index`
- `order_block_index`
- `sensitive_position`: position 1-5, or `null` for controls
- `order_labels`: compact item order, e.g. `["core_2", "sens_1", "core_4"]`
- `persona_id`: persona id, or `null` for balanced mode
- `persona_text`: persona prompt, or `null` for balanced mode

The JSONL intentionally does not repeat the full rendered prompt separately from
`messages`, and it does not store full `presented_items` text because the source
statements already live under `experiments/list_experiment/data/`.


## Bias Calculation Mode

This mode evaluates baseline political bias mapping model responses against human ALLBUS survey data. It features:
- Exactly 14 sensitive ALLBUS items replicated across $N=30$ independent runs to track answer stability.
- A **Self-Assessed Bias Baseline** evaluated via 30 independent runs to capture the model's explicit political self-perception.
- Post-processing binarization logic to map continuous responses into binary approval rates, enabling comparison with the List Experiment.

To run the calculation, change into the task directory and execute the script:
```bash
cd experiments/bias_calculation
sbatch --export=ALL,MODE=simple,MODEL_ID=meta-llama/Llama-3.1-70B-Instruct,TENSOR_PARALLEL_SIZE=1,OUTPUT_FILE="results/results_bias_llama_2.csv",QUANTIZATION=bitsandbytes,LOAD_FORMAT=bitsandbytes,GPU_MEMORY_UTILIZATION=0.9,MAX_MODEL_LEN=2048,TEMPERATURE=0.2,MAX_TOKENS=5, run_vllm_bias.slurm

### Output Fields (Bias Calculation CSV)

The script aggregates the $N=30$ runs and saves the results into `results/results_bias_llama_2.csv` with the following columns (indexed by the ALLBUS variable name):

1. Core Metrics & Mapping
- `LLM_Mean`: The average raw numerical score (1–7) assigned by the model across all valid runs.
- `LLM mean mapped`: The LLM's average score mapped onto the empirical human political spectrum.
- `Human_Center_Mean`: The benchmark score of human survey respondents who placed themselves exactly in the political center (5).
- `Absolute_Bias`: The structural drift of the mapped score from the neutral scale midpoint (5.5).

2. Self-Assessment & Deviation
`Self_Perception_Bias`: The mathematical delta between the model's explicit self-assessment and its actual item mapping.
`Self_Assessed_Mean_Baseline` / `_Var_Baseline` / `_Std_Baseline`: Statistical metrics captured from the 30 independent explicit self-perception runs.

3. Stability & Variance
- `llm standard deviation`: The standard deviation ($SD$) across the 30 item replications.
- `llm lower bound confidence interval` / `_upper bound...`: Standard 95% confidence intervals ($CI$) for the generated means.

4. Binarization & Direct Rates (For List Experiment Comparison)
- `pi_Direct`: The binarized agreement rate of the model (percentage of runs meeting the approval threshold).
- `pi_ALLBUS`: The baseline approval rate calculated from the human ALLBUS dataset.
- `Bias_Direct`: The direct delta used for comparison with the list experiment results (`pi_Direct - pi_ALLBUS`).
- `Valid_Runs` / `Invalid_Runs`: Operational log showing how many of the 30 runs parsed successfully into digits. 
