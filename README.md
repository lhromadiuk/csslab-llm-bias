# csslab-llm-bias

This repository stores code and results for the Computational Social Science Lab @ RWTH Aachen University.

## Repository structure

- `data/`: Raw data files are not published openly in this repository, please check the corresponding README file.
- `logs/`: logs generated during experiments.
- `mwe/`: minimal working examples for model execution.
- `main.py`: project entrypoint for running vLLM over prompts from a file.
- `prompt_loader.py`: prompt-file loading and optional prompt-order randomization.
- `prompts.txt`: example prompt file.
- `requirements.txt`: Python dependencies needed to run the project code.
- `setup_env.sh`: setup script that creates or updates a Python virtual environment in the parent folder.
- `run_vllm.slurm`: Slurm job script for running `main.py`.
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

## List Experiment Mode

This repository supports a list experiment setup with:

- exactly 4 core statements from `core_statements.txt`
- 1 sensitive statement at a time from `sensitive_statements.txt`
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
sbatch --export=ALL,EXPERIMENT_MODE=list,LIST_TRIAL_KIND=core_control,CORE_CONTROL_REPLICATES=4,SEED=42 run_vllm.slurm
```

Run sensitive-treatment trials via Slurm:

```bash
sbatch --export=ALL,EXPERIMENT_MODE=list,LIST_TRIAL_KIND=sensitive_treatment,REPLICATES_PER_SENSITIVE=5,SEED=42 run_vllm.slurm
```

Run both controls and treatments via Slurm:

```bash
sbatch --export=ALL,EXPERIMENT_MODE=list,LIST_TRIAL_KIND=both,CORE_CONTROL_REPLICATES=4,REPLICATES_PER_SENSITIVE=5,SEED=42 run_vllm.slurm
```

Default generation parameters are:

- `--temperature 0.2`
- `--max-tokens 80`

Results are written to:

- `logs/list_experiment/<run_id>.generations.jsonl`
- `logs/list_experiment/<run_id>.errors.jsonl`

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

The JSONL intentionally does not repeat the full rendered prompt separately from
`messages`, and it does not store full `presented_items` text because the source
statements already live in `core_statements.txt` and `sensitive_statements.txt`.

Error rows include the attempted trial metadata plus:

- `error_type`
- `error`
