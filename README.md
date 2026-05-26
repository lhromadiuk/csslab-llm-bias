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
