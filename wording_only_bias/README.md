# LLM Bias Calculation with vLLM

This directory contains scripts for calculating bias metrics of a Large Language Model (LLM) using ALLBUS survey data.  
The model is evaluated using different prompt wording variations, and the generated responses are compared with human response distributions from the ALLBUS dataset.

The pipeline calculates:

- mean LLM responses
- response standard deviation
- confidence intervals
- bias measures (`Bias_Direct`) between approval ratesX
- comparison with ALLBUS population statistics

---

## Directory Structure

```
bias_calc_wording/
│
├── bias_calc_wording.py
│   Main script for the bias calculation
│
├── run_vllm_wording.slurm
│   SLURM job script for running the calculation on the cluster
│
├── prompts_wording_variations.txt
│   Contains the different prompt variations
│
└── results/
    └── bias_results_wording.csv
        Output file containing the results
```

---

# Requirements

The following dependencies are required:

- Python 3.11
- vLLM
- pandas
- numpy
- pyreadstat (for reading SPSS files)
- HuggingFace access (if required by the model)

A Python virtual environment is expected.

Example:

```bash
source ../../.venv/bin/activate
```

---

# Dataset

The script uses the ALLBUS dataset:

```
data/ZA8831_v1-3-0.sav
```

The dataset is loaded automatically using:

```python
pd.read_spss(...)
```

The analyzed variables are:

```python
[
"ma01b",
"ma02",
"ma03",
"ma04",
"mp02",
"ca13",
"ca08",
"fr10",
"fr04b",
"fr03b",
"pi08",
"mm03",
"mm04",
"mm05"
]
```

---

# Running the Pipeline

## Local execution

The Python script can be executed directly:

```bash
python bias_calc_wording.py
```

---

## Running on the cluster (SLURM)

The recommended way to run the experiment is:

```bash
sbatch run_vllm_wording.slurm
```

The SLURM script:

- loads Python
- activates the virtual environment
- allocates a GPU
- configures vLLM
- starts the bias calculation

---

# SLURM Configuration

Current resource configuration:

```bash
#SBATCH --partition=c23g
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
```

The job uses:

- 1 GPU
- 4 CPU cores
- 32 GB RAM

---

# Model Configuration

Default model:

```
mistralai/Mistral-7B-Instruct-v0.3
```

The model can be changed:

```bash
MODEL_ID="model/name" sbatch run_vllm_wording.slurm
```

---

# Configuration Parameters

All important parameters can be modified using environment variables.

## Model parameters

| Variable | Default | Description |
|---|---|---|
| MODEL_ID | mistralai/Mistral-7B-Instruct-v0.3 | LLM model used |
| QUANTIZATION | empty | Optional model quantization |
| LOAD_FORMAT | empty | Model loading format |

---

## Generation parameters

| Variable | Default | Description |
|---|---|---|
| MAX_TOKENS | 80 | Maximum number of generated tokens |
| TEMPERATURE | 0.0 | Sampling temperature |
| MAX_MODEL_LEN | 4096 | Maximum context length |

---

## Prompt parameters

| Variable | Default |
|---|---|
| PROMPTS_FILE | prompts_wording_variations.txt |
| PROMPT_MODE | lines |

Example:

```bash
PROMPTS_FILE=my_prompts.txt sbatch run_vllm_wording.slurm
```

---

# Output

The results are saved to:

```
results/bias_results_wording.csv
```

The CSV file contains:

| Column | Description |
|---|---|
| LLM_Mean | Mean model response |
| llm standard deviation | Standard deviation of responses |
| confidence intervals | Uncertainty interval of the estimate |
| pi_Direct | Fraction of LLM responses classified as positive |
| pi_ALLBUS | Corresponding fraction in ALLBUS |
| Bias_Direct | Difference between LLM and human response distribution |
| Valid_Runs | Number of valid model responses |
| Invalid_Runs | Number of rejected responses |

---

# Calculation Workflow

For each sensitive variable:

1. Load prompt variations
2. Query the LLM 30 times
3. Extract a single numeric response
4. Remove invalid responses
5. Calculate:
   - mean response
   - standard deviation
   - confidence interval
6. Convert responses into binary categories
7. Compare LLM results with ALLBUS statistics
8. Export results as CSV

---

# Example Usage

Start the experiment:

```bash
sbatch run_vllm_wording.slurm
```

Check the output log:

```bash
cat logs/slurm-<JOB_ID>.out
```

The final results can be found at:

```
results/bias_results_wording.csv
```

---

# Notes

- The model is instructed to output only a single digit.
- Responses that cannot be parsed as valid numeric answers are discarded.
- Bias results depend on the chosen binarization rules for each survey variable.
- Changing prompt wording can influence model responses and therefore bias estimates.
