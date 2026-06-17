#!/bin/bash
#SBATCH --job-name=vllm_mwe
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:hopper:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/home/ej530777/logs/mwe/slurm-%j.out

module load Python/3.11.5

cd /rwthfs/rz/cluster/home/ej530777/csslab-llm-bias

source /home/ej530777/.venv/bin/activate

export PYTHONPATH=/rwthfs/rz/cluster/home/ej530777/csslab-llm-bias:$PYTHONPATH
export VLLM_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"

echo "Running on host: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "PWD: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Venv: $VIRTUAL_ENV"
echo "Model: $VLLM_MODEL_ID"
echo "Purpose: vLLM setup smoke test"

srun python -m mwe.main_vllm
