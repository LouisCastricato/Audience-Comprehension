#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --account=louis
#SBATCH --job-name=chatbot-experiment
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/fsx/$(whoami)/Audience-Comprehension/outputs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --comment carper

export FSX_HOME=/fsx/$(whoami)
# Point common large caches to fsx
export BASE_CACHE=$FSX_HOME/.cache
export XDG_CACHE_HOME=$BASE_CACHE
export HF_HOME=$BASE_CACHE/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export JUPYTER_CONFIG_DIR=/fsx/$(whoami)/.jupyter
export JUPYTER_DATA_DIR=/fsx/$(whoami)/.local/share/jupyter/

source /fsx/louis/Audience-Comprehension/.env/bin/activate
srun --comment carper python3.8 /fsx/louis/Audience-Comprehension/main.py