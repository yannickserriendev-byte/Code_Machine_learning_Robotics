# Example SLURM job script for model training on GPU cluster
# Usage: Adjust resource requests and script name as needed for your environment

#!/bin/bash
#SBATCH --job-name=TrainModel
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=2
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3900
#SBATCH --mail-type=START,END,FAIL
#SBATCH --account=Education-ME-MSc-BE
#SBATCH --output=slurm-%j.out

# === Load modules ===
module purge
module load 2024r1
module load Python/3.10.13
module load cuda/12.1
module load ffmpeg
module load miniconda3
conda activate AugmentDataCorrect

# === Run your training script ===
python train_resnet18_multitask.py
