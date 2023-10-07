#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 4 -c 16                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=4		# Specify number of tasks per node
#SBATCH --gpus-per-node=4		        # Specify total number of GPUs
#SBATCH -t 82:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200067                       # Specify project name
#SBATCH -J Test_mixed                          # Specify job name

module restore
module load Miniconda3
module load PrgEnv-gnu
module load cpe-cuda
module load cudatoolkit/22.7_11.7
module load craype-accel-nvidia80
module load aws-ofi-nccl

conda deactivate
conda activate /project/lt200056-opgpth/new/TinyLlama/.conda-torch-2.1-new

export WANDB_MODE=offline

python -m xformers.info


srun python pretrain/tinyllama.py \
    --train_data_dir data/thepile \
    --val_data_dir data/thepile_validation \
    --devices 4 \
    --num_nodes 4 \
    # --resume out/tinyllama_1b/iter-002000-ckpt.pth