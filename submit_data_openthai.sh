#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 64                      # Specify number of nodes and processors per task     
#SBATCH --gpus-per-task=0               # Specify the number of GPUs
#SBATCH --ntasks-per-node=1             # Specify tasks per node
#SBATCH -t 10:00:00                      # Specify maximum time limit (hour: minute: second)   
#SBATCH -A lt200056                     # Specify project name

module restore
module load Miniconda3
module load PrgEnv-gnu
module load cpe-cuda
module load cudatoolkit/22.7_11.7
module load craype-accel-nvidia80
module load aws-ofi-nccl

conda deactivate
conda activate /project/lt200056-opgpth/new/TinyLlama/.conda-torch-2.1-new

python scripts/prepare_openthaigpt.py \
    --source_path /scratch/lt200056-opgpth/HF_V6_Colassal_deduplicated_128_09_decontaminated_128_03_blinded_json/ \
    --split train --percentage 1.0 \
    --tokenizer_path /project/lt200056-opgpth/tokenizer_spm_v5 \
    --destination_path data/openthaigpt_v6

python scripts/prepare_openthaigpt.py \
    --source_path /scratch/lt200056-opgpth/HF_V6_Colassal_deduplicated_128_09_decontaminated_128_03_blinded_json/ \
    --split eval --percentage 1.0 \
    --tokenizer_path /project/lt200056-opgpth/tokenizer_spm_v5 \
    --destination_path data/openthaigpt_v6_validation