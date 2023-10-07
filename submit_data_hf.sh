#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 64                      # Specify number of nodes and processors per task     
#SBATCH --ntasks-per-node=1             # Specify tasks per node
#SBATCH -t 08:00:00                      # Specify maximum time limit (hour: minute: second)   
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

# python scripts/prepare_hf_datasets.py \
#  --input /scratch/lt200056-opgpth/HF_V5_555_Dataset_deduplicated_128_09_decontaminated_128_03/ \
#  --output /scratch/lt200056-opgpth/HF_V5_555_Dataset_deduplicated_128_09_decontaminated_128_03_jsonl/

python scripts/prepare_hf_datasets.py \
    /scratch/lt200056-opgpth/HF_V6_Colassal_deduplicated_128_09_decontaminated_128_03_blinded \
    /scratch/lt200056-opgpth/HF_V6_Colassal_deduplicated_128_09_decontaminated_128_03_blinded_json

# python scripts/prepare_hf_datasets.py \
#  --input /project/lt200056-opgpth/public_datasets/pile_hf \
#  --output /project/lt200056-opgpth/public_datasets/pile_hf_jsonl_for_tiny_llama