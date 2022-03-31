#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

dataset=$1

python -u scripts/filter_whole_catted_dataset.py \
    --dataset $dataset 