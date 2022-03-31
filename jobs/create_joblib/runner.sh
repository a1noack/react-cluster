#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

dataset=$1

python -u scripts/filtered_csv_to_joblib.py \
    --dataset $dataset 