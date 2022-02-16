#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate textattack-0.2.11

dataset=$1
group_size=$2
n=$3
novel_attacks=$4
novelty_prediction=$5
where_to_avg=$6

python3 scripts/train_clf.py \
    --dataset=$dataset \
    --group_size=$group_size \
    --n=$n \
    --novel_attacks=$novel_attacks \
    --novelty_prediction=$novelty_prediction \
    --where_to_avg=$where_to_avg
