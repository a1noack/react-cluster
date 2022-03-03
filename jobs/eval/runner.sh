#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate textattack-0.2.11

model=$1
dataset=$2
features=$3
n=$4
in_dir=$5
out_dir=$6
siamese_net_dir=$7
group_size=$8

python3 scripts/eval_siamese.py \
  --model $model \
  --dataset $dataset \
  --features $features \
  --n $n \
  --in_dir $in_dir \
  --out_dir $out_dir \
  --siamese_net_dir $siamese_net_dir \
  --group_size $group_size
