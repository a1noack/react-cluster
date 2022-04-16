#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

model=$1
dataset=$2
features=$3
detection_bert_root=$4
n=$5
in_dir=$6
out_dir=$7
siamese_net_dir=$8
group_size=$9
epoch=${10}

python -u scripts/eval_siamese.py \
  --model $model \
  --dataset $dataset \
  --features $features \
  --detection_bert_root $detection_bert_root \
  --n $n \
  --in_dir $in_dir \
  --out_dir $out_dir \
  --siamese_net_dir $siamese_net_dir \
  --group_size $group_size \
  --epoch $epoch
