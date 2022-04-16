#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

model=$1
dataset=$2
in_dir=$3
out_dir=$4
features=$5
detection_bert_root=$6
lr=$7
batch_size=$8
hid_size=$9
out_size=${10}
n=${11}
n_layer=${12}
group_size=${13}
where_to_avg=${14}
max_epochs=${15}
early_stop=${16}

python -u scripts/train_siamese.py \
    --model $model \
    --dataset $dataset \
    --in_dir $in_dir \
    --out_dir $out_dir \
    --features $features \
    --detection_bert_root $detection_bert_root \
    --lr $lr \
    --batch_size $batch_size \
    --hid_size $hid_size \
    --out_size $out_size \
    --n $n \
    --n_layer $n_layer \
    --group_size $group_size \
    --where_to_avg $where_to_avg \
    --max_epochs $max_epochs \
    --early_stop $early_stop