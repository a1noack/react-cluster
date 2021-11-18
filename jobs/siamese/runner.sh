#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
#SBATCH --constraint=kepler
module load miniconda
conda activate textattack-0.2.11

model=$1
dataset=$2
in_dir=$3
out_dir=$4
features=$5
lr=$6
batch_size=$7
hid_size=$8
out_size=$9
n=${10}
n_layer=${11}
group_size=${12}
where_to_avg=${13}
max_epochs=${14}
early_stop=${15}

python3 scripts/train_siamese.py \
    --model $model \
    --dataset $dataset \
    --in_dir $in_dir \
    --out_dir $out_dir \
    --features $features \
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
