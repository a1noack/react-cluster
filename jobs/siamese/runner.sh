#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
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
val=${10}
n=${11}
n_layer=${12}
group_size=${13}
held_out=('iga_wang' 'faster_genetic' 'genetic')

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
    --select_on_val $val \
    --n $n \
    --n_layer $n_layer \
    --group_size $group_size \
    --held_out "${held_out[@]}"

