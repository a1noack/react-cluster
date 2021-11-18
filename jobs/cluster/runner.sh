#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate textattack-0.2.11

model=$1
dataset=$2
in_dir=$3
n=$4
out_dir=$5
features=$6
n_dim=$7
siamese_net_dir=$8
reduced_set=$9
compress_with_siamese=${10}
group_size=${11}
#algos=('aff_prop' 'agg_clust' 'birch' 'dbscan' 'kmeans' 'mean_shift' 'optics' 'spec_clust')

python3 scripts/cluster.py --model $model --dataset $dataset --in_dir $in_dir --n $n \
        --out_dir $out_dir --compress_features $n_dim --features $features \
        --siamese_net_dir $siamese_net_dir --use_reduced_attack_set $reduced_set \
        --compress_with_siamese $compress_with_siamese  --group_size $group_size
