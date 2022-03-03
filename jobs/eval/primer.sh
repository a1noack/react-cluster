model=$1
dataset=$2
features=$3
n=$4
in_dir=$5
out_dir=$6
siamese_net_dir=$7
group_size=$8

mem=$9
time=${10}
partition=${11}
emb_size=${12}

job_name="SiE_${model}_${dataset}_${features}_${group_size}_${emb_size}"


sbatch --mem=${mem}G \
    --time=$time \
    --partition=$partition \
    --gres=gpu:1 \
    --job-name=$job_name \
    --output=jobs/logs/$job_name \
    --error=jobs/errors/$job_name \
    jobs/eval/runner.sh $model $dataset $features $n $in_dir $out_dir $siamese_net_dir $group_size
