model=$1
dataset=$2
features=$3
lr=$4
batch_size=$5
hid_size=$6
out_size=$7
mem=$8
n=${9}
n_layer=${10}
time=${11}
group_size=${12}
where_to_avg=${13}
max_epochs=${14}
early_stop=${15}
partition=${16}


in_dir="data/"
out_dir="output/"
job_name="Si_${model}_${dataset}_${features}_${lr}_${batch_size}_${hid_size}_${out_size}_${n_layer}_${group_size}"


sbatch --mem=${mem}G \
    --time=$time \
    --partition=$partition \
    --gres=gpu:1 \
    --job-name=$job_name \
    --output=jobs/logs/$job_name \
    --error=jobs/errors/$job_name \
    jobs/siamese/runner.sh $model $dataset $in_dir $out_dir $features $lr $batch_size $hid_size \
      $out_size $n $n_layer $group_size $where_to_avg $max_epochs $early_stop
