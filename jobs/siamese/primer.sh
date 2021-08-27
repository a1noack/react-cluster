model=$1
dataset=$2
features=$3
lr=$4
batch_size=$5
hid_size=$6
out_size=$7
mem=$8
val=$9
n=${10}
n_layer=${11}
time=${12}
group_size=${13}
partition=${14}

in_dir="data/"
out_dir="output/"
job_name="Si_${model}_${dataset}_${lr}_${batch_size}_${hid_size}_${out_size}_${n_layer}_val-${val}_${group_size}"


sbatch --mem=${mem}G \
    --time=$time \
    --partition=$partition \
    --gres=gpu:1 \
    --job-name=$job_name \
    --output=jobs/logs/$job_name \
    --error=jobs/errors/$job_name \
    jobs/siamese/runner.sh $model $dataset $in_dir $out_dir $features $lr $batch_size $hid_size $out_size $val $n $n_layer $group_size
