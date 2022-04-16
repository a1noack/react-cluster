model=$1
dataset=$2
features=$3
detection_bert_root=$4
lr=$5
batch_size=$6
hid_size=$7
out_size=$8
n=$9
n_layer=${10}
group_size=${11}
where_to_avg=${12}
max_epochs=${13}
early_stop=${14}
nodelist_=${15}

in_dir="filtered_data/${dataset}"
out_dir="output/"
job_name="Si_finetuned_bert_${model}_${dataset}_${features}_${lr}_${batch_size}_${hid_size}_${out_size}_${n_layer}_${group_size}"
mem_=400

sbatch --mem=${mem_}G \
       --time=4320 \
       --partition=ava_s.p \
       --nodelist=${nodelist_} \
       --gpus=1 \
       --cpus-per-task=4 \
       --job-name=$job_name \
       --output=jobs/logs/finetuned_bert/$job_name \
       --error=jobs/errors/finetuned_bert/$job_name \
       --account=ucinlp.a \
       jobs/siamese_ava/runner.sh $model $dataset $in_dir $out_dir $features $detection_bert_root $lr $batch_size $hid_size \
       $out_size $n $n_layer $group_size $where_to_avg $max_epochs $early_stop
