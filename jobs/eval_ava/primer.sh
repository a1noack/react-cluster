model=$1
dataset=$2
features=$3
detection_bert_root=$4
n=$5
in_dir=$6
out_dir=$7
siamese_net_dir=$8
group_size=$9
emb_size=${10}
epoch=${11}
nodelist_=${12}

job_name="SiE_ft-bert_${model}_${dataset}_${features}_${group_size}_${emb_size}_${epoch}"
mem_=400

sbatch --mem=${mem_}G \
       --time=1440 \
       --partition=ava_s.p \
       --nodelist=${nodelist_} \
       --gpus=1 \
       --cpus-per-task=4 \
       --job-name=$job_name \
       --output=jobs/logs/finetuned_bert/$job_name \
       --error=jobs/errors/finetuned_bert/$job_name \
       --account=ucinlp.a \
       jobs/eval_ava/runner.sh $model $dataset $features $detection_bert_root $n $in_dir $out_dir $siamese_net_dir $group_size $epoch 
