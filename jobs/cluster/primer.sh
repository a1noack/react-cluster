model=$1
dataset=$2
n=$3
features=$4
n_dim=$5
mem=$6
time=1440
partition=short

in_dir="data/"
out_dir="output/"
siamese_net_dir="output/siamese_bert_sst_0.0001_32_128_32_val-1_3_1630689049"
reduced_set=$7
compress_with_siamese=$8
group_size=$9
job_name="Clu_${model}_${dataset}_${features}_${n_dim}"


sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=$job_name \
       --output=jobs/logs/$job_name \
       --error=jobs/errors/$job_name \
       jobs/cluster/runner.sh $model $dataset $in_dir $n $out_dir $features $n_dim \
       $siamese_net_dir $reduced_set $compress_with_siamese $group_size
