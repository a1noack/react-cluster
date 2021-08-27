model=$1
dataset=$2
n=$3
features=$4
n_dim=$5
mem=100
time=1440
partition=short

in_dir="data/"
out_dir="output/"
job_name="Cl_${model}_${dataset}_${features}_${n_dim}"

sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=$job_name \
       --output=jobs/logs/$job_name \
       --error=jobs/errors/$job_name \
       jobs/cluster/runner.sh $model $dataset $in_dir $n $out_dir $features $n_dim
