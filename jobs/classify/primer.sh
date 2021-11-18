dataset=$1
group_size=$2
n=$3
novel_attacks=$4
partition=$5
time=$6
mem=$7

in_dir="data/"
out_dir="output/"
job_name="Cl_${dataset}_${group_size}_${novel_attacks}"


sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --gres=gpu:1 \
       --constraint=kepler \
       --job-name=$job_name \
       --output=jobs/logs/$job_name \
       --error=jobs/errors/$job_name \
       jobs/classify/runner.sh $dataset $group_size $n $novel_attacks
