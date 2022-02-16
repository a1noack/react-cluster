dataset=$1
group_size=$2
n=$3
novel_attacks=$4
novelty_prediction=$5
where_to_avg=$6

partition=$7
time=$8
mem=$9

in_dir="data/"
out_dir="output/"
job_name="Cl_nov-pred${novelty_prediction}_${dataset}_${group_size}_${novel_attacks}"


sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=$job_name \
       --output=jobs/logs/$job_name \
       --error=jobs/errors/$job_name \
       jobs/classify/runner.sh $dataset $group_size $n $novel_attacks $novelty_prediction $where_to_avg
