dist_metric=$1  # bert or equals

mem=$2
time=$3
partition=$4
job_name=variant_comparison_${dist_metric}


sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=$job_name \
       --output=jobs/logs/$job_name \
       --error=jobs/errors/$job_name \
       jobs/variants/runner.sh $dist_metric
