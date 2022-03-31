dataset=$1
nodelist_=$2

job_name="filter_${dataset}"
mem_=50

sbatch --mem=${mem_}G \
       --time=1440 \
       --partition=ava_s.p \
       --nodelist=${nodelist_} \
       --cpus-per-task=1 \
       --job-name=$job_name \
       --output=jobs/logs/$job_name \
       --error=jobs/errors/$job_name \
       --account=ucinlp.a \
       jobs/filter_csv/runner.sh $dataset
