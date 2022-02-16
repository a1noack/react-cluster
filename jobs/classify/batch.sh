# template
#./jobs/classify/primer.sh [dataset] [group_size] [partition] [time] [mem]

group_size=1
n=5000  # equal to 25000 / 5
partition='short'
time=300

#datasets=("wikipedia" "wikipedia_personal" "hatebase" "civil_comments" "imdb" "reddit_dataset" "gab_dataset")

## not novelty prediction
#for dataset in "${datasets[@]}"; do
#  ./jobs/classify/primer.sh $dataset $group_size $n 1 0 'input' $partition $time 100
#done

datasets=("civil_comments" "sst")
group_size=15
# novelty prediction
for dataset in "${datasets[@]}"; do
  ./jobs/classify/primer.sh $dataset $group_size $n 1 1 'output' $partition $time 50
done