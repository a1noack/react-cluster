# template
#./jobs/classify/primer.sh [dataset] [group_size] [partition] [time] [mem]

group_size=1
n=5000  # equal to 25000 / 5
partition='short'
time=300

#for novel_attacks in 0 1
#do
#  datasets=("wikipedia" "wikipedia_personal" "hatebase" "civil_comments" "imdb" "reddit_dataset" "gab_dataset")
#  for dataset in "${datasets[@]}"
#  do
#    ./jobs/classify/primer.sh $dataset $group_size $n $novel_attacks $partition $time 100
#  done
#
#  ./jobs/classify/primer.sh 'sst' $group_size $n $novel_attacks $partition $time 100
#done

for novel_attacks in 0 1
do
  ./jobs/classify/primer.sh 'climate-change_waterloo' $group_size $n $novel_attacks 'gpu' $time 200
done