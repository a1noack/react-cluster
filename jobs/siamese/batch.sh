# template
#./jobs/siamese/primer.sh [model] [dataset] [features] [lr] [batch sz] [hid l. sz] [emb. l. sz] [mem]
#     [samples per class] [no. layers] [time] [samples per group] [where to avg] [max epochs] [early stop] [partition]

#n=25000  # was 25,000
#
#for group_size in 1
#do
#  mem=100
#  datasets=("wikipedia" "wikipedia_personal" "hatebase" "civil_comments" "imdb" "reddit_dataset" "gab_dataset")
#  for dataset in "${datasets[@]}"
#  do
#    ./jobs/siamese/primer.sh 'all' $dataset 'btlc' .0001 32 128 32 $mem $n 3 1440 $group_size 'embedding' 50 10 gpu
#  done
#
#  mem=150
#  ./jobs/siamese/primer.sh 'all' 'sst' 'btlc' .0001 32 128 32 $mem $n 3 1440 $group_size 'embedding' 50 10 gpu
#
#  mem=200
#  ./jobs/siamese/primer.sh 'all' 'climate-change_waterloo' 'btlc' .0001 32 128 32 $mem $n 3 1440 $group_size 'embedding' 50 10 gpu
#done

mem=200
n=60000
#./jobs/siamese/primer.sh 'roberta' 'hatebase' 'btlc' .0001 32 128 32 50 1000 3 1440 1 'embedding' 10 2 gpu
./jobs/siamese/primer.sh 'all' 'all' 'btlc_' .0001 32 128 32 $mem $n 3 2880 3 'embedding' 100 15 longgpu
./jobs/siamese/primer.sh 'all' 'all' 'btlc_' .0001 32 128 32 $mem $n 3 2880 5 'embedding' 100 15 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc_' .0001 32 128 32 $mem $n 3 2880 7 'embedding' 100 15 longgpu
./jobs/siamese/primer.sh 'all' 'all' 'btlc_' .0001 32 128 32 $mem $n 3 2880 9 'embedding' 100 15 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc_' .0001 32 128 32 $mem $n 3 2880 11 'embedding' 100 15 longgpu
./jobs/siamese/primer.sh 'all' 'all' 'btlc_' .0001 32 128 32 $mem $n 3 2880 13 'embedding' 100 15 longgpu
