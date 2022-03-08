# template
#./jobs/siamese/primer.sh [model] [dataset] [features] [lr] [batch sz] [hid l. sz] [emb. l. sz] [mem]
#     [samples per class] [no. layers] [time] [samples per group] [where to avg] [max epochs] [early stop] [partition]

# ========== JOBS FOR NEW DEADLINE ==========
mem=200

group_size=15
n=25000

#datasets=("climate-change_waterloo" "imdb" "sst" "wikipedia"
#          "hatebase" "civil_comments" "abuse" "sentiment")
datasets=("sst" "civil_comments")
feature_sets=("b" "c" "btlc")

for dataset in "${datasets[@]}"; do
  for feature_set in "${feature_sets[@]}"; do
    ./jobs/siamese/primer.sh 'all' $dataset $feature_set .0001 32 128 32 $mem $n 3 4320 $group_size 'embedding' 60 999 longgpu
  done
done


# ========== JOBS FOR DIFFERENT SIZED EMBEDDING LAYERS ==========
#out_sizes=(3 5 10)
#
#for out_size in "${out_sizes[@]}"; do
#  for dataset in "${datasets[@]}"; do
#    for feature_set in "${feature_sets[@]}"; do
#      ./jobs/siamese/primer.sh 'all' $dataset $feature_set .0001 32 128 $out_size $mem $n 3 1440 $group_size 'embedding' 30 5 gpu
#    done
#  done
#done



# ========== JOBS FOR DIFFERENT SIZED GROUPS AND ALL SAMPLES ==========
#mem=200
#n=60000
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 3 'embedding' 28 5 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 5 'embedding' 28 5 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 7 'embedding' 28 5 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 9 'embedding' 28 5 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 11 'embedding' 28 5 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 13 'embedding' 28 5 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 15 'embedding' 28 5 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 17 'embedding' 28 5 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 19 'embedding' 28 5 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 21 'embedding' 28 5 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 23 'embedding' 28 5 longgpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 $mem $n 3 4320 25 'embedding' 28 5 longgpu
