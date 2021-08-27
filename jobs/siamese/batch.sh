# template
#./jobs/siamese/primer.sh [model] [dataset] [features] [init. lr] [batch size] [hid layer sz] [emb. layer sz] [mem. in GB]
# [select on val data] [no. samples per class] [no. layers] [time] [no. samples per cluster] [partition]

./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 100 1 25000 3 1440 1 gpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 100 1 25000 3 1440 3 gpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 100 1 25000 3 1440 5 gpu
#./jobs/siamese/primer.sh 'all' 'all' 'btlc' .0001 32 128 32 100 1 25000 3 1440 7 gpu

