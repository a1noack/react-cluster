# hatebase

./jobs/siamese_ava/primer.sh 'all' 'hatebase' 'b' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s3
./jobs/siamese_ava/primer.sh 'all' 'hatebase' 'c' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s1
./jobs/siamese_ava/primer.sh 'all' 'hatebase' 'btlc' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s0

# imdb

./jobs/siamese_ava/primer.sh 'all' 'imdb' 'b' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s3
./jobs/siamese_ava/primer.sh 'all' 'imdb' 'c' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s1
./jobs/siamese_ava/primer.sh 'all' 'imdb' 'btlc' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s3

# hatebase with finetuned bert features
# ./jobs/siamese_ava/primer.sh [target models] [dataset] [path to bert model to use for re-encode bert features] [learning rate] [batch size] [hidden layer size] [embedding layer size] [n -> samples per class ] [number of layers] [group size] [where to avg] [max epochs] [early stop] [ava node]


## multliclass with clean (started)
./jobs/siamese_ava/primer.sh 'all' 'hatebase' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s0
./jobs/siamese_ava/primer.sh 'all' 'hatebase' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s1
./jobs/siamese_ava/primer.sh 'all' 'hatebase' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s0

## clean vs all

./jobs/siamese_ava/primer.sh 'all' 'hatebase' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/clean_vs_all/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s0
./jobs/siamese_ava/primer.sh 'all' 'hatebase' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/clean_vs_all/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s1
./jobs/siamese_ava/primer.sh 'all' 'hatebase' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/clean_vs_all/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s0

# imdb with finetured bert features

## multiclass with clean (started)
./jobs/siamese_ava/primer.sh 'all' 'imdb' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s1
./jobs/siamese_ava/primer.sh 'all' 'imdb' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s0
./jobs/siamese_ava/primer.sh 'all' 'imdb' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s1

# sst with finetuned bert features

## multiclass with clean 
./jobs/siamese_ava/primer.sh 'all' 'sst' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/sst/multiclass_with_clean/bert/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s3
./jobs/siamese_ava/primer.sh 'all' 'sst' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/sst/multiclass_with_clean/bert/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s0
./jobs/siamese_ava/primer.sh 'all' 'sst' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/sst/multiclass_with_clean/bert/BERT_DETECTION' .0001 32 128 32 25000 3 15 'embedding' 60 999 ava-s0

