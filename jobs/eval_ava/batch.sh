# n=4000
# gs=15
# mem=40
# time=1440
# in_dir='data/'
# out_dir='output/'

# imdb

jobs/eval_ava/primer.sh 'all' 'imdb' 'b' 4000 'data/' 'output/' 'output/siamese_all_imdb_b_0.0001_32_128_32_15_embedding_1646885991' 15 32 ava-s1
jobs/eval_ava/primer.sh 'all' 'imdb' 'c' 4000 'data/' 'output/' 'output/siamese_all_imdb_c_0.0001_32_128_32_15_embedding_1646886027' 15 32 ava-s0
jobs/eval_ava/primer.sh 'all' 'imdb' 'btlc' 4000 'data/' 'output/' 'output/siamese_all_imdb_btlc_0.0001_32_128_32_15_embedding_1646899487' 15 32 ava-s1

# hatebase

jobs/eval_ava/primer.sh 'all' 'hatebase' 'b' 4000 'data/' 'output/' 'output/siamese_all_hatebase_b_0.0001_32_128_32_15_embedding_1646885983' 15 32 ava-s3
jobs/eval_ava/primer.sh 'all' 'hatebase' 'c' 4000 'data/' 'output/' 'output/siamese_all_hatebase_c_0.0001_32_128_32_15_embedding_1646886761' 15 32 ava-s1
jobs/eval_ava/primer.sh 'all' 'hatebase' 'btlc' 4000 'data/' 'output/' 'output/siamese_all_hatebase_btlc_0.0001_32_128_32_15_embedding_1646898845' 15 32 ava-s0

# eval with detection_bert_root

# jobs/eval_ava/primer.sh [target models] [dataset] [featuers] [bert model to use for re-encode bert features] [n -> number of samples per class] [folder that contains 'dataset'.joblib] [output folder] [trained siamese network dir] [group size] [embedding size] [epoch - if set to 0, will evaluate the siamese network with lowest val loss] [ava node]
# hatebase

jobs/eval_ava/primer.sh 'all' 'hatebase' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_b_0.0001_32_128_32_15_embedding_ft-bert_1648703387' 15 32 0 ava-s1
jobs/eval_ava/primer.sh 'all' 'hatebase' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_c_0.0001_32_128_32_15_embedding_ft-bert_1648703661' 15 32 0 ava-s1
jobs/eval_ava/primer.sh 'all' 'hatebase' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648703667' 15 32 0 ava-s1

# hatebase-epoch-grid-search
# jobs/eval_ava/primer.sh 'all' 'hatebase' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_b_0.0001_32_128_32_15_embedding_ft-bert_1648703387' 15 32 50 ava-s1
# jobs/eval_ava/primer.sh 'all' 'hatebase' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_b_0.0001_32_128_32_15_embedding_ft-bert_1648703387' 15 32 52 ava-s1
jobs/eval_ava/primer.sh 'all' 'hatebase' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_b_0.0001_32_128_32_15_embedding_ft-bert_1648703387' 15 32 54 ava-s1
jobs/eval_ava/primer.sh 'all' 'hatebase' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_b_0.0001_32_128_32_15_embedding_ft-bert_1648703387' 15 32 56 ava-s1
jobs/eval_ava/primer.sh 'all' 'hatebase' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_b_0.0001_32_128_32_15_embedding_ft-bert_1648703387' 15 32 58 ava-s1

# jobs/eval_ava/primer.sh 'all' 'hatebase' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_c_0.0001_32_128_32_15_embedding_ft-bert_1648703661' 15 32 49 ava-s3
# jobs/eval_ava/primer.sh 'all' 'hatebase' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_c_0.0001_32_128_32_15_embedding_ft-bert_1648703661' 15 32 51 ava-s3
jobs/eval_ava/primer.sh 'all' 'hatebase' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_c_0.0001_32_128_32_15_embedding_ft-bert_1648703661' 15 32 53 ava-s3
jobs/eval_ava/primer.sh 'all' 'hatebase' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_c_0.0001_32_128_32_15_embedding_ft-bert_1648703661' 15 32 55 ava-s3
jobs/eval_ava/primer.sh 'all' 'hatebase' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_c_0.0001_32_128_32_15_embedding_ft-bert_1648703661' 15 32 57 ava-s3

# jobs/eval_ava/primer.sh 'all' 'hatebase' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648703667' 15 32 50 ava-s0
# jobs/eval_ava/primer.sh 'all' 'hatebase' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648703667' 15 32 52 ava-s0
jobs/eval_ava/primer.sh 'all' 'hatebase' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648703667' 15 32 54 ava-s0
jobs/eval_ava/primer.sh 'all' 'hatebase' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648703667' 15 32 56 ava-s0
jobs/eval_ava/primer.sh 'all' 'hatebase' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/hatebase/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/hatebase' 'output/' 'output/siamese_all_hatebase_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648703667' 15 32 58 ava-s0

# imdb
jobs/eval_ava/primer.sh 'all' 'imdb' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_b_0.0001_32_128_32_15_embedding_ft-bert_1648704135' 15 32 0 ava-s0
jobs/eval_ava/primer.sh 'all' 'imdb' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_c_0.0001_32_128_32_15_embedding_ft-bert_1648704309' 15 32 0 ava-s3
jobs/eval_ava/primer.sh 'all' 'imdb' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648704288' 15 32 0 ava-s0

# imdb-epoch-grid-search
# jobs/eval_ava/primer.sh 'all' 'imdb' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_b_0.0001_32_128_32_15_embedding_ft-bert_1648704135' 15 32 49 ava-s0
# jobs/eval_ava/primer.sh 'all' 'imdb' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_b_0.0001_32_128_32_15_embedding_ft-bert_1648704135' 15 32 51 ava-s0
jobs/eval_ava/primer.sh 'all' 'imdb' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_b_0.0001_32_128_32_15_embedding_ft-bert_1648704135' 15 32 53 ava-s0
jobs/eval_ava/primer.sh 'all' 'imdb' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_b_0.0001_32_128_32_15_embedding_ft-bert_1648704135' 15 32 55 ava-s0
jobs/eval_ava/primer.sh 'all' 'imdb' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_b_0.0001_32_128_32_15_embedding_ft-bert_1648704135' 15 32 57 ava-s0

# jobs/eval_ava/primer.sh 'all' 'imdb' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_c_0.0001_32_128_32_15_embedding_ft-bert_1648704309' 15 32 50 ava-s1
# jobs/eval_ava/primer.sh 'all' 'imdb' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_c_0.0001_32_128_32_15_embedding_ft-bert_1648704309' 15 32 52 ava-s1
jobs/eval_ava/primer.sh 'all' 'imdb' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_c_0.0001_32_128_32_15_embedding_ft-bert_1648704309' 15 32 54 ava-s1
jobs/eval_ava/primer.sh 'all' 'imdb' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_c_0.0001_32_128_32_15_embedding_ft-bert_1648704309' 15 32 56 ava-s1
jobs/eval_ava/primer.sh 'all' 'imdb' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_c_0.0001_32_128_32_15_embedding_ft-bert_1648704309' 15 32 58 ava-s1

# jobs/eval_ava/primer.sh 'all' 'imdb' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648704288' 15 32 49 ava-s3
# jobs/eval_ava/primer.sh 'all' 'imdb' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648704288' 15 32 51 ava-s3
jobs/eval_ava/primer.sh 'all' 'imdb' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648704288' 15 32 53 ava-s3
jobs/eval_ava/primer.sh 'all' 'imdb' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648704288' 15 32 55 ava-s3
jobs/eval_ava/primer.sh 'all' 'imdb' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/imdb/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/imdb' 'output/' 'output/siamese_all_imdb_btlc_0.0001_32_128_32_15_embedding_ft-bert_1648704288' 15 32 57 ava-s3

#sst
jobs/eval_ava/primer.sh 'all' 'sst' 'b' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/sst/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/sst' 'output/' 'output/siamese_all_sst_b_0.0001_32_128_32_15_embedding_ft-bert_1649132355' 15 32 0 ava-s0
jobs/eval_ava/primer.sh 'all' 'sst' 'c' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/sst/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/sst' 'output/' 'output/siamese_all_sst_c_0.0001_32_128_32_15_embedding_ft-bert_1649049750' 15 32 0 ava-s1
jobs/eval_ava/primer.sh 'all' 'sst' 'btlc' '/extra/ucinlp0/kasthana/react-cluster/detection-experiments/sst/multiclass_with_clean/bert/BERT_DETECTION' 4000 'filtered_data/sst' 'output/' 'output/siamese_all_sst_btlc_0.0001_32_128_32_15_embedding_ft-bert_1649050250' 15 32 0 ava-s3
