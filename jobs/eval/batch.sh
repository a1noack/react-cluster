n=4000  # number of samples per class
gs=15  # group size
mem=40  # memory in GB
time=1440
in_dir='data/'
out_dir='output/'

epoch=50  # the epoch for the siamese network weights; if 0, load the model weights with lowest val. loss


# SST
#jobs/eval/primer.sh 'all' 'sst' 'b' $n $in_dir $out_dir 'output/siamese_all_sst_b_0.0001_32_128_32_15_embedding_1641839900' $gs $mem $time gpu 32 $epoch
#jobs/eval/primer.sh 'all' 'sst' 'c' $n $in_dir $out_dir 'output/siamese_all_sst_c_0.0001_32_128_32_15_embedding_1641839901' $gs 48 $time gpu 32 $epoch
jobs/eval/primer.sh 'all' 'sst' 'btlc' $n $in_dir $out_dir 'output/siamese_all_sst_btlc_0.0001_32_128_32_15_embedding_1641839937' $gs $mem $time gpu 32 $epoch

# different Siamese embedding layer sizes
#jobs/eval/primer.sh 'all' 'sst' 'btlc' $n $in_dir $out_dir 'output/siamese_all_sst_btlc_0.0001_32_128_3_15_embedding_1642528822' $gs $mem $time gpu 3 $epoch
#jobs/eval/primer.sh 'all' 'sst' 'btlc' $n $in_dir $out_dir 'output/siamese_all_sst_btlc_0.0001_32_128_5_15_embedding_1642529183' $gs $mem $time gpu 5 $epoch
#jobs/eval/primer.sh 'all' 'sst' 'btlc' $n $in_dir $out_dir 'output/siamese_all_sst_btlc_0.0001_32_128_10_15_embedding_1642531044' $gs $mem $time gpu 10 $epoch
#
## Civil Comments
#jobs/eval/primer.sh 'all' 'civil_comments' 'b' $n $in_dir $out_dir 'output/siamese_all_civil_comments_b_0.0001_32_128_32_15_embedding_1641333704' $gs $mem $time gpu 32 $epoch
#jobs/eval/primer.sh 'all' 'civil_comments' 'c' $n $in_dir $out_dir 'output/siamese_all_civil_comments_c_0.0001_32_128_32_15_embedding_1641333939' $gs $mem $time gpu 32 $epoch
jobs/eval/primer.sh 'all' 'civil_comments' 'btlc' $n $in_dir $out_dir 'output/siamese_all_civil_comments_btlc_0.0001_32_128_32_15_embedding_1641333954' $gs $mem $time gpu 32 $epoch
#
## different Siamese embedding layer sizes
#jobs/eval/primer.sh 'all' 'civil_comments' 'btlc' $n $in_dir $out_dir 'output/siamese_all_civil_comments_btlc_0.0001_32_128_3_15_embedding_1642529002' $gs $mem $time gpu 3 $epoch
#jobs/eval/primer.sh 'all' 'civil_comments' 'btlc' $n $in_dir $out_dir 'output/siamese_all_civil_comments_btlc_0.0001_32_128_5_15_embedding_1642530460' $gs $mem $time gpu 5 $epoch
#jobs/eval/primer.sh 'all' 'civil_comments' 'btlc' $n $in_dir $out_dir 'output/siamese_all_civil_comments_btlc_0.0001_32_128_10_15_embedding_1642531230' $gs $mem $time gpu 10 $epoch
