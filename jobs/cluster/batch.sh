n=500  # the number of samples to keep in each class
n_dim=0  # the number of dimensions to compress feature vectors down to

for model in 'bert' 'roberta' 'xlnet'; do
  for dataset in 'climate-change_waterloo' 'sst' 'wikipedia' 'hatebase' 'civil_comments'; do
    for features in 'b' 'bt' 'btl' 'btlc'; do
      ./jobs/cluster/primer.sh $model $dataset $n $features $n_dim
    done
  done
done

for model in 'bert' 'roberta' 'xlnet'; do
  for dataset in 'climate-change_waterloo' 'sst' 'wikipedia' 'hatebase' 'civil_comments'; do
    for n_dim in 2 8 32; do
      ./jobs/cluster/primer.sh $model $dataset $n 'btlc' $n_dim
    done
  done
done
