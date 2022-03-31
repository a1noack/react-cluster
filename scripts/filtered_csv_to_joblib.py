"""
Make an experiment, i.e., construct joblib train.joblib and test.joblib based on
train.csv and test.csv in dir.
"""
import os
import joblib
import resource
import argparse
from collections import defaultdict

import pandas as pd

from utils import grab_joblibs
from utils import load_json
from utils import get_pk_tuple
from utils import no_duplicate_index
from magic_vars import SUPPORTED_TARGET_MODELS
from magic_vars import SUPPORTED_TARGET_MODEL_DATASETS
from magic_vars import SUPPORTED_ATTACKS

from filter_whole_catted_dataset import check_joblib_dict


def load_known_instances(rootdir_with_joblib_file, target_model_dataset, lazy_loading):
    """
    load all joblib from some root dir
    """

    assert target_model_dataset in SUPPORTED_TARGET_MODEL_DATASETS

    print(f'\n--- loading known instances from {rootdir_with_joblib_file}')
    print(f'\n--- target_model_dataset: {target_model_dataset}')
    all_jblb = grab_joblibs(rootdir_with_joblib_file)

    if lazy_loading:
        all_jblb = [jb for jb in all_jblb if target_model_dataset + '_' in jb]

    print(f'--- No. joblib files of varied size to read: {len(all_jblb):,}')

    known_samples = {}
    for i, jblb in enumerate(all_jblb):
        print(f'[{i}] {jblb}')  # to avoid name collision

        holder = check_joblib_dict(joblib.load(jblb))

        for idx in holder.keys():
            instance = holder[idx]
            pk = instance['primary_key']
            pk = tuple(pk)
            assert isinstance(pk, tuple)
            known_samples[pk] = instance

    print(f'\ndone, no. unique instances with extracted features: {len(known_samples):,}')
    return known_samples


def df_to_instance_subset(df, known_samples):
    """
    Use df to get a subset of instances from
        <known_samples> as returned by build_known_instances above
    """
    assert no_duplicate_index(df)

    out = {}
    no_repr_count = 0

    for idx in df.index:
        pk = get_pk_tuple(df, idx)
        pk = tuple(pk)

        if pk in known_samples:
            out[pk] = known_samples[pk]
            del known_samples[pk]

            out[pk]['attack_name'] = pk[0]  # ok so this is because react convention, pk[0] is attack_name
            out[pk]['binary_label'] = 'clean' if pk[0] == 'clean' else 'perturbed'

            try:  # just in case we need the actual text
                out[pk]['perturbed_text'] = df.at[idx, 'perturbed_text']
                out[pk]['original_text'] = df.at[idx, 'original_text']

            except:
                pass

        else:
            no_repr_count += 1

    print(f'    cannot find repr. for {no_repr_count:,} / {len(df):,} instances')

    return out


def main(args, out_dir):

    print('making exp into... ', out_dir)
    lazy_loading = True
   
    known_instances = load_known_instances('./data/extracted_features',
                                            target_model_dataset=args.dataset,
                                            lazy_loading=lazy_loading)

    print('\n--- creating joblib')
    df = pd.read_csv(os.path.join(out_dir, 'data.csv'))
    data = df_to_instance_subset(df, known_instances)
    joblib.dump(data, os.path.join(out_dir, 'data.joblib'))

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f'max_rss: {max_rss:,}b')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--in_dir', type=str, help='directory off a distributed exp to be made')
    parser.add_argument('--dataset', type=str, help='same as dir. in dropbox')
    args = parser.parse_args()

    out_dir = os.path.join(os.getcwd(), 'filtered_data', args.dataset)
    assert os.path.exists(os.path.join(out_dir, 'data.csv')), 'cannot find csv in ' + out_dir

    main(args, out_dir)