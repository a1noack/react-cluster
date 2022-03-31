"""
filter out samples from specified dataset and create new csv
"""
import os
import argparse

import pandas as pd

from utils import mkdir_if_dne, mkfile_if_dne, no_duplicate_index
from utils import no_duplicate_entry
from utils import drop_for_column_outside_of_values
from utils import get_pk_tuple_from_pandas_row
from utils import get_src_instance_identifier_from_pandas_row

from magic_vars import SUPPORTED_TARGET_MODEL_DATASETS
from magic_vars import UCLMR_SUPPORTED_ATTACKS
from magic_vars import SUPPORTED_ATTACKS


def check_joblib_dict(samples_dict):
    """
    Checks to make sure each example has the correct
        number of fields in the primary key; if not,
        this method attempts to fill in the missing values.

    Input
        samples_dict: dict of extracted features
            format: key - sample index, value - dict.

    Return
        samples_dict, with updated primary keys and unique IDs.
    """
    for key, d in samples_dict.items():
        assert len(d['primary_key']) == 6
    return samples_dict


def get_dataset_df(idf_dir='./original_data/whole_catted_dataset.csv'):
    """
    Read in the whole_catted_dataset.csv. Do some sanity check on it as well
    Pad useful column called pk, and another src instance identifier
    """
    odf = pd.read_csv(idf_dir)
    odf['pk'] = odf.apply(lambda row: get_pk_tuple_from_pandas_row(row), axis=1)
    odf['unique_src_instance_identifier'] = odf.apply(lambda row: get_src_instance_identifier_from_pandas_row(row), axis=1 )
    assert no_duplicate_index(odf)
    assert no_duplicate_entry(odf, 'pk')
    return odf


def main(args):

    print('--- reading data')
    df = get_dataset_df('./original_data/whole_catted_dataset.csv')
    df = df[~(df['target_model_dataset'] == 'nuclear_energy')]
    assert 'nuclear_energy' not in df['target_model_dataset'].unique()

    df = df[~(df['target_model_dataset'] == 'fnc1')]
    assert 'fnc1' not in df['target_model_dataset'].unique()

    print('--- filtering data')
    df = drop_for_column_outside_of_values(df, 'target_model_dataset', [args.dataset])
    df = drop_for_column_outside_of_values(df, 'attack_name', SUPPORTED_ATTACKS)

    out_path = os.path.join(os.getcwd(), 'filtered_data', args.dataset)
    mkdir_if_dne(out_path)

    df.to_csv(os.path.join(out_path, 'data.csv'))
    print('--- dumped filtered csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hatebase', help='same as dir. in dropbox')
    args = parser.parse_args()
    assert args.dataset in SUPPORTED_TARGET_MODEL_DATASETS
    main(args)