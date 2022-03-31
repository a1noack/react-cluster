import joblib
import os
import random
import torch
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from model_loading import load_target_model
import sys
from magic_vars import PRIMARY_KEY_FIELDS


BERT_FEATS = ['tp_bert']
TEXT_FEATS = ['tp_bert', 'tp_avg_word_length', 'tp_is_first_word_lowercase', 'tp_num_alpha_chars',
              'tp_num_cased_letters', 'tp_num_cased_word_switches', 'tp_num_chars', 'tp_num_digits',
              'tp_num_lowercase_after_punctuation', 'tp_num_mixed_case_words', 'tp_num_multi_spaces',
              'tp_num_non_ascii', 'tp_num_punctuation', 'tp_num_single_lowercase_letters', 'tp_num_words']
LANG_FEATS = ['lm_perplexity', 'lm_proba_and_rank']
CLF_FEATS = ['tm_activation', 'tm_gradient', 'tm_posterior', 'tm_saliency']

FEATURES = {
    'b': BERT_FEATS,
    'bt': BERT_FEATS + TEXT_FEATS,
    'btl': BERT_FEATS + TEXT_FEATS + LANG_FEATS,
    'btlc': BERT_FEATS + TEXT_FEATS + LANG_FEATS + CLF_FEATS,
    'c': CLF_FEATS
}

DATASET_GROUPS = {
    'abuse': ['wikipedia', 'hatebase', 'civil_comments'],
    'sentiment': ['climate-change_waterloo', 'imdb', 'sst'],
    'all': ['wikipedia', 'hatebase', 'civil_comments', 'climate-change_waterloo', 'imdb', 'sst']
}


def load_joblib_data(model, dataset, dir_path, n_dims, feats, logger=None, verbosity=0,
                     keep_prob=.45, attacks=None, use_variants=True, detection_bert_root=None):
    """
    Load all of the joblib files in dir_path. Each of the joblib
    files should contain the extracted features for attack instances
    for one target model / domain dataset / attack method combination.
    """

    # have keep prob be large if we are only dealing with single datasets
    # if dataset not in DATASET_GROUPS:
    #     keep_prob = 1.0

    # create containers
    samples = {}
    labels = []
    keys = []  # save the primary keys so we can get original text data

    attack_counts = {}  # frequency dictionary for attacks

    # load all of the joblib files under dir_path
    thresh = inc = .1

    for ii, filename in enumerate(os.listdir(dir_path)):
        if ii / len(os.listdir(dir_path)) > thresh and logger is not None:
            logger.info(f'\tLoaded {thresh * 100: .1f}% of samples')
            thresh += inc
        first_sample = True

        # only load those joblib files for the specified model
        if model not in filename and model != 'all':
            continue

        if '.joblib' not in filename:
            continue

        # filter based on dataset
        if dataset not in filename:
            if dataset in DATASET_GROUPS:
                # check to see if dataset is in group of datasets
                included = False
                for d in DATASET_GROUPS[dataset]:
                    if d in filename:
                        included = True
                if not included:
                    continue
            else:
                continue
        if dataset == 'wikipedia' and 'wikipedia_personal' in filename:
            # prevent wikipedia personal samples from being loaded when wikipedia is the dataset
            continue
        if dataset in ['all', 'abuse', 'sentiment'] and 'climate-change' in filename:
            # if using multiple domain datasets to train, don't inlcude climate-change because of different TM shapes
            continue
        if 'nuclear' in filename or 'fnc1' in filename:
            # skip these deprecated datasets
            continue

        # filter based on attacks
        if attacks is not None and dataset != "hatebase":  # ALL hatebase attacks are in one joblib file per tgt. model
            # check to see if this attack is included
            included = False
            for a in attacks:
                if a in filename:
                    included = True
            if not included:
                continue
        if not use_variants:
            if 'v1' in filename or 'v2' in filename or 'v3' in filename:
                continue

        if dataset == 'hatebase':
            filename = 'hatebase.joblib'

        try:
            instances = joblib.load(os.path.join(dir_path, filename))
        except ValueError:
            logger.info(f'Error reading file with filename: {filename}')

        if detection_bert_root != None:
            bert_detector = load_bert_detector(detection_bert_root)
            instances = re_encode_with_bertclassifier(instances, bert_detector, logger)
            logger.info(f'\t Done with Re Encoding Features')
            del bert_detector

        # load the information for one instance
        for num, instance in instances.items():
            if keep_prob != 1 and random.random() > keep_prob:
                continue

            # get the name of the attack that created this instance
            attack = instance['primary_key'][0]

            # check to make sure required features are present
            required_features = set(FEATURES[feats].copy())
            present_features = set(list(instance['deliverable'].keys()))

            if not required_features.issubset(present_features):
                continue

            # update frequency dictionary
            if attack not in attack_counts:
                attack_counts[attack] = 1
            else:
                attack_counts[attack] += 1

            for feature_name, feature_values in instance['deliverable'].items():
                if feature_name in FEATURES[feats]:
                    # concatenate all of the feature values for this sample
                    if verbosity > 0 and logger is not None and first_sample:
                        logger.info(f'\t{feature_name}: {feature_values.shape[1]}')
                    feature_values = feature_values.flatten()
                    if feature_name == 'tm_posterior' and len(feature_values) > 1:
                        feature_values = np.array([feature_values[-1]])
                    if feature_name in samples:
                        samples[feature_name].append(feature_values)
                    else:
                        samples[feature_name] = [feature_values]
            first_sample = False

            # add instance's feature and label to the containers
            labels.append(attack)
            keys.append(instance['primary_key'])

    # make sure all 11 attacks included
    # assert len(set(labels)) in [11, 12], f"Only found joblib files for {len(set(labels))} attacks!"

    # convert feature groups to numpy arrays
    for feature_name, feature_values in samples.items():
        try:
            samples[feature_name] = np.vstack(feature_values)
        except ValueError:
            logger.info(f'Shape mismatches for feature name = {feature_name}')
            shapes = {}
            for i, sample in enumerate(feature_values):
                if sample.shape not in shapes:
                    shapes[sample.shape] = [1, {keys[i][4]: 1}]
                else:
                    shapes[sample.shape][0] += 1
                    if keys[i][4] not in shapes[sample.shape][1]:
                        shapes[sample.shape][1][keys[i][4]] = 1
                    else:
                        shapes[sample.shape][1][keys[i][4]] += 1

            logger.info(f'Shapes counts: {shapes}')

    # if n_dims is not 0, compress number of dimensions for this feature group
    if n_dims != 0:
        for feature_name, feature_values in samples.items():
            # only compress certain groups of features
            if feature_name in ['tp_bert', 'lm_proba_and_rank', 'tm_activation', 'tm_gradient'] and \
                    feature_values.shape[1] > n_dims:
                pca = PCA(n_components=n_dims)
                samples[feature_name] = pca.fit_transform(feature_values)
            else:
                samples[feature_name] = feature_values

    # merge all feature groups into one feature vector
    all_data = []
    for feature_name, feature_values in samples.items():
        all_data.append(feature_values)
    samples = np.concatenate(all_data, axis=1)

    return samples, labels, keys, attack_counts


def downsample(df, n=100, cut_to_min_group=False):
    """Downsample dataset so all attacks have the
    same n samples"""

    # if n is None, downsample until all classes have same
    # number of instances as class with least samples
    if cut_to_min_group:
        n = min(n, min(df.groupby('label').size()))

    dfs = []
    for attack_name in df.label.unique():
        df_ = df[df['label'] == attack_name]
        if len(df_) > n:
            df_ = df_.sample(n=n)
        dfs.append(df_)

    df = pd.concat(dfs)

    return df


def plot_tsne(data, labels, out_dir, ppl=20):
    """Takes embedded samples and creates visualization using t-SNE"""
    # perform PCA on dataframe before doing t-SNE to reduce computational burden (this is recommended)
    if data.shape[1] > 50:
        pca = PCA(n_components=50)
        pca_result = pca.fit_transform(data)
    else:
        pca_result = data

    # fit t-SNE on output produced by PCA
    tsne = TSNE(n_components=2, verbose=0, perplexity=ppl, n_iter=10000)
    tsne_results = tsne.fit_transform(pca_result)

    # set graph size
    size = 15
    diff = 0
    params = {
        'font.size': size,
        'axes.titlesize': size,
        'axes.labelsize': size - diff,
        'xtick.labelsize': size - diff,
        'ytick.labelsize': size - diff,
        'figure.figsize': (18, 11),
    }
    plt.rcParams.update(params)

    # create dataframe for plotting
    _df = pd.DataFrame()
    _df['tsne-one'] = tsne_results[:, 0]
    _df['tsne-two'] = tsne_results[:, 1]
    _df['label'] = labels
    _df = _df.sort_values(by=['label'])

    # choose colors for attacks
    palette = {'genetic': "gray",
               'deepwordbug': "orange",
               'clean': 'green',
               'pruthi': 'indigo',
               'hotflip': 'red',
               'iga_wang': 'black',
               'faster_genetic': 'magenta',
               'textbugger': 'violet',
               'pruthiv1': "yellow",
               'pruthiv2': 'blue',
               'pruthiv3': 'cyan',
               'deepwordbugv1': 'lightblue',
               'deepwordbugv2': 'lightgreen',
               'deepwordbugv3': 'pink',
               'textbuggerv1': 'maroon',
               'textbuggerv2': 'purple',
               'textbuggerv3': 'beige'}

    # plot the figure
    plt.figure(figsize=(18, 11))
    try:
        sns.scatterplot(
            x="tsne-one", y="tsne-two",
            hue="label",
            palette=palette,
            data=_df,
            legend="full",
            alpha=0.8
        )
    except ValueError:
        sns.scatterplot(
            x="tsne-one", y="tsne-two",
            hue="label",
            palette=sns.color_palette("Paired", len(set(labels))),
            data=_df,
            legend="full",
            alpha=0.8
        )
    plt.title(f't-SNE with perplexity = {ppl}')
    plt.savefig(os.path.join(out_dir, f'tsne{ppl}.png'))
    plt.close()

def load_bert_detector(bert_detection_model_root):

    if 'clean_vs_all' in bert_detection_model_root:
        num_labels = 2
    else:
        num_labels = len(pd.read_csv(
            os.path.join(bert_detection_model_root, 'train.csv')
        )['attack_name'].unique())
        
    model_metadata = np.load(
        os.path.join(bert_detection_model_root, 'BERT_DETECTION/bert/results.npy'),
        allow_pickle = True
    ).item()
    max_seq_len = model_metadata['max_seq_len']
    target_model_name = 'bert'
    dir_target_model = os.path.join(bert_detection_model_root, 'BERT_DETECTION/bert')
    bert_detector = load_target_model(
        target_model_name, 
        dir_target_model,  
        max_seq_len ,
        torch.device('cuda'), 
        num_labels
    )
    return bert_detector

def re_encode_with_bertclassifier(data_dict, bert_detector, logger=None):
    with torch.no_grad():
        for key in tqdm(list(data_dict.keys())):
            instance = data_dict[key]
            logger.info(f'\t Keys: {instance.keys()}')
            deliverable = instance['deliverable']
            logger.info(f'\t Deliverable Keys: {deliverable.keys()}')
            # logger.info(f'\t Instance: {instance}')
            old_bert_shape = instance['deliverable']['tp_bert'].shape
            # logger.info(f'\t OLD TP BERT SHAPE: {old_bert_shape}')
            new_tp_bert = bert_detector.get_cls_repr([instance['perturbed_text']]).cpu().numpy()
            assert new_tp_bert.shape == old_bert_shape
            instance['deliverable']['tp_bert'] = new_tp_bert 
    return data_dict

# pandas operations

def get_pk_tuple_from_pandas_row(pandas_row):
    return tuple([pandas_row[_PK] for _PK in PRIMARY_KEY_FIELDS])

def drop_for_column_outside_of_values(df, column_name, values):
    return df[df[column_name].isin(values)]

def drop_for_column_inside_values(df, column_name, values):
    return df[~df[column_name].isin(values)]

def no_duplicate_entry(df, column_name):
    return df[column_name].is_unique

def no_duplicate_index(df):
    return df.index.is_unique

def get_src_instance_identifier_from_pandas_row(pandas_row):
    """
    returns an tuple idenfier unique to src instance

    e.g. #777 sentence in SST is attacked by 7 attacks, those will share this identifier
    """
    return tuple([pandas_row['target_model_dataset'], pandas_row['test_index']])

# file I/O

def grab_joblibs(root_dir):
    """
    grab all joblibs in a root dir
    """
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(root_dir):
        for file in f:
            if 'joblib' in file:
                files.append(os.path.join(r, file))
    return files

def load_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data

def mkfile_if_dne(fpath):
    if not os.path.exists(os.path.dirname(fpath)):
        print('mkfile warning: making fdir since DNE: ',fpath)
        os.makedirs(os.path.dirname(fpath))
   
def mkdir_if_dne(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath)

# hashing

def get_pk_tuple(df, index):
    _PK = sorted(['scenario', 'target_model_dataset', 'target_model',
                  'attack_toolchain', 'attack_name', 'test_index'])
    _PK = [df.at[index, i] for i in _PK]
    return _PK