import joblib
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA


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
    'btlc': BERT_FEATS + TEXT_FEATS + LANG_FEATS + CLF_FEATS
}


def load_joblib_data(model, dataset, dir_path, n_dims, feats, logger=None, verbosity=0):
    """
    Load all of the joblib files in dir_path. Each of the joblib
    files should contain the extracted features for attack instances
    for one target model / domain dataset / attack method combination.
    """

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

        # only load those joblib files for the specified model and dataset
        if verbosity > 0 and logger is not None:
            logger.info(filename)
        if model not in filename and model != 'all':
            if verbosity > 0 and logger is not None:
                logger.info('\tskipping!')
            continue
        if dataset not in filename and dataset != 'all':
            if verbosity > 0 and logger is not None:
                logger.info('\tskipping!')
            continue
        if 'nuclear' in filename or 'waterloo' in filename or 'fnc1' in filename:
            continue

        instances = joblib.load(os.path.join(dir_path, filename))

        # load the information for one instance
        for num, instance in instances.items():

            # get the name of the attack that created this instance
            attack = instance['primary_key'][0]

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
                    if feature_name in samples:
                        samples[feature_name].append(feature_values)
                    else:
                        samples[feature_name] = [feature_values]
            first_sample = False

            # add instance's feature and label to the containers
            labels.append(attack)
            keys.append(instance['primary_key'])

    # make sure all 11 attacks included
    assert len(set(labels)) in [11, 12], f"Only found joblib files for {len(set(labels))} attacks!"

    # convert feature groups to numpy arrays
    for feature_name, feature_values in samples.items():
        samples[feature_name] = np.vstack(feature_values)

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


def downsample(df, n=100):
    """Downsample dataset so all attacks have the
    same n samples"""

    # if n is None, downsample until all classes have same
    # number of instances as class with least samples
    n = min(n, min(df.groupby('label').size()))

    dfs = []
    for attack_name in df.label.unique():
        new_df = df[df['label'] == attack_name].sample(n=n)
        dfs.append(new_df)

    df = pd.concat(dfs)

    return df
