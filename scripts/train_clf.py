"""This script is for training a boosted tree classifier in order to provide
intuition about where the upper bound on clustering ability might be"""
import joblib
import logging
import os
from pathlib import Path
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import configargparse
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from utils import load_joblib_data, downsample


# ATTACKS = ['hotflip', 'deepwordbug', 'textbugger', 'pruthi', 'clean', 'iga_wang', 'faster_genetic', 'genetic']
ATTACKS = ['hotflip', 'deepwordbug', 'textbugger', 'pruthi', 'clean']
NOVEL_ATTACKS = ['iga_wang', 'faster_genetic', 'genetic']


if __name__ == '__main__':
    # create logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    # parse the command line arguments
    cmd_opt = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    cmd_opt.add('--model', type=str, default='all', help="Name of target model for which the attacks were created.")
    cmd_opt.add('--dataset', type=str, default='hatebase', help="Name of dataset used to create the attacks.")
    cmd_opt.add('--features', type=str, default='btlc', help='feature groups to include: b, bt, btl, or btlc.')
    cmd_opt.add('--compress_features', type=int, default=0,
                help='compress all features with more than one dimension down to have at most this many dimensions.')
    cmd_opt.add('--group_size', type=int, default=5, help='group size for averaging in input space')
    cmd_opt.add('--n', type=int, default=0, help="The number of attacks to keep per attack method.")
    cmd_opt.add('--novel_attacks', type=int, help='If 1, use novel attack methods IN ADDITION to base elite attacks')
    cmd_opt.add('--in_dir', type=str, default='data/',
                help='The path to the folder containing the joblib files for the extracted samples.')
    cmd_opt.add('--out_dir', type=str, default='output/',
                help='Where to save the results for this clustering experiment.')
    args = cmd_opt.parse_args()

    # create output directory
    out_dir = os.path.join(args.out_dir,
                           "classify_{}_{}_{}_{}_{}".format(
                               args.model,
                               args.dataset,
                               args.features,
                               args.compress_features,
                               args.novel_attacks
                           ))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_file_handler = logging.FileHandler(os.path.join(out_dir, 'output.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    output_file_handler.setFormatter(formatter)
    logger.addHandler(output_file_handler)

    # load the extracted feature data
    logger.info(f'Training classifer on attacks for {args.model}/{args.dataset}.')
    s = time.time()
    logger.info(f'Loading the data. Features: {args.features}')
    dir_path = os.path.join(args.in_dir, 'extracted_features')  # , f'{args.model}_{args.dataset}')

    samples, labels, keys, attack_counts = load_joblib_data(args.model, args.dataset, dir_path,
                                                            args.compress_features, args.features, logger)

    # remove attacks that aren't in new, smaller group
    if args.novel_attacks:
        attacks_to_use = ATTACKS + NOVEL_ATTACKS
    else:
        attacks_to_use = ATTACKS
    new_samples, new_labels = [], []
    for (sample, label) in zip(samples, labels):
        if label in attacks_to_use:
            new_samples.append(sample)
            new_labels.append(label)
    logger.info(f'Loaded the data. {time.time() - s:.2f}s.')

    # # load original data containing strings and filter based on target model and dataset
    # df_orig = pd.read_csv(os.path.join(args.in_dir, 'original_data', 'whole_catted_dataset.csv'))
    # df_orig = df_orig[(df_orig.target_model == args.model) & (df_orig.target_model_dataset == args.dataset)]
    # logger.info(f'Loaded original textual data.')

    # scale the data
    scaler = StandardScaler()
    samples = np.array(new_samples)

    # average samples in the input space
    if args.group_size > 1:
        samples_avg = []
        labels_avg = []
        for attack in set(new_labels):
            samples_ = samples[np.array(new_labels) == attack]
            n_keep = len(samples_) - (len(samples_) % args.group_size)  # nearest no. of samples divisible by group size
            samples_ = samples_[:n_keep]
            samples_ = samples_.reshape(-1, args.group_size, samples_.shape[-1]).mean(axis=1)
            labels_ = [attack] * len(samples_)

            samples_avg.append(samples_)
            labels_avg.extend(labels_)

        samples = np.vstack(samples_avg)
        new_labels = labels_avg

    df = pd.DataFrame(samples)
    df['label'] = new_labels
    logger.info(f'Attack counts after averaging: {dict(df.label.value_counts())}')

    # downsample for clustering efficiency
    if args.n != 0:
        df = downsample(df, args.n)
        logger.info(f'Downsampled data so all classes have at most {args.n} samples.')
    labels = list(df['label']).copy()
    df = df.drop(['label'], axis=1)
    n_attack_methods = len(set(labels))

    attack_counts = {}
    for label in labels:
        if label in attack_counts:
            attack_counts[label] += 1
        else:
            attack_counts[label] = 0
    logger.info(f'Attack counts: {attack_counts}')

    # build stratified train and test splits
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=.2, stratify=labels)

    # fit the specified clustering algorithms on the data
    logger.info(f'Fitting classifier on the data.')
    clf = LGBMClassifier(n_estimators=100, max_depth=5, num_leaves=32, random_state=0, class_weight=None)
    param_grid = {'n_estimators': [50, 100],
                  'max_depth': [3, 5],
                  'num_leaves': [2, 15],
                  'boosting_type': ['gbdt']}
    pipeline = make_pipeline(scaler, clf)
    clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, verbose=3)

    # perform grid search
    s = time.time()
    clf.fit(X_train, y_train)
    logger.info(f'\tFinished fitting. {time.time() - s:.2f}s')

    y_test_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    logger.info(f'Test set accuracy: {acc:.4f}')
    weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')  # get weighted F1 score
    logger.info(f'Test set weighted F1 = {weighted_f1:.4f}')
    counts = np.unique(y_test, return_counts=True)  # get attack counts
    counts_dict = dict(zip(list(counts[0]), list(counts[1])))
    logger.info(f'Attack counts for test set: {counts_dict}')
    with open(os.path.join(out_dir, 'scores.csv'), 'w') as f:
        f.write(f'Accuracy = {acc:.4f}\n')
        f.write(f'Weighted F1 score = {weighted_f1:.4f}\n')
        f.write(f'Attack counts dictionary = {counts_dict}\n')
    logger.info(f'Saved accuracy and F1 scores to file.')

    # # save predictions, labels, and model
    joblib.dump(clf, os.path.join(out_dir, f'lgbm.mdl'))
    logger.info(f'\tSaved model file.')

    logger.info(f'DONE.')
