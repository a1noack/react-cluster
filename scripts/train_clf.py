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
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler  # scaling isn't necessary for decision trees
import yaml


from utils import load_joblib_data, downsample


# ATTACKS = ['hotflip', 'deepwordbug', 'textbugger', 'pruthi', 'clean', 'iga_wang', 'faster_genetic', 'genetic']
ATTACKS = ['hotflip', 'deepwordbug', 'textbugger', 'pruthi', 'clean']
NOVEL_ATTACKS = ['iga_wang', 'faster_genetic', 'genetic',
                 'textbuggerv1', 'textbuggerv2', 'textbuggerv3', 'textbuggerv4',
                 'pruthiv1', 'pruthiv2', 'pruthiv3', 'pruthiv4',
                 'deepwordbugv1', 'deepwordbugv2', 'deepwordbugv3', 'deepwordbugv4']


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
    cmd_opt.add('--where_to_avg', type=str, default='input',
                help='where to average the samples when group size > 1; either "embedding" or "input"')
    cmd_opt.add('--novelty_prediction', type=int, default=0,
                help='if 1, perform novelty prediction using output entropy.')
    cmd_opt.add('--out_dir', type=str, default='output/',
                help='Where to save the results for this clustering experiment.')
    args = cmd_opt.parse_args()

    # create output directory
    out_dir = os.path.join(args.out_dir,
                           "classify_{}_{}_{}_{}_{}_nov_pred-{}".format(
                               args.model,
                               args.dataset,
                               args.features,
                               args.compress_features,
                               args.novel_attacks,
                               args.novelty_prediction
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
                                                            0, args.features, logger, keep_prob=.2,
                                                            attacks=ATTACKS + NOVEL_ATTACKS)
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
    # scaler = StandardScaler()
    samples = np.array(new_samples)

    # average samples in the input space
    if args.group_size > 1 and args.where_to_avg == 'input':
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
    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    y_train, y_test = np.array(y_train), np.array(y_test)
    # TODO: make representation of novel attacks 1/k

    # remove novel attacks from the training set
    if args.novelty_prediction:
        mask = np.isin(y_train, ATTACKS)
        X_train = X_train[mask]
        y_train = list(np.array(y_train)[mask])
        logger.info(f'Training set set of labels = {set(y_train)}')
        logger.info(f'Test set set of labels = {set(y_test)}')

    # fit the specified clustering algorithms on the data
    logger.info(f'Fitting classifier on the data.')
    clf = LGBMClassifier(n_estimators=100, max_depth=5, num_leaves=32, random_state=0, class_weight=None)
    param_grid = {'n_estimators': [50, 100],
                  'max_depth': [3, 5],
                  'num_leaves': [2, 15],
                  'boosting_type': ['gbdt']}
    # pipeline = make_pipeline(scaler, clf)
    clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, verbose=3)  # this was taking clf in the estimator parameter, so scaling wasn't happening

    # perform grid search
    s = time.time()
    clf.fit(X_train, y_train)
    logger.info(f'\tFinished fitting. {time.time() - s:.2f}s')

    if args.where_to_avg == 'input' and not args.novelty_prediction:
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

    elif args.where_to_avg == 'output' and args.novelty_prediction:
        y_test_probs = clf.predict_proba(X_test)
        logger.info(f'y_test_probs.shape = {y_test_probs.shape}')
        y_test_copy = y_test.copy()

        # generate group predictions
        scores_dict = {}
        group_entavgs = []
        group_avgents = []
        for i in range(len(y_test)):
            current_class = y_test[i]  # get sample's class
            class_probs = y_test_probs[y_test == current_class]  # get only those predictions for the current_class
            group_probs = class_probs[np.random.randint(len(class_probs), size=args.group_size)]  # randomly sample

            # calculate entropy for group by averaging predictions and getting entropy of average
            group_probs_avg = group_probs.mean(axis=0)
            assert group_probs_avg.shape[-1] == y_test_probs.shape[-1], f'Shapes not matching! {group_probs_avg.shape}'
            group_entavg = entropy(group_probs_avg, base=len(set(y_test)))
            assert group_entavg <= 1, f'Entropy is greater than one! {group_entavg}'

            # calculate entropy for group by averaging the entropies of all of the individual predictions in the group
            group_avgent = entropy(group_probs, axis=1, base=len(set(y_test))).mean()
            assert group_avgent <= 1, f'Entropy is greater than one! {group_avgent}'

            group_entavgs.append(group_entavg)
            group_avgents.append(group_avgent)
        group_entavgs = np.array(group_entavgs)
        group_avgents = np.array(group_avgents)

        # convert labels seen at training time into 'known'
        np.save(os.path.join(out_dir, 'y_test_orig.npy'), y_test)
        y_test_kno_v_unk = np.where(np.isin(y_test, ATTACKS), 'known', y_test)
        np.save(os.path.join(out_dir, 'y_test_kno_v_unk.npy'), y_test_kno_v_unk)

        # get AUROC for each attack
        for attack in NOVEL_ATTACKS:
            if attack == 'known' or attack not in set(y_test_kno_v_unk):
                continue
            group_entavgs_subset = group_entavgs[np.isin(y_test_kno_v_unk, ['known', attack])]
            group_avgents_subset = group_avgents[np.isin(y_test_kno_v_unk, ['known', attack])]
            y_test_subset = y_test_kno_v_unk[np.isin(y_test_kno_v_unk, ['known', attack])]
            is_novel = np.where(y_test_subset == 'known', 0, 1)

            auroc_entavgs = float(roc_auc_score(is_novel, group_entavgs_subset))
            auroc_avgents = float(roc_auc_score(is_novel, group_avgents_subset))

            scores_dict[attack] = [auroc_entavgs, auroc_avgents]

            logger.info(f'\t{attack} AUROC (ent. of avg.) = {auroc_entavgs:.4f}, (avg. of ent.) = {auroc_avgents:.4f}')

        # write scores to file
        with open(os.path.join(out_dir, 'scores.yml'), 'w') as outfile:
            yaml.dump(scores_dict, outfile, default_flow_style=False)

        # save predictions and entropies to files
        np.save(os.path.join(out_dir, 'y_test_probs.npy'), y_test_probs)
        np.save(os.path.join(out_dir, 'group_entavgs.npy'), group_entavgs)
        np.save(os.path.join(out_dir, 'group_avgents.npy'), group_avgents)

        # get test set accuracy numbers on known attacks
        X_test_subset = X_test[np.isin(y_test_copy, ATTACKS)]
        y_test_subset = y_test_copy[np.isin(y_test_copy, ATTACKS)]
        y_test_pred = clf.predict(X_test_subset)
        acc = accuracy_score(y_test_subset, y_test_pred)
        logger.info(f'Test set accuracy: {acc:.4f}')
        weighted_f1 = f1_score(y_test_subset, y_test_pred, average='weighted')  # get weighted F1 score
        logger.info(f'Test set weighted F1 = {weighted_f1:.4f}')
        counts = np.unique(y_test_subset, return_counts=True)  # get attack counts
        counts_dict = dict(zip(list(counts[0]), list(counts[1])))
        logger.info(f'Attack counts for test set: {counts_dict}')
        with open(os.path.join(out_dir, 'scores.csv'), 'w') as f:
            f.write(f'Accuracy = {acc:.4f}\n')
            f.write(f'Weighted F1 score = {weighted_f1:.4f}\n')
            f.write(f'Attack counts dictionary = {counts_dict}\n')
        logger.info(f'Saved accuracy and F1 scores to file.')

    if args.novelty_prediction:

        # ========== DO NOVELTY PREDICTION USING LOF ==========

        lof = LOF(novelty=True)  # novelty detection needs to be set to True
        lof.fit(X_train)

        # logger.info(f'X_test.shape = {X_test.shape}')
        y_score = lof.decision_function(X_test)  # # large values correspond to inliers
        y_score = -y_score  # flip so that large values are outliers
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())  # scale between zero and one
        # logger.info(f'y_score.shape = {y_score.shape}')
        # logger.info(f'y_test.shape = {y_test.shape}')

        # get individual AUROC per novel attack
        # logger.info(f'y_test = {y_test}')
        logger.info(f'LOF AUROCs:')
        scores_dict = {}
        for novel_attack in NOVEL_ATTACKS:
            try:
                y_score_subset = y_score[np.isin(y_test, ATTACKS + [novel_attack])]
                # logger.info(f'y_score_subset.shape = {y_score_subset.shape}')
                y_true_subset = y_test[np.isin(y_test, ATTACKS + [novel_attack])]
                # logger.info(f'y_true_subset.shape = {y_true_subset.shape}')
                y_true_subset = np.where(np.isin(y_true_subset, NOVEL_ATTACKS), 1, 0)  # 1 => novel, 0 => known
                # logger.info(f'y_true_subset.shape = {y_true_subset.shape}')

                auroc = roc_auc_score(np.array(y_true_subset), y_score_subset)
                logger.info(f'\t{novel_attack} AUROC = {auroc:.4f}')
                scores_dict[novel_attack] = float(auroc)
            except ValueError as e:
                logger.info(f'Error {e}')
                logger.info(f'Only one class present in y_true for {novel_attack}.')

        # write scores to file
        with open(os.path.join(out_dir, 'lof_scores.yml'), 'w') as outfile:
            yaml.dump(scores_dict, outfile, default_flow_style=False)

    # save model to file
    joblib.dump(clf, os.path.join(out_dir, f'lgbm.mdl'))
    logger.info(f'\tSaved model file.')

    # save training and testing data
    np.save(os.path.join(out_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
    logger.info(f'Saved training/testing data to file')

    logger.info(f'DONE.')
