"""This script is for evaluating the clustering ability of a
Siamese network that has already been trained"""
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
from scipy.spatial.distance import cdist
from scipy.stats import percentileofscore, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import LocalOutlierFactor as LOF
import torch
from torch.utils.data import DataLoader
import yaml

from models import SiameseNet, NormalDataset, SiameseDataset
from utils import load_joblib_data, downsample, plot_tsne

# use only the attacks below when determining cluster centers
ATTACKS_CONSIDERED = ['hotflip', 'deepwordbug', 'textbugger', 'pruthi', 'clean']
HELD_OUT = ['iga_wang', 'faster_genetic', 'genetic',
            'textbuggerv1', 'textbuggerv2', 'textbuggerv3', 'textbuggerv4',
            'pruthiv1', 'pruthiv2', 'pruthiv3', 'pruthiv4',
            'deepwordbugv1', 'deepwordbugv2', 'deepwordbugv3', 'deepwordbugv4']


if __name__ == '__main__':

    # ========== CREATE LOGGER AND PARSE ARGUMENTS ==========

    # create logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    # parse the command line arguments
    cmd_opt = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    cmd_opt.add('--model', type=str, default='roberta', help="Name of target model for which the attacks were created.")
    cmd_opt.add('--dataset', type=str, default='sst', help="Name of dataset used to create the attacks.")
    cmd_opt.add('--features', type=str, default='btlc', help='feature groups to include: b, c, bt, btl, or btlc.')
    cmd_opt.add('--n', type=int, default=100, help="The number of attacks to keep per attack method.")
    cmd_opt.add('--in_dir', type=str,
                help='The path to the folder containing the joblib files for the extracted samples.')
    cmd_opt.add('--out_dir', type=str, help='Where to save the results for this clustering experiment.')
    cmd_opt.add('--siamese_net_dir', type=str,
                help='the path to where the Siamese network AND the training sample keys AND scaler is saved')
    cmd_opt.add('--group_size', type=int, default=0, help='size of the groups of samples to embed with the Siamese net')
    args = cmd_opt.parse_args()

    # create output directory
    si_emb_size = int(args.siamese_net_dir.split('_')[-4])
    out_dir = os.path.join(args.out_dir, "eval_siamese_{}_{}_{}_{}_{}".format(
                               args.model, args.dataset, args.features, int(time.time()), si_emb_size))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_file_handler = logging.FileHandler(os.path.join(out_dir, 'output.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    output_file_handler.setFormatter(formatter)
    logger.addHandler(output_file_handler)

    # ========== LOAD THE DATA ==========

    # load the extracted feature data
    logger.info(f'Evaluating Siamese network at: {args.siamese_net_dir}.')
    logger.info(f'Experiment arguments: {args}')
    s = time.time()
    logger.info(f'Features to load for each sample: {args.features}')
    logger.info(f'Loading the data...')
    dir_path = os.path.join(args.in_dir, 'extracted_features')
    samples, labels, keys, attack_counts = load_joblib_data(args.model, args.dataset, dir_path,
                                                            0, args.features, logger, keep_prob=.2,
                                                            attacks=ATTACKS_CONSIDERED+HELD_OUT)
    logger.info(f'Loaded the data. {time.time() - s:.2f}s.')
    logger.info(f'Attack counts: {attack_counts}')

    # create dataframe from the samples and labels
    df = pd.DataFrame(np.asarray(samples))  # np.asarray instead of np.array, because np.array copies data
    df['label'] = labels.copy()
    df['key'] = keys.copy()

    # remove samples that aren't in new, smaller group
    df = df[df.label.isin(ATTACKS_CONSIDERED+HELD_OUT)]
    logger.info(f'Removed all samples not produced by one of the following attack methods: {ATTACKS_CONSIDERED+HELD_OUT}')
    logger.info(f'Attack counts now: {dict(df.label.value_counts())}')

    # downsample for clustering efficiency
    if args.n != 0:
        df = downsample(df, args.n)
        df = df.reset_index(drop=True)
        logger.info(f'Downsampled data so all classes have at most {args.n} samples.')
        logger.info(f'Attack counts now: {dict(df.label.value_counts())}')
    else:
        logger.info(f'No downsampling performed.')

    # ========== COMPRESS THE SAMPLES WITH THE SIAMESE NET ==========

    # set device
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')

    # load saved Siamese network
    with open(os.path.join(args.siamese_net_dir, 'siamese_net_args.yml'), 'r') as infile:
        net_args = yaml.safe_load(infile)
        logger.info(f'Loaded Siamese network architecture args.')
    net = SiameseNet(in_size=df.shape[1]-2, hid_size=net_args['hid_size'],
                     out_size=net_args['out_size'], n_layer=net_args['n_layer'],
                     drop_prob=net_args['drop_prob'], group_size=args.group_size)
    net.load_state_dict(torch.load(os.path.join(args.siamese_net_dir, 'siamese_net.pt'), map_location=device))
    net.to(device)
    logger.info(f'Loaded saved model weights.')

    # load the StandardScaler
    siamese_scaler = joblib.load(os.path.join(args.siamese_net_dir, 'scaler.pkl'))
    logger.info(f'Loaded fitted scaler.')

    try:
        # load training and validation set keys
        train_keys = pd.read_csv(os.path.join(args.siamese_net_dir, 'train_keys.csv'))
        val_keys = pd.read_csv(os.path.join(args.siamese_net_dir, 'val_keys.csv'))
        used_keys = list(pd.concat([train_keys, val_keys])['key'])
        logger.info(f'Loaded keys for samples that were used for training Siamese network.')
    except FileNotFoundError:
        logger.info(f'Unable to load keys used for training so no data filtering was performed.')

    try:
        # filter out those samples in df that were used to train and validate the Siamese network
        s = time.time()
        to_drop = []
        for i, key in enumerate(df['key']):
            if key in used_keys:
                to_drop.append(i)
        logger.info(f'Found all samples used for training Siamese network.')
        df = df.drop(to_drop)
        # couldn't get the below to work when the keys were in lists... stupid
        # df = df[~df.key.isin(used_keys.key)].reset_index(drop=True)
        logger.info(f'Removed samples that were used to train Siamese network. {time.time() - s:.2f}s')
    except MemoryError as me:
        logger.info(f'Failed with a memory error when i = {i}')

    # create normal dataset for feeding samples into network in groups
    df_orig = df
    dataset_test = NormalDataset(df.copy(), group_size=args.group_size, return_keys=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # compress the samples with one of the twins
    embedded_groups = []
    non_embedded_groups = []
    non_embedded_labels = []
    labels = []
    keys = []
    s = time.time()
    for batch_id, (x, y, k) in enumerate(dataloader_test, 1):
        # logger.info(f'x.shape = {x.shape}')

        # if multiple samples in group, reshape inputs so that they can pass through network nicely
        orig_shape = x.shape
        if args.group_size > 1:
            x = x.view(-1, orig_shape[-1])  # x's shape after this line should be (group_size, n_features)
            non_embedded_groups.append(x)
            non_embedded_labels.append(np.array([y]*x.shape[0]))

        # scale and pass through network
        x = torch.Tensor(siamese_scaler.transform(x)).to(device)  # scale the samples
        x_ = net.forward_one(x)  # embed the samples

        # reshape inputs so that they can pass through network nicely
        if args.group_size > 1:
            x_ = x_.view(orig_shape[0], args.group_size, net.out_size)
            x_ = x_.mean(dim=1)  # take mean of embedded samples within each group

        embedded_groups.append(x_)
        labels.append(y)
        keys.append(k)

    logger.info(f'Embedded all remaining samples in groups. {time.time() - s:.2f}s')

    # ========== SPLIT DATA INTO TRAIN AND TEST SETS ==========

    # concatenate the samples
    samples = torch.cat(embedded_groups, dim=0).detach().cpu().numpy()
    labels = np.hstack(labels)
    keys = np.array(keys)

    unemb_samples = torch.cat(non_embedded_groups).detach().cpu().numpy()
    unemb_labels = np.hstack(non_embedded_labels)

    # save these for the demo
    torch.save(samples, os.path.join(out_dir, 'group_samples.pt'))
    np.save(os.path.join(out_dir, 'group_labels.npy'), labels)
    np.save(os.path.join(out_dir, 'group_keys.npy'), keys)
    logger.info(f'Saved t-SNE input.')

    # perform t-SNE on compressed samples
    s = time.time()
    for ppl in [2, 5, 10, 30, 50, 100]:  # try full range of perplexity values
        plot_tsne(samples, labels, out_dir, ppl)
    logger.info(f'Plotted embedded samples using t-SNE. {time.time() - s:.2f}s')

    # put samples into dataframe
    s = time.time()
    df = pd.DataFrame(samples)
    df['label'] = labels.copy()
    # df['key'] = keys

    # remove variants from the "training data"
    df_known = df[df.label.isin(ATTACKS_CONSIDERED)]
    df_train, df_known_test = train_test_split(df_known, test_size=.3)
    df_train, df_known_test = df_train.reset_index(drop=True), df_known_test.reset_index(drop=True)
    df_unknown = df[df.label.isin(HELD_OUT)]
    df_test = pd.concat([df_known_test, df_unknown]).reset_index(drop=True)

    # split into x and y
    # df_x_train, y_train, keys_train = df_train.drop(['label', 'key'], axis=1), df_train['label'], df_train['key']
    # df_x_test, y_test, keys_test = df_test.drop(['label', 'key'], axis=1), df_test['label'], df_test['key']
    df_x_train, y_train = df_train.drop(['label'], axis=1), df_train['label']
    df_x_test, y_test = df_test.drop(['label'], axis=1), df_test['label']

    logger.info(f'Split into train and test sets. {time.time() - s:.2f}s')

    # ========== TRAIN LGBM CLASSIFIER ON KNOWN ATTACKS AND EVAL ENTROPY ON NOVEL ATTACKS ==========

    df = pd.DataFrame(unemb_samples)

    un_df_x_train, un_df_x_test, un_y_train, un_y_test = train_test_split(df, unemb_labels,
                                                                          test_size=.2, stratify=unemb_labels)
    un_df_x_train, un_df_x_test = un_df_x_train.reset_index(drop=True), un_df_x_test.reset_index(drop=True)
    un_y_train, un_y_test = np.array(un_y_train), np.array(un_y_test)

    # fit the specified clustering algorithms on the data
    logger.info(f'Fitting classifier on the data.')
    clf = LGBMClassifier(n_estimators=100, max_depth=5, num_leaves=32, random_state=0, class_weight=None)
    param_grid = {'n_estimators': [50, 100],
                  'max_depth': [3, 5],
                  'num_leaves': [2, 15],
                  'boosting_type': ['gbdt']}
    # pipeline = make_pipeline(scaler, clf)
    clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3,
                       verbose=3)  # this was taking clf in the estimator parameter, so scaling wasn't happening

    # perform grid search
    s = time.time()
    clf.fit(un_df_x_train, np.array(un_y_train))
    logger.info(f'\tFinished fitting. {time.time() - s:.2f}s')

    y_test_probs = clf.predict_proba(un_df_x_test)
    logger.info(f'y_test_probs.shape = {y_test_probs.shape}')
    y_test_copy = np.array(un_y_test).copy()

    # generate group predictions
    scores_dict = {}
    group_entavgs = []
    group_avgents = []
    for i in range(len(y_test_copy)):
        current_class = y_test_copy[i]  # get sample's class
        class_probs = y_test_probs[y_test_copy == current_class]  # get only those predictions for the current_class
        group_probs = class_probs[np.random.randint(len(class_probs), size=args.group_size)]  # randomly sample

        # calculate entropy for group by averaging predictions and getting entropy of average
        group_probs_avg = group_probs.mean(axis=0)
        assert group_probs_avg.shape[-1] == y_test_probs.shape[-1], f'Shapes not matching! {group_probs_avg.shape}'
        group_entavg = entropy(group_probs_avg, base=len(set(y_test_copy)))
        assert group_entavg <= 1, f'Entropy is greater than one! {group_entavg}'

        # calculate entropy for group by averaging the entropies of all of the individual predictions in the group
        group_avgent = entropy(group_probs, axis=1, base=len(set(y_test_copy))).mean()
        assert group_avgent <= 1, f'Entropy is greater than one! {group_avgent}'

        group_entavgs.append(group_entavg)
        group_avgents.append(group_avgent)
    group_entavgs = np.array(group_entavgs)
    group_avgents = np.array(group_avgents)

    # convert labels seen at training time into 'known'
    np.save(os.path.join(out_dir, 'y_test_orig.npy'), y_test_copy)
    y_test_kno_v_unk = np.where(np.isin(y_test_copy, ATTACKS_CONSIDERED), 'known', y_test_copy)
    np.save(os.path.join(out_dir, 'y_test_kno_v_unk.npy'), y_test_kno_v_unk)

    # get AUROC for each attack
    for attack in HELD_OUT:
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
    X_test_subset = un_df_x_test[np.isin(y_test_copy, ATTACKS_CONSIDERED)]
    y_test_subset = y_test_copy[np.isin(y_test_copy, ATTACKS_CONSIDERED)]
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

    # ========== GENERATE MEAN EMBEDDINGS FOR KNOWN ATTACKS ==========

    # get the mean embedding
    s = time.time()
    attack_mean_and_distances = {}
    for known_attack in ATTACKS_CONSIDERED:  # this WAS incorrectly (I think) set to HELD_OUT_ATTACKS

        # get attacks in training set that were created by the known_attack attack method
        known_attack_samples = df_x_train[y_train == known_attack]
        if len(known_attack_samples) == 0:
            continue
        known_attack_mean_embedding = known_attack_samples.to_numpy().mean(axis=0)  # get mean embedding for this attack method

        # get distance from this mean embedding to every training sample
        distances = cdist(df_x_train, known_attack_mean_embedding.reshape(1, -1)).flatten()
        # distances.sort()

        attack_mean_and_distances[known_attack] = (known_attack_mean_embedding, distances)

    logger.info(f'Computed mean embeddings and distances for {list(attack_mean_and_distances.keys())}. {time.time() - s:.2f}s')

    # ========== ASSIGN TRAINING SAMPLES TO MEAN EMBEDDINGS ==========

    # for each sample in the training set, determine which mean embedding it's closest to
    s = time.time()
    attack_mean_and_distances_filtered = {a: list([None, list([])]) for a in attack_mean_and_distances}
    for i in range(len(df_x_train)):

        # determine attack method with closest mean embedding
        closest_attack = None
        smallest_dist = np.inf
        for attack, (mean_embedding, distances) in attack_mean_and_distances.items():
            distance = distances[i]
            if distance < smallest_dist:
                closest_attack = attack
                smallest_dist = distance

        # copy mean embedding over to new dictionary
        attack_mean_and_distances_filtered[closest_attack][0] = attack_mean_and_distances[closest_attack][0]

        # append the distance from this sample to the mean embedding that it is closest
        if closest_attack in attack_mean_and_distances_filtered:
            attack_mean_and_distances_filtered[closest_attack][1].append(smallest_dist)
        else:
            raise ValueError(f'Closest Attack is {closest_attack}!')

    logger.info(f'Assigned training samples to mean embeddings. {time.time() - s:.2f}s')

    # ========== PREDICT ON TEST DATA ==========

    s = time.time()
    logger.info(f'Evaluating mean embeddings for prediction:')
    scores_dict = {}

    for novel_attack in HELD_OUT:
        try:
            # filter out those attacks produced by novel attacks that aren't "novel_attack"
            df_x_test_subset = df_x_test[y_test.isin(ATTACKS_CONSIDERED + [novel_attack])]
            y_test_subset = y_test[y_test.isin(ATTACKS_CONSIDERED + [novel_attack])]
            # keys_test_subset = keys_test[y_test.isin(ATTACKS_CONSIDERED + [novel_attack])]

            # for each sample in the test set, see how close it is (in percentile terms) to each mean embedding
            smallest_percentiles = []
            is_novel = []
            closest_attacks = []
            closest_attacks_dists = []
            smallest_distances = []
            # keys_to_save = []
            for i, sample in df_x_test_subset.iterrows():
                sample = sample.to_numpy()

                # determine distance percentiles for each attack method
                smallest_percentile = 1.0
                closest_attack = None
                smallest_dist = np.inf
                closest_attack_dist = None
                for attack, (mean_embedding, distances) in attack_mean_and_distances_filtered.items():
                    distance = np.linalg.norm(sample - mean_embedding)
                    percentile = percentileofscore(distances, distance) / 100

                    # get closest attack in terms of percentile of distances
                    if percentile < smallest_percentile:
                        smallest_percentile = percentile
                        closest_attack = attack

                    # get closest attack in terms of absolute distance
                    if distance < smallest_dist:
                        smallest_dist = distance
                        closest_attack_dist = attack

                closest_attacks.append(closest_attack)
                closest_attacks_dists.append(closest_attack_dist)
                smallest_distances.append(smallest_dist)

                # get the label for this sample
                is_novel.append(0 if y_test_subset[i] in ATTACKS_CONSIDERED else 1)
                smallest_percentiles.append(smallest_percentile)
                # keys_to_save.append(keys_test_subset[i])

            # convert to numpy arrays
            is_novel = np.array(is_novel)
            smallest_percentiles = np.array(smallest_percentiles)
            closest_attacks = np.array(closest_attacks)
            closest_attacks_dists = np.array(closest_attacks_dists)
            smallest_distances = np.array(smallest_distances)
            # keys_to_save = np.array(keys_to_save)

            # save outputs for this novel attack
            np.save(os.path.join(out_dir, f'is_novel_{novel_attack}.npy'), is_novel)
            np.save(os.path.join(out_dir, f'smallest_percentiles_{novel_attack}.npy'), smallest_percentiles)
            np.save(os.path.join(out_dir, f'closest_attacks_{novel_attack}.npy'), closest_attacks)
            np.save(os.path.join(out_dir, f'closest_attacks_dists_{novel_attack}.npy'), closest_attacks_dists)
            np.save(os.path.join(out_dir, f'smallest_distances_{novel_attack}.npy'), smallest_distances)
            # np.save(os.path.join(out_dir, f'keys_to_save_{novel_attack}.npy'), keys_to_save)

            # compute AUROC for this novel attack
            auroc = roc_auc_score(is_novel, smallest_percentiles)

            # determine percentage of time novel attacks are closest to original attack mean embedding
            original_attack = novel_attack[:-2]  # last two characters of each novel attack's name are the version num
            nearest_original_prob = (np.array(closest_attacks) == original_attack).mean()

            # save in dictionary and log
            logger.info(f'\t{novel_attack} AUROC = {auroc:.4f}, nearest original prob = {nearest_original_prob:.4f}')
            scores_dict[novel_attack] = [float(auroc), float(nearest_original_prob)]
        except ValueError:
            logger.info(f'Only one class present in y_true for {novel_attack}.')

    logger.info(f'Computed AUROC for each novel attack. {time.time() - s:.2f}s')

    # write scores to file
    with open(os.path.join(out_dir, 'me_scores.yml'), 'w') as outfile:
        yaml.dump(scores_dict, outfile, default_flow_style=False)

    logger.info(f'Wrote scores to file.')

    # ========== DO NOVELTY PREDICTION USING LOF ==========

    lof = LOF(novelty=True)  # novelty detection needs to be set to True
    lof.fit(df_x_train)

    y_score = lof.decision_function(df_x_test)  # # large values correspond to inliers
    y_score = -y_score  # flip so that large values are outliers
    y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())  # scale between zero and one

    # get individual AUROC per novel attack
    logger.info(f'LOF AUROCs:')
    scores_dict = {}
    for novel_attack in HELD_OUT:
        try:
            y_score_subset = y_score[y_test.isin(ATTACKS_CONSIDERED + [novel_attack])]
            y_true_subset = y_test[y_test.isin(ATTACKS_CONSIDERED + [novel_attack])]
            y_true_subset = np.where(np.isin(y_true_subset, HELD_OUT), 1, 0)  # ones are novel attacks, zeros are seen

            auroc = roc_auc_score(np.array(y_true_subset), y_score_subset)
            logger.info(f'\t{novel_attack} AUROC = {auroc:.4f}')
            scores_dict[novel_attack] = float(auroc)
        except ValueError:
            logger.info(f'Only one class present in y_true for {novel_attack}.')

    # write scores to file
    with open(os.path.join(out_dir, 'lof_scores.yml'), 'w') as outfile:
        yaml.dump(scores_dict, outfile, default_flow_style=False)

    logger.info(f'Wrote scores to file.')

    # ========== TEST TO SEE HOW OFTEN TWO GROUPS PRODUCED BY DIFFERENT METHODS ARE PREDICTED DIFFERENT ==========

    df_orig_unknown = df_orig[df_orig.label.isin(HELD_OUT)]  # get uncompressed samples produced by novel attack methods

    logger.info(f'Same/different attack method AUROCs:')
    scores_dict = {}
    for group_size in range(2, 25):
        ys = []
        outputs = []

        dataset_test = SiameseDataset(df_orig_unknown, freq=.5, group_size=group_size)
        dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

        # compress the samples with one of the twins
        s = time.time()
        for batch_id, (x1, x2, y) in enumerate(dataloader_test, 1):
            if group_size > 1:  # this assumes this network was trained so samples were averaged in embedding space
                x1 = x1.view(-1, x1.shape[-1])  # x1 & x2 should originally have shape (batch_size, group_size, n_feats)
                x2 = x2.view(-1, x2.shape[-1])

            # prepare the data
            x1, x2 = torch.Tensor(siamese_scaler.transform(x1)), torch.Tensor(siamese_scaler.transform(x2))  # scale the samples
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)  # move the samples and label to the device

            output = net.forward(x1, x2, group_size=group_size)
            output = output.view_as(y)

            # convert from torch to numpy
            y = y.detach().cpu().flatten().numpy()
            output = output.detach().cpu().flatten().numpy()

            ys.append(y)
            outputs.append(output)

        # ys = torch.cat(ys).flatten()
        # outputs = torch.cat(outputs).flatten()
        ys = np.concatenate(ys)
        outputs = np.concatenate(outputs)

        torch.cuda.empty_cache()

        # auroc = roc_auc_score(ys.detach().cpu(), outputs.detach().cpu())
        auroc = roc_auc_score(ys, outputs)
        logger.info(f'\tGroup size = {group_size} AUROC = {auroc:.4f}')
        scores_dict[group_size] = float(auroc)

    # write scores to file
    with open(os.path.join(out_dir, 'same_diff_auroc_scores.yml'), 'w') as outfile:
        yaml.dump(scores_dict, outfile, default_flow_style=False)

    logger.info('DONE.')
