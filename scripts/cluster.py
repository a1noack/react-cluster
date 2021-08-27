"""Script for clustering attacks using various clustering algorithms"""
import joblib
import logging
import os
from pathlib import Path
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import configargparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, \
    Birch, DBSCAN, KMeans, MeanShift, OPTICS, SpectralClustering
from sklearn.metrics import rand_score, normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from utils import load_joblib_data, downsample


ALGOS = {
    'aff_prop': AffinityPropagation,
    'agg_clust': AgglomerativeClustering,
    'birch': Birch,
    'dbscan': DBSCAN,
    'kmeans': KMeans,
    'mean_shift': MeanShift,
    'optics': OPTICS,
    'spec_clust': SpectralClustering
}

PARAM_GRIDS = {
    'aff_prop': {'damping': [.5, .6, .7, .8, .9, .99], 'max_iter': [10000]},
    'agg_clust': {'n_clusters': [-1]},
    'birch': {'threshold': [.01, .25, .5, .75, .99]},
    'dbscan': {'eps': [.01, .03, .1, .3, 1, 3, 10, 30, 100], 'min_samples': [50, 75, 100]},
    'kmeans': {'n_clusters': [-1], 'n_init': [20], 'max_iter': [300, 3000]},
    'mean_shift': {'bandwidth': [None, .1, 1, 10, 100, 1000]},  # 'bandwidth' is the main param to tune for mean_shift
    'optics': {},
    'spec_clust': {'n_clusters': [-1]}  # works well for small number of clusters, but not recommended for many clusters
}


def get_score(predictions, labels, metrics):
    """Returns scores for this pair of predictions and labels."""
    scores = {}
    for metric in metrics:
        if metric == 'rand_ind':
            scores['rand_ind'] = rand_score(predictions, labels)
        elif metric == 'nmi':
            scores['nmi'] = normalized_mutual_info_score(predictions, labels)
        else:
            raise ValueError('This metric is not supported!')

    return scores


def get_freq_dict(labels, predictions, cluster):
    """Returns a frequency dictionary for the specified cluster number."""

    assert cluster in predictions, f'Cluster number {cluster} is not a valid cluster!'

    fd = {}
    for i, prediction in enumerate(predictions):
        if prediction == cluster:
            predicted_attack = labels[i]
            if predicted_attack in fd:
                fd[predicted_attack] += 1
            else:
                fd[predicted_attack] = 1

    return fd


def plot_distributions(labels, predictions, algo, save_path):
    """Plot the distribution of attack methods for each cluster"""

    # set font and figure sizes
    size = 14
    params = {
        'font.size': size,
        'axes.titlesize': size,
        'axes.labelsize': size,
        'xtick.labelsize': size,
        'ytick.labelsize': size,
        'figure.figsize': (15, len(set(predictions))*2)
    }
    plt.rcParams.update(params)

    # create figure with as many subplots as there are clusters
    attacks = sorted(list(set(labels)))
    n_clusters = len(set(predictions))
    if n_clusters <= 1 or n_clusters > 30:
        return  # if number of clusters is 1 or 0, don't plot
    fig, axs = plt.subplots(n_clusters, 1, sharey=True)
    fig.text(0.00, 0.5, 'Number of samples', va='center', rotation='vertical', size=size + 3)

    # plot the distribution of attacks for each cluster
    for i, cluster in enumerate(sorted(list(set(predictions)))):

        # get the number of predicted samples for each attack for this cluster as plot
        freq_dict = get_freq_dict(labels, predictions, cluster)
        attack_counts = []
        for attack in attacks:
            try:
                attack_counts.append(freq_dict[attack])
            except KeyError:
                attack_counts.append(0)
        axs[i].bar(attacks, attack_counts)

        # add x-label info and title
        if i == len(attacks) - 1:
            axs[i].set_xlabel('Attack method', size=size + 3)
        axs[i].set_title(f'{algo} cluster {cluster}', size=size)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{algo}_cluster_distributions.pdf'), transparent=False, pad_inches=0)


def eval_cluster_centers(df_orig, keys, data, cluster_centers, k=30):
    """Get the original text for the k samples nearest to the cluster centers"""

    # a list to hold the lists of samples belonging to each cluster
    all_samples = []

    # get nearest samples for each cluster
    for i in range(len(cluster_centers)):

        # container for the nearest samples for this cluster center
        samples = []

        # get the indices of the samples nearest to this cluster center
        center = cluster_centers[i].reshape(1, -1)  # get the i-th cluster center
        dists = np.linalg.norm(center - data, axis=1)  # get distances between center & all points
        idxs = np.argpartition(dists, min(k, len(dists)-1))  # get indices closest to center

        # get the information for each of the samples
        for idx in idxs[:k]:
            attack, toolchain, scenario, model, dataset, test_ndx = keys[idx]  # get the unique key for this sample
            row = df_orig[(df_orig.attack_name == attack) & (df_orig.attack_toolchain == toolchain) &
                          (df_orig.scenario == scenario) & (df_orig.target_model == model) &
                          (df_orig.target_model_dataset == dataset) & (df_orig.test_ndx == test_ndx)]
            if len(row) > 1:
                samples.append(('collision error', 'collision error'))
            elif len(row) == 0:
                samples.append(('search error', 'search error'))
            else:
                perturbed_text = row['perturbed_text'].values[0]
                attack = row['attack_name'].values[0]
                samples.append((attack, perturbed_text))

        all_samples.append(samples)

    return all_samples


if __name__ == '__main__':
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
    cmd_opt.add('--cluster_algos', type=str, nargs='+', help="The clustering algorithms to try out.",
                default=['aff_prop', 'agg_clust', 'birch', 'dbscan', 'kmeans', 'mean_shift', 'optics', 'spec_clust'])
    cmd_opt.add('--features', type=str, default='btlc', help='feature groups to include: b, bt, btl, or btlc.')
    cmd_opt.add('--compress_features', type=int, default=0,
                help='compress all features with more than one dimension down to have at most this many dimensions.')
    cmd_opt.add('--metrics', type=str, nargs='+', default=['rand_ind', 'nmi'],
                help="""The metrics used to score the cluster methods; the first metric
                in the list is the one used to select the clustering methods\' hyperparameters.""")
    cmd_opt.add('--n', type=int, default=100, help="The number of attacks to keep per attack method.")
    cmd_opt.add('--in_dir', type=str,
                help='The path to the folder containing the joblib files for the extracted samples.')
    cmd_opt.add('--out_dir', type=str, help='Where to save the results for this clustering experiment.')
    args = cmd_opt.parse_args()

    # create output directory
    out_dir = os.path.join(args.out_dir, f"{args.model}_{args.dataset}_{args.features}_{args.compress_features}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # load the extracted feature data
    logger.info(f'Clustering attacks for {args.model}/{args.dataset}.')
    logger.info(f'Using the following algorithms: {args.cluster_algos}')
    s = time.time()
    logger.info(f'Loading the data. Features: {args.features}')
    dir_path = os.path.join(args.in_dir, 'extracted_features')  # , f'{args.model}_{args.dataset}')
    samples, labels, keys, attack_counts = load_joblib_data(args.model, args.dataset, dir_path,
                                                            args.compress_features, args.features)
    logger.info(f'Loaded the data. {time.time() - s:.2f}s.')
    logger.info(f'Attack counts: {attack_counts}')

    # load original data containing strings and filter based on target model and dataset
    df_orig = pd.read_csv(os.path.join(args.in_dir, 'original_data', 'whole_catted_dataset.csv'))
    df_orig = df_orig[(df_orig.target_model == args.model) & (df_orig.target_model_dataset == args.dataset)]
    logger.info(f'Loaded original textual data.')

    # scale the data
    scaler = StandardScaler().fit(samples)
    samples = scaler.transform(samples)

    # create dataframe from the samples and labels
    df = pd.DataFrame(np.array(samples))
    df['label'] = labels.copy()
    df['key'] = keys.copy()

    # downsample for clustering efficiency
    if args.n != -1:
        df = downsample(df, args.n)
    labels = list(df['label']).copy()
    keys = list(df['key']).copy()
    n_attack_methods = len(set(labels))
    logger.info(f'Downsampled data so all classes have {args.n} samples.')

    # create copy of data without the labels
    data = df.drop(['label', 'key'], axis=1).copy()

    # fit the specified clustering algorithms on the data
    logger.info(f'Fitting algorithms on the data.')
    scores = {}
    for algo in args.cluster_algos:

        # create instance of the algorithm's class
        model = ALGOS[algo]()

        # some clustering algorithms (e.g. hierarchical cluster algorithms)
        # don't create cluster centers, so they cannot predict scores for new data
        if not hasattr(model, 'predict'):
            logger.info(f'{algo} cannot predict on new data. Skipping to next algorithm...')
            continue

        # get the parameters to consider for this algorithm
        param_grid = PARAM_GRIDS[algo]

        # replace -1 in the param grid with the actual number of attack methods for this experiment
        if 'n_clusters' in param_grid:
            n_clusters_list = param_grid['n_clusters']
            param_grid['n_clusters'] = [n_attack_methods if n == -1 else n for n in n_clusters_list]

        # create grid search object
        clf = GridSearchCV(estimator=model, scoring='rand_score',
                           param_grid=param_grid, verbose=3)

        # perform grid search
        s = time.time()
        logger.info(f'\tFitting {algo} on the data.')
        logger.info(f'\tParam grid = {param_grid}')
        clf.fit(data, labels)
        logger.info(f'\tFinished fitting. {time.time() - s:.2f}s')

        # evaluate the clustering algorithm
        predictions = clf.predict(data)
        score = get_score(predictions, labels, args.metrics)
        scores[algo] = score
        logger.info(f'\tFinished fitting. Performance: {score}')
        logger.info(f'\tBest params = {clf.best_params_}')

        # save predictions, labels, and model
        np.save(os.path.join(out_dir, 'predictions.npy'), np.array(predictions))
        np.save(os.path.join(out_dir, 'labels.npy'), np.array(labels))
        joblib.dump(clf, os.path.join(out_dir, f'{algo}.mdl'))
        logger.info(f'\tSaved predictions and labels and model file.')

        # get samples near cluster centers
        if hasattr(clf.best_estimator_, 'cluster_centers_'):
            s = time.time()
            nearest_samples = eval_cluster_centers(df_orig, keys, data, clf.best_estimator_.cluster_centers_)
            np.save(os.path.join(out_dir, 'nearest_samples.npy'), nearest_samples)
            logger.info(f'\tSaved samples nearest cluster centers. {time.time() - s:.2f}s')
        # plot the clusters' attack distributions
        plot_distributions(labels, predictions, algo, out_dir)
        logger.info(f'\tPlotted cluster distributions.')

        # write results to file
        with open(os.path.join(out_dir, f'performance.txt'), 'a') as f:
            f.write(f'{algo}: ')
            for metric, value in scores[algo].items():
                f.write(f'({metric}, {value*100:.1f}) ')
            f.write(f'\n')
        logger.info(f'\tSaved performance scores to file.')
        logger.info(f'')

    logger.info(f'DONE.')
