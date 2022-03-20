"""This script compares the attacked text samples produced by the variant attack methods
to their respective original attack methods; it does this using a simple string equals
comparison, but also a BERT score comparison"""
import logging
import os

import bert_score
from bert_score.utils import (
    get_model,
    get_tokenizer,
    lang2model,
    model2layers
)
import configargparse
import pandas as pd
import torch


# define original attacks and variant attacks
variant_attacks = [
    'pruthiv1', 'pruthiv2', 'pruthiv3', 'pruthiv4',
    'deepwordbugv1', 'deepwordbugv2', 'deepwordbugv3', 'deepwordbugv4',
    'textbuggerv1', 'textbuggerv2', 'textbuggerv3', 'textbuggerv4']
# original_attacks = ['pruthi', 'deepwordbug', 'textbugger']
original_attacks = ['textbugger']
original_to_variants = {
    'pruthi': ['pruthiv1', 'pruthiv2', 'pruthiv3', 'pruthiv4'],
    'deepwordbug': ['deepwordbugv1', 'deepwordbugv2', 'deepwordbugv3', 'deepwordbugv4'],
    'textbugger': ['textbuggerv1', 'textbuggerv2', 'textbuggerv3', 'textbuggerv4']}


if __name__ == '__main__':
    # create logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    # parse the command line arguments
    cmd_opt = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    # args for configuring the training and test data for the Siamese network
    cmd_opt.add('--distance_metric', type=str, default='bert',
                help="Name of target model for which the attacks were created")
    args = cmd_opt.parse_args()

    # load BERT model if needed
    if args.distance_metric == 'bert':
        model_type = lang2model['en']
        num_layers = model2layers[model_type]

        tokenizer = get_tokenizer(model_type)
        model = get_model(model_type, num_layers, all_layers=False)

    # set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # load the attacked samples for the different variants
    logger.info(f'Loading variant attacks...')
    dfs = []
    for attack in variant_attacks:
        try:
            df = pd.read_csv(
                os.path.join('/projects/uoml/anoack2/react-cluster/data/textattack_variants', attack, 'results.csv'))
            dfs.append(df)
        except FileNotFoundError:
            logger.info('cannot find attacks for:', attack)
            pass
    df_variants = pd.concat(dfs, axis=0).reset_index(drop=True)
    logger.info(f'Variants attack counts: {dict(df_variants.attack_name.value_counts())}')
    logger.info(f'Finished loading variant attacks.')

    # load the original attacks' data and merge with the variants
    logger.info(f'Loading original attacks...')
    df_all = pd.read_csv('/projects/uoml/anoack2/react-cluster/data/whole_dataset_with_meta.csv')
    logger.info(f'Finished loading original attacks.')

    # filter out those attacks that aren't for SST and aren't successes
    df_all = df_all[df_all.dataset == 'sst']
    df_all = df_all[df_all.status == 'success']

    # rename certain columns in the variants dataframe so they match the original attacks dataframe
    df_variants = df_variants.rename({'test_index': 'test_ndx', 'target_model_dataset': 'dataset'}, axis=1)

    # merge the two dataframes
    df = pd.concat([df_variants, df_all], axis=0).reset_index(drop=True)
    logger.info(f'Variants attack counts: {dict(df.attack_name.value_counts())}')

    # for each variant, measure how often the variant produces output that is different
    # from the original attack method's output
    diff_counter = {}

    # loop through the original attacks
    logger.info(f'Comparing attacks...')
    for original_attack in original_attacks:

        logger.info(f'Comparing {original_attack} to:')

        # loop through the variants for this original attack
        for variant_attack in original_to_variants[original_attack]:

            logger.info(f'\t{variant_attack}')

            key = f'{original_attack}_{variant_attack}'  # the key into the diff_counter for this original/variant pair

            # make sure the samples being compared were for the same target model
            for model in ['roberta', 'bert', 'xlnet']:

                # filter dataframe
                df_ = df[(df.attack_name == original_attack) | (df.attack_name == variant_attack)]
                df_ = df_[df_.target_model == model]
                # logger.info(f'Length of dataset subset = {len(df_)}')

                # group the resultant dataframe by the original_text column
                for group in df_.groupby('original_text'):

                    group_df = group[1]  # get the dataframe for this group

                    # if len(group_df) > 1:
                    #     logger.info(f'length of samples = {len(group_df)}')
                    #     logger.info(f'attack names = {list(group_df.attack_name)}')

                    # if the original attack AND the variant attack BOTH successfully attacked this sample
                    if original_attack in set(group_df.attack_name) and variant_attack in set(group_df.attack_name):

                        # logger.info(f'In here')

                        # get the original attack's output and the variant attack's output
                        orig_attack_df = group_df[group_df.attack_name == original_attack].reset_index(drop=True)
                        variant_attack_df = group_df[group_df.attack_name == variant_attack].reset_index(drop=True)
                        orig_attack_output = orig_attack_df.loc[0, 'perturbed_text']
                        variant_attack_output = variant_attack_df.loc[0, 'perturbed_text']

                        if args.distance_metric == 'bert':
                            all_preds, hash_code = bert_score.score(
                                [orig_attack_output],
                                [variant_attack_output],
                                model_type=None,
                                num_layers=num_layers,
                                verbose=False,
                                idf=False,
                                device=device,
                                batch_size=1,
                                lang='en',
                                return_hash=True,
                                rescale_with_baseline=False,
                                baseline_path=None,
                            )
                            similarity = all_preds[2].item()

                        elif args.distance_metric == 'equals':
                            # determine if the outputs are different
                            if orig_attack_output == variant_attack_output:
                                similarity = 1
                            else:
                                similarity = 0

                        if key not in diff_counter:
                            diff_counter[key] = [similarity]
                        else:
                            diff_counter[key].append(similarity)

    for key in diff_counter:
        diff_counter[key] = sum(diff_counter[key]) / len(diff_counter[key])

    # print out the differences
    for key in diff_counter:
        logger.info(f'{key}, {diff_counter[key]:.4f}')
