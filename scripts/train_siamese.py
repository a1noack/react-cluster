"""Script for training a Siamese network to embed attacks"""
import joblib
import logging
import os
from pathlib import Path
import random
import time

import configargparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from utils import load_joblib_data, downsample, plot_tsne
from models import NormalDataset, SiameseDataset, SiameseNet


# use only the attacks below for training the Siamese network
ELITE_ATTACKS = ['hotflip', 'deepwordbug', 'textbugger', 'pruthi', 'clean']


def plot_losses(train_loss_list, val_loss_list):
    """Saves a plot showing both training and validation loss"""
    plt.close()
    epochs = list(range(1, len(train_loss_list) + 1))

    # set graph size
    size = 15
    diff = 0
    params = {
        'font.size': size,
        'axes.titlesize': size,
        'axes.labelsize': size - diff,
        'xtick.labelsize': size - diff,
        'ytick.labelsize': size - diff,
        'figure.figsize': (8*1.618, 8),
    }
    plt.rcParams.update(params)

    plt.plot(epochs, train_loss_list, 'red', label='Training loss')
    plt.plot(epochs, val_loss_list, 'blue', label='Validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy loss')
    plt.legend()
    plt.title(f'Training and validation loss vs. epoch')

    plt.savefig(os.path.join(out_dir, f'losses_plot.png'))


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
    cmd_opt.add('--model', type=str, default='roberta', help="Name of target model for which the attacks were created")
    cmd_opt.add('--dataset', type=str, default='sst', help="Name of dataset used to create the attacks")
    cmd_opt.add('--features', type=str, default='btlc', help='feature groups to include: b, bt, btl, or btlc')
    cmd_opt.add('--compress_features', type=int, default=0,
                help='compress all features with > 1 dim. down to at most this many dims.')
    cmd_opt.add('--n', type=int, default=0, help="The number of attacks to keep per attack method")
    cmd_opt.add('--held_out', type=str, nargs='+', default=[],
                help='which attacks to hold out; if name of an attack is "rand", a random attack will be held out')
    cmd_opt.add('--use_elite_attackers_only', type=int, default=1, help='if 1, use only the elite attack methods')

    # args defining Siamese network architecture
    cmd_opt.add('--hid_size', type=int, default=128, help='the number of neurons per hidden layer')
    cmd_opt.add('--n_layer', type=int, default=4, help='the number of hidden layers – plus one – in network')
    cmd_opt.add('--out_size', type=int, default=32, help='the size of the embedding layer, i.e. output layer')
    cmd_opt.add('--drop_prob', type=float, default=.5, help='the dropout probability for dropout layers in the network')

    # args defining training
    cmd_opt.add('--same_freq', type=float, default=.5, help='frequency that two samples with same label should appear')
    cmd_opt.add('--lr', type=float, default=.0001, help='the initial learning rate')
    cmd_opt.add('--batch_size', type=int, default=32, help='the number of samples in a minibatch')
    cmd_opt.add('--max_epochs', type=int, default=200, help='the max number of epochs to train for')
    cmd_opt.add('--early_stop', type=int, default=10, help='no. of epochs of non-improvement before ending training')
    cmd_opt.add('--select_on_val', type=int, default=1, help='select on val. loss if 1, training loss if 0')
    cmd_opt.add('--group_size', type=int, default=1, help='size of the groups of samples the network is trained on')
    cmd_opt.add('--device', type=int, default=0, help='the number of the device to be used')

    # args for I/O
    cmd_opt.add('--in_dir', type=str, help='path to folder containing joblib files for extracted samples')
    cmd_opt.add('--out_dir', type=str, help='where to save the results for this clustering experiment')

    args = cmd_opt.parse_args()

    # set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # create output directory
    out_dir = os.path.join(args.out_dir, "siamese_{}_{}_{}_{}_{}_{}_val-{}_{}_{}".format(
        args.model, args.dataset, args.lr, args.batch_size,
        args.hid_size, args.out_size, args.select_on_val, args.group_size, int(time.time())))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_file_handler = logging.FileHandler(os.path.join(out_dir, 'output.log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    output_file_handler.setFormatter(formatter)
    logger.addHandler(output_file_handler)

    # load the extracted feature data
    logger.info(f'Training Siamese Network for attacks on {args.model}/{args.dataset}.')
    logger.info(f'Experiment arguments: {args}')
    s = time.time()
    logger.info(f'Features to load for each sample: {args.features}')
    logger.info(f'Loading the data...')
    dir_path = os.path.join(args.in_dir, 'extracted_features')  # , f'{args.model}_{args.dataset}')
    samples, labels, keys, attack_counts = load_joblib_data(args.model, args.dataset, dir_path,
                                                            args.compress_features, args.features, logger)
    logger.info(f'Loaded the data. {time.time() - s:.2f}s.')
    logger.info(f'Attack counts: {attack_counts}')

    # create dataframe from the samples and labels
    df = pd.DataFrame(np.asarray(samples))  # np.asarray instead of np.array, because np.array copies data
    df['label'] = labels.copy()

    # remove samples that aren't in new, smaller group
    if args.use_elite_attackers_only:
        elite_attacks = list(set(ELITE_ATTACKS + args.held_out))  # add held out attacks to ELITE_ATTACKS
        df = df[df.label.isin(elite_attacks)]
        logger.info(f'Removed all samples not produced by one of the following attack methods: {elite_attacks}')
        logger.info(f'Attack counts now: {dict(df.label.value_counts())}')

    # downsample for clustering efficiency
    if args.n != 0:
        df = downsample(df, args.n)
        logger.info(f'Downsampled data so all classes have at most {args.n} samples.')
        logger.info(f'Attack counts now: {dict(df.label.value_counts())}')
    else:
        logger.info(f'No downsampling performed.')

    # set up dataloaders for training and testing
    # TODO: save which indices are in train and test
    df_train, df_nottrain = train_test_split(df, stratify=df['label'], test_size=.2 if len(df) < 40000 else 10000)
    df_val, df_test = train_test_split(df_nottrain, stratify=df_nottrain['label'], test_size=0.5)
    if len(args.held_out) > 0:  # remove specified attacks from train and validation samples (keep in test set tho)
        removed_attacks = []
        for attack in args.held_out:
            if attack == 'rand':  # randomly choose one of remaining attack methods to remove
                attack = random.choice(df_train.label.unique())
            df_train = df_train[df_train['label'] != attack]
            df_val = df_val[df_val['label'] != attack]
            removed_attacks.append(attack)
        for attack in removed_attacks:
            assert attack not in list(df_train.label) and attack not in list(df_val.label), \
                'Held out attacks not removed successfully!'
        logger.info(f'Removed {removed_attacks} from training and validation sets...')
    logger.info(f'No. train = {len(df_train)}, no. val = {len(df_val)}, no. test = {len(df_test)}')
    dataset_train = SiameseDataset(df_train, freq=args.same_freq, group_size=args.group_size)
    dataset_val = SiameseDataset(df_val, freq=args.same_freq, group_size=args.group_size)
    dataset_test = NormalDataset(df_test, group_size=args.group_size)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    # scale the data
    scaler = StandardScaler().fit(dataset_train.data)
    logger.info(f'Fitted scaler to training data.')
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'))
    logger.info(f'Saved fitted sklearn StandardScaler to output directory.')

    # instantiate net, turn on drop out, and move to GPU
    net = SiameseNet(in_size=df_train.shape[1]-1, hid_size=args.hid_size, out_size=args.out_size,
                     n_layer=args.n_layer, drop_prob=args.drop_prob, group_size=args.group_size)
    logger.info(f'Instantiated Siamese Network with layers: {net.hidden}')
    net.to(device)

    # define loss and optimizer
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs)

    # instantiate accumulator variables
    lowest_loss = np.inf

    # define train/validation function
    def train():

        # make pass over training dataset and update model according to loss
        net.train()  # turn dropout on
        total_train_loss = 0
        for batch_id, (x1, x2, y) in enumerate(dataloader_train, 1):

            # reshape inputs so that they can pass through network nicely
            if args.group_size > 1:
                x1 = x1.view(-1, x1.shape[-1])
                x2 = x2.view(-1, x2.shape[-1])

            # prepare the data
            x1, x2 = torch.Tensor(scaler.transform(x1)), torch.Tensor(scaler.transform(x2))  # scale the samples
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)  # move the samples and label to the device

            # forward pass to get loss
            optimizer.zero_grad()
            output = net.forward(x1, x2)
            loss = loss_fn(output, y)
            total_train_loss += loss.item()

            # back propagate and update weights
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(dataloader_train)

        # make pass over validation set
        net.eval()  # turn dropout off
        total_val_loss = 0
        for batch_id, (x1, x2, y) in enumerate(dataloader_val, 1):

            # reshape inputs so that they can pass through network nicely
            if args.group_size > 1:
                x1 = x1.view(-1, x1.shape[-1])
                x2 = x2.view(-1, x2.shape[-1])

            # prepare the data
            x1, x2 = torch.Tensor(scaler.transform(x1)), torch.Tensor(scaler.transform(x2))  # scale the samples
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)  # move the samples and label to the device

            # forward pass to get loss
            output = net.forward(x1, x2)
            loss = loss_fn(output, y)
            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(dataloader_val)

        return avg_val_loss, avg_train_loss

    # initialize results containers
    val_loss_list = []
    train_loss_list = []
    no_improve_ctr = 0  # used to determine when to early stop

    # training loop
    for epoch in range(args.max_epochs):
        val_loss, train_loss = train()
        val_loss_list.append(val_loss)
        train_loss_list.append(train_loss)
        loss = val_loss if args.select_on_val else train_loss  # which loss to select model on

        # log loss
        logger.info(f'Epoch {epoch}: train loss = {train_loss}, val. loss = {val_loss}')

        # save model weights if lowest loss so far
        if loss < lowest_loss:
            torch.save(net.state_dict(), os.path.join(out_dir, 'siamese_net.pt'))
            logger.info(f'\tLowest {"val." if args.select_on_val else "train"} loss so far. Saved model weights.')
            lowest_loss = loss
            no_improve_ctr = 0
        else:
            no_improve_ctr += 1

        # early stop if no recent improvement
        if no_improve_ctr >= args.early_stop:
            logger.info('No improvement in {} loss in {} epochs. Ending training.'.format(
                "val." if args.select_on_val else "train", no_improve_ctr))
            break

        # adjust learning rate
        scheduler.step()

    # load the state dict and evaluate the network's performance on test set
    net.load_state_dict(torch.load(os.path.join(out_dir, 'siamese_net.pt')))
    net.eval()  # turn dropout off

    # compress the test set samples with one of the twins
    all_embedded_samples = []
    labels = []
    s = time.time()
    for batch_id, (x, y) in enumerate(dataloader_test, 1):

        # if multiple samples in group, reshape inputs so that they can pass through network nicely
        orig_shape = x.shape
        if args.group_size > 1:
            x = x.view(-1, orig_shape[-1])

        # scale and pass through network
        x = torch.Tensor(scaler.transform(x)).to(device)  # scale the samples
        x_ = net.forward_one(x)

        # reshape inputs so that they can pass through network nicely
        if args.group_size > 1:
            x_ = x_.view(orig_shape[0], args.group_size, args.out_size)
            x_ = x_.mean(dim=1)

        all_embedded_samples.append(x_)
        labels.append(y)
    logger.info(f'Embedded all test set samples. {time.time() - s}s')

    data = torch.cat(all_embedded_samples, dim=0).detach().cpu().numpy()
    labels = np.hstack(labels)

    # save embedded samples
    s = time.time()
    np.save(os.path.join(out_dir, 'embedded_samples.npy'), data)
    np.save(os.path.join(out_dir, 'labels.npy'), labels)
    logger.info(f'Saved embedded samples. {time.time() - s}s')

    # perform t-SNE on compressed samples
    s = time.time()
    for ppl in [2, 5, 10, 30, 50, 100]:  # try full range of perplexity values
        plot_tsne(data, labels, out_dir, ppl)
    logger.info(f'Plotted compressed samples using t-SNE. {time.time() - s}s')

    # plot training and validation losses
    s = time.time()
    plot_losses(train_loss_list, val_loss_list)
    logger.info(f'Plotted loss curves. {time.time() - s}s')

    logger.info(f'DONE.')
