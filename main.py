import torch
import pickle
import Config as cfg
import argparse
from NeuralNet import NeuralNet
from Datasets import Datasets, FeatureClassification
from Evaluation import auroc

# Command line arguments setup
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--label', type=int,
                    help='label to use', required=True)
parser.add_argument('-t', '--trans', type=int, default=8,
                    help='number of transformations (default: 4)')
parser.add_argument('-e', '--epochs', type=int, default=128,
                    help='total epochs (default: 128)')
parser.add_argument('--dataset', default='FashionMNIST',
                    help='MNIST, FashionMNIST (default), CIFAR10')

args = parser.parse_args()

label = args.label
epochs = args.epochs
trans = args.trans
dataset = args.dataset

# Initialize channel and input dimensions for WRN to support the different datasets
if dataset == 'MNIST' or dataset == 'FashionMNIST':
    cfg.INPUT_CH = 1
    cfg.INPUT_DIM = 28
elif dataset == 'CIFAR10':
    cfg.INPUT_CH = 3
    cfg.INPUT_DIM = 32

cfg.INPUT_PAD = 16
cfg.INPUT_DIM_PADDED = cfg.INPUT_DIM + 2 * cfg.INPUT_PAD


def train():
    """
    Train the network on the specific label given as a command line argument.
    The function dumps its evaluation data for further analysis.
    """
    nn = NeuralNet()
    test_import = Datasets.get(dataset, cfg.DATASET_DIR)._test_importer()
    train_import = Datasets.get(dataset, cfg.DATASET_DIR)._train_importer()

    # Not the cleanest code ever, but I reuse the MNIST generation code of ClassificationDataset class
    # to take MNIST dataset
    test_dataset = FeatureClassification(test_import, target_label=None, transforms=cfg.NUM_TRANS)
    train_dataset = FeatureClassification(train_import, target_label=label, transforms=cfg.NUM_TRANS)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                                               sampler=torch.utils.data.RandomSampler(train_dataset),
                                               num_workers=4, pin_memory=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
                                              sampler=None,
                                              num_workers=4, pin_memory=1)

    nn.train(train_loader, test_loader, epochs=int(epochs/cfg.NUM_TRANS), lr=0.01, lr_plan={5: 0.001, 10: 0.0001})
    score_func, labels = nn.evaluate(test_loader)

    pickle_out = open('dump.pickle', 'wb')
    pickle.dump({'score_func': score_func, 'labels': labels}, pickle_out)
    pickle_out.close()


def main():
    print("=> Dataset: {}, Label: {}, Transformations: {}, Total Epochs: {}".format(dataset, label, trans, epochs))
    cfg.NUM_TRANS = trans
    train()

    pickle_in = open("dump.pickle", "rb")
    dump = pickle.load(pickle_in)
    score_func, labels = dump['score_func'], dump['labels']

    auroc(score_func, labels, target=label, samples=1e3)


if __name__ == '__main__':
    main()
