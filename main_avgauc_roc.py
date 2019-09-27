import torch
import pickle
import Config as cfg
from NeuralNet import NeuralNet
from Datasets import Datasets, FeatureClassification
from Evaluation import auroc
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs', metavar='N', type=int, default=20,
                    help='number of epochs')
parser.add_argument('--target', metavar='N', type=int, default=0,
                    help='class target')

args = parser.parse_args()

epochs = args.epochs
target = args.target

def train():
    nn = NeuralNet()
    # dataset = Datasets.get('MNIST', cfg.DATASET_DIR)
    dataset = Datasets.get('FashionMNIST', cfg.DATASET_DIR)

    test_gen, _ = dataset.testset(batch_size=cfg.BATCH_SIZE, max_samples=cfg.TEST_SET_SIZE, specific_label=target)
    (train_gen, _), (_, _) = dataset.trainset(batch_size=cfg.BATCH_SIZE, valid_size=0.0, specific_label=target)

    # Not the cleanest code ever, but I reuse the MNIST generation code of ClassificationDataset class
    # to take MNIST dataset
    mnist_train_dataset = FeatureClassification(train_gen.dataset, target_label=target, transforms=cfg.NUM_TRANS)
    mnist_test_dataset = FeatureClassification(test_gen.dataset, target_label=target, transforms=cfg.NUM_TRANS)
    mnist_eval_dataset = FeatureClassification(test_gen.dataset, target_label=None, transforms=cfg.NUM_TRANS)

    train_loader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size=cfg.BATCH_SIZE,
                                               sampler=torch.utils.data.RandomSampler(mnist_train_dataset),
                                               num_workers=4, pin_memory=1)
    test_loader = torch.utils.data.DataLoader(mnist_test_dataset, batch_size=cfg.BATCH_SIZE,
                                              sampler=None,
                                              num_workers=4, pin_memory=1)
    eval_loader = torch.utils.data.DataLoader(mnist_eval_dataset, batch_size=cfg.BATCH_SIZE,
                                              sampler=None,
                                              num_workers=4, pin_memory=1)

    nn.train(train_loader, test_loader, epochs=epochs, lr=0.01, lr_plan={0: 0.001, 10: 0.0001})
    score_func, labels = nn.evaluate(eval_loader)

    pickle_out = open('dump.pickle', 'wb')
    pickle.dump({'score_func': score_func, 'labels': labels}, pickle_out)
    pickle_out.close()


def main():
    auc_roc = []

    for i in range(0,5,1):
        print("===== Iteration : {} ======".format(i))
        train()

        pickle_in = open("dump.pickle", "rb")
        dump = pickle.load(pickle_in)
        score_func, labels = dump['score_func'], dump['labels']

        auc_roc.append(auroc(score_func, labels, target=target, samples=1e3, epochs=epochs))

    auc_rod_mean = np.mean(auc_roc)
    auc_rod_std = np.std(auc_roc)
    print("AUC-ROC for class {} is {:.3f}+-{:.3f}".format(target, 100*auc_rod_mean, 100*auc_rod_std))


if __name__ == '__main__':
    main()