import os
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import matplotlib.pyplot as plt
import numpy as np
import Config as cfg
from PIL import Image


# ----------------------------------------------------------------------------------------------------------------------
#                                        Feature Classification Dataset
# ----------------------------------------------------------------------------------------------------------------------
class FeatureClassification(torch.utils.data.Dataset):
    def __init__(self, dataset, target_label, transforms):
        self.dataset = dataset

        # Taking images from the dataset
        if type(dataset.data).__name__ != 'Tensor':
            self.data = torch.Tensor(dataset.data)
        else:
            self.data = dataset.data

        if type(dataset.targets).__name__ != 'Tensor':
            self.labels = torch.Tensor(dataset.targets)
        else:
            self.labels = dataset.targets

        if target_label != None:
            indices = [pos for pos, label in enumerate(dataset.targets) if label == target_label]
            # Taking only the relevant images from the dataset
            self.data = self.data[indices]
            self.labels = self.labels[indices]
            # Duplicating the inputs 'features' times, since the same input will be used with different transformation
            self.data = torch.repeat_interleave(self.data, transforms, dim=0)
            self.labels = torch.repeat_interleave(self.labels, transforms, dim=0) #TODO: it is redundent here, but might be usefull to remove if not ('None' maybe)
        else:
            # Duplicating the inputs 'features' times, since the same input will be used with different transformation
            self.data = torch.repeat_interleave(self.data, transforms, dim=0)
            self.labels = torch.repeat_interleave(self.labels, transforms, dim=0)

        self.targets = torch.zeros([transforms])
        for i in range(transforms):
            self.targets[i] = i
        # self.targets = self.targets.repeat(len(indices))
        self.targets = self.targets.repeat(len(self.data)//transforms)

    def __getitem__(self, index):
        # TODO: this is specific for MNIST, should make it general
        img, target, labels = self.data[index], int(self.targets[index]), self.labels[index]

        # Doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy().astype('uint8'))

        if self.dataset.transform is not None:
            img = self.dataset.transform(img)

        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)

        return img, target, labels

    def __len__(self):
        return len(self.data)


# ----------------------------------------------------------------------------------------------------------------------
#                                                Base Class
# ----------------------------------------------------------------------------------------------------------------------
class ClassificationDataset:
    def __init__(self, data_dir, class_labels, shape, testset_size, trainset_size, expected_files):
        # Basic Dataset Info
        self._class_labels = tuple(class_labels)
        self._shape = tuple(shape)
        self._testset_size = testset_size
        self._trainset_size = trainset_size
        self._data_dir = data_dir

        if not isinstance(expected_files, list):
            self._expected_files = [expected_files]
        else:
            self._expected_files = expected_files

        self._download = True if any(
            not os.path.isfile(os.path.join(self._data_dir, file)) for file in self._expected_files) else False

    def name(self):
        assert self.__class__.__name__ != 'ClassificationDataset'
        return self.__class__.__name__

    def num_classes(self):
        return len(self._class_labels)

    def class_labels(self):
        return self._class_labels

    def input_channels(self):
        return self._shape[0]

    def shape(self):
        return self._shape

    def max_test_size(self):
        return self._testset_size

    def max_train_size(self):
        return self._trainset_size

    def testset(self, batch_size, max_samples, specific_label=None, device='cuda'):

        if device.lower() == 'cuda' and torch.cuda.is_available():
            num_workers, pin_memory = 1, True
        else:
            print('Warning: Did not find working GPU - Loading dataset on CPU')
            num_workers, pin_memory = 4, False

        test_dataset = self._test_importer()

        # If there is not specific label set than take data from the dataset with no regard to the labels
        if specific_label is None:
            if max_samples < self._testset_size:
                testset_sz = max_samples
                # TODO: inference sampler shouldn't be random, although it probably doesn't chagne the results
                test_sampler = SubsetRandomSampler(list(range(max_samples)))
            else:
                test_sampler = None
                testset_sz = self._testset_size

        # Otherwise, pick all data with the specific label
        else:
            indices = [pos for pos, label in enumerate(test_dataset.test_labels) if label == specific_label]
            # TODO: same here, sampler shouldn't be random during inference
            test_sampler = SubsetRandomSampler(indices)
            testset_sz = len(indices)

        test_gen = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)

        return test_gen, testset_sz

    def trainset(self, batch_size=128, valid_size=0.1, max_samples=None, shuffle=True, random_seed=None,
                 specific_label=None, show_sample=False, device='cuda'):

        if device.lower() == 'cuda' and torch.cuda.is_available():
            num_workers, pin_memory = 1, True
        else:
            print('Warning: Did not find working GPU - Loading dataset on CPU')
            num_workers, pin_memory = 4, False

        max_samples = self._trainset_size if max_samples is None else min(self._trainset_size, max_samples)
        assert ((valid_size >= 0) and (valid_size <= 1)), "[!] Valid_size should be in the range [0, 1]."

        train_dataset = self._train_importer()
        # print(sum(1 for _ in train_dataset)) #Can be used to discover the trainset size if needed

        val_dataset = self._train_importer()

        if specific_label is None:
            indices = list(range(self._trainset_size))
            if shuffle:
                if random_seed is not None:
                    np.random.seed(random_seed)
                np.random.shuffle(indices)

            indices = indices[:max_samples]  # Truncate to desired size
        else:
            indices = [pos for pos, label in enumerate(train_dataset.train_labels) if label == specific_label]

        # Split validation
        split = int(np.floor(valid_size * max_samples))
        train_ids, valid_ids = indices[split:], indices[:split]

        num_train = len(train_ids)
        num_valid = len(valid_ids)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(train_ids), num_workers=num_workers,
                                                   pin_memory=pin_memory)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_ids), num_workers=num_workers,
                                                   pin_memory=pin_memory)

        return (train_loader, num_train), (valid_loader, num_valid)

    def _train_importer(self):
        raise NotImplementedError

    def _test_importer(self):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
#                                                  Implementations
# ----------------------------------------------------------------------------------------------------------------------
class CIFAR10(ClassificationDataset):
    def __init__(self, data_dir):
        super().__init__(
            data_dir=data_dir,
            class_labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            shape=(3, 32, 32),
            testset_size=10000,
            trainset_size=50000,
            expected_files=os.path.join('CIFAR10', 'cifar-10-python.tar.gz')
        )

    def _train_importer(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return datasets.CIFAR10(root=os.path.join(self._data_dir, 'CIFAR10'), train=True, download=self._download,
                                transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.Pad(cfg.INPUT_PAD, padding_mode='reflect'),
                                                              transforms.ToTensor(),
                                                              normalize]))

    def _test_importer(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return datasets.CIFAR10(root=os.path.join(self._data_dir, 'CIFAR10'), train=False, download=self._download,
                                transform=transforms.Compose([transforms.Pad(cfg.INPUT_PAD, padding_mode='reflect'),
                                                              transforms.ToTensor(),
                                                              normalize]))


class MNIST(ClassificationDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir=data_dir,
                         class_labels=['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
                         shape=(1, 28, 28), testset_size=10000, trainset_size=60000,
                         expected_files=[os.path.join('MNIST', 'processed', 'training.pt'),
                                         os.path.join('MNIST', 'processed', 'test.pt')])

    def _train_importer(self):
        ops = [transforms.Pad(cfg.INPUT_PAD, padding_mode='reflect'),
               transforms.ToTensor(),
               transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
        return datasets.MNIST(root=os.path.join(self._data_dir, 'MNIST'), train=True, download=self._download,
                              transform=transforms.Compose(ops))

    def _test_importer(self):
        ops = [transforms.Pad(cfg.INPUT_PAD, padding_mode='reflect'),
               transforms.ToTensor(),
               transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
        return datasets.MNIST(root=os.path.join(self._data_dir, 'MNIST'), train=False, download=self._download,
                              transform=transforms.Compose(ops))


class FashionMNIST(ClassificationDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir=data_dir,
                         class_labels=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
                                       'Sneaker', 'Bag', 'Ankle boot'],
                         shape=(1, 28, 28), testset_size=10000, trainset_size=60000,
                         expected_files=[os.path.join('FashionMNIST', 'processed', 'training.pt'),
                                         os.path.join('FashionMNIST', 'processed', 'test.pt')])

    def _train_importer(self):
        ops = [transforms.Pad(cfg.INPUT_PAD, padding_mode='reflect'), transforms.ToTensor()]
        return datasets.FashionMNIST(root=os.path.join(self._data_dir, 'FashionMNIST'), train=True,
                                     download=self._download,
                                     transform=transforms.Compose(ops))

    def _test_importer(self):
        ops = [transforms.Pad(cfg.INPUT_PAD, padding_mode='reflect'), transforms.ToTensor()]
        return datasets.FashionMNIST(root=os.path.join(self._data_dir, 'FashionMNIST'), train=False,
                                     download=self._download,
                                     transform=transforms.Compose(ops))


# ----------------------------------------------------------------------------------------------------------------------
#                                                  Implementations
# ----------------------------------------------------------------------------------------------------------------------
class Datasets:
    _implemented = {
        'MNIST': MNIST,
        'CIFAR10': CIFAR10,
        'FashionMNIST': FashionMNIST
    }

    @staticmethod
    def which():
        return tuple(Datasets._implemented.keys())

    @staticmethod
    def get(dataset_name, data_dir):
        return Datasets._implemented[dataset_name](data_dir)
