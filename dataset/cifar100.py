from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from dataset.cifar10 import *



def train_val_split100(labels, n_labeled_per_class, mode = None):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []
    # Set some strategies
    print(f'Current sampling strategy is {mode}')
    if mode == 'PureRandom':
        idxs = np.arange(labels.shape[0])
        np.random.shuffle(idxs)
        train_labeled_idxs = idxs[:n_labeled_per_class*100]
        train_unlabeled_idxs = idxs[n_labeled_per_class*100:-5000]
        val_idxs = idxs[-5000:]
    else:
        for i in range(100):
            idxs = np.where(labels == i)[0]
            np.random.shuffle(idxs)
            unbalance = False
            rebalance = False
            rebalance_un = False
            num_per_class = 500
            ratio_max = 2
            # unbalance
            if unbalance:
                if i < 50:
                    train_labeled_idxs.extend(idxs[:n_labeled_per_class])
                    train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-50])
                    val_idxs.extend(idxs[-50:])
                    unlabel_per_class = num_per_class - n_labeled_per_class - 50
                else:
                    train_labeled_idxs.extend(idxs[:n_labeled_per_class//ratio_max])
                    if rebalance:
                        for i in range(ratio_max-1):
                            train_labeled_idxs.extend(idxs[:n_labeled_per_class//ratio_max])

                    train_unlabeled_idxs.extend(idxs[n_labeled_per_class//ratio_max:unlabel_per_class//ratio_max + n_labeled_per_class//ratio_max])
                    if rebalance_un:
                        for i in range(ratio_max-1):
                            train_unlabeled_idxs.extend(idxs[n_labeled_per_class//ratio_max:unlabel_per_class//ratio_max + n_labeled_per_class//ratio_max])
                    val_idxs.extend(idxs[unlabel_per_class//ratio_max + n_labeled_per_class//ratio_max:])
            else:
                # balance
                train_labeled_idxs.extend(idxs[:n_labeled_per_class])
                train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-50])
                val_idxs.extend(idxs[-50:])
        print(f'train_labeled_idxs are {train_labeled_idxs}')
    rec = np.zeros(100)
    for i in range(100):
        rec[i] = np.sum(labels[train_labeled_idxs] == i)
    print(f'We sampled {rec} instances for each class')
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


def get_cifar100(root, n_labeled,
                 transform_train=None, transform_val=None,
                 download=True, mode = None):

    base_dataset = CIFAR100(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split100(base_dataset.targets, int(n_labeled/100), mode = mode)
    print(f'The first 10 train labels are: {train_labeled_idxs[:10]}')
    train_labeled_dataset = CIFAR100_labeled(root, train_labeled_idxs, train=True, transform=transform_train, fine = False)
    train_unlabeled_dataset = CIFAR100_unlabeled(root, train_unlabeled_idxs, train=True, transform=TransformTwice(transform_train))
    val_dataset = CIFAR100_labeled(root, val_idxs, train=True, transform=transform_val, download=True, fine = False)
    test_dataset = CIFAR100_labeled(root, train=False, transform=transform_val, download=True, fine = False)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset



class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, fine = True):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.fine = fine
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    if self.fine:
                        self.targets.extend(entry['fine_labels'])
                    else:
                        self.fine_targets = []
                        self.targets.extend(entry['coarse_labels'])
                        self.fine_targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.fine: # target is the coarse target
            fine_target = self.fine_targets[index]
            return img, target, fine_target
        else:
            return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


cifar100_mean = (0.5071, 0.4867, 0.4408) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar100_std = (0.2675, 0.2565, 0.2761) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise100(x, mean=cifar100_mean, std=cifar100_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

class CIFAR100_labeled(CIFAR100):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False, fine = True):
        super(CIFAR100_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download, fine = fine)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            print(self.targets.shape)
            if not self.fine: # target is the coarse target
                self.fine_targets = np.array(self.fine_targets)[indexs]
        self.data = transpose(normalise100(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.fine: # target is the coarse target
            fine_target = self.fine_targets[index]
            return img, target, fine_target
        else:
            return img, target
        # return img, target


class CIFAR100_unlabeled(CIFAR100_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])