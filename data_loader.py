import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset

# for generative model
from mea.examples.mnist import mnist_model
from attention_target_dataset import AttentionTargetDataset, MixtureDataset


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
      In the paper, this number is set to 0.1.
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=trans
    )

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


class GenModelDataset(Dataset):
    def __init__(self,
                 transform,
                 epoch_size,
                 fix_data,
                 fix_offset=0):
        """
        transform: preprocessing step for data
        fix_data: if True, will use index as random seed when returning a sample
        """
        self.transform = transform
        self.epoch_size = epoch_size
        self.fix_data = fix_data
        self.fix_offset = fix_offset

    def __getitem__(self, index):
        if self.fix_data:
            rng_state = torch.get_rng_state()
            torch.manual_seed(index+self.fix_offset)
        trace_dict = mnist_model()
        if self.fix_data:
            torch.set_rng_state(rng_state)
        raw = trace_dict['image'].data.view(1, 28, 28)
        image = self.transform(raw)
        digit_label = trace_dict['label']
        return image, digit_label

    def __len__(self):
        return self.epoch_size


def get_gen_model_loader(batch_size,
                         epoch_size,
                         fix_data,
                         fix_offset=0,
                         num_workers=4,
                         pin_memory=False):
    """
    imitates data loaders but supplies infinite stream from generative model

    fix random seed externally if desired

    actually very silly
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    dataset = GenModelDataset(normalize,
                              epoch_size,
                              fix_data=fix_data,
                              fix_offset=fix_offset)

    sampler = SubsetRandomSampler(range(epoch_size))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def get_supervised_attention_loader(batch_size,
                                    num_workers=4,
                                    pin_memory=False):

    # define transforms
    normalize = transforms.Normalize((0.1307*255,), (0.3081*255,))

    dataset = AttentionTargetDataset("attention_target_data",
                                     transform=normalize)

    sampler = SubsetRandomSampler(range(len(dataset)))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def get_partially_supervised_attention_loader(batch_size,
                                              fake_epoch_size,
                                              supervised_prob,
                                              num_workers=4,
                                              pin_memory=False):

    unsupervised_normalize = transforms.Normalize((0.1307,), (0.3081,))
    supervised_normalize = transforms.Normalize((0.1307*255,), (0.3081*255,))

    unsupervised_dataset = GenModelDataset(unsupervised_normalize,
                                           epoch_size=None,
                                           fix_data=False)

    supervised_dataset = AttentionTargetDataset("attention_target_data",
                                                transform=supervised_normalize)

    mixture_dataset = MixtureDataset(supervised_dataset,
                                     unsupervised_dataset,
                                     supervised_prob)

    # sampler only needs the __len__ attribute to be big
    sampler = SequentialSampler(range(fake_epoch_size))
    data_loader = torch.utils.data.DataLoader(
        mixture_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
