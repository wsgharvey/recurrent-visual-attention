import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset

# for generative model
from mea.examples.mnist import mnist_model
from attention_target_dataset import AttentionTargetDataset

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
                 fix_offset,
                 supervise_attention_freq):
        """
        transform: preprocessing step for data
        fix_data: if True, will use index as random seed when returning a sample
        """
        self.transform = transform
        self.epoch_size = epoch_size
        self.fix_data = fix_data
        self.fix_offset = fix_offset
        self.supervise_attention_freq = supervise_attention_freq
        self.supervise_indicator = 0

    def __getitem__(self, index):
        self.supervise_indicator += supervise_attention_freq
        if self.supervise_indicator >= 1:
            self.supervise_indicator -= 1
            raise NotImplementedError("Mixed in attention targets are not yet implemented.")
        else:
            return self._generate_unsupervised(index)

    def _generate_unsupervised(self, index):
        if self.fix_data:
            rng_state = torch.get_rng_state()
            torch.manual_seed(index+self.fix_offset)
        trace_dict = mnist_model()
        if self.fix_data:
            torch.set_rng_state(rng_state)
        raw = trace_dict['image'].data.view(1, 28, 28)
        image = self.transform(raw)
        digit_label = trace_dict['label']
        attention_target = torch.zeros(10)
        attention_target_exists = torch.tensor(0.)
        return image, digit_label, target, target_exists

    def __len__(self):
        return self.epoch_size

def get_gen_model_loader(batch_size,
                         epoch_size,
                         fix_data,
                         fix_offset=0,
                         num_workers=4,
                         pin_memory=False,
                         supervise_attention_freq=0):
    """
    imitates data loaders but supplies infinite stream from generative model

    fix random seed externally if desired

    supervise_attention_freq (float in range [0, 1]) is how often to give a
    sample with attention targets

    actually very silly
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    dataset = GenModelDataset(normalize,
                              epoch_size,
                              fix_data=fix_data,
                              fix_offset=fix_offset,
                              supervise_attention_freq=supervise_attention_freq)

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
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    dataset = AttentionTargetDataset("attention_target_data",
                                     transform=trans)

    sampler = SubsetRandomSampler(range(len(dataset)))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
