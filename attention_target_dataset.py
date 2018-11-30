import numpy as np
import pickle
import os

from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import makedir_exist_ok


class AttentionTargetDataset(Dataset):
    data_file_name = 'training.pt'

    def __init__(self, root, transform=None, target_transform=None, loadfrom=None):
        self.processed_folder = os.path.join(root, "processed")
        self.training_path = os.path.join(self.processed_folder, self.data_file_name)
        self.transform = transform
        self.target_transform = target_transform

        if loadfrom is not None:
            self._process(loadfrom)

        if not self._processed_data_exists():
            raise RuntimeError('No processed data found.')

        self.images, self.labels, self.attention_targets, self.posterior_targets =\
            torch.load(self.training_path)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        attention_target = self.attention_targets[index]
        posterior_target = self.posterior_targets[index]

        image = Image.fromarray(image.numpy(), mode='L')

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, label, attention_target, posterior_target

    def _process(self, raw_path):
        """
        Loads data from specified folder, which should have format:
         - <raw_path>
           - images
             - 0.png
             - 1.png
             ...
           - logs
             - 0.p
             - 1.p
             ...
        where each *.p is a pickled dictionary containing optimal attention
        locations in format ... with key ...
        """
        makedir_exist_ok(self.processed_folder)

        images = torch.Tensor([])
        labels = torch.LongTensor([])
        attention_targets = torch.Tensor([])
        posterior_targets = torch.Tensor([])

        for img_path, log_path in self._iter_raw_files(raw_path):
            image = torch.Tensor(np.array(Image.open(img_path))).view(1, 28, 28)
            log = pickle.load(open(log_path, 'rb'))
            label = log["true_digit"].view(1)
            attention_target = torch.Tensor(log["designs"]).view(1, 10)
            posterior_target = torch.cat([dp.view(1, 10) for dp in log['digit_posteriors']]).view(1, 11, 10)

            images = torch.cat((images, image), dim=0)
            labels = torch.cat((labels, label), dim=0)
            attention_targets = torch.cat((attention_targets, attention_target), dim=0)
            posterior_targets = torch.cat((posterior_targets, posterior_target), dim=0)

        data = (images, labels, attention_targets, posterior_targets)
        with open(self.training_path, 'wb') as f:
            torch.save(data, f)

    def _iter_raw_files(self, raw_path):
        i = 0
        while True:
            img_path = os.path.join(raw_path, 'images', f"{i}.png")
            log_path = os.path.join(raw_path, 'logs', f"{i}.p")
            if os.path.exists(img_path):
                yield img_path, log_path
                i += 1
            else:
                break

    def _processed_data_exists(self):
        return os.path.exists(self.training_path)

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import argparse

    arg_lists = []
    parser = argparse.ArgumentParser(description='Attention target dataset')
    parser.add_argument('raw_data_folder', type=str,
                        help='path of raw data folder')
    args = parser.parse_args()

    AttentionTargetDataset("attention_target_data", loadfrom=args.raw_data_folder)
