import torch

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config
from data_loader import get_gen_model_loader


def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

    # instantiate data loaders
    if config.is_train:
        train_loader = get_gen_model_loader(
            config.batch_size,
            epoch_size=54000,
            fix_data=False,
            **kwargs
        )
        valid_loader = get_gen_model_loader(
            config.batch_size,
            epoch_size=6000,
            fix_data=True,
            **kwargs
        )
        data_loader = (train_loader, valid_loader)
    else:
        data_loader = get_gen_model_loader(
            config.batch_size,
            epoch_size=10000,
            fix_data=True,
            **kwargs
        )

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
