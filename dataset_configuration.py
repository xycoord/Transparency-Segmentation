from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import os
import sys
sys.path.append("..")

from dataloader_trans10k.trans10k import TransSegmentation as Trans10k


def get_trans10k_train_loader(dataset_path, batch_size=1, logger=None):
    return prepare_dataloader('trans10k', dataset_path, split='train', mode='train', shuffle=True, batch_size=batch_size, logger=logger)

def get_trans10k_val_loader(dataset_path, difficulty='mix', batch_size=1, logger=None):
    return prepare_dataloader('trans10k', dataset_path, split='validation', mode='val', difficulty=difficulty, shuffle=False, batch_size=batch_size, logger=logger)

def get_trans10k_test_loader(dataset_path, difficulty='mix', batch_size=1, logger=None):
    return prepare_dataloader('trans10k', dataset_path, split='test', mode='testval', difficulty=difficulty, shuffle=False, batch_size=batch_size, logger=logger)


def prepare_dataloader(data_name,
                    dataset_path,
                    split,
                    mode,
                    difficulty=None,
                    shuffle=False,
                    batch_size=1,
                    datathread=4, # 4 seems fine in testing
                    logger=None):
    
    assert data_name == 'trans10k' # only one dataset is supported

    if split in ['validation', 'test']:
        assert difficulty in ['easy', 'hard', 'mix']

    datathread = check_datathread(datathread, logger)

    # Load Datasets
    image_size = 1024
    dataset_kwargs = {'transform': ToTensor(), 'base_size': image_size,}
    dataset = Trans10k(dataset_path, split=split, mode=mode, **dataset_kwargs)

    # Set up data loaders
    train_loader = DataLoader(dataset, batch_size = batch_size, \
                            shuffle = shuffle, num_workers = datathread, \
                            pin_memory = True)

    return train_loader


def check_datathread(datathread, logger=None):
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))

    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)

    return datathread

