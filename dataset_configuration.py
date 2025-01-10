import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import sys
sys.path.append("..")

from dataloader_trans10k.trans10k import TransSegmentation


def prepare_dataset(data_name,
                    dataset_path,
                    batch_size=1,
                    test_batch=1,
                    datathread=4,
                    logger=None):
    
    # set the config parameters
    dataset_config_dict = dict()
    
    assert data_name == 'trans10k' # only one dataset is supported

    to_tensor = transforms.ToTensor()
    image_size = 1024

    # Load Datasets
    train_kwargs = {'transform': to_tensor, 'base_size': image_size,}
    val_kwargs = train_kwargs.copy() 
    test_kwargs = train_kwargs.copy()

    trans10k = TransSegmentation
    train_dataset = trans10k(dataset_path, split='train', mode='train', **train_kwargs)
    val_dataset = trans10k(dataset_path, split='validation', mode='val', difficulty='mix', **val_kwargs)
    test_dataset = trans10k(dataset_path, split='test', mode='testval', difficulty='easy', **test_kwargs)


    img_height, img_width = image_size, image_size

    datathread=4 # 4 seems fine in testing
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))

    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)

    # Set up data loaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    val_loader = DataLoader(val_dataset, batch_size = test_batch, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = test_batch, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    
    num_batches_per_epoch = len(train_loader)
    
    dataset_config_dict['num_batches_per_epoch'] = num_batches_per_epoch
    dataset_config_dict['img_size'] = (img_height,img_width)
    
    
    return (train_loader, val_loader, test_loader), dataset_config_dict


def normalize_mask(mask):
    clipped_mask = torch.clamp(mask, 0, 1)
    normalized_mask = (clipped_mask - 0.5) *2
    return normalized_mask