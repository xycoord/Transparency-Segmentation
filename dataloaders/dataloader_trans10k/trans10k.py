"""Prepare Trans10K dataset"""
import os
import torch
import numpy as np
import logging

from PIL import Image

from .settings import cfg

class TransSegmentation(object):
    """Trans10K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Trans10K folder. Default is './datasets/Trans10K'
    split: string
        'train', 'validation', 'test'
    difficulty: string
        'easy', 'hard', 'mix'
    mode : string
        'train', 'val', 'testval'
    transform : callable, optional
        A function that transforms the image
    augmentation : callable, optional
        A function that augments the image 
    ignore_class : bool, optional
        Whether to ignore the class 'things' and 'stuff' and only consider 'transparency'
    """
    NUM_CLASS = 3


    def __init__(self, root='./datasets/Trans10K', split='train', difficulty='mix', mode=None, transform=None, augmentation=None, ignore_class=True):
        super(TransSegmentation, self).__init__()

        self.root = os.path.join(cfg.ROOT_PATH, root)
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/Trans10K"

        self.split = split
        self.mode = mode if mode is not None else split

        self.transform = transform
        self.augmentation = augmentation 

        self.ignore_class = ignore_class


        self.images, self.mask_paths = _get_trans10k_pairs(self.root, self.split, difficulty=difficulty)

        assert (len(self.images) == len(self.mask_paths))

        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

        if ignore_class:
            self.valid_classes = [0,1]
            self._key = np.array([0,1])
        else:
            self.valid_classes = [0,1,2]
            self._key = np.array([0,1,2])
        
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32') + 1


    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)

        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)


    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index])

        img = np.array(img)
        mask = np.array(mask)[:,:,:3].mean(-1)

        # Things and Stuff
        if self.ignore_class:
            mask[mask!=0] = 1
            assert mask.max()<=1, mask.max()
        else:
            mask[mask==85.0] = 1
            mask[mask==255.0] = 2 
            assert mask.max()<=2, mask.max()

        # Augmentation and Normalization
        augmented = self.augmentation(image=img, mask=mask)
        augmented = self.transform(image=augmented['image'], mask=augmented['mask'])
        img_aug = augmented['image'].to(torch.float32)
        mask_aug = augmented['mask'].to(torch.float32)

        transformed = self.transform(image=img, mask=mask)
        img = transformed['image'].to(torch.float32)
        mask = transformed['mask'].to(torch.float32)
        
        return img, mask, img_aug, mask_aug, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def get_img_size(self):
        return 512, 512


    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        if self.ignore_class:
            return ('background', 'transparency')
        return ('background', 'things', 'stuff')

    @property
    def num_class(self):
        """Number of categories."""
        if self.ignore_class:
            return 2
        return self.NUM_CLASS
    

def _get_trans10k_pairs(folder, split='train', difficulty='mix'):

    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        imgs = os.listdir(img_folder)

        for imgname in imgs:
            imgpath = os.path.join(img_folder, imgname)
            maskname = imgname.replace('.jpg', '_mask.png')
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                logging.info('cannot find the mask or image:', imgpath, maskpath)

        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths


    if split == 'train':
        img_folder = os.path.join(folder, split, 'images')
        mask_folder = os.path.join(folder, split, 'masks')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    else:
        assert split == 'validation' or split == 'test'
        easy_img_folder = os.path.join(folder, split, 'easy', 'images')
        easy_mask_folder = os.path.join(folder, split, 'easy', 'masks')
        hard_img_folder = os.path.join(folder, split, 'hard', 'images')
        hard_mask_folder = os.path.join(folder, split, 'hard', 'masks')
        easy_img_paths, easy_mask_paths = get_path_pairs(easy_img_folder, easy_mask_folder)
        hard_img_paths, hard_mask_paths = get_path_pairs(hard_img_folder, hard_mask_folder)
        if difficulty == 'easy':
            return easy_img_paths, easy_mask_paths
        elif difficulty == 'hard':
            return hard_img_paths, hard_mask_paths
        else:
            assert difficulty == 'mix'
            easy_img_paths.extend(hard_img_paths)
            easy_mask_paths.extend(hard_mask_paths)
            img_paths = easy_img_paths
            mask_paths = easy_mask_paths

    return img_paths, mask_paths

if __name__ == '__main__':
    dataset = TransSegmentation()
