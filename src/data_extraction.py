from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as albu
from albumentations.pytorch import ToTensorV2

root_path = Path('../data')
BATCH_SIZE = 4


class SegDataset(Dataset):
    def __init__(self, root_dir, imageFolder, maskFolder, transform=None, seed=None, fraction=None, subset=None):
        '''
        A custom images dataset; takes photos and masks from a given path an transfroms them using augmentations listed
        in transforms param
        :param root_dir: A base folder with whole dataset
        :param imageFolder: Photos subfolder
        :param maskFolder: Masks subfolder
        :param transform: Callable data transforms, i.e PyTorch/Albumentations augmentations
        :param seed: Random seed
        :param fraction: Train split fraction (the rest is used on validation and test stages)
        :param subset: Train/Val/Test mode
        '''
        self.root_dir = root_dir
        self.transform = transform
        if not fraction:
            self.image_names = sorted(
                glob.glob(os.path.join(self.root_dir, imageFolder, '*')))
            self.mask_names = sorted(
                glob.glob(os.path.join(self.root_dir, maskFolder, '*')))
        else:
            assert (subset in ['Train', 'Valid', 'Test'])
            self.fraction = fraction
            self.image_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, imageFolder, '*'))))
            self.mask_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, maskFolder, '*'))))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            l_bound = int(np.floor(len(self.image_list) * (1 - self.fraction)))
            if subset == 'Train':
                self.image_names = self.image_list[:l_bound]
                self.mask_names = self.mask_list[:l_bound]
            else:
                middle = (len(self.image_list) - l_bound) // 2
                if subset == 'Valid':
                    self.image_names = self.image_list[l_bound: l_bound + middle]
                    self.mask_names = self.mask_list[l_bound:l_bound + middle]
                if subset == 'Test':
                    self.image_names = self.image_list[l_bound + middle:]
                    self.mask_names = self.mask_list[l_bound + middle:]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = Image.open(img_name)
        image = np.array(image)
        mask_name = self.mask_names[idx]
        mask = np.load(mask_name)
        sample = {'image':image, 'mask':mask}
        if self.transform:
            sample = self.transform(image=image, mask=mask)
        sample['image_name'] = img_name
        sample['mask_name'] = mask_name
        return sample


def pixelwise_transforms():
    result = albu.Compose([
        albu.HorizontalFlip(p=0.5)
    ])

    return result


def resize_transforms():
    result = albu.Compose([
        albu.Resize(768, 128, p=1)
      ],
        p=1)
    return result



def get_dataloader_single_folder(data_dir, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                 imageFolder='photos', maskFolder='matrixes',
                                 fraction=0.2,
                                 batch_size=BATCH_SIZE):
    '''
    Make iterable PyTorch DataLoader using instances of SegDataset class
    :param data_dir: A base folder with whole dataset
    :param mean: Parameter used in Normalization transform, set to imagenet mean by default
    :param std: Parameter used in Normalization transform, set to imagenet std by default
    :param imageFolder: Photos subfolder
    :param maskFolder: Masks subfolder
    :param fraction: Train split fraction (the rest is used on validation and test stages)
    :param batch_size: Number of photo-mask pairs in one batch
    '''
    data_transforms = {
        'Train': albu.Compose([resize_transforms(),
                               pixelwise_transforms(),
                               albu.Normalize(mean, std),
                               ToTensorV2()
                               ]),
        'Valid': albu.Compose([resize_transforms(),
                               albu.Normalize(mean, std),
                              ToTensorV2()
                              ]),
        'Test': albu.Compose([resize_transforms(),
                              albu.Normalize(mean, std),
                              ToTensorV2()])
    }
    image_datasets = {
        x: SegDataset(data_dir, imageFolder=imageFolder, maskFolder=maskFolder, seed=100, fraction=fraction, subset=x,
                      transform=data_transforms[x])
        for x in ['Train', 'Valid']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['Train', 'Valid']}
    dataloaders['Test'] = DataLoader(SegDataset(data_dir, imageFolder=imageFolder, maskFolder=maskFolder, seed=100,
                                                fraction=fraction, subset='Test',
                                                transform=data_transforms['Test']),
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=4)
    return dataloaders
