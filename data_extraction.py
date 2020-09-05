from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image

root_path = Path('data')


class SegDataset(Dataset):
    """Segmentation Dataset"""
    def __init__(self, root_dir, imageFolder, maskFolder, transform=None, seed=None, fraction=None, subset=None,
                 imagecolormode='rgb', maskcolormode='grayscale'):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Mask
            -----Mask 1
            -----Mask N
            imageFolder (string) = 'Images' : Name of the folder which contains the Images.
            maskFolder (string)  = 'Masks : Name of the folder which contains the Masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed: Specify a seed for the train and test split
            fraction: A float value from 0 to 1 which specifies the validation split fraction
            subset: 'Train' or 'Test' to select the appropriate set.
            imagecolormode: 'rgb' or 'grayscale'
            maskcolormode: 'rgb' or 'grayscale'
        """
        self.color_dict = {'rgb': 1, 'grayscale': 0}
        assert (imagecolormode in ['rgb', 'grayscale'])
        assert (maskcolormode in ['rgb', 'grayscale'])

        self.imagecolorflag = self.color_dict[imagecolormode]
        self.maskcolorflag = self.color_dict[maskcolormode]
        self.root_dir = root_dir
        self.transform = transform
        if not fraction:
            self.image_names = sorted(
                glob.glob(os.path.join(self.root_dir, imageFolder, '*')))
            self.mask_names = sorted(
                glob.glob(os.path.join(self.root_dir, maskFolder, '*')))
        else:
            assert (subset in ['Train', 'Test'])
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
            if subset == 'Train':
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[int(
                    np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = Image.open(img_name)
        msk_name = self.mask_names[idx]
        mask = np.load(msk_name)['arr_0']
        mask = Image.fromarray(mask, 'L')
        sample = {'image': image, 'mask': mask}
        if self.transform:
            res = {'image': self.transform(sample['image']), 'mask': self.transform(sample['mask'])}
            return res

        return sample


def get_dataloader_single_folder(data_dir, imageFolder='Images', maskFolder='Masks', fraction=0.2, batch_size=4):
    """
        Create training and testing dataloaders from a single folder.
    """
    data_transforms = {
        'Train': transforms.Compose([transforms.Resize((512, 512)),
                                     transforms.ToTensor(),
                                     ]),
        'Test': transforms.Compose([transforms.Resize((512, 512)),
                                    transforms.ToTensor(),
                                    ]),
    }

    image_datasets = {
        x: SegDataset(data_dir, imageFolder=imageFolder, maskFolder=maskFolder, seed=100, fraction=fraction, subset=x,
                      transform=data_transforms[x])
        for x in ['Train', 'Test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['Train', 'Test']}
    return dataloaders


if __name__ == '__main__':
    train = get_dataloader_single_folder(root_path / 'Dataset')['Train']
    test = get_dataloader_single_folder(root_path / 'Dataset')['Test']
    for tensor in iter(train).next()['mask']:
        print(tensor.shape)
