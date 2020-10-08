from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as albu
from albumentations.pytorch import ToTensorV2, ToTensor
from matplotlib import pyplot as plt
from segmentation_models_pytorch.encoders import get_preprocessing_fn

root_path = Path('data')
IMAGE_SIZE = 224
BATCH_SIZE = 4


class SegDataset(Dataset):
    def __init__(self, root_dir, imageFolder, maskFolder, transform=None, seed=None, fraction=None, subset=None,
                 preprocess=None, training_mode = 'deafult'):
        self.preprocess = preprocess
        self.root_dir = root_dir
        self.transform = transform
        self.training_mode = training_mode
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
        mask = np.load(mask_name)['arr_0']
        #mask = Image.fromarray(mask, 'L')
        sample = {'image':image, 'mask':mask}
        if self.transform:
            sample = self.transform(image=image, mask=mask)
        if self.preprocess:
            sample = self.preprocess(image=sample['image'], mask=sample['mask'])
        #sample['image'] = np.transpose(sample['image'], (2, 0, 1))
        sample['image_name'] = img_name
        sample['mask_name'] = mask_name
        return sample #доделать вариант для обучения через smp


def pre_transforms(image_size=IMAGE_SIZE):
    return albu.Resize(image_size, image_size, p=1)


def pw_transforms(): #дополнить
    result = albu.Compose([
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        #albu.HueSaturationValue(p=1),
        #albu.Blur(blur_limit=2)
        #albu.Downscale()
    ])

    return result


def resize_transforms(image_size=IMAGE_SIZE):
    pre_size = int(image_size * 1.5)
    random_crop = albu.Compose([
      albu.SmallestMaxSize(pre_size, p=1),
      albu.RandomCrop(
          image_size, image_size, p=1
      )
    ])
    rescale = albu.Resize(image_size, image_size, p=1)
    result = albu.OneOf([
          random_crop,
          rescale,
      ], p=1)
    return result





def get_dataloader_single_folder(data_dir, imageFolder='Images', maskFolder='Masks', fraction=0.2, batch_size=BATCH_SIZE):
    data_transforms = {
        'Train': albu.Compose([resize_transforms(),
                               pw_transforms(),
                               ToTensor()
                               ]),
        'Valid': albu.Compose([pre_transforms(), #Прочекать среднее и std для Normilise()
                              ToTensor()
                              ]),
        'Test': albu.Compose([albu.Resize(512, 512, p=1),
                              ToTensorV2()])
    }
    image_datasets = {
        x: SegDataset(data_dir, imageFolder=imageFolder, maskFolder=maskFolder, seed=100, fraction=fraction, subset=x,
                      transform=data_transforms[x])
        for x in ['Train', 'Valid',  'Test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['Train', 'Valid', 'Test']}
    return dataloaders


def show_batch(dataloader, batch_size):
    for bn, sample in enumerate(dataloader):
        print(f'Batch №{bn}')
        print(sample['image'].shape, sample['mask'].shape)
        print(f"Images: \n {sample['image_name']},"
              f"\n Masks: \n {sample['mask_name']}")
        fig, ax = plt.subplots(batch_size, 4)
        fig.suptitle(f'Batch №{bn}')
        ax[0, 0].set_title('Original image')
        ax[0, 1].set_title('Mask')
        ax[0, 2].set_title('Transformed image')
        ax[0, 3].set_title('Transformed mask')
        for i in range(batch_size):
            ax[i, 0].imshow(Image.open(sample['image_name'][i]))
            ax[i, 1].imshow(np.load(sample['mask_name'][i])['arr_0'])
            ax[i, 2].imshow(np.transpose(sample['image'][i], (1, 2, 0)))
            ax[i, 3].imshow(np.transpose(sample['mask'][i], (1, 2, 0)))
        plt.show()



if __name__ == '__main__':
    train = get_dataloader_single_folder(root_path / 'Dataset')['Train']
    valid = get_dataloader_single_folder(root_path / 'Dataset')['Valid']
    test = get_dataloader_single_folder(root_path / 'Dataset')['Test']
    for tensor in iter(test).next()['image']:
        print(tensor.shape, type(tensor))
        print(tensor)
    show_batch(train, BATCH_SIZE)
