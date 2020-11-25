from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from src.data_extraction import root_path
import numpy as np


def show_test_result(image_id: str, csv_path = root_path / 'result' / 'test.csv'):
    '''
    Show an arbitrary photo, mask and model prediction
    :param image_id: Number of a photo, string type
    :param csv_path: Path to csv with stored photo ids and metric values
    '''
    test_df = pd.read_csv(csv_path)
    iou_value = test_df[test_df['Id'] == image_id]['IoU'].values
    print(iou_value)
    fig, ax = plt.subplots(1, 3, figsize=(20, 40))
    image = Image.open(root_path / 'preprocessed_dataset' / 'photos' / f'{image_id}.jpeg')
    mask = np.load(root_path / 'preprocessed_dataset' / 'matrixes' / f'{image_id}.npy')
    result = np.load(root_path / 'result' / f'{image_id}.npy')
    fig.suptitle(f'Image:{image_id}, IoU:{iou_value}')
    ax[0].set_title('Preprocessed image')
    ax[1].set_title('Mask')
    ax[2].set_title('Prediction')
    ax[0].imshow(image)
    ax[1].imshow(mask)
    ax[2].imshow(result)
    plt.show()


def show_batch(dataloader, n_samples):
    '''
    Visualize first n batches in a given dataloader
    :param dataloader: PyTorch DataLoader
    :param n_samples: number of batches
    '''
    iterator = iter(dataloader)
    for bn in range(n_samples):
        sample = iterator.next()
        print(f'Batch №{bn}')
        print(sample['image'].shape, sample['mask'].shape)
        print(f"Images: \n {sample['image_name']},"
              f"\n Masks: \n {sample['mask_name']}")
        for i in range(sample['image'].shape[0]):
            fig, ax = plt.subplots(1, 4, figsize=(20, 10))
            plt.subplots_adjust(wspace=5)
            fig.suptitle(f'Batch №{bn}')
            ax[0].set_title('Original img')
            ax[1].set_title('Mask')
            ax[2].set_title('Transformed img')
            ax[3].set_title('Transformed mask')
            ax[0].imshow(Image.open(sample['image_name'][i]))
            ax[1].imshow(np.load(sample['mask_name'][i]))
            ax[2].imshow(np.transpose(sample['image'][i], (1, 2, 0)))
            ax[3].imshow(sample['mask'][i])
            plt.show()

