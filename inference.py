import torch
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from data_extraction import get_dataloader_single_folder, root_path
from PIL import Image


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#model = smp.Unet('efficientnet-b3', classes=1, activation='sigmoid')
model = torch.load('best_model.pt')
model.to(device)
model.eval()

test = get_dataloader_single_folder(root_path / 'Dataset')['Test']

for sample in test:
    image, mask = sample['image'], sample['mask']
    names_list = sample['image_name']
    print(names_list)
    image = image.cuda()
    image = image.float()
    predict = model(image)
    predict = predict.squeeze().cpu().detach().numpy().round()
    fig, ax = plt.subplots(4, 3)
    fig.suptitle('Test')
    ax[0, 0].set_title('Original image')
    ax[0, 1].set_title('Mask')
    ax[0, 2].set_title('Prediction')
    mask = mask.cpu().numpy()
    for i in range(4):
        image = Image.open(names_list[i])
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask[i])
        ax[i, 2].imshow(predict[i])
    plt.show()



