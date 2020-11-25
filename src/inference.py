import pandas as pd
import numpy as np
import re


def test(test_loader, device, model, metric, output_path):
    '''
    Get model outputs and given metrics scores for images in test split

    :param test_loader: PyTorchDataLoader for test split
    :param device: PyTorch device abstraction
    :param model:  Trained model
    :param metric: Test metric; could be PyTorch or Catalyst metrics
    :param output_path: Path to save model output matrices and overall data csv to
    '''
    res_df = pd.DataFrame(columns=['Id', 'IoU'])
    score = 0
    j = 0
    for sample in test_loader:
        j += 1
        image, mask = sample['image'], sample['mask']
        image_name = re.findall(r'\d{7}_\d', sample['image_name'][0])[0]
        image = image.to(device)
        image = image.float()
        mask = mask.to(device)
        predict = model(image)
        single_score = np.round(metric(predict, mask).item(), 2)
        score += single_score
        res_df.loc[j] = [image_name] + [single_score]
        predict = predict.squeeze().cpu().detach().numpy().round()
        np.save(output_path / f'{image_name}.npy', predict)
    score /= len(test_loader.dataset)
    print(f'Overall test IoU: {score}')
    res_df.to_csv(output_path / 'test.csv')

