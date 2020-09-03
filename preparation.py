import numpy as np
import pandas as pd
from pathlib import Path
import os
import shutil


root_path = Path('data')
img_path = Path('/home/iref/PycharmProjects/CSA_new_pipeline/core-sample-snakemake/data/raw/photos/')
segments_to_labels = {
    'Аргиллит': 0,
    'Алевролит': 1,
    'Песчаник': 2,
    'Переслаивание пород': 3
}


def create_folders():
    if not os.path.exists(root_path / 'interim'):
        os.mkdir(root_path / 'Dataset')
        os.mkdir(root_path / 'Dataset/Images')
        os.mkdir(root_path / 'Dataset/Masks')


def prepare_data(data_path=root_path / 'raw/segmented_data.csv',
                 experts_only=True):
    data = pd.read_csv(data_path)
    white_list = ['Песчаник', 'Алевролит', 'Аргиллит', 'Переслаивание пород']
    data = data[data['segment_value'].isin(white_list) & (data['photo_type'] == 'ДС')
                & (data['segment_type'] == 'Порода')]
    data = data[['photo_id','task_id','user','segment_num','segment_value']]
    if experts_only:
        users_list = ['markup_expert01', 'markup_expert02']
        data = data[data['user'].isin(users_list)]
    return data


def relocate_images(df: pd.DataFrame):
    f_dict = {}
    for (dirpath, dirnames, filenames) in os.walk(img_path):
        for file in filenames:
            f_dict[file[:-5]] = img_path / dirpath / file

    for idd in df['photo_id'].values:
        shutil.copy(f_dict[str(idd)], root_path/ f'Dataset/Images/{idd}.jpeg')


def convert_matrixes(df: pd.DataFrame):
    photos = df['photo_id'].unique()
    for p_id in photos:
        temp_df = df[df['photo_id'] == p_id]
        task_id = temp_df['task_id'].values[0]
        file_name = f'matrix_{p_id}__{task_id}.npz'
        matrix = np.load(root_path / 'raw/matrixes' / file_name)['data']

        for index, row in temp_df.iterrows():
            seg_num = row['segment_num']
            seg_val = row['segment_value']
            res = np.where(matrix == seg_num, segments_to_labels[seg_val], matrix)
            np.savez_compressed(root_path / 'Dataset/Masks' / file_name, res)


if __name__ == '__main__':
    create_folders()
    main_df = prepare_data()
    main_df.to_csv(root_path / 'cleared.csv')
    relocate_images(main_df)
    convert_matrixes(main_df)
