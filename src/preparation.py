import numpy as np
import pandas as pd
from pathlib import Path
import os
import shutil


root_path = Path('data')
img_path = Path(root_path / 'photos')


def create_folders():
    if not os.path.exists(root_path / 'preprocessed'):
        os.mkdir(root_path / 'preprocessed')
    if not os.path.exists(root_path / 'dataset'):
        os.mkdir(root_path / 'dataset')

def prepare_data(data_path=root_path / 'raw/data.csv',
                 experts_only=True):
    data = pd.read_csv(data_path)
    ruin = set(data[data['segment_type'] == 'Разрушенность']['photo_id'].values)
    core = set(data[data['segment_type'] == 'Порода']['photo_id'].values)
    inter = ruin.intersection(core)
    data = data[data['photo_id'].isin(inter)]
    data = data[['photo_id','task_id','user','segment_num', 'segment_type', 'segment_value']]
    if experts_only:
        users_list = ['markup_expert01']
        data = data[data['user'].isin(users_list)]
    data.to_csv(root_path / 'preprocessed' / 'filtered.csv')
    return data

