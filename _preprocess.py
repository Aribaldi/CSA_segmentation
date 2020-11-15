import os 
import numpy as np
import pandas as pd
import warnings

from skimage import io
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def get_binary_mask(slice_df_ruin, image_path, matrix_path, photo_id, user):
    '''
    Returns ruin binary mask as PIL.Image.
    1 -- ruined.
    '''
    task_df = slice_df_ruin[(slice_df_ruin['photo_id']==photo_id) & (slice_df_ruin['user']==user)]
    task_id = task_df['task_id'].unique()[0]
    source_mask = np.load(f'{matrix_path}matrix_{photo_id}__{task_id}.npz')['data']
    ruin_list = task_df['segment_num'].to_numpy()
    binary_mask = np.isin(source_mask, ruin_list)
    binary_mask = Image.fromarray(binary_mask, 'L')
    
    return binary_mask

def get_ruin_mask(slice_df_ruin, image_path, matrix_path, photo_id, user):
    '''
    Returns ruin mask with breaks and probas as PIL.Image.
    1 -- break,
    2 -- proba.
    '''
    task_df = slice_df_ruin[(slice_df_ruin['photo_id']==photo_id) & (slice_df_ruin['user']==user)]
    task_id = task_df['task_id'].unique()[0]
    source_mask = np.load(f'{matrix_path}matrix_{photo_id}__{task_id}.npz')['data']
    break_list = task_df[task_df['segment_value']=='Разлом']['segment_num'].to_numpy()
    proba_list = task_df[task_df['segment_value']=='Проба']['segment_num'].to_numpy()
    ruin_mask = np.isin(source_mask, break_list)
    ruin_mask[np.isin(source_mask, proba_list)] = 2
    ruin_mask = Image.fromarray(ruin_mask, 'L')

    return ruin_mask

def get_image(raw_df, image_path, photo_id):
    '''
    Returns source image as PIL.Image.
    '''
    photo_df = raw_df[raw_df['Id']==photo_id]
    field, well = photo_df['Field'].to_numpy()[0], photo_df['Well'].to_numpy()[0]
    image = Image.open(f'{image_path}{field}_{well}/{photo_id}.jpeg')
    
    return image

def preprocess(raw_df, slice_df, image_path, matrix_path, output_path, user_list=['markup_expert01'], output_size=(1536,256), proba=False):
    '''
    Preprocess' and saves images as .jpeg in output_path/photos/ and groundtruth labels as .npy in output_path/matrixes/.

    Preprocessing:
    1) Scale source image rateably to output_size[1];
    2) Crop scaled image by output_size[0];
    3) Fill segments outside of boundaries with black.
    '''
    slice_df_ruin = slice_df[slice_df['segment_type']=='Разрушенность'] 

    if not os.path.exists(f'{output_path}photos/'):
        os.makedirs(f'{output_path}photos/')
    if not os.path.exists(f'{output_path}matrixes/'):
        os.makedirs(f'{output_path}matrixes/')
    
    warnings.filterwarnings("ignore")
    for user in np.intersect1d(slice_df_ruin['user'].unique(), user_list):
        for photo_id in tqdm(slice_df_ruin[slice_df_ruin['user']==user]['photo_id'].unique()):
            if proba:
                mask = get_ruin_mask(slice_df_ruin, image_path, matrix_path, photo_id, user)
            else:
                mask = get_binary_mask(slice_df_ruin, image_path, matrix_path, photo_id, user)
            image = get_image(raw_df, image_path, photo_id)
            
            resize_image = transforms.Resize(size=output_size[1])
            resize_mask = transforms.Resize(size=output_size[1], interpolation=Image.NEAREST)
            
            image_scaled = np.array(resize_image(image))
            mask_scaled = np.array(resize_mask(mask))
            
            h_idx = 0
            crop_idx = 0
            while h_idx < image_scaled.shape[0]:
                image_scaled_crop = image_scaled[h_idx : h_idx+output_size[0], :]
                mask_scaled_crop = mask_scaled[h_idx : h_idx+output_size[0], :]
                
                crop_h = image_scaled_crop.shape[0]
                if crop_h < output_size[0]:
                    image_scaled_crop = np.pad(image_scaled_crop, ((0,output_size[0]-crop_h),(0,0),(0,0)), 'constant')
                    mask_scaled_crop = np.pad(mask_scaled_crop, ((0,output_size[0]-crop_h),(0,0)), 'constant')
                
                io.imsave(f'{output_path}photos/{str(photo_id)}_{str(crop_idx)}.jpeg', image_scaled_crop, quality=100)
                np.save(f'{output_path}matrixes/{str(photo_id)}_{str(crop_idx)}.npy', mask_scaled_crop)
                
                h_idx += output_size[0]
                crop_idx += 1