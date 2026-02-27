from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import openslide
import cv2
from .utils import get_mpp_from_slide

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

Image.MAX_IMAGE_PIXELS = None


class ACROBATDataset(Dataset):
    def __init__(self, meta_data_path, data_dir, target_level, target_mpp):
        super().__init__()
        self.meta_data = pd.read_csv(meta_data_path)
        self.anon_id_list = self.meta_data['anon_id'].unique()
        self.data_dir = Path(data_dir)
        self.target_level = target_level
        self.target_mpp = target_mpp
        self.tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.anon_id_list)

    def _load_slide(self, fpath):
        slide = openslide.OpenSlide(fpath)
        mpp_lv0 = get_mpp_from_slide(slide)
        mpp_target = mpp_lv0 * slide.level_downsamples[self.target_level]
        full_size_at_level = slide.level_dimensions[self.target_level]

        image = slide.read_region(
            location=(0,0), 
            level=self.target_level,
            size=full_size_at_level
        ).convert("RGB")
        slide.close()
        return np.array(image), mpp_lv0, mpp_target

    def _resize_image(self, image, mpp, target_mpp):
        scale = mpp / target_mpp
        if scale > 1:
            raise ValueError(f"Scale is greater than 1: {scale}")
        
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return resized_image
    
    def __getitem__(self, index):
        anon_id = self.anon_id_list[index]
        group_df = self.meta_data[self.meta_data['anon_id'] == anon_id]

        assert len(group_df['anon_filename_he'].unique()) == 1, \
            f"Multiple HE files found for anon_id: {anon_id}"
        assert len(group_df['anon_filename_ihc'].unique()) == 1, \
            f"Multiple IHC files found for anon_id: {anon_id}"

        he_filename = group_df['anon_filename_he'].values[0].replace('.ndpi', '.tiff')
        ihc_filename = group_df['anon_filename_ihc'].values[0].replace('.ndpi', '.tiff')
        mpp_lv0_he = group_df['mpp_he_10X'].values[0]
        mpp_lv0_ihc = group_df['mpp_ihc_10X'].values[0]


        img_he, _, mpp_he = self._load_slide(self.data_dir / he_filename)      #  (H x W x C)
        img_ihc, _, mpp_ihc = self._load_slide(self.data_dir / ihc_filename)   #  (H x W x C)
        
        img_he = self._resize_image(img_he, mpp_he, self.target_mpp)
        img_ihc = self._resize_image(img_ihc, mpp_ihc, self.target_mpp)
        
        gray_img_he = np.array(img_he)[..., 1]
        gray_img_ihc = np.array(img_ihc)[..., 1]
        
        gray_img_he = cv2.cvtColor(gray_img_he, cv2.COLOR_GRAY2RGB)
        gray_img_he = self.tensor_transform(gray_img_he)    #  (C x H x W)

        gray_img_ihc = cv2.cvtColor(gray_img_ihc, cv2.COLOR_GRAY2RGB)
        gray_img_ihc = self.tensor_transform(gray_img_ihc)  #  (C x H x W)
        
        target = {
            'slide_id': anon_id,
            'stain': group_df['ihc_antibody'].values[0],
            'mpp_lv0_he': mpp_lv0_he,
            'mpp_lv0_ihc': mpp_lv0_ihc,
            'mpp_he': self.target_mpp,
            'mpp_ihc': self.target_mpp,
            'landmarks_he': group_df[['he_x', 'he_y']].values,
            'landmarks_ihc': group_df[['ihc_x', 'ihc_y']].values,
        }

        return gray_img_he, gray_img_ihc, target


def save_combined_preprocessed_images(img1, img2, output_path):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    max_h = max(h1, h2)
    total_w = w1 + w2
    
    if len(img1.shape) == 3:
        combined = np.full((max_h, total_w, img1.shape[2]), 255, dtype=np.uint8)
    else:
        combined = np.full((max_h, total_w), 255, dtype=np.uint8)
    
    combined[:h1, :w1] = img1
    combined[:h2, w1:] = img2
    
    cv2.imwrite(str(output_path), combined)
    return combined