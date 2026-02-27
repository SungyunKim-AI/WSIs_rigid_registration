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


class HyReCoDataset(Dataset):
    def __init__(self, data_dir, target_level, target_mpp=8):
        super().__init__()
        self.data_list = []
        self.target_level = target_level
        self.target_mpp = target_mpp

        slide_list = ['29', '108', '361', '464', '533', '611', '628', '644', '679']
        ihc_list = ['CD8', 'CD45', 'KI67', 'PHH3']
        
        data_dir = Path(data_dir)
        for slide_id in slide_list:
            for stain in ihc_list:
                he_path = data_dir / 'HE' / slide_id
                ihc_path = data_dir / stain / slide_id
                self.data_list.append((slide_id, stain, str(he_path), str(ihc_path)))
        self.tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data_list)

    def _load_slide(self, fpath):
        slide = openslide.OpenSlide(fpath)
        mpp_lv0 = get_mpp_from_slide(slide)
        mpp_target = mpp_lv0 * slide.level_downsamples[self.target_level]

        w, h = slide.level_dimensions[0]  # Original Level 0 size
        full_size_at_level = slide.level_dimensions[self.target_level]
        image = slide.read_region(
            location=(0,0), 
            level=self.target_level,
            size=full_size_at_level
        ).convert("RGB")
        slide.close()

        return np.array(image), (h,w), mpp_lv0, mpp_target

    def _load_landmarks(self, fpath):
        df = pd.read_csv(fpath, header=None)
        landmarks = df.iloc[:, :-1].values
        landmarks *= 1000
        return landmarks.astype(np.float32)

    def _resize_image(self, image, mpp, target_mpp):
        scale = mpp / target_mpp
        if scale > 1:
            raise ValueError(f"Scale is greater than 1: {scale}")
        
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return resized_image

    def __getitem__(self, index):
        slide_id, stain, path_he, path_ihc = self.data_list[index]

        landmarks_he = self._load_landmarks(f"{path_he}.csv")
        landmarks_ihc = self._load_landmarks(f"{path_ihc}.csv")

        img_he, size_lv0_he, mpp_lv0_he, mpp_he = self._load_slide(f"{path_he}.tif")      #  (H x W x C)
        img_ihc, size_lv0_ihc, mpp_lv0_ihc, mpp_ihc = self._load_slide(f"{path_ihc}.tif")   #  (H x W x C)

        img_he = self._resize_image(img_he, mpp_he, self.target_mpp)
        img_ihc = self._resize_image(img_ihc, mpp_ihc, self.target_mpp)
        
        green_channel = img_he[..., 1]
        gray_img_he = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2RGB)
        gray_img_he = self.tensor_transform(gray_img_he)    #  (C x H x W)

        green_channel = img_ihc[..., 1]
        gray_img_ihc = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2RGB)
        gray_img_ihc = self.tensor_transform(gray_img_ihc)  #  (C x H x W)

        target = {
            'slide_id': slide_id,
            'stain': stain,
            'size_lv0_he': size_lv0_he,
            'mpp_lv0_he': mpp_lv0_he,
            'mpp_he': self.target_mpp,
            'size_lv0_ihc': size_lv0_ihc,
            'mpp_lv0_ihc': mpp_lv0_ihc,
            'mpp_ihc': self.target_mpp,
            'landmarks_he': landmarks_he,
            'landmarks_ihc': landmarks_ihc,
        }

        return gray_img_he, gray_img_ihc, target
