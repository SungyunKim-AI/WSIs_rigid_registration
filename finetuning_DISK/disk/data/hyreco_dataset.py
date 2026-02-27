from pathlib import Path
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class HyReCoDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_list = []

        slide_list = ['29', '108', '361', '464', '533', '611', '628', '644', '679']
        ihc_list = ['CD8', 'CD45', 'KI67', 'PHH3']
        
        data_dir = Path(data_dir)
        for slide_id in slide_list:
            for stain in ihc_list:
                self.data_list.append({
                    'slide_id': slide_id,
                    'stain': stain,
                    'he_image_path': str(data_dir / "image_mpp_8" / f"{slide_id}_HE.png"),
                    'he_landmarks_path': str(data_dir / "landmarks_mpp_8" / f"{slide_id}_HE.csv"),
                    'ihc_image_path': str(data_dir / "image_mpp_8" / f"{slide_id}_{stain}.png"),
                    'ihc_landmarks_path': str(data_dir / "landmarks_mpp_8" / f"{slide_id}_{stain}.csv")
                })
        self.tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]

        landmarks_he = np.loadtxt(data['he_landmarks_path'], delimiter=',', dtype=float)
        landmarks_ihc = np.loadtxt(data['ihc_landmarks_path'], delimiter=',', dtype=float)

        img_he = cv2.imread(data['he_image_path'], cv2.IMREAD_COLOR)
        img_he = cv2.cvtColor(img_he, cv2.COLOR_BGR2RGB)
        img_he = self.tensor_transform(img_he)    #  (C x H x W)

        img_ihc = cv2.imread(data['ihc_image_path'], cv2.IMREAD_COLOR)
        img_ihc = cv2.cvtColor(img_ihc, cv2.COLOR_BGR2RGB)
        img_ihc = self.tensor_transform(img_ihc)  #  (C x H x W)

        target = {
            'landmarks_he': landmarks_he,
            'landmarks_ihc': landmarks_ihc,
        }

        return img_he, img_ihc, target
