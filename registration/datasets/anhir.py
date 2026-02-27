from pathlib import Path
import pandas as pd
import numpy as np
import cv2

from torch.utils.data import Dataset
import torchvision.transforms as transforms

MPP_DICT = {
    'lung-lesion': 0.174,
    'lung-lobes': 1.274,
    'mammary-gland': 2.294,
    'mice-kidney': 0.227,
    'COAD': 0.468,
    'gastric': 0.2528,
    'breast': 0.2528,
    'kidney': 0.2528
}

class ANHIRDataset(Dataset):
    def __init__(self, data_dir, target_mpp=8):
        self.data_dir = Path(data_dir)
        self.target_mpp = target_mpp
        self.tensor_transform = transforms.ToTensor()

        meta_file = pd.read_csv(self.data_dir / "dataset_medium.csv")
        meta_file = meta_file[meta_file["status"] == "training"].reset_index(drop=True)

        self.data_list = []
        for i, row in meta_file.iterrows():
            slide_id = row['Source image'].split('/')[0]
            slide_name = slide_id.split('_')[0]
            scale_pc = int(row['Source image'].split('/')[1].split('-')[-1].replace('pc', ''))
            mpp = MPP_DICT[slide_name] * (100/scale_pc)
            diagonal = float(row['Image diagonal [pixels]'])

            src_stain = row['Source image'].split('/')[2].split('.')[0]
            src_image_path = str(self.data_dir / row['Source image'])
            src_landmarks_path = str(self.data_dir / row['Source landmarks'])
            
            tgt_stain = row['Target image'].split('/')[2].split('.')[0]
            tgt_image_path = str(self.data_dir / row['Target image'])
            tgt_landmarks_path = str(self.data_dir / row['Target landmarks'])

            self.data_list.append({
                'slide_id': slide_id,
                'scale_pc': scale_pc,
                'mpp': mpp,
                'diagonal': diagonal,
                'src_stain': src_stain,
                'tgt_stain': tgt_stain,
                'src_image_path': src_image_path,
                'src_landmarks_path': src_landmarks_path,
                'tgt_image_path': tgt_image_path,
                'tgt_landmarks_path': tgt_landmarks_path
            })

    def __len__(self):
        return len(self.data_list)

    def _load_image(self, fpath):
        image = cv2.imread(fpath, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.array(image)

    def _load_landmarks(self, fpath):
        df = pd.read_csv(fpath)
        landmarks = df.iloc[:, 1:].values
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
        data_dict = self.data_list[index]

        landmarks_src = self._load_landmarks(data_dict['src_landmarks_path'])
        landmarks_tgt = self._load_landmarks(data_dict['tgt_landmarks_path'])

        img_src = self._load_image(data_dict['src_image_path'])   #  (H x W x C)
        img_tgt = self._load_image(data_dict['tgt_image_path'])   #  (H x W x C)

        try:
            img_src = self._resize_image(img_src, data_dict['mpp'], self.target_mpp)
            img_tgt = self._resize_image(img_tgt, data_dict['mpp'], self.target_mpp)
            actual_mpp = self.target_mpp
        except ValueError:
            actual_mpp = data_dict['mpp']

        green_channel = img_src[..., 1]
        gray_img_src = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2RGB)
        gray_img_src = self.tensor_transform(gray_img_src)    #  (C x H x W)

        green_channel = img_tgt[..., 1]
        gray_img_tgt = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2RGB)
        gray_img_tgt = self.tensor_transform(gray_img_tgt)  #  (C x H x W)

        target = {
            'slide_id': data_dict['slide_id'],
            'src_stain': data_dict['src_stain'],
            'tgt_stain': data_dict['tgt_stain'],
            'mpp_lv0': data_dict['mpp'],
            'mpp_target': actual_mpp,
            'diagonal': data_dict['diagonal'],
            'landmarks_src': landmarks_src,
            'landmarks_tgt': landmarks_tgt,
        }

        return gray_img_src, gray_img_tgt, target
