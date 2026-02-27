"""
Simply load images from a folder or nested folders (does not have any split),
and apply affine transformations to it. Yields an image pair without border
artifacts.
"""

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from omegaconf import OmegaConf

from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .augmentations_wsi import random_aug_transform
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class AffinePatch_Dataset(BaseDataset):
    default_conf = {
        # image search
        "data_dir": None, # TODO: Dataset directory
        "glob": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        # splits
        "train_size": 100,
        "val_size": 10,
        "shuffle_seed": 0,  # or None to skip
        # image loading
        "reseed": False,
        "right_only": True,
        "affine": {
            "color_jitter": True,
            "mix_channel": True,
            "max_translate": 0.08,
            "max_rotate": 30,
            "scale_range": (0.85, 1.15),
            "distortion_prob": 0.5,
            "elastic_weight": 0.5,
            "alpha_range": (40, 70),
            "sigma_range": (8, 12),
            "grid_grid": (4, 4),
            "magnitude_range": (20, 40),
            "patch_shape": [512, 512]
        }
    }

    def _init(self, conf):
        data_dir = Path(conf.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(data_dir)
        
        images = []
        for subfolder in data_dir.iterdir():
            crop_folder = subfolder / "crops"
            if crop_folder.exists() and crop_folder.is_dir():
                images.extend(crop_folder.glob("*.png"))
            else:
                raise ValueError(f"Cannot find any image in folder: {crop_folder}.")
        
        # images = [i.relative_to(data_dir).as_posix() for i in images]
        images = sorted(images)  # for deterministic behavior
        logger.info("Found %d images in data directory.", len(images))

        if conf.shuffle_seed is not None:
            np.random.RandomState(conf.shuffle_seed).shuffle(images)
        train_images = images[: conf.train_size]
        val_images = images[conf.train_size : conf.train_size + conf.val_size]
        self.images = {"train": train_images, "val": val_images}

    def get_dataset(self, split):
        return _Dataset(self.conf, self.images[split], split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, images, split):
        self.conf = conf
        self.split = split
        self.images = np.array(images)

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def _read_view(self, img, crop_size):
        grid_image = cv2.imread(img)
        grid_image = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
        image0 = grid_image[0:crop_size[0], 0:crop_size[1]]

        grid_gray = cv2.cvtColor(grid_image, cv2.COLOR_RGB2GRAY)
        mask_region = grid_gray[0:crop_size[0], crop_size[1]:crop_size[1]*2]
        mask0 = (mask_region >= 128).astype(np.float32)

        src_img, src_mask, src_coord = random_aug_transform(
            image0,
            mask0,
            color_jitter=self.conf.affine.color_jitter,
            mix_channel=self.conf.affine.mix_channel,
            max_translate=self.conf.affine.max_translate,
            max_rotate=self.conf.affine.max_rotate,
            scale_range=self.conf.affine.scale_range,
            distortion_prob=self.conf.affine.distortion_prob,
            elastic_weight=self.conf.affine.elastic_weight,
            alpha_range=self.conf.affine.alpha_range,
            sigma_range=self.conf.affine.sigma_range,
            grid_grid=self.conf.affine.grid_grid,
            magnitude_range=self.conf.affine.magnitude_range
        )

        if self.conf.right_only:
            green_channel = image0[..., 1]
            tgt_img = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2RGB)
            tgt_mask = mask0

            x_coords, y_coords = np.meshgrid(np.arange(image0.shape[1]), np.arange(image0.shape[0]))
            tgt_coord = np.stack([x_coords, y_coords], axis=-1).astype(np.float32) # Shape: (H, W, 2)

        else:
            tgt_img, tgt_mask, tgt_coord = random_aug_transform(
                image0,
                mask0,
                color_jitter=self.conf.affine.color_jitter,
                mix_channel=self.conf.affine.mix_channel,
                max_translate=self.conf.affine.max_translate,
                max_rotate=self.conf.affine.max_rotate,
                scale_range=self.conf.affine.scale_range,
                distortion_prob=self.conf.affine.distortion_prob,
                elastic_weight=self.conf.affine.elastic_weight,
                alpha_range=self.conf.affine.alpha_range,
                sigma_range=self.conf.affine.sigma_range,
                grid_grid=self.conf.affine.grid_grid,
                magnitude_range=self.conf.affine.magnitude_range
            )

        data0 = {
            "image": ToTensor()(tgt_img),
            "mask": tgt_mask,
            "coord": tgt_coord,
            "image_size": np.array(crop_size, dtype=np.float32)
        }

        data1 = {
            "image": ToTensor()(src_img),
            "mask": src_mask,
            "coord": src_coord,
            "image_size": np.array(crop_size, dtype=np.float32)
        }

        return data0, data1

    def getitem(self, idx):
        image = self.images[idx]
        ps = self.conf.affine.patch_shape

        data0, data1 = self._read_view(image, ps)

        data = {
            "name": image.name,
            "original_image_size": np.array(ps),
            "coord0": data0["coord"],
            "coord1": data1["coord"],
            "idx": idx,
            "view0": data0,
            "view1": data1,
        }

        return data

    def __len__(self):
        return len(self.images)


def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = AffinePatch_Dataset(conf)
    loader = dataset.get_data_loader("train")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
            )
    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
