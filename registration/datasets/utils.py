from typing import Tuple
import openslide

import torch
import torch.nn.functional as F


def resize_longest_side(
    image: torch.Tensor, 
    target_size: int
) -> Tuple[torch.Tensor, float]:
    
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor, but got {type(image)}. "
            f"Only torch.Tensor is supported."
        )
    
    was_2d = False
    if image.dim() == 2:
        image = image.unsqueeze(0)  # (H, W) -> (1, H, W)
        was_2d = True
    elif image.dim() != 3:
        raise ValueError(
            f"Expected 2D tensor (H, W) or 3D tensor (C, H, W), but got {image.dim()}D tensor."
        )
    
    _, height, width = image.shape
    if width > height:
        new_width = target_size
        resizing_factor = target_size / width
        new_height = int(height * resizing_factor)
    else:
        new_height = target_size
        resizing_factor = target_size / height
        new_width = int(width * resizing_factor)
    
    image_batch = image.unsqueeze(0)  # (1, C, H, W) or (1, 1, H, W)
    resized_image = F.interpolate(
        image_batch, 
        size=(new_height, new_width), 
        mode='bilinear', 
        align_corners=False
    )
    resized_image = resized_image.squeeze(0)  # (C, H, W) or (1, H, W)
    
    if was_2d:
        resized_image = resized_image.squeeze(0)  # (1, H, W) -> (H, W)
    
    return resized_image, resizing_factor


def get_mpp_from_slide(slide: openslide.OpenSlide) -> float:
    metadata = slide.properties
    if 'tiff.ResolutionUnit' in metadata and 'tiff.XResolution' in metadata:
        unit = metadata['tiff.ResolutionUnit']
        try:
            # Convert resolution value to float
            resolution = float(metadata['tiff.XResolution'])
        except ValueError:
            raise ValueError(f"Error: cannot convert tiff.XResolution to float: {metadata['tiff.XResolution']}")

        if unit == 'centimeter':
            return (1.0 / resolution) * 10000.0
        elif unit == 'inch':
            return (1.0 / resolution) * 25400.0
        else:
            raise ValueError(f"Error: 'tiff.ResolutionUnit' is {unit}. Must be 'centimeter' or 'inch'.")
