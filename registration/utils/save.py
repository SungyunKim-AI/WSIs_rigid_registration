import numpy as np
from pathlib import Path
from typing import Union, Optional
import torch
import cv2
import matplotlib.pyplot as plt
from models import viz2d

def save_matching_result(
    image0: Union[torch.Tensor, np.ndarray], 
    m_kpts0: np.ndarray, 
    image1: Union[torch.Tensor, np.ndarray], 
    m_kpts1: np.ndarray, 
    output_dir: Union[str, Path], 
    filename: str,
    kpts0: Optional[np.ndarray] = None,
    kpts1: Optional[np.ndarray] = None,
    angle: Optional[int] = None, 
    flip: Optional[bool] = None
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    filename = f"{filename}_{angle}_{flip}" if angle is not None and flip is not None else filename
    kpts_path = output_dir / f"{filename}_kpts.png"
    matches_path = output_dir / f"{filename}_mkpts.png"

    if kpts0 is not None and kpts1 is not None:
        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints([kpts0, kpts1], colors="lime", ps=1)
        viz2d.save_plot(kpts_path)

    viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", ps=1, lw=0.2)
    viz2d.save_plot(matches_path)

    plt.close()

def save_landmarks(
    ref_image: torch.Tensor, 
    landmarks0: np.ndarray, 
    landmarks1: np.ndarray, 
    output_dir: Union[str, Path], 
    filename: str
) -> None:
    image = ref_image.cpu().numpy().transpose(1,2,0)
    image = (image*255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for coord in landmarks0:
        cv2.circle(image, tuple(coord), radius=2, color=(0, 0, 255), thickness=-1)  # red
    
    for coord in landmarks1:
        cv2.circle(image, tuple(coord), radius=2, color=(0, 255, 0), thickness=-1)  # green
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / f"landmarks_{filename}.png"
    cv2.imwrite(str(save_path), image)

def save_landmarks_pair(
    image0: torch.Tensor, 
    image1: torch.Tensor, 
    landmarks0: np.ndarray, 
    landmarks1: np.ndarray, 
    output_dir: Union[str, Path], 
    filename: str
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / f"landmarks_{filename}.png"

    viz2d.plot_images([image0, image1])
    viz2d.plot_matches(landmarks0, landmarks1, color="lime", ps=1, lw=0.2)
    viz2d.save_plot(save_path)

def save_local_refinement_result(
    tile0: np.ndarray,
    tile1: np.ndarray,
    local_m_kpts0_cropped: np.ndarray,
    local_m_kpts1_cropped: np.ndarray,
    output_dir: Union[str, Path],
    file_name: str,
    local_matrix_crop: Optional[np.ndarray] = None,
) -> None:
    """
    Save separated tiles (tile0, tile1). If local_matrix_crop is given,
    also save tile0 aligned by local_matrix once more.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1) Save separated tiles (matching result visualization)
    save_matching_result(
        tile0, local_m_kpts0_cropped,
        tile1, local_m_kpts1_cropped,
        output_dir, file_name,
    )

    # 2) Save tiles aligned by local_matrix (concatenate two tile images only)
    if local_matrix_crop is not None:
        h, w = tile0.shape[:2]
        tile0_aligned = cv2.warpAffine(tile0, local_matrix_crop, (w, h))
        padding_width = 10
        black_strip = np.zeros((h, padding_width, tile0.shape[2]), dtype=tile0.dtype)
        concatenated = np.concatenate([tile0_aligned, black_strip, tile1], axis=1)
        aligned_path = output_dir / f"{file_name}_aligned.png"
        cv2.imwrite(str(aligned_path), cv2.cvtColor(concatenated, cv2.COLOR_RGB2BGR))
