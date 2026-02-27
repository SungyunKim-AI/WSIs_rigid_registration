import numpy as np
from typing import List, Tuple, Optional, Union
import torch
import cv2

# tensor (B, C, H, W) or (C, H, W) -> numpy array (H, W, C)
def tensor_to_numpy(data: torch.Tensor) -> np.ndarray:
    assert isinstance(data, torch.Tensor), "Please input tensor type"
    if data.dim() == 3:
        numpy_array = data.detach().cpu().numpy()
    elif data.dim() == 4:
        numpy_array = data.squeeze(0).detach().cpu().numpy()
    else:
        raise ValueError(f"Invalid tensor dimension: {data.dim()}")
    return np.transpose(numpy_array, (1, 2, 0))  

# numpy array (H, W, C) -> tensor (C, H, W)
def numpy_to_tensor(data: np.ndarray) -> torch.Tensor:
    assert isinstance(data, np.ndarray), "Please input numpy array type"
    if data.ndim == 3:
        if data.max() > 1.1:
            _data = data.astype(np.float32) / 255.0
        else:
            _data = data.astype(np.float32)
        _data = _data.transpose(2,0,1)
        tensor = torch.from_numpy(_data).float()
    else:
        raise ValueError(f"Invalid numpy array dimension: {data.ndim}")
    return tensor


# micrometer -> pixel
def downscale_landmarks(
    landmarks: np.ndarray,
    mpp: float,
    level: int=0
) -> np.ndarray:
    scale_factor = mpp * (2**level)
    return (landmarks / scale_factor).astype(float)


# pixel -> micrometer
def upscale_landmarks(
    landmarks: np.ndarray,
    mpp: float,
    level: int=0
) -> np.ndarray:
    scale_factor = mpp * (2**level)
    return (landmarks * scale_factor).astype(float)


def rotate_and_flip(
    image_size: Tuple[int, int],    # (W, H)
    coord: np.ndarray, # (N, 2)
    angle: int, 
    flip: bool, 
) -> np.ndarray: # (N, 2)
    w, h = image_size
    x = coord[:, 0].copy() # (N,)
    y = coord[:, 1].copy() # (N,)
    
    if angle == 90:
        new_x = h - 1 - y
        new_y = x
        current_w, current_h = h, w
    elif angle == 180:
        new_x = w - 1 - x
        new_y = h - 1 - y
        current_w, current_h = w, h
    elif angle == 270:
        new_x = y
        new_y = w - 1 - x
        current_w, current_h = h, w
    else:
        new_x = x
        new_y = y
        current_w, current_h = w, h

    if flip:
        new_x = current_w - 1 - new_x
    return np.stack([new_x, new_y], axis=1)


def get_tile_bboxes(
    dsize: Tuple[int, int], 
    tile_size: Union[int, Tuple[int, int]]
) -> List[Tuple[int, int, int, int]]:
    width, height = dsize
    bboxes = []

    if isinstance(tile_size, int):
        t_w = t_h = tile_size
    else:
        t_w, t_h = tile_size
    
    for y_min in range(0, height, t_h):
        for x_min in range(0, width, t_w):
            x_max = min(x_min + t_w, width)
            y_max = min(y_min + t_h, height)
            
            bboxes.append((x_min, y_min, x_max, y_max))
            
    return bboxes

def find_tile_bbox(
    bboxes: List[Tuple[int, int, int, int]], 
    points: Tuple[float, float]
) -> Optional[int]:
    # bounding_boxes: List of bounding boxes, each defined as [x_min, y_min, x_max, y_max]
    # points: List of points, each defined as [x, y]
    x, y = points
    for i, (x_min, y_min, x_max, y_max) in enumerate(bboxes):
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return i
    return None


def get_affine_matrix(
    m_kpts0: np.ndarray, 
    m_kpts1: np.ndarray, 
    match_threshold: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    if m_kpts0.shape[0] < match_threshold:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([], dtype=np.int32)
    else:
        matrix, inliers = cv2.estimateAffinePartial2D(m_kpts0, m_kpts1, method=cv2.RANSAC)
        return matrix, inliers


def decompose_affine_matrix(matrix: np.ndarray) -> Tuple[float, float, float, float, float]:
    # Extract affine matrix components
    a, b, tx = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    c, d, ty = matrix[1, 0], matrix[1, 1], matrix[1, 2]

    # Rotation angle (radian)
    rotation_angle_rad = np.arctan2(b, a)
    # Rotation angle (degrees)
    rotation_angle_deg = np.degrees(rotation_angle_rad)

    # Scaling (x, y directions)
    scale_x = np.sqrt(a**2 + b**2)
    scale_y = np.sqrt(c**2 + d**2)

    # Translation
    translation_x = tx
    translation_y = ty

    # return rotation_angle_deg, scale_x, scale_y, translation_x, translation_y
    return np.array([rotation_angle_deg, scale_x, scale_y, translation_x, translation_y], dtype=np.float32)


def reconstruct_affine_matrix(params: np.ndarray) -> np.ndarray:
    # Reconstruct 2x3 Affine Matrix from the five decomposed parameters.
    rot_deg, sx, sy, tx, ty = params.tolist()
    rot_rad = np.radians(rot_deg)
    cos_v = np.cos(rot_rad)
    sin_v = np.sin(rot_rad)

    # Reassemble: assume order Scale -> Rotation -> Translation (typical Affine)
    # Matrix = [[sx*cos, -sy*sin, tx], [sx*sin, sy*cos, ty]]
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0] = sx * cos_v
    matrix[0, 1] = -sy * sin_v
    matrix[0, 2] = tx
    matrix[1, 0] = sx * sin_v
    matrix[1, 1] = sy * cos_v
    matrix[1, 2] = ty
    return matrix

# Convert affine matrix (m) estimated from downscaled image to original-scale affine matrix (M)
def convert_matrix_to_original_scale(
    matrix_m: np.ndarray, 
    resize_factor_0: float, 
    resize_factor_1: float
) -> np.ndarray:
    m_3x3 = np.vstack([matrix_m, [0, 0, 1]])
    T_A = np.array([
        [resize_factor_0, 0,   0], 
        [0,   resize_factor_0, 0],
        [0,   0,   1]
    ], dtype=float)

    if resize_factor_1 == 0:
         raise ValueError("f_B (target resize factor) cannot be zero.")
    inv_f_B = 1.0 / resize_factor_1
    T_B_inv = np.array([
        [inv_f_B, 0,       0], 
        [0,       inv_f_B, 0], 
        [0,       0,       1]
    ], dtype=float)
    M_3x3 = T_B_inv @ m_3x3 @ T_A
    matrix_M = M_3x3[:2, :]
    return matrix_M

def get_TRE_list(
    landmarks0: np.ndarray, 
    landmarks1: np.ndarray
) -> float:
    if landmarks0.shape != landmarks1.shape:
        raise ValueError(f"Two landmark arrays must have the same shape (N, D). "
                        f"A: {landmarks0.shape}, B: {landmarks1.shape}")
    
    diff = landmarks0 - landmarks1
    squared_diff = np.square(diff)
    sum_squared_diff = np.sum(squared_diff, axis=1)
    tre_per_landmark = np.sqrt(sum_squared_diff)
    return tre_per_landmark


def filter_keypoints_by_mask(
    mask_he: np.ndarray, 
    mask_ihc: np.ndarray, 
    kpts_he: np.ndarray, 
    kpts_ihc: np.ndarray, 
    m_kpts_he: np.ndarray, 
    m_kpts_ihc: np.ndarray, 
    matches: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    def create_tissue_map(kpts, mask):
        H, W = mask.shape
        N = len(kpts)
        
        kpts_yx = kpts[:, ::-1].astype(int)
        
        valid_y = np.logical_and(kpts_yx[:, 0] >= 0, kpts_yx[:, 0] < H)
        valid_x = np.logical_and(kpts_yx[:, 1] >= 0, kpts_yx[:, 1] < W)
        kpts_valid_mask = np.logical_and(valid_y, valid_x)
        
        kpts_tissue_mask = np.zeros(N, dtype=bool)
        kpts_tissue_mask[kpts_valid_mask] = (
            mask[kpts_yx[kpts_valid_mask, 0], kpts_yx[kpts_valid_mask, 1]] == 0
        )
        
        survival_indices = np.where(kpts_tissue_mask)[0]
        new_indices = np.arange(len(survival_indices))
        old_to_new_index_map = np.full(N, -1, dtype=int)
        old_to_new_index_map[survival_indices] = new_indices

        filtered_kpts = kpts[kpts_tissue_mask]
        return filtered_kpts, kpts_tissue_mask, old_to_new_index_map

    filtered_kpts_he, kpts_he_tissue_mask, he_index_map = create_tissue_map(kpts_he, mask_he)
    filtered_kpts_ihc, kpts_ihc_tissue_mask, ihc_index_map = create_tissue_map(kpts_ihc, mask_ihc)

    is_m_in_tissue_he_by_idx = kpts_he_tissue_mask[matches[:, 0]]
    is_m_in_tissue_ihc_by_idx = kpts_ihc_tissue_mask[matches[:, 1]]
    valid_matches_mask = np.logical_and(is_m_in_tissue_he_by_idx, is_m_in_tissue_ihc_by_idx)
    
    filtered_matches_old_idx = matches[valid_matches_mask]
    filtered_m_kpts_he = m_kpts_he[valid_matches_mask]
    filtered_m_kpts_ihc = m_kpts_ihc[valid_matches_mask]

    new_he_indices = he_index_map[filtered_matches_old_idx[:, 0]]
    new_ihc_indices = ihc_index_map[filtered_matches_old_idx[:, 1]]
    filtered_matches_new = np.stack((new_he_indices, new_ihc_indices), axis=1)
    
    
    return (
        filtered_kpts_he, 
        filtered_kpts_ihc, 
        filtered_m_kpts_he, 
        filtered_m_kpts_ihc, 
        filtered_matches_new
    )
