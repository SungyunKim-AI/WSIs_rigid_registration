import cv2
import numpy as np
from typing import Tuple, Optional, Union, List

import torch
import torch.nn.functional as F
from disk import Features
from disk.model import CycleMatcher


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

def get_affine_matrix(
    m_kpts0: np.ndarray, 
    m_kpts1: np.ndarray, 
    match_threshold: int = 3
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    if m_kpts0.shape[0] < match_threshold:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([], dtype=np.int32)
    result = cv2.estimateAffinePartial2D(m_kpts0, m_kpts1, method=cv2.RANSAC)
    if result is None:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([], dtype=np.int32)
    matrix, inliers = result
    if matrix is None:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if inliers is None:
        inliers = np.array([], dtype=np.int32)
    return matrix, inliers

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
    # return np.mean(tre_per_landmark)
    return tre_per_landmark


def extract_keypoints(
    disk,
    image: torch.Tensor,
    tile_size: int,
    max_kpts: int = 2048
) -> Features:
    """Extract keypoints/descriptors from a single image with DISK. Returns one DISK Features object."""
    device = image.device
    _, H, W = image.shape
    bboxes = get_tile_bboxes((W, H), tile_size)

    all_kpts = []
    all_scores = []
    all_descs = []

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        tile_h, tile_w = y_max - y_min, x_max - x_min

        if tile_h < 32 or tile_w < 32:
            continue

        tile = image[:, y_min:y_max, x_min:x_max].unsqueeze(0)  # (1, C, H, W)

        # U-Net requirement: height/width must be multiples of 2^n (default 16)
        MULT = 16
        H_pad = ((tile_h + MULT - 1) // MULT) * MULT
        W_pad = ((tile_w + MULT - 1) // MULT) * MULT
        if tile_h != H_pad or tile_w != W_pad:
            # (left, right, top, bottom) for last 2 dims
            tile = F.pad(tile, (0, W_pad - tile_w, 0, H_pad - tile_h), mode='constant', value=0)

        with torch.no_grad():
            features_ = disk.features(tile, kind='nms')
            feat = features_.reshape(tile.shape[0])[0]  # single DISK Features

        kpts = feat.kp
        scores = feat.kp_logp if feat.kp_logp is not None else torch.zeros(feat.kp.shape[0], device=device)
        descs = feat.desc

        # Remove keypoints in padding region (original tile coords: [0, tile_w) x [0, tile_h))
        valid = (kpts[:, 0] < tile_w) & (kpts[:, 1] < tile_h)
        kpts = kpts[valid]
        scores = scores[valid]
        descs = descs[valid]

        offset = torch.tensor([[x_min, y_min]], device=device, dtype=torch.float32)
        kpts = kpts + offset

        all_kpts.append(kpts)
        all_scores.append(scores)
        all_descs.append(descs)

    if len(all_kpts) == 0:
        desc_dim = getattr(disk, 'desc_dim', 128)
        all_kpts = torch.empty((0, 2), device=device)
        all_scores = torch.empty((0,), device=device)
        all_descs = torch.empty((0, desc_dim), device=device)
    else:
        all_kpts = torch.cat(all_kpts, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_descs = torch.cat(all_descs, dim=0)

    if max_kpts > 0 and all_scores.shape[0] > max_kpts:
        top_k_indices = torch.topk(all_scores, max_kpts).indices
        all_kpts = all_kpts[top_k_indices]
        all_scores = all_scores[top_k_indices]
        all_descs = all_descs[top_k_indices]

    return Features(all_kpts, all_descs, all_scores)

def global_matching(
    matcher: CycleMatcher,
    feats_src: Features,
    feats_tgt: Features
) -> Tuple[np.ndarray, int, int]:
    """Match two Features with CycleMatcher then RANSAC affine; returns (matrix, num_matches, num_inliers)."""
    # [1, 2]: one scene with src and tgt images
    features = np.array([[feats_src, feats_tgt]], dtype=object)
    with torch.no_grad():
        matched_pairs = matcher.match_pairwise(features)

    pairs = matched_pairs[0, 0]
    kps1 = pairs.kps1
    kps2 = pairs.kps2
    matches = pairs.matches

    num_matches = matches.shape[1] if matches.numel() > 0 else 0
    if num_matches > 0:
        m_kpts0 = kps1[matches[0]].cpu().numpy().astype(np.float32)
        m_kpts1 = kps2[matches[1]].cpu().numpy().astype(np.float32)
        global_affine, inliers = get_affine_matrix(m_kpts0, m_kpts1)
        num_inliers = int(np.sum(inliers)) if inliers is not None and inliers.size > 0 else 0
    else:
        global_affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        num_inliers = 0

    return global_affine, num_matches, num_inliers





def run_registration(
    disk,
    matcher,
    src_image: torch.Tensor, 
    tgt_image: torch.Tensor,
    landmarks_src: np.ndarray,
    landmarks_tgt: np.ndarray
) -> Tuple[float, float, int, int]:
    """Return TRE (90 percentile, mean), num_matches, num_inliers from disk, matcher, HE/IHC images and landmarks."""
    # Remove batch dim: (1, C, H, W) -> (C, H, W)
    if src_image.dim() == 4:
        src_image = src_image.squeeze(0)
    if tgt_image.dim() == 4:
        tgt_image = tgt_image.squeeze(0)
    if torch.is_tensor(landmarks_src):
        landmarks_src = landmarks_src.cpu().numpy()
    if torch.is_tensor(landmarks_tgt):
        landmarks_tgt = landmarks_tgt.cpu().numpy()
    if landmarks_src.ndim == 3:
        landmarks_src = landmarks_src[0]
    if landmarks_tgt.ndim == 3:
        landmarks_tgt = landmarks_tgt[0]

    landmarks_src = np.asarray(landmarks_src, dtype=np.float64)
    landmarks_tgt = np.asarray(landmarks_tgt, dtype=np.float64)

    tile_size = 512
    feats_src = extract_keypoints(disk, src_image, tile_size, max_kpts=0)
    feats_tgt = extract_keypoints(disk, tgt_image, tile_size, max_kpts=0)

    try:
        global_affine, num_matches, num_inliers = global_matching(
            matcher, feats_src, feats_tgt
        )
    except RuntimeError:
        # Matching failed (e.g. zero keypoints in one image)
        global_affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        num_matches = 0
        num_inliers = 0

    # If matches are too few, retry with downscaled images
    if num_matches < 5000:
        _, H_src, W_src = src_image.shape
        _, H_tgt, W_tgt = tgt_image.shape

        src_image_downscaled = F.interpolate(
            src_image.unsqueeze(0),
            size=(H_src // 2, W_src // 2),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        tgt_image_downscaled = F.interpolate(
            tgt_image.unsqueeze(0),
            size=(H_tgt // 2, W_tgt // 2),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        tile_size_downscaled = tile_size // 2
        feats_src_downscaled = extract_keypoints(disk, src_image_downscaled, tile_size_downscaled, max_kpts=0)
        feats_tgt_downscaled = extract_keypoints(disk, tgt_image_downscaled, tile_size_downscaled, max_kpts=0)

        try:
            global_affine_downscaled, num_matches, num_inliers = global_matching(
                matcher, feats_src_downscaled, feats_tgt_downscaled
            )
            global_affine = convert_matrix_to_original_scale(
                global_affine_downscaled,
                resize_factor_0=0.5,
                resize_factor_1=0.5
            )
        except RuntimeError:
            pass

    warped_landmarks = cv2.transform(
        landmarks_src.reshape(-1, 1, 2), global_affine
    ).reshape(-1, 2)    # (N, 2)

    tre_list_px = get_TRE_list(warped_landmarks, landmarks_tgt)  # in pixels (MPP mpp)
    tre_list_um = tre_list_px * 15  # convert to µm: 1 pixel = mpp µm
    tre_90s = np.percentile(tre_list_um, 90)
    tre_m = np.mean(tre_list_um)

    return tre_90s, tre_m, num_matches, num_inliers

