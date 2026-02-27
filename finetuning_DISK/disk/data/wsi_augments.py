import numpy as np
import cv2
from skimage.transform import PiecewiseAffineTransform, warp
from typing import Optional, Union, Tuple
from skimage import color

def stain_color_jitter(image, brightness=0.6, contrast=0.6, saturation=0.28, hue=0.1, gamma_range=(0.6, 1.5)):
    """
    Color augmentation to simulate stain variation in pathology images.
    (Recommended to apply in RGB before extracting Green channel.)
    
    Args:
        image: RGB image (H, W, 3) - uint8 or float
        brightness: brightness adjustment range
        contrast: contrast adjustment range
        saturation: saturation adjustment range (for RGB)
        hue: hue adjustment range (simulates H&E color shift)
        gamma_range: gamma correction range (simulates tissue thickness / stain concentration difference)
    """
    # Convert to 0~1 float for computation
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.copy()

    # 1. Random Gamma Correction (non-linear brightness -> tissue thickness / stain concentration)
    if np.random.rand() < 0.5:
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        img = np.power(img, gamma)

    # 2. HSV conversion then Color Jittering
    # Adjusting in HSV preserves stain balance better than per-channel multiplication in RGB
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)

    # Hue - subtle change of H&E purple/pink tones
    hue_factor = np.random.uniform(-hue, hue)
    h = (h + hue_factor * 360) % 360

    # Saturation - faded or intense stain feel
    sat_factor = np.random.uniform(1 - saturation, 1 + saturation)
    s = np.clip(s * sat_factor, 0, 1)

    # Value (brightness)
    val_factor = np.random.uniform(1 - brightness, 1 + brightness)
    v = np.clip(v * val_factor, 0, 1)

    # Merge back
    img_hsv = cv2.merge([h, s, v])
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    # 3. Contrast
    cont_factor = np.random.uniform(1 - contrast, 1 + contrast)
    img_rgb = (img_rgb - 0.5) * cont_factor + 0.5
    
    # Clip to valid range
    img_rgb = np.clip(img_rgb, 0, 1)

    # Restore original dtype
    if image.dtype == np.uint8:
        return (img_rgb * 255).astype(np.uint8)
    return img_rgb
    

def channel_mixing_green_extraction(image, mix_limit=0.1):
    """
    When using Green channel only, mix other channels slightly to simulate
    scanner filter differences or stain spectrum shift.
    
    Returns:
        gray_image: (H, W) Green-based grayscale image
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32)

    r, g, b = cv2.split(image)

    # Use Green as base, mix R and B randomly
    # e.g. G_new = 0.95*G + 0.02*R + 0.03*B
    
    # 1. Mix ratios (sum need not be 1; will normalize)
    # Green weight: 1.0 - mix_limit ~ 1.0
    w_g = np.random.uniform(1.0 - mix_limit, 1.0)
    # Remaining weights
    w_r = np.random.uniform(0, mix_limit)
    w_b = np.random.uniform(0, mix_limit)

    mixed = w_g * g + w_r * r + w_b * b
    
    # Clipping and type conversion
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)
    mixed = cv2.cvtColor(mixed, cv2.COLOR_GRAY2RGB)
    return mixed


def affine_transform(image, roi_mask, coord_grid, angle, scale, tx, ty):
    """
    Apply random translation, rotation, and scaling to input numpy image.
    
    Args:
        image (np.ndarray): input image (H, W) or (H, W, C)
        coord_grid (np.ndarray): coordinate grid (H, W, 2)
        angle (float): rotation angle (degrees)
        scale (float): scaling factor
        tx (float): translation x
        ty (float): translation y
    
    Returns:
        transformed (np.ndarray): transformed image
        transformed_roi_mask (np.ndarray): transformed mask
        coord_affine (np.ndarray): transformed coordinate grid (H, W, 2)
        M (np.ndarray): applied affine matrix (2x3)
    """
    h, w = image.shape[:2]

    # Build rotation (around center) + scale matrix
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Add translation
    M[0, 2] += tx
    M[1, 2] += ty

    # Apply affine transform
    transformed = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    transformed_roi_mask = cv2.warpAffine(
        roi_mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT
    )

    coord_affine = cv2.warpAffine(coord_grid, M, (w, h), flags=cv2.INTER_LINEAR)

    return transformed, transformed_roi_mask, coord_affine, M


def elastic_transform(image, roi_mask, coord_grid, dx, dy, alpha=30, sigma=4):
    """
    Apply elastic deformation to image (non-linear).
    
    Args:
        image: input image
        roi_mask: input mask
        coord_grid: coordinate grid (H, W, 2)
        dx: displacement x
        dy: displacement y
        alpha: deformation strength (pixels; higher = more distortion)
        sigma: smoothness of deformation (smoothing; higher = gentler bend)
    
    Returns:
        distorted_image: distorted image
        distorted_mask: distorted mask
        map_x, map_y: coordinate mapping used for deformation (for flow field computation)
    """
    h, w = image.shape[:2]

    # Gaussian Blur & Apply alpha
    # If sigma is large, the wave will be thick, and if it is small, it will be rough.
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    # Create grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Add displacement to grid
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    # Remap function to reposition pixels (Interpolation)
    distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    distorted_mask = cv2.remap(roi_mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    distorted_coord_grid = cv2.remap(coord_grid, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return distorted_image, distorted_mask, distorted_coord_grid


def piecewise_affine_transform(image, mask, coord, grid_rows=4, grid_cols=4, magnitude=8.0, random_state=None):
    """
    Transform image, mask, and coordinate system using skimage's PiecewiseAffineTransform.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    h, w = image.shape[:2]

    # 1. Source control points - create regular grid
    rows = np.linspace(0, h, grid_rows)
    cols = np.linspace(0, w, grid_cols)
    rows_grid, cols_grid = np.meshgrid(rows, cols)
    src_cols = cols_grid.flatten()
    src_rows = rows_grid.flatten()
    src = np.vstack([src_cols, src_rows]).T  # (N, 2) [x, y] format

    # 2. Destination control points - add noise
    # Edge (image boundary) points can be fixed or damped; here we add noise to all and let warp handle boundaries
    dst = src.copy()
    
    # Apply random shift to grid points (here applied to all)
    shift_x = random_state.uniform(-magnitude, magnitude, src.shape[0])
    shift_y = random_state.uniform(-magnitude, magnitude, src.shape[0])
    
    dst[:, 0] += shift_x
    dst[:, 1] += shift_y

    # 3. Build transform (dst -> src for inverse mapping)
    # Warping needs inverse (where did target pixel come from in source)
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)  # estimate transform from src to dst

    # 4. Warp image and coordinates (warp uses inverse mapping internally)
    # output_shape same as input. order=1 (Linear), order=0 (Nearest)
    
    # Image (linear interpolation recommended)
    img_warped = warp(image, tform.inverse, output_shape=(h, w), order=1, 
                      mode='reflect', preserve_range=True).astype(image.dtype)
    
    # Mask (nearest interpolation required - values must not blend)
    mask_warped = warp(mask, tform.inverse, output_shape=(h, w), order=0, 
                       mode='reflect', preserve_range=True).astype(mask.dtype)
    
    # Coordinates (linear interpolation - values should vary smoothly)
    coord_warped = warp(coord, tform.inverse, output_shape=(h, w), order=1, 
                        mode='reflect', preserve_range=True).astype(np.float32)

    return img_warped, mask_warped, coord_warped


def random_aug_transform(
    image: np.ndarray,
    roi_mask: np.ndarray,
    color_jitter: bool = True,
    mix_channel: bool = True,
    max_translate: float = 0.1,
    max_rotate: float = 15,
    scale_range: tuple = (0.9, 1.1),
    distortion_prob: float = 0.5,     # Total probability of applying distortion (Elastic or Piecewise)
    elastic_weight: float = 0.5,      # When distorting, probability of choosing Elastic (rest is Piecewise)
    alpha_range: tuple = (20, 50),
    sigma_range: tuple = (4, 6),
    grid_grid: tuple = (4, 4),        # (rows, cols)
    magnitude_range: tuple = (10, 30),
    random_state: Optional[Union[int, np.random.RandomState]] = None
):
    if random_state is None:
        random_state = np.random.RandomState(None)
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    h, w = image.shape[:2]

    # Create coordinate grid
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    coord_grid = np.stack([x_coords, y_coords], axis=-1).astype(np.float32) # Shape: (H, W, 2)

    # 1. Color Jitter & Mixing
    img_jit = stain_color_jitter(image) if color_jitter else image
    img_mix = channel_mixing_green_extraction(img_jit) if mix_channel else img_jit

    # 2. Random Affine Transform (Global Geometric)
    angle = random_state.uniform(-max_rotate, max_rotate)
    scale = random_state.uniform(scale_range[0], scale_range[1])
    tx = random_state.uniform(-max_translate, max_translate) * w
    ty = random_state.uniform(-max_translate, max_translate) * h

    img_affine, mask_affine, coord_affine, affine_matrix = affine_transform(
        img_mix, roi_mask, coord_grid, angle, scale, tx, ty
    )
    
    # 3. Random Non-rigid Transform (Elastic OR Piecewise)
    img_final, mask_final, coord_final = img_affine, mask_affine, coord_affine

    if random_state.rand() < distortion_prob:
        if random_state.rand() < elastic_weight:
            # Elastic Deformation
            dx = random_state.rand(h, w) * 2 - 1
            dy = random_state.rand(h, w) * 2 - 1
            alpha = random_state.uniform(alpha_range[0], alpha_range[1])
            sigma = random_state.uniform(sigma_range[0], sigma_range[1])
            
            img_final, mask_final, coord_final = elastic_transform(
                img_affine, mask_affine, coord_affine, dx, dy, alpha, sigma
            )
            
        else:
            # Piecewise Affine Transformation
            magnitude = random_state.uniform(magnitude_range[0], magnitude_range[1])
            
            img_final, mask_final, coord_final = piecewise_affine_transform(
                image=img_affine,
                mask=mask_affine,
                coord=coord_affine,
                grid_rows=grid_grid[0],
                grid_cols=grid_grid[1],
                magnitude=magnitude,
                random_state=random_state
            )

    return img_final, mask_final, coord_final
