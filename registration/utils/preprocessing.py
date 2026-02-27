import numpy as np
import cv2
from PIL import Image
from skimage import exposure, filters
from sklearn.cluster import MiniBatchKMeans
from scipy.interpolate import Akima1DInterpolator
import colour

def rgb2jab(rgb, cspace='CAM16UCS'):
    eps = np.finfo("float").eps
    if np.issubdtype(rgb.dtype, np.integer) and rgb.max() > 1:
        rgb01 = rgb / 255.0
    else:
        rgb01 = rgb
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        jab = colour.convert(rgb01 + eps, 'sRGB', cspace)
    return jab

def jab2rgb(jab, cspace='CAM16UCS'):
    eps = np.finfo("float").eps
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb = colour.convert(jab + eps, cspace, 'sRGB')
    return rgb

def rgb2jch(rgb, cspace='CAM16UCS'):
    jab = rgb2jab(rgb, cspace)
    jch = colour.models.Jab_to_JCh(jab)
    return jch

def rgb2od(rgb_img):
    eps = np.finfo("float").eps
    if np.issubdtype(rgb_img.dtype, np.integer) or rgb_img.max() > 1:
        rgb01 = rgb_img / 255.0
    else:
        rgb01 = rgb_img
    od = -np.log10(rgb01 + eps)
    od[od < 0] = 0
    return od

def stainmat2decon(stain_mat_srgb255):
    od_mat = rgb2od(stain_mat_srgb255)
    # Handle singular matrix case roughly
    M = od_mat / (np.linalg.norm(od_mat, axis=1, keepdims=True) + 1e-9)
    M[np.isnan(M)] = 0
    D = np.linalg.pinv(M)
    return D

def deconvolve_img(rgb_img, D):
    od_img = rgb2od(rgb_img)
    deconvolved_img = np.dot(od_img, D)
    deconvolved_img[deconvolved_img < 0] = 0
    return deconvolved_img

def calc_background_color_dist(img, brightness_q=0.99, mask=None):
    eps = np.finfo("float").eps
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        if 1 < img.max() <= 255 and np.issubdtype(img.dtype, np.integer):
            cam = colour.convert(img / 255 + eps, 'sRGB', 'CAM16UCS')
        else:
            cam = colour.convert(img + eps, 'sRGB', 'CAM16UCS')

    if mask is None:
        brightest_thresh = np.quantile(cam[..., 0], brightness_q)
    else:
        if np.sum(mask) == 0:
            brightest_thresh = np.max(cam[..., 0])
        else:
            brightest_thresh = np.quantile(cam[..., 0][mask > 0], brightness_q)

    brightest_idx = np.where(cam[..., 0] >= brightest_thresh)
    if len(brightest_idx[0]) == 0:
        bright_cam = np.array([100, 0, 0]) 
    else:
        brightest_pixels = cam[brightest_idx]
        bright_cam = brightest_pixels.mean(axis=0)
        
    cam_d = np.sqrt(np.sum((cam - bright_cam)**2, axis=2))
    return cam_d

def mask2contours(mask, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_dilated = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask_dilated)
    for cnt in contours:
        cv2.drawContours(contour_mask, [cnt], 0, 255, -1)
    return contour_mask

def combine_masks_by_hysteresis(mask_list):
    if len(mask_list) == 0: return None
    mshape = mask_list[0].shape
    to_hyst_mask = np.zeros(mshape)
    for m in mask_list:
        to_hyst_mask[m > 0] += 1
    
    hyst_mask = 255 * filters.apply_hysteresis_threshold(to_hyst_mask, 0.5, len(mask_list) - 0.5).astype(np.uint8)
    return hyst_mask

def create_tissue_mask_from_rgb(img, brightness_q=0.99, kernel_size=3, gray_thresh=0.075, light_gray_thresh=0.875, dark_gray_thresh=0.7):
    jch = rgb2jch(img)
    light_greys = 255 * ((jch[..., 1] < gray_thresh) & (jch[..., 0] < light_gray_thresh)).astype(np.uint8)
    dark_greys = 255 * ((jch[..., 1] < gray_thresh) & (jch[..., 0] < dark_gray_thresh)).astype(np.uint8)
    grey_mask = combine_masks_by_hysteresis([light_greys, dark_greys])
    
    color_mask = 255 - grey_mask
    cam_d = calc_background_color_dist(img, brightness_q=brightness_q, mask=color_mask)
    
    vert_knl = np.ones((1, 5), dtype=np.uint8)
    no_v_lines = cv2.morphologyEx(cam_d.astype(np.float32), cv2.MORPH_OPEN, vert_knl)
    
    horiz_knl = np.ones((5, 1), dtype=np.uint8)
    no_h_lines = cv2.morphologyEx(cam_d.astype(np.float32), cv2.MORPH_OPEN, horiz_knl)
    
    cam_d_no_lines = np.minimum(no_v_lines, no_h_lines)
    
    try:
        masked_cam_d = cam_d_no_lines[grey_mask == 0]
        if masked_cam_d.size > 0:
            cam_d_t = filters.threshold_multiotsu(masked_cam_d)[0]
        else:
            cam_d_t = 0
    except:
        try:
            cam_d_t = filters.threshold_otsu(cam_d_no_lines)
        except:
            cam_d_t = 0

    tissue_mask = np.zeros(cam_d_no_lines.shape, dtype=np.uint8)
    tissue_mask[cam_d_no_lines >= cam_d_t] = 255
    
    concave_tissue_mask = mask2contours(tissue_mask, kernel_size)
    return tissue_mask, concave_tissue_mask

def get_channel_stats(img):
    # 1st, 5th, Mean, 95th, 99th percentiles
    img_stats = [None] * 5
    img_stats[0] = np.percentile(img, 1)
    img_stats[1] = np.percentile(img, 5)
    img_stats[2] = np.mean(img)
    img_stats[3] = np.percentile(img, 95)
    img_stats[4] = np.percentile(img, 99)
    return np.array(img_stats)

def norm_img_stats(img, target_stats, mask=None):
    if mask is None:
        src_stats_flat = get_channel_stats(img)
    else:
        src_stats_flat = get_channel_stats(img[mask > 0])

    lower_knots = np.array([0])
    upper_knots = np.array([300, 350, 400, 450]) 
    src_stats_flat = np.hstack([lower_knots, src_stats_flat, upper_knots]).astype(float)
    target_stats_flat = np.hstack([lower_knots, target_stats, upper_knots]).astype(float)

    eps = 10 * np.finfo(float).resolution
    eps_array = np.arange(len(src_stats_flat)) * eps
    src_stats_flat = src_stats_flat + eps_array
    target_stats_flat = target_stats_flat + eps_array

    src_order = np.argsort(src_stats_flat)
    src_stats_flat = src_stats_flat[src_order]
    target_stats_flat = target_stats_flat[src_order]

    cs = Akima1DInterpolator(src_stats_flat, target_stats_flat)

    if mask is None:
        normed_img = cs(img.reshape(-1)).reshape(img.shape)
    else:
        normed_img = img.astype(float).copy()
        fg_px = np.where(mask > 0)
        if len(fg_px[0]) > 0:
            normed_img[fg_px] = cs(img[fg_px])
    
    normed_img = np.clip(normed_img, 0, 255)
    return normed_img


def stain_flatten_process(image, n_stains=100, q=95):
    fg_mask, _ = create_tissue_mask_from_rgb(image)
    
    tissue_area = np.sum(fg_mask > 0)
    if tissue_area == 0:
        fg_rgb = image.reshape(-1, 3)
    else:
        fg_rgb = image[fg_mask > 0]
        
    if fg_rgb.shape[0] > 10000:
        indices = np.random.choice(fg_rgb.shape[0], 10000, replace=False)
        fg_rgb_sample = fg_rgb[indices]
    else:
        fg_rgb_sample = fg_rgb

    fg_to_cluster = rgb2jab(fg_rgb_sample)
    
    clusterer = MiniBatchKMeans(n_clusters=n_stains, n_init=3, batch_size=1024, random_state=42)
    clusterer.fit(fg_to_cluster)
    
    stain_rgb = jab2rgb(clusterer.cluster_centers_)
    stain_rgb = np.clip(stain_rgb, 0, 1)
    
    if np.sum(fg_mask == 0) > 0:
        mean_bg_rgb = np.mean(image[fg_mask == 0], axis=0) / 255.0
    else:
        mean_bg_rgb = np.array([1.0, 1.0, 1.0]) 
        
    stain_rgb = np.vstack([255 * stain_rgb, mean_bg_rgb * 255])
    
    # 3. Deconvolve
    D = stainmat2decon(stain_rgb)
    deconvolved = deconvolve_img(image, D)
    
    # 4. Normalize channels (95th percentile clipping)
    eps = np.finfo("float").eps
    d_flat = deconvolved.reshape(-1, deconvolved.shape[2])
    dmax = np.percentile(d_flat, q, axis=0)

    for i in range(deconvolved.shape[2]):
        c_dmax = dmax[i] + eps
        deconvolved[..., i] = np.clip(deconvolved[..., i], 0, c_dmax)
        deconvolved[..., i] /= c_dmax
        
    # 5. Summarize
    summary_img = deconvolved.mean(axis=2)
    
    # 6. Adaptive Equalization
    summary_img = exposure.equalize_adapthist(summary_img)
    processed_img = exposure.rescale_intensity(summary_img, in_range="image", out_range=(0, 255)).astype(np.uint8)
    return processed_img, fg_mask


def preprocess_images(pil_img1: Image.Image, pil_img2: Image.Image, 
                      src_mpp1: float, src_mpp2: float, target_mpp: float):
    imgs = [pil_img1, pil_img2]
    mpps = [src_mpp1, src_mpp2]
    
    processed_imgs = []
    masks = []
    
    for i, (img, mpp) in enumerate(zip(imgs, mpps)):
        # Convert to numpy
        np_img = np.array(img.convert("RGB"))
        h, w = np_img.shape[:2]
        
        # Calculate Scaling Factor based on MPP
        scaling = mpp / target_mpp
        new_w, new_h = int(w * scaling), int(h * scaling)
        
        if scaling != 1.0:
            np_img = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        # Stain Flattening
        flat_img, mask = stain_flatten_process(np_img, n_stains=100, q=95)
        
        # Resize mask if dimensions slightly off due to rounding or internal processing
        if mask.shape != flat_img.shape:
             mask = cv2.resize(mask, (flat_img.shape[1], flat_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Background masking
        flat_img[mask == 0] = 0
        
        processed_imgs.append(flat_img)
        masks.append(mask)

    all_pixels = []
    for i, (p_img, mask) in enumerate(zip(processed_imgs, masks)):
        if np.sum(mask) > 0:
            pixels = p_img[mask > 0]
        else:
            pixels = p_img.flatten()
        all_pixels.append(pixels)
            
    all_pixels_concat = np.concatenate(all_pixels)
    target_stats = get_channel_stats(all_pixels_concat)
    
    final_imgs = []
    for i, (p_img, mask) in enumerate(zip(processed_imgs, masks)):
        normed_img = norm_img_stats(p_img, target_stats, mask=mask)
        normed_img = exposure.rescale_intensity(normed_img, out_range=(0, 255)).astype(np.uint8)
        final_imgs.append(normed_img)
        
    return final_imgs[0], final_imgs[1]
