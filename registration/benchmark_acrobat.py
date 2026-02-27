import time, cv2, logging, click
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from typing import Tuple, Optional, Dict, Union, Any

import torch
import torch.nn.functional as F
from models import DISK, SuperPoint, LightGlue, CycleMatcher
from datasets import ACROBATDataset
from utils.calculator import *
from utils.save import *
from utils.logger import *


def rotate_tensor_and_points(image_tensor: torch.Tensor, points: np.ndarray, angle: float, device: torch.device):
    # image_tensor: (C, H, W) on device
    # points: (N, 2) numpy array
    
    if angle == 0:
        return image_tensor, points

    # Convert to numpy for rotation
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy() # H, W, C
    h, w = img_np.shape[:2]
    center = (w / 2, h / 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    
    rotated_img = cv2.warpAffine(img_np, M, (nW, nH))
    if len(rotated_img.shape) == 2:
        rotated_img = rotated_img[..., None]
        
    rotated_tensor = torch.from_numpy(rotated_img).permute(2, 0, 1).to(device)
    
    # Rotate points
    if points is not None and len(points) > 0:
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        rotated_points = M.dot(points_ones.T).T
    else:
        rotated_points = points
    
    return rotated_tensor, rotated_points


class WSI_Registration:
    def __init__(
        self, 
        extractor: str,  # extractor weights path
        matcher: str, # matcher weights path or weigths
        reference: str = 'HE',  # HE or IHC
        tile_size: Optional[Union[int, Tuple[int, int]]] = 512, 
        max_num_keypoints: int = 512, 
        do_tiling: bool = False, 
        local_refinement_top_k: int = 350,
        save_result: bool = False, 
        output_dir: Optional[Union[str, Path]] = None, 
        device: str = 'cpu', 
        logger: Optional[logging.Logger] = None
     ) -> None:
        assert reference in ['HE', 'IHC'], f"Reference must be 'HE' or 'IHC', but got {reference}"
        self.reference = reference
        self.tile_size = tile_size
        self.max_num_keypoints = max_num_keypoints
        self.do_tiling = do_tiling
        self.local_refinement_top_k = local_refinement_top_k
        self.save_result = save_result
        self.output_dir = output_dir
        self.device = torch.device(device)
        self.logger = logger

        # extractor & matcher
        if 'disk' in extractor.name:
            self.extractor = DISK(weights=extractor.weights, max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
        else:
            raise ValueError(f"Invalid extractor: {extractor.name}")
        
        if 'cycle_matcher' in matcher.name:
            self.matcher = CycleMatcher()
        else:
            self.matcher = LightGlue(features=extractor.name, weights=matcher.weights).eval().to(self.device)
        
    
    def extract_keypoints(
        self, 
        image: torch.Tensor,
        tile_size: int, 
        max_kpts: int = 2048
    ) -> Dict[str, torch.Tensor]:
        image = image.to(self.device)
        _, H, W = image.shape

        if tile_size <= 0:
            bboxes = [(0, 0, W, H)]  # Extract from whole image at once
        else:
            bboxes = get_tile_bboxes((W, H), tile_size)
        
        all_kpts = []
        all_scores = []
        all_descs = []
        
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            tile_h, tile_w = y_max - y_min, x_max - x_min
            
            if tile_h < 32 or tile_w < 32:
                continue
                
            tile = image[:, y_min:y_max, x_min:x_max].unsqueeze(0) # (1, C, H, W)
            
            # Extract features from tile
            try:
                with torch.no_grad():
                    feats = self.extractor({"image": tile}) 
                
                kpts = feats['keypoints'][0]      # (N, 2)
                scores = feats['keypoint_scores'][0] # (N,)
                descs = feats['descriptors'][0]    # (N, D)
                
                if len(kpts) == 0:
                    continue
                
                offset = torch.tensor([[x_min, y_min]], device=self.device, dtype=torch.float32)
                kpts = kpts + offset
                
                all_kpts.append(kpts)
                all_scores.append(scores)
                all_descs.append(descs)
            except (IndexError, RuntimeError) as e:
                if self.logger:
                    self.logger.debug(f"Skipping tile ({x_min}, {y_min}, {x_max}, {y_max}): {str(e)}")
                continue
        
        # Handle case where no features were found
        if len(all_kpts) == 0:
            desc_dim = 128
            all_kpts = torch.empty((0, 2), device=self.device)
            all_scores = torch.empty((0,), device=self.device)
            all_descs = torch.empty((0, desc_dim), device=self.device)
        else:
            all_kpts = torch.cat(all_kpts, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_descs = torch.cat(all_descs, dim=0)
        
        # Select top-K features by confidence score
        if max_kpts > 0 and len(all_scores) > max_kpts:
            top_k_indices = torch.topk(all_scores, max_kpts).indices
            all_kpts = all_kpts[top_k_indices]
            all_scores = all_scores[top_k_indices]
            all_descs = all_descs[top_k_indices]
        
        return {
            "keypoints": all_kpts.unsqueeze(0),       # (1, K, 2)
            "keypoint_scores": all_scores.unsqueeze(0), # (1, K)
            "descriptors": all_descs.unsqueeze(0),     # (1, K, D)
            "image_size": torch.tensor([(W, H)], device=self.device)
        }

    def global_matching(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        feats0: Dict[str, torch.Tensor],
        feats1: Dict[str, torch.Tensor],
        angle: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        
        with torch.no_grad():
            matches01 = self.matcher({"image0": feats0, "image1": feats1})
        
        # Remove batch dim
        kpts0 = feats0["keypoints"][0]
        kpts1 = feats1["keypoints"][0]
        matches = matches01["matches"][0] # (M, 2) indices
        
        if len(matches) > 0:
            m_kpts0 = kpts0[matches[:, 0]]
            m_kpts1 = kpts1[matches[:, 1]]
        else:
            m_kpts0 = torch.empty((0, 2), device=kpts0.device)
            m_kpts1 = torch.empty((0, 2), device=kpts1.device)
            
        # Convert to numpy
        m_kpts0 = m_kpts0.cpu().numpy().astype(np.float32)
        m_kpts1 = m_kpts1.cpu().numpy().astype(np.float32)
        matches = matches.cpu().numpy().astype(np.int32)

        global_affine, inliers = get_affine_matrix(m_kpts0, m_kpts1)
        num_inliers = inliers.sum() if inliers is not None else 0
        self.logger.info(f"\tGlobal Matching | angle: {angle}, matches: {num_inliers}/{len(matches)} (keypoints: {len(kpts0)}, {len(kpts1)})")

        if self.save_result:
            output_dir = Path(self.output_dir) / "global_matching_result"
            inliers_mask = (inliers > 0).squeeze()
            # m_kpts0_inliers = m_kpts0[inliers_mask]
            # m_kpts1_inliers = m_kpts1[inliers_mask]

            save_matching_result(
                image0, m_kpts0, # m_kpts0_inliers, 
                image1, m_kpts1, # m_kpts1_inliers, 
                output_dir, self.file_name,
                kpts0=kpts0, kpts1=kpts1, angle=angle
            )
        
        return m_kpts0, m_kpts1, matches, global_affine, num_inliers

    def local_refinement_matching(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        m_kpts0: np.ndarray,
        m_kpts1: np.ndarray,
        landmarks: np.ndarray,
        global_affine: np.ndarray,
        top_k: int = 100
    ) -> np.ndarray:
        """
        Refine registration locally around landmarks using existing matched keypoints.
        """
        src_landmarks_warped = cv2.transform(landmarks.reshape(-1, 1, 2), global_affine).reshape(-1, 2)
        if len(m_kpts0) < 100:
            self.logger.warning(f"Matched keypoints too low: {len(m_kpts0)} < 100, return global affine only")
            return src_landmarks_warped
        if len(m_kpts0) < top_k:
            self.logger.warning(f"Matched keypoints lower than top_k: {len(m_kpts0)} < {top_k}")

        m_kpts0_warped = cv2.transform(m_kpts0.reshape(-1, 1, 2), global_affine).reshape(-1, 2)
        src_landmarks_warped = cv2.transform(landmarks.reshape(-1, 1, 2), global_affine).reshape(-1, 2)
        
        refined_landmarks = []
        for idx, (pt_src, pt_tgt_est) in enumerate(zip(landmarks, src_landmarks_warped)):
            # Find top K nearest keypoints
            dist = np.linalg.norm(m_kpts0 - pt_src, axis=1)
            if len(dist) > top_k:
                nearest_indices = np.argpartition(dist, top_k)[:top_k]
            else:
                nearest_indices = np.arange(len(dist))
                
            local_m_kpts0_warped = m_kpts0_warped[nearest_indices]
            local_m_kpts1 = m_kpts1[nearest_indices]
            
            if len(local_m_kpts0_warped) < 4:
                refined_landmarks.append(pt_tgt_est)
                continue
                
            # Estimate local affine transformation
            local_matrix, inliers = cv2.estimateAffinePartial2D(local_m_kpts0_warped, local_m_kpts1, method=cv2.RANSAC)
            
            if local_matrix is None:
                refined_landmarks.append(pt_tgt_est)
                continue
            
            # Transform the source landmark using local matrix
            pt_src_warped = cv2.transform(pt_src.reshape(-1, 1, 2), global_affine).reshape(2)
            refined_pt = cv2.transform(pt_src_warped.reshape(-1, 1, 2), local_matrix).reshape(2)
            refined_landmarks.append(refined_pt)

            self.logger.info(f"\tLocal Refinement | landmark: {idx}, inliers: {inliers.sum()}/{len(local_m_kpts0_warped)}")

            if self.save_result:
                output_dir = Path(self.output_dir) / "tiling_matching_result"
                file_name = f"{self.file_name}_landmarks_{idx:04d}"
                save_local_refinement_result(
                    image0, image1, 
                    local_m_kpts0_warped, local_m_kpts1, 
                    inliers, global_affine, 
                    output_dir, file_name
                )
                
        return np.array(refined_landmarks)

    def run_registration(
        self, 
        img_he: torch.Tensor, 
        img_ihc: torch.Tensor,
        target: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.file_name = f"{target['slide_id']}_{target['stain']}"
        self.logger.info(f"slide_id: {target['slide_id']}, HE: {img_he.shape}({target['mpp_he']:.2f}), \
            {target['stain']}: {img_ihc.shape}({target['mpp_ihc']:.2f})")
        
        # Reference setting
        if self.reference == 'HE':
            src_image, tgt_image = img_he, img_ihc
            landmarks_src = target['landmarks_he']
            mpp_src, mpp_tgt = target['mpp_he'], target['mpp_ihc']
        else:
            src_image, tgt_image = img_ihc, img_he
            landmarks_src = target['landmarks_ihc']
            mpp_src, mpp_tgt = target['mpp_ihc'], target['mpp_he']
        
        # Ensure images are on the correct device
        src_image = src_image.to(self.device)
        tgt_image = tgt_image.to(self.device)
        
        # Landmarks to pixels (current image level)
        landmarks_src_px = downscale_landmarks(landmarks_src, mpp_src)
            
        start = time.time()
        
        # Extract Features for Target (Static)
        feats_tgt = self.extract_keypoints(tgt_image, self.tile_size, max_kpts=0)
        
        # Global Matching with Rotation
        best_num_matches = -1
        best_results = None
        
        # Loop 0 to 330 degrees
        for angle in range(0, 360, 30):
            # Rotate source image and landmarks
            src_image_rot, landmarks_src_px_rot = rotate_tensor_and_points(
                src_image, landmarks_src_px, angle, self.device
            )
            
            # Extract Features for Rotated Source
            feats_src_rot = self.extract_keypoints(src_image_rot, self.tile_size, max_kpts=0)
            
            # Global Matching
            m_kpts_src, m_kpts_tgt, matches, global_affine, num_inliers = self.global_matching(
                src_image_rot, tgt_image, feats_src_rot, feats_tgt, angle
            )
            
            if len(matches) > best_num_matches:
                best_num_matches = len(matches)
                best_results = {
                    'angle': angle,
                    'src_image': src_image_rot,
                    'landmarks_src_px': landmarks_src_px_rot,
                    'm_kpts_src': m_kpts_src,
                    'm_kpts_tgt': m_kpts_tgt,
                    'matches': matches,
                    'global_affine': global_affine,
                    'num_inliers': num_inliers,
                }

        # Use best results
        src_image = best_results['src_image']
        landmarks_src_px = best_results['landmarks_src_px']
        m_kpts_src = best_results['m_kpts_src']
        m_kpts_tgt = best_results['m_kpts_tgt']
        matches = best_results['matches']
        global_affine = best_results['global_affine']
        best_angle = best_results['angle']
        num_inliers = best_results['num_inliers']
        
        self.logger.info(f"\tBest Rotation | angle: {best_angle}, matches: {num_inliers}/{len(matches)}")

        # If matches are too few, retry with downscaled images
        if len(matches) < 5000:
            self.logger.warning(f"Number of matches is less than 5000: {len(matches)}, retrying with downscaled images")
            # Downscale images to half size
            _, H_src, W_src = src_image.shape
            _, H_tgt, W_tgt = tgt_image.shape
            
            src_image_downscaled = F.interpolate(
                src_image.unsqueeze(0), 
                size=(H_src // 2, W_src // 2), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)

            # Convert landmarks to downscaled coordinate system then rotate (coordinate system must match)
            landmarks_src_px_scaled = landmarks_src_px / 2.0
            src_image_rot_downscaled, landmarks_src_px_rot = rotate_tensor_and_points(
                src_image_downscaled, landmarks_src_px_scaled, best_angle, self.device
            )
            
            tgt_image_downscaled = F.interpolate(
                tgt_image.unsqueeze(0), 
                size=(H_tgt // 2, W_tgt // 2), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Extract features from downscaled images
            tile_size = self.tile_size // 2
            feats_src_downscaled = self.extract_keypoints(src_image_rot_downscaled, tile_size, max_kpts=0)
            feats_tgt_downscaled = self.extract_keypoints(tgt_image_downscaled, tile_size, max_kpts=0)

            # Global Matching with downscaled images
            m_kpts_src_downscaled, m_kpts_tgt_downscaled, matches_downscaled, global_affine_downscaled, num_inliers_downscaled = self.global_matching(
                src_image_rot_downscaled, tgt_image_downscaled, feats_src_downscaled, feats_tgt_downscaled, best_angle
            )
            
            # Scale up the affine matrix and keypoints to original resolution
            # For affine matrix: convert from downscaled (0.5x) to original scale
            # Both source and target images were downscaled by 0.5x
            global_affine = convert_matrix_to_original_scale(
                global_affine_downscaled, 
                resize_factor_0=0.5,  # source image downscale factor
                resize_factor_1=0.5   # target image downscale factor
            )
            
            # Scale up keypoints and landmarks to original resolution
            m_kpts_src = m_kpts_src_downscaled * 2.0
            m_kpts_tgt = m_kpts_tgt_downscaled * 2.0
            matches = matches_downscaled
            # landmarks_src_px_rot: restore from downscaled coords to original scale (global_affine is for original scale)
            landmarks_src_px = landmarks_src_px_rot * 2.0
            num_inliers = num_inliers_downscaled
                
        # Local Refinement around landmarks
        if self.do_tiling:
            warped_landmarks_px = self.local_refinement_matching(
                src_image, tgt_image, m_kpts_src, m_kpts_tgt, 
                landmarks_src_px, global_affine, self.local_refinement_top_k
            )   # (N, 2)
        else:
            self.logger.info("\tApplying Global Affine only")
            warped_landmarks_px = cv2.transform(
                landmarks_src_px.reshape(-1, 1, 2), global_affine
            ).reshape(-1, 2)    # (N, 2)

        warped_landmarks_px[:, 0] = np.clip(warped_landmarks_px[:, 0], 0, tgt_image.shape[2]-1)
        warped_landmarks_px[:, 1] = np.clip(warped_landmarks_px[:, 1], 0, tgt_image.shape[1]-1)
        warped_landmarks_um = upscale_landmarks(warped_landmarks_px, mpp_tgt)
        
        runtime = time.time() - start
        
        match_results = {
            'runtime': runtime,
            'num_matches': len(matches),
            'warped_landmarks': warped_landmarks_um,
            'M': global_affine
        }

        return match_results


@click.command()
@click.option('--config', type=click.Path(exists=True), default="./configs/config_disk_acrobat.yaml")
def main(config: str) -> None:
    cfg = OmegaConf.load(config)
    output_dir = Path(cfg.output_dir) / f"{cfg.ver}"
    output_dir.mkdir(exist_ok=True, parents=True)
    cfg.output_dir = str(output_dir)
    OmegaConf.save(config=cfg, f=f"{cfg.output_dir}/config.yaml")

    logger = get_logger()
    save_logger(output_dir, logger)
    
    dataset = ACROBATDataset(cfg.meta_data_path, cfg.data_dir, cfg.target_level, cfg.target_mpp)

    registrar = WSI_Registration(
        cfg.extractor,
        cfg.matcher,
        reference="IHC",
        tile_size=cfg.tile_size,
        max_num_keypoints=cfg.max_num_keypoints,
        do_tiling=cfg.tiling, 
        local_refinement_top_k=cfg.local_refinement_top_k,
        save_result=cfg.result_save, 
        output_dir=cfg.output_dir,
        device="cuda",
        logger=logger,
    )

    runtimes, match_counts = [], []
    meta_data = dataset.meta_data.copy()
    for idx in range(len(dataset)):
        logger.info(f"{idx+1}/{len(dataset)}")
        img_he, img_ihc,target = dataset[idx]
        match_results = registrar.run_registration(img_he, img_ihc, target)
        
        mask = meta_data['anon_id'] == target['slide_id']
        meta_data.loc[mask, 'he_x'] = match_results['warped_landmarks'][:, 0]
        meta_data.loc[mask, 'he_y'] = match_results['warped_landmarks'][:, 1]

        runtimes.append(match_results['runtime'])
        match_counts.append(match_results['num_matches'])

    meta_data.to_csv(Path(cfg.output_dir)/"results.csv", index=False)

    runtime_list = np.array(runtimes)
    match_count_list = np.array(match_counts)
    
    registrar.logger.info(f"runtime: {np.mean(runtimes):.4f} ± {runtime_list.std()*2:.4f}")
    registrar.logger.info(f"match_count: {np.mean(match_counts):.4f} ± {match_count_list.std()*2:.4f}")


if __name__ == "__main__":
    main()
