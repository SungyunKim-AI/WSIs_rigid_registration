import time, cv2, logging, click
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from typing import Tuple, Optional, Dict, Union, Any

import torch
import torch.nn.functional as F
from models import DISK, SuperPoint, LightGlue, CycleMatcher
from datasets import ANHIRDataset
from utils.calculator import *
from utils.save import *
from utils.logger import *

class WSI_Registration:
    def __init__(
        self, 
        extractor: str,  # extractor weights path
        matcher: str, # matcher weights path or weigths
        tile_size: Optional[Union[int, Tuple[int, int]]] = 512, 
        max_num_keypoints: int = 512, 
        do_tiling: bool = False, 
        local_refinement_top_k: int = 350,
        save_result: bool = False, 
        output_dir: Optional[Union[str, Path]] = None, 
        device: str = 'cpu', 
        logger: Optional[logging.Logger] = None
     ) -> None:
        self.tile_size = tile_size
        self.do_tiling = do_tiling
        self.local_refinement_top_k = local_refinement_top_k
        self.save_result = save_result
        self.output_dir = output_dir
        self.device = torch.device(device)
        self.logger = logger

        # extractor & matcher
        if 'disk' in extractor.name:
            self.extractor = DISK(weights=extractor.weights, max_num_keypoints=max_num_keypoints).eval().to(self.device)
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
        feats1: Dict[str, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
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
        self.logger.info(f"\tGlobal Matching | matches: {num_inliers}/{len(matches)} (keypoints: {len(kpts0)}, {len(kpts1)})")

        if self.save_result:
            output_dir = Path(self.output_dir) / "global_matching_result"
            inliers_mask = (inliers > 0).squeeze()
            m_kpts0_inliers = m_kpts0[inliers_mask]
            m_kpts1_inliers = m_kpts1[inliers_mask]

            save_matching_result(
                image0, m_kpts0_inliers, 
                image1, m_kpts1_inliers, 
                output_dir, self.file_name,
                kpts0=kpts0, kpts1=kpts1
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

        # VER 4.0 : pixel distance limit
        # Use 1/4 of image size as distance limit (m_kpts0 is in source image coordinate system)
        _, H_src, W_src = image0.shape
        max_dist = min(H_src, W_src) / 2.0
        
        refined_landmarks = []
        for idx, (pt_src, pt_tgt_est) in enumerate(zip(landmarks, src_landmarks_warped)):
            # Find top K nearest keypoints (only within 1/4 image distance)
            dist = np.linalg.norm(m_kpts0 - pt_src, axis=1)
            within_range = dist < max_dist
            valid_indices = np.where(within_range)[0]
            valid_dist = dist[valid_indices]
            
            if len(valid_indices) > top_k:
                top_k_local = min(top_k, len(valid_indices))
                nearest_local = np.argpartition(valid_dist, top_k_local)[:top_k_local]
                nearest_indices = valid_indices[nearest_local]
            else:
                nearest_indices = valid_indices
                
            local_m_kpts0_warped = m_kpts0_warped[nearest_indices]
            local_m_kpts1 = m_kpts1[nearest_indices]
            
            # TODO : to be updated
            if len(local_m_kpts0_warped) < 100:
                refined_landmarks.append(pt_tgt_est)
                self.logger.warning(f"\tMatched keypoints lower than 100 | landmark: {idx}, inliers: {len(local_m_kpts0_warped)} < 100")
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
        img_src: torch.Tensor, 
        img_tgt: torch.Tensor,
        target: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.file_name = f"{target['slide_id']}_{target['src_stain']}_{target['tgt_stain']}"
        self.logger.info(f"slide_id: {target['slide_id']}, {target['src_stain']}: {img_src.shape}({target['mpp_target']:.2f}), \
            {target['tgt_stain']}: {img_tgt.shape}({target['mpp_target']:.2f})")
        
        # Reference setting
        src_image, tgt_image = img_src.to(self.device), img_tgt.to(self.device)
        landmarks_src, landmarks_tgt = target['landmarks_src'], target['landmarks_tgt']
        mpp_lv0, mpp_target = target['mpp_lv0'], target['mpp_target']
            
        start = time.time()
        
        # Extract Features
        feats_src = self.extract_keypoints(src_image, self.tile_size, max_kpts=0)
        feats_tgt = self.extract_keypoints(tgt_image, self.tile_size, max_kpts=0)
        
        # Global Matching
        m_kpts_src, m_kpts_tgt, matches, global_affine, num_inliers = self.global_matching(
            src_image, tgt_image, feats_src, feats_tgt
        )

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
            
            tgt_image_downscaled = F.interpolate(
                tgt_image.unsqueeze(0), 
                size=(H_tgt // 2, W_tgt // 2), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Extract features from downscaled images
            tile_size = self.tile_size // 2
            feats_src_downscaled = self.extract_keypoints(src_image_downscaled, tile_size, max_kpts=0)
            feats_tgt_downscaled = self.extract_keypoints(tgt_image_downscaled, tile_size, max_kpts=0)

            # Global Matching with downscaled images
            m_kpts_src_downscaled, m_kpts_tgt_downscaled, matches_downscaled, global_affine_downscaled, num_inliers_downscaled = self.global_matching(
                src_image_downscaled, tgt_image_downscaled, feats_src_downscaled, feats_tgt_downscaled
            )

            # VER 4.0 : if num_inlier is fewer than original method, keep using original method
            if num_inliers_downscaled > num_inliers:
                global_affine = convert_matrix_to_original_scale(
                    global_affine_downscaled, 
                    resize_factor_0=0.5,  # source image downscale factor
                    resize_factor_1=0.5   # target image downscale factor
                )
                
                m_kpts_src = m_kpts_src_downscaled * 2.0
                m_kpts_tgt = m_kpts_tgt_downscaled * 2.0
                matches = matches_downscaled
                num_inliers = num_inliers_downscaled
            else:
                self.logger.info(f"\tDownscaled num_inliers ({num_inliers_downscaled}) <= original ({num_inliers}), keeping original results")

                
        # Local Refinement around landmarks
        # target['landmarks_*']: original image (pre-resize) pixel coords -> convert to resized image coordinate system
        # Dataset resize: scale = mpp_lv0/mpp_target, resized_shape = orig_shape * scale
        resize_scale = mpp_lv0 / mpp_target  # original pixel -> resized pixel
        landmarks_src_px = landmarks_src.astype(np.float64) * resize_scale  # (N, 2) resized image coordinates
        if self.do_tiling:
            warped_landmarks_px = self.local_refinement_matching(
                src_image, tgt_image, m_kpts_src, m_kpts_tgt, 
                landmarks_src_px, global_affine, self.local_refinement_top_k
            )   # (N, 2) resized image coordinates
        else:
            self.logger.info("\tApplying Global Affine only")
            warped_landmarks_px = cv2.transform(
                landmarks_src_px.reshape(-1, 1, 2), global_affine
            ).reshape(-1, 2)    # (N, 2)

        # Restore resized image coordinates -> original image pixel coordinates
        warped_landmarks_original = warped_landmarks_px / resize_scale

        runtime = time.time() - start

        match_results = {
            'runtime': runtime,
            'num_matches': len(matches),
            'warped_landmarks': warped_landmarks_original,
            'M': global_affine
        }

        if self.save_result:
            landmarks_tgt = target['landmarks_tgt']
            landmarks_tgt_px = (landmarks_tgt.astype(np.float64) * resize_scale).astype(int)  # resized coords for visualization
            landmarks_warped_px = (warped_landmarks_px).astype(int)
            
            output_dir_lm = Path(self.output_dir) / 'landmarks_result'
            save_landmarks(
                tgt_image, landmarks_warped_px, landmarks_tgt_px, 
                output_dir_lm, self.file_name
            )

        return match_results


@click.command()
@click.option('--config', type=click.Path(exists=True), default="./configs/config_disk_anhir.yaml")
def main(config: str) -> None:
    cfg = OmegaConf.load(config)
    output_dir = Path(cfg.output_dir) / f"{cfg.ver}"
    output_dir.mkdir(exist_ok=True, parents=True)
    cfg.output_dir = str(output_dir)
    OmegaConf.save(config=cfg, f=f"{cfg.output_dir}/config.yaml")

    logger = get_logger()
    save_logger(output_dir, logger)
    
    dataset = ANHIRDataset(cfg.data_dir, cfg.target_mpp)

    registrar = WSI_Registration(
        cfg.extractor,
        cfg.matcher,
        tile_size=cfg.tile_size,
        max_num_keypoints=cfg.max_num_keypoints, 
        do_tiling=cfg.tiling, 
        local_refinement_top_k=cfg.local_refinement_top_k,
        save_result=cfg.result_save, 
        output_dir=cfg.output_dir,
        device="cuda",
        logger=logger
    )

    results = []
    rtre_results = []
    for idx in range(len(dataset)):
        logger.info(f"{idx+1}/{len(dataset)}")
        img_he, img_ihc, target = dataset[idx]
        match_results = registrar.run_registration(img_he, img_ihc, target)

        try:
            tre_list = get_TRE_list(match_results['warped_landmarks'], target['landmarks_tgt'])   # unit: px
        except ValueError:
            logger.warning(f"Invalid landmarks: {target['slide_id']}, {target['src_stain']}, {target['tgt_stain']}")
            continue

        rtre_list = tre_list / target['diagonal']
        match_results['rTRE_median'] = np.median(rtre_list)
        match_results['rTRE_mean'] = np.mean(rtre_list)
        registrar.logger.info(f"\trTRE median: {match_results['rTRE_median']:.4f}, rTRE mean: {match_results['rTRE_mean']:.4f}, Runtime: {match_results['runtime']:.4f} sec")
        
        for i, rtre in enumerate(rtre_list):
            rtre_results.append({
                "slide_id": target['slide_id'], "src_stain": target['src_stain'], "tgt_stain": target['tgt_stain'], 
                "landmark_index": i, 'target_landmark': target['landmarks_tgt'][i], 
                "warped_landmark": match_results['warped_landmarks'][i], "rTRE": rtre,
            })

        results.append({
            "slide_id": target['slide_id'], "src_stain": target['src_stain'], "tgt_stain": target['tgt_stain'], 
            "rTRE_median": match_results['rTRE_median'], "rTRE_mean":match_results['rTRE_mean'], "num_matches": match_results['num_matches'], 
            "runtime": match_results['runtime']
        })

    df = pd.DataFrame(results)
    df.to_csv(Path(cfg.output_dir)/"results.csv", index=False)

    df_tre = pd.DataFrame(rtre_results)
    df_tre.to_csv(Path(cfg.output_dir)/"all_landmarks_results.csv", index=False)

    rTRE_median_list = df['rTRE_median'].values
    rTRE_mean_list = df['rTRE_mean'].values
    registrar.logger.info(f"Med TRE: {np.mean(rTRE_median_list):.4f} ({rTRE_median_list.std():.4f})")
    registrar.logger.info(f"Avg TRE: {np.mean(rTRE_mean_list):.4f} ({rTRE_mean_list.std()*2:.4f})")

    runtime_list = df['runtime'].values
    registrar.logger.info(f"runtime: {np.mean(runtime_list):.4f} ({runtime_list.std():.4f})")

    matches_list = df['num_matches'].values
    registrar.logger.info(f"num_matches: {np.mean(matches_list):.4f} ({matches_list.std():.4f})")


if __name__ == "__main__":
    main()
