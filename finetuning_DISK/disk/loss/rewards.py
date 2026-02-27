import torch
import torch.nn.functional as F
from torch_dimcheck import dimchecked

from disk import Image
from disk.geom.epi import asymmdist_from_imgs

class IdentityGridReward:
    """
    Non-linear correspondence reward based on Identity Grid (coords).
    - Uses coord map that includes both Affine and Elastic transforms.
    - Projects two points to original coordinate system and computes distance.
    """

    def __init__(self, th=2., lm_tp=1., lm_fp=-0.25, lm_roi_fp=-0.25):
        self.th = th
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp
        self.lm_roi_fp = lm_roi_fp

    def get_orig_coords(self, kps, coord_map):
        """
        Convert image coordinates (kps) to original coordinates via coord_map (Identity Grid).
        Args:
            kps: (N, 2) tensor [x, y]
            coord_map: (H, W, 2) tensor [orig_x, orig_y]
        """
        h, w = coord_map.shape[:2]
        device = kps.device

        # 1. Normalize kps to grid_sample format [-1, 1]
        grid_kps = kps.clone()
        grid_kps[:, 0] = (grid_kps[:, 0] / (w - 1)) * 2.0 - 1.0
        grid_kps[:, 1] = (grid_kps[:, 1] / (h - 1)) * 2.0 - 1.0
        grid_input = grid_kps.unsqueeze(0).unsqueeze(0) # (1, 1, N, 2)

        # 2. Reshape coord_map to (1, 2, H, W)
        map_input = coord_map.permute(2, 0, 1).unsqueeze(0).to(device)

        # 3. Sample original coordinates via bilinear interpolation
        sampled_orig = F.grid_sample(map_input, grid_input, 
                                     mode='bilinear', padding_mode='border', align_corners=True)
        
        return sampled_orig.squeeze(0).squeeze(1).t()  # (N, 2)

    @dimchecked
    def __call__(
        self,
        kps1: ['N', 2],
        kps2: ['M', 2],
        img1: Image,
        img2: Image
    ) -> ['N', 'M']:
        device = kps1.device
        
        # 1. ROI mask
        in_roi1 = img1.is_in_roi(kps1.T)  # (N,)
        in_roi2 = img2.is_in_roi(kps2.T)  # (M,)
        roi_mask = in_roi1[:, None] & in_roi2[None, :]  # (N, M)

        # 2. Valid range mask (image boundary check)
        mask1 = img1.in_range_mask(kps1.T) # (N,)
        mask2 = img2.in_range_mask(kps2.T) # (M,)
        valid_range_mask = mask1[:, None] & mask2[None, :]

        # 3. Back-project to original coordinates (non-rigid correspondence)
        # Assume img1.coords and img2.coords are (H, W, 2) Identity Grids
        orig_kps1 = self.get_orig_coords(kps1, img1.coords) # (N, 2)
        orig_kps2 = self.get_orig_coords(kps2, img2.coords) # (M, 2)

        # 4. Distance in original coordinate system (L2 norm)
        diff = orig_kps1[:, None, :] - orig_kps2[None, :, :] # (N, M, 2)
        dist = torch.norm(diff, dim=-1) # (N, M)

        # 5. Success by distance
        is_close = dist < self.th 
           
        # --- Reward assignment ---
        reward = torch.zeros_like(dist)
        
        # Case 1: Success (inside ROI & close)
        tp_mask = roi_mask & is_close & valid_range_mask
        reward[tp_mask] = self.lm_tp
        
        # Case 2: Match failure (inside ROI & far)
        fp_match_mask = roi_mask & (~is_close) & valid_range_mask
        reward[fp_match_mask] = self.lm_fp
        
        # Case 3: Location failure (outside ROI)
        fp_roi_mask = (~roi_mask) & valid_range_mask
        reward[fp_roi_mask] = self.lm_roi_fp
        
        return reward


class AffineReward:
    """
    2D affine-transform-based reward for pathology images.
    - Uses known 2x3 affine matrices between image pairs.
    - No explicit keypoint matches required.
    """

    def __init__(self, th=2., lm_tp=1., lm_fp=-0.25, use_roi_mask=True):
        """
        Args:
            th (float): threshold for distance
            lm_tp (float): positive-penalty for true positives
            lm_fp (float): positive-penalty for false positives
        """
        self.th   = th
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp
        self.use_roi_mask = use_roi_mask

    def warp_points(self, points, affine):
        """
        Apply 2D affine transform to keypoints.

        Args:
            points: (N, 2) tensor
            affine: (2, 3) tensor

        Returns:
            warped_points: (N, 2) tensor
        """
        ones = torch.ones((points.shape[0], 1), device=points.device)
        homo = torch.cat([points, ones], dim=-1)  # (N, 3)
        warped = torch.matmul(homo, affine.T)     # (N, 2)
        return warped

    @dimchecked
    def __call__(
        self,
        kps1: ['N', 2],
        kps2: ['M', 2],
        img1: Image,
        img2: Image
    ) -> ['N', 'M']:
        """
        Compute reward given two sets of keypoints and affine transform A→B.

        Args:
            keypoints_a: (N, 2) tensor (predicted from image A)
            keypoints_b: (N, 2) tensor (predicted from image B)
            affine_ab: (2, 3) tensor (known transform from A to B)
            match_prob: (N,) tensor (optional, descriptor confidence)

        Returns:
            scalar reward tensor
        """
        # ROI mask filtering
        if self.use_roi_mask:
            in_roi1 = img1.is_in_roi(kps1.T)  # (N,)
            in_roi2 = img2.is_in_roi(kps2.T)  # (M,)
            roi_mask = in_roi1[:, None] & in_roi2[None, :]  # (N, M)
        else:
            roi_mask = torch.ones((kps1.shape[0], kps2.shape[0]), 
                                dtype=torch.bool, device=kps1.device)


        # Compute pairwise distance between warped A and B points
        affine_ab = img1.affine_matrix.to(kps1.device)
        warped_a = self.warp_points(kps1, affine_ab)

        warped_a_in_range = img2.in_range_mask(warped_a.T)
        range_mask = warped_a_in_range[:, None] & roi_mask

        diff = warped_a[:, None, :] - kps2[None, :, :]  
        dist = torch.norm(diff, dim=-1)

        good_pairs = dist < self.th
        reward = self.lm_tp * good_pairs + self.lm_fp * (~good_pairs)
        reward = reward * range_mask.float()
        return reward


class SymmetricAffineReward:
    """
    2D affine-transform-based reward for pathology images (Symmetric Version).
    - Uses known 2x3 affine matrices between image pairs.
    - Calculates Symmetric Transfer Error (Forward + Backward).
    """

    def __init__(self, th=2., lm_tp=1., lm_fp=-0.25, use_roi_mask=True):
        """
        Args:
            th (float): threshold for distance (applies to the symmetric sum or average)
            lm_tp (float): positive-penalty for true positives
            lm_fp (float): positive-penalty for false positives
        """
        self.th   = th
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp
        self.use_roi_mask = use_roi_mask

    def get_inverse_affine(self, affine):
        """
        Compute the inverse of a 2x3 affine matrix.
        Args:
            affine: (2, 3) tensor
        Returns:
            inv_affine: (2, 3) tensor
        """
        # Extend (2, 3) to (3, 3) to compute inverse
        zeros = torch.tensor([0., 0., 1.], device=affine.device).view(1, 3)
        mat3x3 = torch.cat([affine, zeros], dim=0)
        inv_mat = torch.linalg.inv(mat3x3)
        return inv_mat[:2, :]  # return as (2, 3)

    def warp_points(self, points, affine):
        """
        Apply 2D affine transform to keypoints.
        Args:
            points: (N, 2) tensor
            affine: (2, 3) tensor
        Returns:
            warped_points: (N, 2) tensor
        """
        ones = torch.ones((points.shape[0], 1), device=points.device)
        homo = torch.cat([points, ones], dim=-1)  # (N, 3)
        warped = torch.matmul(homo, affine.T)     # (N, 2)
        return warped

    @dimchecked
    def __call__(
        self,
        kps1: ['N', 2],
        kps2: ['M', 2],
        img1: Image,
        img2: Image
    ) -> ['N', 'M']:
        """
        Compute symmetric reward given two sets of keypoints and affine transform.
        """
        device = kps1.device
        if self.use_roi_mask:
            in_roi1 = img1.is_in_roi(kps1.T)  # (N,)
            in_roi2 = img2.is_in_roi(kps2.T)  # (M,)
            roi_mask = in_roi1[:, None] & in_roi2[None, :]  # (N, M)
        else:
            roi_mask = torch.ones((kps1.shape[0], kps2.shape[0]), 
                                dtype=torch.bool, device=device)

        # Forward Distance (1 -> 2)
        affine_1to2 = img1.affine_matrix.to(device)
        warped_kps1 = self.warp_points(kps1, affine_1to2) # (N, 2)
        mask_fwd = img2.in_range_mask(warped_kps1.T) # (N,)

        diff_fwd = warped_kps1[:, None, :] - kps2[None, :, :] # (N, M, 2)
        dist_fwd = torch.norm(diff_fwd, dim=-1) # (N, M)

        # Backward Distance (2 -> 1)
        affine_2to1 = self.get_inverse_affine(affine_1to2)
        warped_kps2 = self.warp_points(kps2, affine_2to1) # (M, 2)
        mask_bwd = img1.in_range_mask(warped_kps2.T) # (M,)

        diff_bwd = kps1[:, None, :] - warped_kps2[None, :, :] # (N, M, 2)
        dist_bwd = torch.norm(diff_bwd, dim=-1) # (N, M)

        # Symmetric Error Sum and Mask Combination
        symmetric_dist = (dist_fwd + dist_bwd) / 2
        range_mask = mask_fwd[:, None] & mask_bwd[None, :] & roi_mask

        # Reward Calculation
        good_pairs = symmetric_dist < self.th
        reward = self.lm_tp * good_pairs + self.lm_fp * (~good_pairs)
        reward = reward * range_mask.float()
        
        return reward


class TissueAwareAffineReward:
    """
    2D affine-transform-based reward for pathology images (Symmetric Version).
    - Uses known 2x3 affine matrices between image pairs.
    - Calculates Symmetric Transfer Error (Forward + Backward).
    """

    def __init__(self, th=2., lm_tp=1., lm_fp=-0.25, lm_roi_fp=-0.25, use_roi_mask=True):
        """
        Args:
            th (float): threshold for distance (applies to the symmetric sum or average)
            lm_tp (float): positive-penalty for true positives
            lm_fp (float): positive-penalty for false positives
        """
        self.th   = th
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp
        self.lm_roi_fp = lm_roi_fp
        self.use_roi_mask = use_roi_mask

    def get_inverse_affine(self, affine):
        """
        Compute the inverse of a 2x3 affine matrix.
        Args:
            affine: (2, 3) tensor
        Returns:
            inv_affine: (2, 3) tensor
        """
        # Extend (2, 3) to (3, 3) to compute inverse
        zeros = torch.tensor([0., 0., 1.], device=affine.device).view(1, 3)
        mat3x3 = torch.cat([affine, zeros], dim=0)
        inv_mat = torch.linalg.inv(mat3x3)
        return inv_mat[:2, :]  # return as (2, 3)

    def warp_points(self, points, affine):
        """
        Apply 2D affine transform to keypoints.
        Args:
            points: (N, 2) tensor
            affine: (2, 3) tensor
        Returns:
            warped_points: (N, 2) tensor
        """
        ones = torch.ones((points.shape[0], 1), device=points.device)
        homo = torch.cat([points, ones], dim=-1)  # (N, 3)
        warped = torch.matmul(homo, affine.T)     # (N, 2)
        return warped

    @dimchecked
    def __call__(
        self,
        kps1: ['N', 2],
        kps2: ['M', 2],
        img1: Image,
        img2: Image
    ) -> ['N', 'M']:
        """
        Compute symmetric reward given two sets of keypoints and affine transform.
        """
        device = kps1.device
        if self.use_roi_mask:
            in_roi1 = img1.is_in_roi(kps1.T)  # (N,)
            in_roi2 = img2.is_in_roi(kps2.T)  # (M,)
            roi_mask = in_roi1[:, None] & in_roi2[None, :]  # (N, M)
        else:
            roi_mask = torch.ones((kps1.shape[0], kps2.shape[0]), 
                                dtype=torch.bool, device=device)

        # Forward Distance (1 -> 2)
        affine_1to2 = img1.affine_matrix.to(device)
        warped_kps1 = self.warp_points(kps1, affine_1to2) # (N, 2)
        mask_fwd = img2.in_range_mask(warped_kps1.T) # (N,)

        diff_fwd = warped_kps1[:, None, :] - kps2[None, :, :] # (N, M, 2)
        dist_fwd = torch.norm(diff_fwd, dim=-1) # (N, M)

        # Backward Distance (2 -> 1)
        affine_2to1 = self.get_inverse_affine(affine_1to2)
        warped_kps2 = self.warp_points(kps2, affine_2to1) # (M, 2)
        mask_bwd = img1.in_range_mask(warped_kps2.T) # (M,)

        diff_bwd = kps1[:, None, :] - warped_kps2[None, :, :] # (N, M, 2)
        dist_bwd = torch.norm(diff_bwd, dim=-1) # (N, M)

        # Symmetric Distance Calculation
        symmetric_dist = (dist_fwd + dist_bwd) / 2
        valid_range_mask = mask_fwd[:, None] & mask_bwd[None, :]
        is_close = symmetric_dist < self.th     
           
        # Case 1: Inside ROI & close -> Success (True Positive)
        tp_mask = roi_mask & is_close & valid_range_mask
        
        # Case 2: Inside ROI & far -> Match failure (False Positive - Mismatch)
        fp_match_mask = roi_mask & (~is_close) & valid_range_mask
        
        # Case 3: Outside ROI -> Location failure (False Positive - Out of ROI)
        fp_roi_mask = (~roi_mask) & valid_range_mask
        
        reward = torch.zeros_like(symmetric_dist)
        reward[tp_mask] = self.lm_tp
        reward[fp_match_mask] = self.lm_fp
        reward[fp_roi_mask] = self.lm_fp
        
        return reward


class EpipolarReward:
    def __init__(self, th=2., lm_tp=1., lm_fp=-0.25):
        self.th   = th
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp

    @dimchecked
    def __call__(
        self,
        kps1: ['N', 2],
        kps2: ['M', 2],
        img1: Image,
        img2: Image
    ) -> ['N', 'M']:
        '''
        assigns all pairs of keypoints across (kps1, kps2) a reward depending
        if the are correct or incorrect under epipolar constraints
        '''
        good = self.classify(kps1, kps2, img1, img2)
        return self.lm_tp * good + self.lm_fp * (~good)

    @dimchecked
    def classify(
        self,
        kps1: ['N', 2],
        kps2: ['M', 2],
        img1: Image,
        img2: Image,
    ) -> ['N', 'M']:
        '''
        classifies all pairs of keypoints across (kps1, kps2) as correct or
        incorrect depending on epipolar error
        '''

        epi_1_to_2 = asymmdist_from_imgs(kps1.T, kps2.T, img1, img2).abs()
        epi_2_to_1 = asymmdist_from_imgs(kps2.T, kps1.T, img2, img1).abs()

        # the distance is asymmetric, so we check if both 2_to_1 is
        # correct and 1_to_2.
        return (epi_1_to_2 < self.th) & (epi_2_to_1 < self.th).T

class DepthReward:
    def __init__(self, th=2., lm_tp=1., lm_fp=-0.25):
        self.th   = th 
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp

        self._epipolar = EpipolarReward(th=th)

    @dimchecked
    def __call__(
        self,
        kps1: ['N', 2],
        kps2: ['M', 2],
        img1: Image,
        img2: Image
    ) -> ['N', 'M']:
        '''
        classifies all (kp1, kp2) pairs as either
        * correct  : within dist_α in reprojection
        * incorrect: above dist_α away in epipolar constraints
        * unknown  : no depth is available and is not incorrect

        and assigns them rewards according to DepthReward parameters
        '''

        # reproject to the other image.
        kps1_r = img2.project(img1.unproject(kps1.T)) # [2, N]
        kps2_r = img1.project(img2.unproject(kps2.T)) # [2, M]

        # compute pixel-space differences between (kp1, repr(kp2))
        # and (repr(kp1), kp2)
        diff1 = kps2_r[:, None, :] - kps1.T[:, :, None] # [2, N, M]
        diff2 = kps1_r[:, :, None] - kps2.T[:, None, :] # [2, N, M]

        # NaNs indicate we had no depth available at this location
        has_depth = (torch.isfinite(diff1) & torch.isfinite(diff2)).all(dim=0)

        # threshold the distances
        close1    = torch.norm(diff1, p=2, dim=0) < self.th
        close2    = torch.norm(diff2, p=2, dim=0) < self.th
        
        epi_bad    = ~self._epipolar.classify(kps1, kps2, img1, img2)
        good_pairs = close1 & close2

        return self.lm_tp * good_pairs + self.lm_fp * epi_bad
