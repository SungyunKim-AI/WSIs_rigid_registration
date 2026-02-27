import torch
import torch.nn.functional as F
import numpy as np
from torch_dimcheck import dimchecked
from disk import MatchedPairs, Image, NpArray


def get_orig_coords(kps: torch.Tensor, coord_map: torch.Tensor):
    """
    Convert augmented image keypoints (kps) to original coordinates via coord_map.
    kps: (N, 2) - [x, y] pixel coordinates
    coord_map: (H, W, 2) - map storing [orig_x, orig_y] (Tensor)
    """
    h, w = coord_map.shape[:2]

    # 1. Normalize kps to grid_sample format [-1, 1]
    # grid_sample expects coordinates in [-1, 1] range.
    grid_kps = kps.clone()
    grid_kps[:, 0] = (grid_kps[:, 0] / (w - 1)) * 2.0 - 1.0  # x
    grid_kps[:, 1] = (grid_kps[:, 1] / (h - 1)) * 2.0 - 1.0  # y
    
    # Reshape to (1, 1, N, 2)
    grid_input = grid_kps.unsqueeze(0).unsqueeze(0)

    # 2. Reshape coord_map to (1, 2, H, W) for sampling
    # coord_map is [H, W, 2], permute to put channels first
    map_input = coord_map.permute(2, 0, 1).unsqueeze(0) # (1, 2, H, W)

    # 3. Sample original coordinates via bilinear interpolation
    # Read "original address" at kps locations in augmented image
    sampled_orig = F.grid_sample(map_input, grid_input, 
                                 mode='bilinear', padding_mode='border', align_corners=True)
    
    # (1, 2, 1, N) -> (N, 2)
    # return sampled_orig.squeeze().t()
    return sampled_orig[0, :, 0, :].t()

@dimchecked
def classify_pairs(kps1: ['N', 2], kps2: ['M', 2], img1: Image, img2: Image, th: float):
    """
    Using Identity Grid (img.coords), determine whether keypoint pairs between two images
    (including Elastic transform) are correct.
    """
    # 1. Back-project each image's keypoints to 'original coordinate system'
    # Assume img1.coords and img2.coords are (H, W, 2) Tensors.
    orig_kps1 = get_orig_coords(kps1, img1.coords) # (N, 2)
    orig_kps2 = get_orig_coords(kps2, img2.coords) # (M, 2)

    # 2. Valid range mask (same as existing logic)
    # Exclude coordinates that fall outside the image
    # (If you need to check warped coords directly due to Elastic transform,
    #  you may use the warped_kps logic inside get_orig_coords separately.)
    in_range_1 = img1.in_range_mask(kps1.T) # (N,)
    in_range_2 = img2.in_range_mask(kps2.T) # (M,)

    # 3. Distance in original coordinate system (Symmetric Distance)
    # Both points are in the same 'original' frame; this distance determines match.
    diff = orig_kps1[:, None, :] - orig_kps2[None, :, :] # (N, M, 2)
    dist = torch.norm(diff, dim=-1) # (N, M)

    # 4. Final decision: treat as correct if two keypoints are within th (pixels) in original image
    valid_range_mask = in_range_1[:, None] & in_range_2[None, :]
    good = (dist < th) & valid_range_mask
    
    return good


class DiscreteMetric(torch.nn.Module):
    def __init__(self, th=2., lm_kp=0., lm_tp=1., lm_fp=-0.25):
        super(DiscreteMetric, self).__init__() 

        self.th   = th
        self.lm_kp = lm_kp
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp

    def forward(
        self,
        images : NpArray[Image],       # [N_scenes, N_per_scene]
        matches: NpArray[MatchedPairs] # [N_scenes, N_per_scene choose 2]
    ):
        N_scenes, N_per_scene = images.shape

        assert matches.shape[0] == N_scenes
        assert matches.shape[1] == ((N_per_scene - 1) * N_per_scene) // 2

        stats = np.zeros(matches.shape, dtype=object)

        for i_scene in range(N_scenes):
            i_match = 0
            scene_matches = matches[i_scene]
            scene_images  = images[i_scene]

            for i_image1 in range(N_per_scene):
                image1 = scene_images[i_image1]

                for i_image2 in range(i_image1+1, N_per_scene):
                    image2 = scene_images[i_image2]

                    stats[i_scene, i_match] = self._loss_one_pair(
                        scene_matches[i_match],
                        image1, image2
                    )

                    i_match += 1

        return stats

    def _loss_one_pair(self, pairs: MatchedPairs, img1: Image, img2: Image):
        n_kps   = pairs.kps1.shape[0] + pairs.kps2.shape[0]

        kps1 = pairs.kps1[pairs.matches[0]]
        kps2 = pairs.kps2[pairs.matches[1]]
        n_pairs = pairs.matches.shape[1]

        good = classify_pairs(kps1, kps2, img1, img2, th=self.th)

        if good.dim() == 2:
            # Each matching should have exactly one good value
            # Here we check whether each match is correct
            good = good.any(dim=1) if good.shape[0] == n_pairs else good.any(dim=0)
        bad  = ~good
        
        n_good = good.to(torch.int64).sum().item()
        n_bad  = bad.to(torch.int64).sum().item()
        prec   = n_good / (n_pairs + 1)

        reward = self.lm_tp * n_good  + \
                 self.lm_fp * n_bad   + \
                 self.lm_kp * n_kps

        stats = {
            'n_kps'    : n_kps,
            'n_pairs'  : n_pairs,
            'tp'       : n_good,
            'fp'       : n_bad,
            'reward'   : reward,
            'precision': prec,
        }

        return stats
