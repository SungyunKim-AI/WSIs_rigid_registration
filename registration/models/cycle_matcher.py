import torch
import torch.nn as nn

SQRT_2 = 1.414213
def distance_matrix(fs1, fs2):
    '''
    Assumes fs1 and fs2 are normalized!
    '''
    return SQRT_2 * (1. - fs1 @ fs2.T).clamp(min=1e-6).sqrt()


class CycleMatcher(nn.Module):
    required_data_keys = ["image0", "image1"]

    def __init__(self):
        super().__init__()

    def match_features(self, feat_1, feat_2):
        dist_m = distance_matrix(feat_1, feat_2)

        if dist_m.shape[0] == 0 or dist_m.shape[1] == 0:
            msg = '''
            Feature matching failed because one image has 0 detected features.
            This likely means that the algorithm has converged to a local
            optimum of detecting no features at all (0 reward). This can arise
            when lambda_fp and lambda_kp penalties are too high. Please check
            that your penalty annealing scheme is sound. It can also be that
            you are using a too low value of --warmup or --chunk-size
            '''
            raise RuntimeError(msg)

        n_amin = torch.argmin(dist_m, dim=1)
        m_amin = torch.argmin(dist_m, dim=0)

        # nearest neighbor's nearest neighbor
        nnnn = m_amin[n_amin]

        # we have a cycle consistent match for each `i` such that
        # nnnn[i] == i. We create an auxiliary array to check for that
        n_ix = torch.arange(dist_m.shape[0], device=dist_m.device)
        mask = nnnn == n_ix

        # Now `mask` is a binary mask and n_amin[mask] is an index array.
        # We use nonzero to turn `n_amin[mask]` into an index array and return
        return torch.stack([
            torch.nonzero(mask, as_tuple=False)[:, 0],
            n_amin[mask],
        ], dim=0)

    def forward(self, data: dict) -> dict:
        """
        Match keypoints and descriptors between two images

        Input (dict):
            image0: dict
                keypoints: [B x M x 2]
                descriptors: [B x M x D]
            image1: dict
                keypoints: [B x N x 2]
                descriptors: [B x N x D]
        Output (dict):
            matches0: [B x M]
            matching_scores0: [B x M]
            matches1: [B x N]
            matching_scores1: [B x N]
            matches: List[[Si x 2]]
            scores: List[[Si]]
            stop: int
            prune0: [B x M]
            prune1: [B x N]
        """
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        desc0, desc1 = data0["descriptors"], data1["descriptors"]
        
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        
        matches0_list = []
        matches1_list = []
        matching_scores0_list = []
        matching_scores1_list = []
        matches_list = []
        scores_list = []
        
        for k in range(b):
            desc0_k = desc0[k]  # [M x D]
            desc1_k = desc1[k]  # [N x D]
            
            match_indices = self.match_features(desc0_k, desc1_k)  # [2, K]
            
            if match_indices.shape[1] == 0:
                m0_k = torch.full((m,), -1, device=device, dtype=torch.long)
                m1_k = torch.full((n,), -1, device=device, dtype=torch.long)
                mscores0_k = torch.zeros((m,), device=device)
                mscores1_k = torch.zeros((n,), device=device)
                matches_k = torch.empty((0, 2), device=device, dtype=torch.long)
                scores_k = torch.empty((0,), device=device)
            else:
                m0_k = torch.full((m,), -1, device=device, dtype=torch.long)
                m0_k[match_indices[0]] = match_indices[1]
                
                m1_k = torch.full((n,), -1, device=device, dtype=torch.long)
                m1_k[match_indices[1]] = match_indices[0]
                
                dist_m = distance_matrix(desc0_k, desc1_k)
                mscores0_k = torch.zeros((m,), device=device)
                mscores1_k = torch.zeros((n,), device=device)
                
                for idx in range(match_indices.shape[1]):
                    i0, i1 = match_indices[0, idx].item(), match_indices[1, idx].item()
                    dist = dist_m[i0, i1]
                    score = 1.0 / (1.0 + dist)
                    mscores0_k[i0] = score
                    mscores1_k[i1] = score
                
                matches_k = match_indices.transpose(0, 1)  # [K x 2]
                scores_k = mscores0_k[match_indices[0]]  # [K]
            
            matches0_list.append(m0_k)
            matches1_list.append(m1_k)
            matching_scores0_list.append(mscores0_k)
            matching_scores1_list.append(mscores1_k)
            matches_list.append(matches_k)
            scores_list.append(scores_k)
        
        matches0 = torch.stack(matches0_list, dim=0)  # [B x M]
        matches1 = torch.stack(matches1_list, dim=0)  # [B x N]
        matching_scores0 = torch.stack(matching_scores0_list, dim=0)  # [B x M]
        matching_scores1 = torch.stack(matching_scores1_list, dim=0)  # [B x N]
        
        return {
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": matching_scores0,
            "matching_scores1": matching_scores1,
            "matches": matches_list,
            "scores": scores_list
        }

