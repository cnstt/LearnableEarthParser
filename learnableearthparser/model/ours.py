import numpy as np
import torch
import torch.nn.functional as F
from .point import decoders as PD

from .base import BaseModel

from ..utils import chamfer_distance
from typing import Tuple

# Only used to visualize mask if necessary
import plotly.graph_objects as go


@torch.jit.script
def compute_freq_loss(choice_K, choice_L, epsilon_freq_K: float, epsilon_freq_L: float, K: int, S: int):
    lK = choice_K.sum(0).sum(0)
    lK = lK / lK.sum()
    lK = - lK.clamp(max=epsilon_freq_K / K) / epsilon_freq_K

    lL = choice_L.sum(0)
    lL = lL / lL.sum()
    lL = - lL.clamp(max=epsilon_freq_L / S) / epsilon_freq_L

    return 1 + lK.sum(), 1 + lL.sum()

@torch.jit.script
def compute_l_XP(kappa_presoftmax: torch.Tensor, choice_L: torch.Tensor, cham_x: torch.Tensor, x_lengths_LK: torch.Tensor, S: int, K: int) -> torch.Tensor:
    for_wsum = kappa_presoftmax[..., 1:]
    for_wsum = torch.exp(for_wsum.unsqueeze(-2) - for_wsum.unsqueeze(-1)).sum(-1)
    cham_x = (cham_x.view(-1, S, K, cham_x.size(-1)) / for_wsum.unsqueeze(-1)).sum(-2)
    x_lengths_L = x_lengths_LK[..., 0]

    sorted_cham_x, indices = torch.sort(cham_x, 1, descending=False)
    epsilon_ordered = torch.gather(choice_L.unsqueeze(-1).repeat(1, 1, indices.size(-1)), 1, indices) #torch.equal(torch.gather(cham_x, 1, indices),sorted_cham_x) is True
    cumprod = torch.cumprod(1 - epsilon_ordered, dim=1)
    cumprod = torch.cat([torch.ones((cumprod.size(0), 1, cumprod.size(-1)), device=cumprod.device, dtype=cumprod.dtype), cumprod[:, :-1]], 1)
    l_XP = sorted_cham_x * epsilon_ordered * cumprod
    l_XP = (l_XP.sum(-1) / x_lengths_L).sum(-1) #l_XP.sum(-1).sum(-1) #(l_XP.sum(-1) / x_lengths).sum(-1)  ---> fixed chamfer loss

    return l_XP

@torch.jit.script
def clip_below_threshold(tensor: torch.Tensor, dist_threshold: float, min_ratio: float, clip_threshold: float) -> torch.Tensor:
    # Count the number of values below the threshold along the last dimension
    below_threshold_count = (tensor < dist_threshold).sum(dim=-1)
    # Create a mask indicating which elements have enough values below the distance threshold (close)
    mask = below_threshold_count >= min_ratio*tensor.shape[-1]
    # Zero out values along the last dimension for elements that satisfy the condition
    tensor[mask] = torch.where(tensor[mask]>clip_threshold, torch.tensor(0.0, device = tensor.device), tensor[mask])
    return tensor

@torch.jit.script
def cartesian_to_spherical(input_tensor: torch.Tensor) -> torch.Tensor:
    batch_size, S, K, num_points, _ = input_tensor.size()
    
    r_values = torch.norm(input_tensor, dim=-1)
    theta_values = torch.atan2(input_tensor[..., 1], input_tensor[..., 0])
    phi_values = torch.acos(input_tensor[..., 2] / r_values)
    
    spherical_coords = torch.stack((r_values, theta_values, phi_values), dim=-1)
    return spherical_coords

def visualize_mask_3d(mask, y):
    b=0
    s=0
    k=0
    y_np = y.detach().cpu().numpy()
    masked_pt = y[b, s, k][mask[b, s, k]].detach().cpu().numpy()
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter3d(x = y_np[b,s,k, :, 0], y = y_np[b,s,k, :, 1], z = y_np[b,s,k, :, 2],
                                       mode='markers', marker=dict(size=5, opacity=0.7),
                                       name='Points'))
    scatter_fig.add_trace(go.Scatter3d(x = masked_pt[:, 0], y=masked_pt[:, 1], z=masked_pt[:, 2],
                                       mode='markers', marker=dict(size=5, opacity=0.2, color='red'),
                                       name='Masked Points'))
    scatter_fig.update_layout(title=f'3D Points and Mask Visualization',
                              scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    scatter_fig.show()

@torch.jit.script
def mask_self_occultation(
    y: torch.Tensor,
    lidar_center: torch.Tensor,
    theta_step: float = 0.08,
    phi_step: float = 0.5,
    epsilon: float = 1.5
)-> torch.Tensor:
    # spherical coordinates
    y_centered = y-lidar_center.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    y_spherical = cartesian_to_spherical(y_centered)
    # binning by grid (theta, phi)
    """
    # There was a missing rad conversion, but let's keep it like this
    # Otherwise the sampling would be too dense for the sparse protos
    theta_step = np.radians(theta_step) * 10 # *10 = make sampling sparser
    phi_step = np.radians(phi_step) * 10
    epsilon=1
    """
    thetas = y_spherical[...,1] / theta_step
    phis = y_spherical[...,2] / phi_step
    theta_indexes = torch.round(phis)
    phi_indexes = torch.round(thetas)
    sphere_indexes = torch.stack((theta_indexes, phi_indexes), dim=-1)
    """
    # Simple first implementation
    mask = torch.zeros(y.size()[:-1], dtype=torch.bool, device=y.device)
    for b in range(y.size(0)):
        for s in range(y.size(1)):
            for k in range(y.size(2)):
                unique_combinations, unique_indices = torch.unique(sphere_indexes[b,s,k], dim=0, return_inverse=True)
                for i in torch.unique(unique_indices):
                    indices = unique_indices == i
                    # TODO: approximation used for now here
                    # The distance is only the shortest distance to the cell, not the center of the ray
                    values_to_consider = y_spherical[b,s,k][indices][..., 0]
                    min_value = torch.min(values_to_consider)
                    # argmin_index = torch.argmin(y[b,s,k][indices])
                    # Create a mask for all indices where the minimum value occurs
                    min_mask = (values_to_consider <= min_value + epsilon)
                    min_indices = indices.clone()
                    min_indices[indices] = min_mask
                    mask[b,s,k][min_indices] = True
    """
    # Efficient implementation
    mask_eff = torch.zeros(y.size()[:-1], dtype=torch.bool, device=y.device)
    unique_combinations_flat = torch.unique(sphere_indexes.view(-1,2), dim=0)
    # max value for where
    max_where = y_spherical[...,0].max()+2000
    for ij in unique_combinations_flat:
        indices_comb = torch.all(sphere_indexes == ij, dim=-1)
        # Arbitrary max value in where
        values_to_consider = torch.where(indices_comb ,y_spherical[..., 0], max_where)
        min_values = torch.min(values_to_consider, dim=-1).values
        # Create a mask for all indices where the minimum value occurs
        min_mask = (values_to_consider <= min_values.unsqueeze(-1) + epsilon)
        min_indices = indices_comb.clone()
        min_indices[indices_comb] = min_mask[indices_comb]
        mask_eff[min_indices] = True
    """
    # Tests to verify correctness of efficient implementation
    assert torch.equal(mask, mask_eff), "Efficient implementation doesn't lead to same result"
    true_count = torch.sum(mask)
    total_count = mask.numel()  # Total number of elements in the mask tensor
    ratio = true_count.float() / total_count
    print("Ratio of True values:", ratio.item())
    """
    return mask_eff

@torch.jit.script
def mask_general_occultation(
    x: torch.Tensor,
    x_lengths: torch.Tensor,
    y: torch.Tensor,
    lidar_center: torch.Tensor,
    theta_step: float = 0.08,
    phi_step: float = 0.5,
    epsilon: float = 1.5,
) -> torch.Tensor:
    y_centered = y-lidar_center.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    x_centered = x-lidar_center.unsqueeze(1)
    # convert to spherical
    y_spherical = cartesian_to_spherical(y_centered)
    x_spherical = cartesian_to_spherical(x_centered)
    # binning by grid (theta, phi), by using real lidar sampling (don't forget conversion to radians)
    theta_step = theta_step/360*torch.pi*2
    phi_step = phi_step/360*torch.pi*2    

    # contrary to self occlusion, the binning here happens on X
    x_thetas = x_spherical[...,1] / theta_step
    x_phis = x_spherical[...,2] / phi_step
    x_theta_indexes = torch.round(x_thetas)
    x_phi_indexes = torch.round(x_phis)
    x_sphere_indexes = torch.stack((x_theta_indexes, x_phi_indexes), dim=-1)    

    y_thetas = y_spherical[...,1] / theta_step
    y_phis = y_spherical[...,2] / phi_step
    y_theta_indexes = torch.round(y_thetas)
    y_phi_indexes = torch.round(y_phis)
    y_sphere_indexes = torch.stack((y_theta_indexes, y_phi_indexes), dim=-1)

    # max value used for depthmap
    max_where = y_spherical[...,0].max()+2000
    # Initialise mask
    mask_y = torch.zeros(y.size()[:-1], dtype=torch.bool, device=y.device)
    # Initialise y depth map    
    y_depth = torch.empty(mask_y.shape, device=y.device).fill_(max_where)
    # batch size
    bs = y.size(0)
    # Size of the z-buffer
    height = int(torch.pi/phi_step)+1 # The +1 fixes the out of range issue for phi=360 in y_sphere_indexes
    width = int(torch.pi*2/theta_step)

    z_buffer = torch.empty((bs, width, height), device=y.device).fill_(max_where)
    x_indexing = x_sphere_indexes.to(torch.long)
    x_indexing[...,0] += width//2-1
    x_mask = (torch.arange(x.size(1), device=x.device)[None] < x_lengths[:, None])
    for i in range(bs):
        z_buffer[i, x_indexing[i,x_mask[i],0], x_indexing[i,x_mask[i],1]] = x_spherical[i, x_mask[i], 0]
    y_indexing = y_sphere_indexes.to(torch.long)
    y_indexing[...,0] += width//2-1
    for i in range(bs):
        y_depth[i,...] = z_buffer[i, y_indexing[i,...,0], y_indexing[i,...,1]]

    # Only mask out points in the 'masked area of the scene'
    # visible_cond = (torch.abs(y_spherical[..., 0] - y_depth) <= epsilon)
    visible_cond = (y_spherical[..., 0] <= y_depth + epsilon)
    mask_y[visible_cond] = True

    return mask_y

@torch.jit.script
def compute_l_PX(
    y: torch.Tensor,
    choice_K: torch.Tensor,
    cham_y: torch.Tensor,
    max_xy: float,
    S: int,
    K: int,
    masking: str = "",
    soft_mask: float = 0,
    lidar_center: torch.Tensor = None,
    x: torch.Tensor = None,
    x_lengths: torch.Tensor = None,
)-> Tuple[torch.Tensor, torch.Tensor]:
    """l_PX corresponds to L_acc in the paper.
    It corresponds to the expected distance between:
    the reconstruction of slot S Ms(X) (the activated prototypes P) and X
    """
    with torch.no_grad():
        mask = torch.logical_and(
            y[..., :2] >= 0,
            y[..., :2] <= max_xy
        ).all(-1)
    
    # Mask out outside of patch values
    cham_y = cham_y * mask
    
    if masking == 'clipping':
        cham_y_reshaped = cham_y.view(-1, S, K, cham_y.size(-1))
        dist_threshold = 1.
        clip_threshold = 2.5
        min_ratio = 1/3
        clamped_cham_y = clip_below_threshold(cham_y_reshaped, dist_threshold, min_ratio, clip_threshold)
        # mask_y_soft is just needed for compatibility with other methods
        mask_y_soft = torch.ones(mask.shape, device=y.device).view(-1, S, K, mask.size(-1))
        cham_y = clamped_cham_y.view(-1, cham_y.size(-1))

    elif masking == 'self':
        cham_y_reshaped = cham_y.view(-1, S, K, cham_y.size(-1))
        y_reshaped = y.view(-1, S, K, y.size(-2), y.size(-1))
        mask_y = mask_self_occultation(y_reshaped, lidar_center)
        # Run the following line to get a visualisation of the masking
        # visualize_mask_3d(mask_y,y_reshaped)
        mask_y_soft = mask_y * (1 - soft_mask) + soft_mask
        masked_cham_y = cham_y_reshaped * mask_y_soft
        cham_y = masked_cham_y.view(-1, cham_y.size(-1))

    elif masking == 'gen':
        cham_y_reshaped = cham_y.view(-1, S, K, cham_y.size(-1))
        y_reshaped = y.view(-1, S, K, y.size(-2), y.size(-1))
        mask_y = mask_general_occultation(x, x_lengths, y_reshaped, lidar_center)
        # Run the following line to get a visualisation of the masking
        # visualize_mask_3d(mask_y,y_reshaped)
        mask_y_soft = mask_y * (1 - soft_mask) + soft_mask
        masked_cham_y = cham_y_reshaped * mask_y_soft
        cham_y = masked_cham_y.view(-1, cham_y.size(-1))

    elif masking == 'gen+self':
        cham_y_reshaped = cham_y.view(-1, S, K, cham_y.size(-1))
        y_reshaped = y.view(-1, S, K, y.size(-2), y.size(-1))
        # General occlusion
        mask_gen = mask_general_occultation(x, x_lengths, y_reshaped, lidar_center)
        # Self occlusion
        mask_self = mask_self_occultation(y_reshaped, lidar_center)
        # Final mask: apply self occlusion to visible parts in general occlusion
        mask_y = mask_gen * mask_self
        mask_y_soft = mask_y * (1 - soft_mask) + soft_mask
        masked_cham_y = cham_y_reshaped * mask_y_soft
        cham_y = masked_cham_y.view(-1, cham_y.size(-1))

    else:
        mask_y_soft = torch.ones(mask.shape, device=y.device).view(-1, S, K, mask.size(-1))

    # Normalize by the combination of both mask and mask_y_soft
    # If no occlusion, then the mask_y_soft contains only ones anyways
    mask_comb = mask * mask_y_soft.view(mask.shape)
    mask_sum = mask_comb.sum(-1)
    mask_sum_is_zero = mask_sum==0
    mask_sum[mask_sum_is_zero] = 1
    cham_y = cham_y.sum(-1) / mask_sum
    cham_y[mask_sum_is_zero] = 1.

    l_PX = choice_K.flatten() * cham_y
    l_PX = l_PX.view(-1, S*K).sum(-1)
    
    return l_PX / S, mask_y_soft


@torch.jit.script
def compute_translate_loss(translation_L: torch.Tensor) -> torch.Tensor:
    xytranslate = F.softshrink(translation_L[..., :2], 1.)**2

    return xytranslate.mean(0).sum()


@torch.jit.script
def triu_flatten(triu: torch.Tensor) -> torch.Tensor: 
    N = triu.size(-1)
    indicies = torch.triu_indices(N, N, offset=1)
    indicies = N * indicies[0] + indicies[1]
    return triu.flatten(-2)[:, indicies]


@torch.jit.script
def compute_overlap_loss(translation_L: torch.Tensor, tile_size: torch.Tensor, threshold: float=3) -> torch.Tensor:
    """
    Compute the overlap loss based on the translations with a minimum threshold.

    Args:
    translation_L (torch.Tensor): Tensor containing the x, y, z translations for each of the slots.
    threshold (float): The minimum threshold for the distance between slots.

    Returns:
    torch.Tensor: The computed overlap loss.

    """
    d = translation_L.squeeze(dim=-2)
    d = tile_size * d 
    distances = torch.cdist(d, d, p=2.)
    upper_triangular = torch.triu(distances, diagonal=1)
    points_dists = triu_flatten(upper_triangular)
    violation_distances = torch.clamp(threshold - points_dists, min=0)
    loss = torch.sum(violation_distances)
    return loss


@torch.jit.script
def compute_gamma_loss(choice_L):
    """Gamma loss corresponds to the sparse slot regularizer
    """
    return choice_L.sum(-1), torch.clamp(1 - choice_L.sum(-1), min=0)


class OursModel(BaseModel):

    def __initnets__(self):
        super().__initnets__()
        
        self.chooser = PD.LinearDecoder(
            dim_in = self.hparams.dim_latent,
            decoder = self.hparams.decoders.decoders, 
            dim_out = self.hparams.K + 1,
            norm = self.hparams.normalization_layer, end_with_bias=False
        )
        self.register_buffer("normalize_proba", 1. / torch.tensor(self.hparams.decoders.decoders[-1])**.5, persistent=False)

        if self.hparams.name == "superquadrics":
            self.SUPERQUADRIC_MODE = "train"

    def compute_logits(self, scene_features):
        return self.chooser(scene_features) * self.normalize_proba

    @torch.profiler.record_function(f"CHOICE")
    def compute_choice(self, protos, encoded, batch, proto_slab, batch_size, out):
        kappa = self.compute_logits(encoded.flatten(0, 1)).view(-1, self.hparams.S, self.hparams.K + 1)
        out["kappa_presoftmax"] = kappa
        
        if hasattr(self, "remove_proto"):
            to_share = kappa[:, :, self.remove_proto].sum(-1, keepdim=True)
            kappa[:, :, 1:] = kappa[:, :, 1:] + to_share / (kappa.size(-1) - 1)
            kappa[:, :, self.remove_proto] = kappa.min()
        
        kappa = torch.nn.functional.softmax(kappa, -1)
        out["kappa_postsoftmax"] = kappa

        choice_L = 1 - kappa[:, :, 0]
        choice_K = kappa[:, :, 1:]
        
        with torch.no_grad():
            if hasattr(self, "remove_proto"):
                choice = kappa.argmax(-1)
            else:
                choice = torch.multinomial(kappa.flatten(0, 1), 1).view(-1, choice_L.size(1))
            zero = choice.sum(-1) == 0
            most_probable_slot = kappa[zero, :, 0].min(-1)[1]
            choice[zero, most_probable_slot] = 1 + kappa[zero, most_probable_slot, 1:].max(-1)[1]

            choice = choice - 1
        
        # If use_gt, only keep the first slots corresponding to gt objects nb
        use_gt = getattr(self.hparams, "use_gt", False)
        if use_gt:
            with torch.no_grad():
                choice_L[:, :batch.gt.size(1)] = 1
                choice_L[:, batch.gt.size(1):] = 0
                # Also force choice_K to stay the same
                # Warning! choice_K has an effect on l_PX
                choice_K = torch.zeros_like(choice_K)
                choice_K[...,0] = 1
                choice[:, :batch.gt.size(1)] = 0
                choice[:, batch.gt.size(1):] = -1

        # Possibility to force the assignment of the prototypes:
        # use same prototypes for same GT objects
        gt_choice = getattr(self.hparams, "gt_choice", False)
        # Additionnal check: index_to_class_mapping dict needs to be defined in the config file
        if gt_choice and hasattr(self.hparams, "index_to_class_mapping"):
            if True:
            #with torch.no_grad():
                translation_L = out["translation_L"][..., 0:2]*self.tile_size[0:2]/2. + batch.lidar_center[...,0:2]
                gt_pos = batch.gt[...,0:2].squeeze(0).squeeze(0)
                distances = torch.cdist(translation_L, gt_pos)
                _, indices = torch.topk(distances, 1, largest=False)
                
                # Retrieve mapping dict set in config file
                # Objects in the scene are mapped to prototype ids
                index_to_class_mapping = self.hparams.index_to_class_mapping

                nearest_class_ids = [index_to_class_mapping[idx.item()] for idx in indices.flatten()]
                a=np.asarray(nearest_class_ids)
                choice_K = torch.zeros_like(choice_K)
                for i in range(len(nearest_class_ids)):
                    # choice_K[:,i][:,nearest_class_ids[i]]=1 
                    # slot proba should be respected in this setting?
                    choice_K[:,i][:,nearest_class_ids[i]]=choice_L[:,i]
                    for j in range(choice.size(0)):
                        choice[0][choice[0]!=-1] = torch.tensor(a, device=choice.device)[choice[0]!=-1]
        
        out["choice"] = choice
        out["choice_L"] = choice_L
        out["choice_K"] = choice_K

    @torch.profiler.record_function(f"LOSS_SUPERQUADRIC")
    def compute_reconstruction_loss(self, tag, batch, batch_size, out, protos, proto_slab, bkg=None, batch_idx=None):

        assert proto_slab.size(0) == batch_size, f"{proto_slab.size(0)} != {batch_size}"

        if self.hparams.distance != "xyz":
            protos = torch.cat([
                protos,
                self.get_protosfeat().unsqueeze(0).unsqueeze(0).unsqueeze(-2).repeat((protos.size(0), protos.size(1), 1, protos.size(-2),1))
            ], -1)

        # Torch.repeat() can been replaced by torch.expand() to reduce unnecessary memory usage
        x_lengths_LK = batch.pos_lenght.unsqueeze(-1).unsqueeze(-1).repeat(1, self.hparams.S, self.hparams.K)
        #x_lengths_LK = batch.pos_lenght.unsqueeze(-1).unsqueeze(-1).expand(batch.pos_lenght.size(0), self.hparams.S, self.hparams.K)
        y = protos.flatten(0, 2)
        
        x_LK = batch.pos_padded.unsqueeze(1).repeat(1, self.hparams.S*self.hparams.K, 1, 1).flatten(0, 1)
        #x_LK = batch.pos_padded.unsqueeze(1).expand(batch.pos_padded.size(0), self.hparams.S*self.hparams.K, batch.pos_padded.shape[-2], batch.pos_padded.shape[-1]).flatten(0, 1)
        cham_x, cham_y, _, _ = chamfer_distance(x_LK*self.lambda_xyz_feat, y*self.lambda_xyz_feat, x_lengths_LK.flatten(), y_lengths = None)
        
        # Visualization
        cham_y_reshaped_log = torch.log(cham_y.view(-1, self.hparams.S, self.hparams.K, cham_y.size(-1)))
        max_value, _ = cham_y_reshaped_log.max(dim=-1)
        min_value, _ = cham_y_reshaped_log.min(dim=-1)
        mean_value = cham_y_reshaped_log.mean(dim=-1)
        std_value = cham_y_reshaped_log.std(dim=-1)
        out["Cham_y/logCham_y"] = torch.log(cham_y).detach().cpu()
        out["Cham_y/logMax_Value"] = max_value.detach().cpu().flatten()
        out["Cham_y/logMin_Value"] = min_value.detach().cpu().flatten()
        out["Cham_y/logMean_Value"] = mean_value.detach().cpu().flatten()
        out["Cham_y/logStd_Value"] = std_value.detach().cpu().flatten()
        
        if hasattr(self.hparams, "masking") and hasattr(self.hparams, "soft_mask"):
            out["l_PX"], out["masks"] = compute_l_PX(y, out["choice_K"], cham_y, self.hparams.data.max_xy, self.hparams.S, self.hparams.K, self.hparams.masking, self.hparams.soft_mask, lidar_center=batch.lidar_center, x=batch.pos_padded, x_lengths=batch.pos_lenght)
        elif hasattr(self.hparams, "masking"):
            out["l_PX"], out["masks"] = compute_l_PX(y, out["choice_K"], cham_y, self.hparams.data.max_xy, self.hparams.S, self.hparams.K, self.hparams.masking, lidar_center=batch.lidar_center, x=batch.pos_padded, x_lengths=batch.pos_lenght)
        else:
            out["l_PX"], out["masks"] = compute_l_PX(y, out["choice_K"], cham_y, self.hparams.data.max_xy, self.hparams.S, self.hparams.K)

        out["l_XP"] = compute_l_XP(out["kappa_presoftmax"], out["choice_L"], cham_x, x_lengths_LK, self.hparams.S, self.hparams.K)

        out["l_gamma"], out["l_gamma0"] = compute_gamma_loss(out["choice_L"])

        out["l_freq_K"], out["l_freq_L"] = compute_freq_loss(
            out["choice_K"], out["choice_L"],
            self.hparams.epsilon_freq_K, self.hparams.epsilon_freq_L,
            self.hparams.K, self.hparams.S
        )

        out["l_xytranslate"] = compute_translate_loss(out["translation_L"])
        out["l_overlap"] = compute_overlap_loss(out["translation_L"], self.tile_size)
        
        if self.hparams.distance != "xyz":
            protos = protos[..., :3]

        with torch.no_grad():
            super().compute_reconstruction_loss(tag, batch, batch_size, out, protos, proto_slab, bkg, batch_idx)
        
    def compute_loss(self, tag, batch_size, out):
        return (
            self.hparams.lambda_XP * out["l_XP"]
            + self.hparams.lambda_PX * out["l_PX"]
            + self.hparams.lambda_gamma * out["l_gamma"]
            + self.hparams.lambda_gamma0 * out["l_gamma0"]
            + self.hparams.lambda_freq_K * out["l_freq_K"]
            + self.hparams.lambda_freq_L * out["l_freq_L"]
            + self.hparams.lambda_xytranslate * out["l_xytranslate"]
            + self.hparams.lambda_overlap * out["l_overlap"]
        ).mean()

    @torch.no_grad()    
    def greedy_histograms(self, batch, out):
        super().greedy_histograms(batch, out)
        
        with torch.no_grad():
            # for i in range(out["Cham_y/logCham_y"].shape[0]):
            #     self.logger.experiment.add_histogram(
            #         f"Cham_y/logCham_y",
            #         out["Cham_y/logCham_y"][i],
            #         global_step=self.current_epoch
            #     )

            self.logger.experiment.add_histogram(
                f"Cham_y/logMax_Value",
                out["Cham_y/logMax_Value"],
                global_step=self.current_epoch
            )

            self.logger.experiment.add_histogram(
                f"Cham_y/logMin_Value",
                out["Cham_y/logMin_Value"],
                global_step=self.current_epoch
            )

            self.logger.experiment.add_histogram(
                f"Cham_y/logMean_Value",
                out["Cham_y/logMean_Value"],
                global_step=self.current_epoch
            )

            self.logger.experiment.add_histogram(
                f"Cham_y/logStd_Value",
                out["Cham_y/logStd_Value"],
                global_step=self.current_epoch
            )
        
            self.logger.experiment.add_histogram(
                f"choice_L",
                out["choice_L"].detach().cpu().flatten(),
                global_step=self.current_epoch
            )

            self.logger.experiment.add_histogram(
                f"choice_K",
                out["choice_K"].detach().cpu().flatten(),
                global_step=self.current_epoch
            )

            self.logger.experiment.add_histogram(
                f"choice_K_max",
                out["choice_K"].max(-1)[0].detach().cpu().flatten(),
                global_step=self.current_epoch
            )

            if hasattr(self, "chooser"):
                self.logger.experiment.add_histogram(
                    f"chooser_LK/weight",
                    self.chooser[-1].weight.detach().cpu().flatten(),
                    global_step=self.current_epoch
                )
                if hasattr(self.chooser[-1], "bias") and self.chooser[-1].bias is not None:
                    self.logger.experiment.add_histogram(
                        f"chooser_LK/bias",
                        self.chooser[-1].bias.detach().cpu().flatten(),
                        global_step=self.current_epoch
                    )