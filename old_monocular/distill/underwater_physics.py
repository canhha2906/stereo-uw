"""Physics-based underwater image formation model (Jaffe-McGlamery style).

Used to synthesize underwater-looking training images from ordinary
"land" RGB images plus a depth map, so a distilled student model can be
supervised without needing real underwater depth ground truth.

Model (per RGB channel c):
    I_c(x) = J_c(x) * t_c(x) + A_c * (1 - t_c(x))
    t_c(x) = exp(-beta_c * d(x))

J_c   : clean, in-air radiance (the "land" image)
d(x)  : scene distance driving light attenuation
beta_c: per-channel attenuation coefficient (red attenuates fastest)
A_c   : per-channel backscatter / veiling-light color
I_c   : the synthesized underwater observation

`d(x)` here comes from a monocular teacher's *relative* depth, which has
no metric scale or origin. We min-max normalize it per-sample and then
randomize an overall depth-scale factor as part of the domain
randomization -- this is a modeling choice standing in for real water
depth, not a calibrated distance.
"""
from dataclasses import dataclass

import torch

Range = tuple


@dataclass
class PhysicsParams:
    beta_r_range: Range = (0.3, 1.2)
    beta_g_range: Range = (0.05, 0.5)
    beta_b_range: Range = (0.02, 0.3)
    backscatter_range: Range = (0.05, 0.6)
    depth_scale_range: Range = (1.0, 8.0)


def _sample_uniform(batch: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.empty(batch, 1, 1, 1, device=device).uniform_(lo, hi)


def synthesize_underwater(clean_rgb: torch.Tensor, rel_depth: torch.Tensor,
                          params: PhysicsParams) -> torch.Tensor:
    """clean_rgb: (B,3,H,W) in [0,1]. rel_depth: (B,1,H,W), teacher's raw
    relative depth (any positive scale; normalized internally).
    Returns the synthesized underwater image, (B,3,H,W) in [0,1]-ish
    (not clamped, since the physics model can slightly exceed 1 under
    strong backscatter -- clamp downstream if a display range is needed).
    """
    B = clean_rgb.shape[0]
    device = clean_rgb.device

    d_min = rel_depth.amin(dim=(1, 2, 3), keepdim=True)
    d_max = rel_depth.amax(dim=(1, 2, 3), keepdim=True).clamp(min=d_min + 1e-6)
    d_norm = (rel_depth - d_min) / (d_max - d_min)

    depth_scale = _sample_uniform(B, *params.depth_scale_range, device)
    d = d_norm * depth_scale  # (B,1,H,W), pseudo-metres

    beta_r = _sample_uniform(B, *params.beta_r_range, device)
    beta_g = _sample_uniform(B, *params.beta_g_range, device)
    beta_b = _sample_uniform(B, *params.beta_b_range, device)
    beta = torch.cat([beta_r, beta_g, beta_b], dim=1)  # (B,3,1,1)

    a_r = _sample_uniform(B, *params.backscatter_range, device)
    a_g = _sample_uniform(B, *params.backscatter_range, device)
    a_b = _sample_uniform(B, *params.backscatter_range, device)
    A = torch.cat([a_r, a_g, a_b], dim=1)  # (B,3,1,1)

    t = torch.exp(-beta * d)  # broadcasts (B,3,1,1) x (B,1,H,W) -> (B,3,H,W)
    return clean_rgb * t + A * (1.0 - t)
