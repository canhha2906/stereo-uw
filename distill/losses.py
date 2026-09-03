"""Scale-and-shift invariant loss for distilling from a *relative* depth
teacher (MiDaS-style output has no metric scale or origin, so raw L1/L2
against it is meaningless without first aligning scale+shift)."""
import torch


def _lstsq_scale_shift(pred: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor):
    """Closed-form scale s and shift t minimizing sum((s*pred+t-target)^2)
    over masked pixels, solved independently per sample in the batch.
    pred/target/mask: (B,1,H,W). Returns s, t as (B,1) tensors."""
    pred = pred.flatten(1)
    target = target.flatten(1)
    mask = mask.flatten(1)

    pred_m = pred * mask
    n = mask.sum(dim=1).clamp(min=1.0)

    sum_p = pred_m.sum(dim=1)
    sum_t = (target * mask).sum(dim=1)
    sum_pp = (pred_m * pred).sum(dim=1)
    sum_pt = (pred_m * target).sum(dim=1)

    denom = (n * sum_pp - sum_p ** 2).clamp(min=1e-6)
    s = (n * sum_pt - sum_p * sum_t) / denom
    t = (sum_t - s * sum_p) / n
    return s.view(-1, 1), t.view(-1, 1)


def scale_shift_invariant_l1(pred: torch.Tensor, target: torch.Tensor,
                             mask: torch.Tensor = None) -> torch.Tensor:
    """pred, target: (B,1,H,W) relative depth. mask: (B,1,H,W) 0/1, defaults
    to all-valid. Aligns pred to target's scale/shift per-sample before
    computing masked L1 -- this is the standard MiDaS-style training loss
    for depth teachers/targets with unknown scale and origin."""
    if mask is None:
        mask = torch.ones_like(target)
    s, t = _lstsq_scale_shift(pred, target, mask)
    pred_aligned = pred * s.view(-1, 1, 1, 1) + t.view(-1, 1, 1, 1)
    err = (pred_aligned - target).abs() * mask
    return err.sum() / mask.sum().clamp(min=1.0)
