import torch

from torchmetrics import ScaleInvariantSignalDistortionRatio

def calc_si_sdr(target, est) -> float:
    alpha = (est * target).sum(dim=1, keepdim=True) / torch.square(target).sum(dim=1, keepdim=True)
    return 10 * torch.log10(torch.square(alpha * target).sum(dim=1) / torch.square(alpha * target - est).sum(dim=1) + 1e-4)

def calc_si_sdri(target, est, source) -> float:
    return calc_si_sdr(target, est) - calc_si_sdr(target, source)
