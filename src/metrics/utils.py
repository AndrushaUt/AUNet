import torch
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

def calc_si_sdr(target, est) -> float:
    alpha = (est * target).sum(dim=1, keepdim=True) / torch.square(target).sum(dim=1, keepdim=True)
    return 10 * torch.log10(torch.square(alpha * target).sum(dim=1) / torch.square(alpha * target - est).sum(dim=1) + 1e-4)

def calc_si_sdri(target, est, source) -> float:
    return calc_si_sdr(target, est) - calc_si_sdr(target, source)

def calc_si_snri(target, est, source) -> float:
    return scale_invariant_signal_noise_ratio(est, target) - scale_invariant_signal_noise_ratio(source, target)
