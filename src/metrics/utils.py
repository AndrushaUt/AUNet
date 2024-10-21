import torch

def calc_si_sdr(target, est) -> float:
    alpha = (est * target).sum() / torch.square(target)
    return 10 * torch.log10(torch.square(alpha * target).sum() / torch.square(alpha * target - est).sum())

def calc_si_sdri(target, est, source) -> float:
    return calc_si_sdr(target, est) - calc_si_sdr(target, source)
