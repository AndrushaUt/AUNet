import torch
from torch import nn

from src.metrics.utils import calc_si_sdr

class SI_SDR_LOSS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s1_audio, s2_audio, s1_estimated, s2_estimated, **batch):
        s1_s1 = calc_si_sdr(s1_audio, s1_estimated)
        s1_s2 = calc_si_sdr(s1_audio, s2_estimated)
        s2_s1 = calc_si_sdr(s2_audio, s1_estimated)
        s2_s2 = calc_si_sdr(s2_audio, s2_estimated)

        # permute_1 = torch.sum((s1_s1 + s2_s2) / 2, axis=-1)
        # permute_2 = torch.sum((s1_s2 + s2_s1) / 2, axis=-1)
        loss = torch.maximum((s1_s1 + s2_s2) / 2, (s1_s2 + s2_s1) / 2)
        return {"loss": -torch.mean(loss)}
