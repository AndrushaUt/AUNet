from torch import Tensor
import torch
from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_si_sdri

class SI_SDRiMetric(BaseMetric):
    def __call__(
        self, 
        s1_audio: Tensor,
        s2_audio: Tensor,
        mix_audio: Tensor,
        s1_estimated: Tensor, 
        s2_estimated: Tensor, 
        **batch,
    ):
        si_sdris = []
        s1_s1 = calc_si_sdri(s1_audio, s1_estimated, mix_audio)
        s1_s2 = calc_si_sdri(s1_audio, s2_estimated, mix_audio)
        s2_s1 = calc_si_sdri(s2_audio, s1_estimated, mix_audio)
        s2_s2 = calc_si_sdri(s2_audio, s2_estimated, mix_audio)
        loss = torch.maximum((s1_s1 + s2_s2) / 2, (s1_s2 + s2_s1) / 2)
        return loss.mean()
