from torch import Tensor
import torch
from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_si_sdri

class SI_SDRiMetric(BaseMetric):
    def __call__(
        self, 
        s1_audio: Tensor,
        s1_audio_length: Tensor,
        s2_audio: Tensor,
        s2_audio_length: Tensor,
        mix_audio: Tensor,
        mix_audio_length: Tensor,
        s1_estimated: Tensor, 
        s2_estimated: Tensor, 
        **batch,
    ):
        si_sdris = []
        # for i in range(len(mix_audio)):
        #     mix_audio_length_ = mix_audio_length[i]
        #     s1_audio_length_ = s1_audio_length[i]
        #     s2_audio_length_ = s2_audio_length[i]
        #     mix_audio_ = mix_audio[i][:mix_audio_length_]
        #     s1_audio_ = s1_audio[i][:s1_audio_length_]
        #     s2_audio_ = s2_audio[i][:s2_audio_length_]
        #     s1_estimated_ = s1_estimated[i][:s1_audio_length_]
        #     s2_estimated_ = s2_estimated[i][:s2_audio_length_]
        #     s1_s1 = calc_si_sdri(s1_audio_, s1_estimated_, mix_audio_)
        #     s1_s2 = calc_si_sdri(s1_audio_, s2_estimated_, mix_audio_)
        #     s2_s1 = calc_si_sdri(s2_audio_, s1_estimated_, mix_audio_)
        #     s2_s2 = calc_si_sdri(s2_audio_, s2_estimated_, mix_audio_)

        #     if s1_s1 + s2_s2 > s1_s2 + s2_s1:
        #         si_sdris.append((s1_s1 + s2_s2) / 2)
        #     else:
        #         si_sdris.append((s1_s2 + s2_s1) / 2)
        s1_s1 = calc_si_sdri(s1_audio, s1_estimated, mix_audio)
        s1_s2 = calc_si_sdri(s1_audio, s2_estimated, mix_audio)
        s2_s1 = calc_si_sdri(s2_audio, s1_estimated, mix_audio)
        s2_s2 = calc_si_sdri(s2_audio, s2_estimated, mix_audio)
        loss = torch.maximum((s1_s1 + s2_s2) / 2, (s1_s2 + s2_s1) / 2)
        return loss.mean()
