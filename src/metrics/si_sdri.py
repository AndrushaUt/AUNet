from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_si_sdri

class SI_SDRiMetric(BaseMetric):
    def __call__(
        self, 
        s1_audios: Tensor,
        s1_audio_lengths: Tensor,
        s2_audios: Tensor,
        s2_audio_lengths: Tensor,
        mix_audios: Tensor,
        mix_audio_lengths: Tensor,
        estimateds_s1: Tensor, 
        estimateds_s2: Tensor, 
        **kwargs,
    ):
        si_sdris = []
        for i in range(len(mix_audios)):
            mix_audio_length = mix_audio_lengths[i]
            s1_audio_length = s1_audio_lengths[i]
            s2_audio_length = s2_audio_lengths[i]
            mix_audio = mix_audios[i][:mix_audio_length]
            s1_audio = s1_audios[i][:s1_audio_length]
            s2_audio = s2_audios[i][:s2_audio_length]
            estimated_s1 = estimateds_s1[i][:s1_audio_length]
            estimated_s2 = estimateds_s2[i][:s2_audio_length]

            si_sdris.append(calc_si_sdri(s1_audio, estimated_s1, mix_audio))
            si_sdris.append(calc_si_sdri(s2_audio, estimated_s2, mix_audio))
        return sum(si_sdris) / len(si_sdris)
