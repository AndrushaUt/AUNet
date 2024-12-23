import torch_audiomentations
from torch import Tensor, nn


class PeakNormalization(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.PeakNormalization(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data.unsqueeze(1)).squeeze(1)
