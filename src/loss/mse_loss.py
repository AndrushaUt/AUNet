import torch
from torch import nn

class CustomMSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, s1_audio, s2_audio, s1_estimated, s2_estimated, **batch):
        s1_s1 = self.loss(s1_audio, s1_estimated)
        s1_s2 = self.loss(s1_audio, s2_estimated)
        s2_s1 = self.loss(s2_audio, s1_estimated)
        s2_s2 = self.loss(s2_audio, s2_estimated)

        permute_1 = torch.sum((s1_s1 + s2_s2) / 2, axis=-1)
        permute_2 = torch.sum((s1_s2 + s2_s1) / 2, axis=-1)
        loss = torch.minimum(permute_1, permute_2)
        return {"loss": torch.mean(loss)}
