import torch.nn as nn

from torch import Tensor

class ConvBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        padding: int, 
        dilation: int,
        residual: bool = True,
    ):
        super().__init__()

        self.convolutions = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size, 
                out_channels=hidden_size, 
                kernel_size=1
            ),
            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_size),

            nn.Conv1d(
                in_channels=hidden_size, 
                out_channels=hidden_size, 
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_size
            ),
            nn.PReLU(),
            nn.GroupNorm(num_groups=1, num_channels=hidden_size),
        )

        self.residual = None
        if residual:
            self.residual = nn.Conv1d(
                in_channels=hidden_size, 
                out_channels=input_size, 
                kernel_size=1
            )

        self.skip_connection = nn.Conv1d(
            in_channels=hidden_size, 
            out_channels=input_size, 
            kernel_size=1
        )

    
    def forward(self, mix_audio: Tensor):
        x = self.convolutions(mix_audio)
        residual = self.residual(x) if self.residual else None
        skip_connection = self.skip_connection(x)

        return residual, skip_connection



