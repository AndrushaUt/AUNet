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
        norm_type: str,
    ):
        super().__init__()

        self.first_conv = nn.Conv1d(
                in_channels=input_size, 
                out_channels=hidden_size, 
                kernel_size=1
            )
        self.first_activation = nn.PReLU()
        
        self.first_norm = nn.GroupNorm(1, hidden_size) if norm_type == 'group' else nn.LayerNorm(hidden_size)


        self.second_conv = nn.Conv1d(
                in_channels=hidden_size, 
                out_channels=hidden_size, 
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_size
            )
        self.second_activation = nn.PReLU()
        self.second_norm = nn.GroupNorm(1, hidden_size) if norm_type == 'group' else nn.LayerNorm(hidden_size)


        self.out = nn.Conv1d(
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
        if isinstance(self.first_norm, nn.GroupNorm):
            x = self.first_activation(self.first_conv(mix_audio))
            x = self.first_norm(x)
            x = self.second_activation(self.second_conv(x))
            x = self.second_norm(x)
        else:
            x = self.first_activation(self.first_conv(mix_audio)).transpose(1, 2)
            x = self.first_norm(x).transpose(1, 2)
            x = self.second_activation(self.second_conv(x)).transpose(1, 2)
            x = self.second_norm(x).transpose(1, 2)
            

        out = self.out(x)
        skip_connection = self.skip_connection(x)

        return out, skip_connection



