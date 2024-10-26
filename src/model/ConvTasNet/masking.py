from src.model.ConvTasNet.convolution import ConvBlock

from torch import nn
from torch import Tensor
import torch

class Mask(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_speakers: int,
        kernel_size: int,
        num_feats: int,
        num_layers: int,
        num_stacks: int,
        device: torch.device = "cuda",
    ):
        super().__init__()

        self.input_size = input_size
        self.num_speakers = num_speakers
        self.device = device

        self.input_normalization = nn.GroupNorm(
            num_groups=1, 
            num_channels=input_size
        )
        self.input_convolution = nn.Conv1d(
            in_channels=input_size, 
            out_channels=num_feats, 
            kernel_size=1
        )

        self.separation_layers = nn.Sequential()

        for stack in range(num_stacks):
            d = 1
            for layer in range(num_layers):
                self.separation_layers.add_module(
                    f"Stack {stack} Layer {layer}",
                    ConvBlock(
                        input_size=num_feats,
                        hidden_size=hidden_size,
                        kernel_size = kernel_size,
                        padding=d,
                        dilation=d,
                        residual=(not layer==num_layers-1 or not stack==num_stacks-1),
                    )
                )
                d *= 2

        self.output_prelu = nn.PReLU()
        self.output_convolution = nn.Conv1d(
            in_channels=num_feats,
            out_channels=input_size * num_speakers,
            kernel_size=1,
        )
        
        self.output_activation=nn.Sigmoid()


    def forward(self, mix_audio: Tensor):
        batch_size = mix_audio.shape[0]
        features = self.input_convolution(self.input_normalization(mix_audio))

        x = torch.zeros_like(features, requires_grad=True)
        for layer in self.separation_layers:
            residual, skip_connection = layer(features)
            # if not x:
            #     x = torch.zeros_like(skip_connection, device=self.device, requires_grad=True)
            if residual is not None:
                features = features + residual

            x =  x + skip_connection

        x = self.output_prelu(x)
        x = self.output_convolution(x)
        output = self.output_activation(x)

        return output.view(batch_size, self.num_speakers, self.input_size, -1)
