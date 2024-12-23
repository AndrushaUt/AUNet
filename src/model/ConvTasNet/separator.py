from src.model.ConvTasNet.convolution import ConvBlock

from torch import nn
from torch import Tensor
import torch

class Separator(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_speakers: int,
        kernel_size: int,
        num_feats: int,
        num_layers: int,
        num_stacks: int,
        norm_type: str,
        activation: str,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_speakers = num_speakers

        self.norm = nn.GroupNorm(1, hidden_size) if norm_type == 'global' else nn.LayerNorm(hidden_size)

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
                        norm_type=norm_type,
                    )
                )
                d *= 2

        self.output_prelu = nn.PReLU()
        self.output_convolution = nn.Conv1d(
            in_channels=num_feats,
            out_channels=input_size * num_speakers,
            kernel_size=1,
        )
        
        self.output_activation = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU()



    def forward(self, mix_audio: Tensor):
        batch_size = mix_audio.shape[0]
        features = None
        if isinstance(self.norm, nn.GroupNorm):
            features = self.input_convolution(self.norm(mix_audio))
        else:
            mix_audio = mix_audio.transpose(1, 2)
            features = self.input_convolution(self.norm(mix_audio).transpose(1, 2))

        x = torch.zeros_like(features, requires_grad=True)
        for layer in self.separation_layers:
            out, skip_connection = layer(features)

            features = features + out
            x =  x + skip_connection

        x = self.output_convolution(self.output_prelu(x))
        output = self.output_activation(x)

        return output.view(batch_size, self.num_speakers, self.input_size, -1)
