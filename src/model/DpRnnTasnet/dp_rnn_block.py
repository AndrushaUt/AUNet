from torch import nn
import torch

from src.model.DpRnnTasnet.rnn_block import DualRnn
from src.model.DpRnnTasnet.utils import GlobalLayerNorm


class DPRNNBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_size: int,
        bidirectional: bool=True,
        num_layers: int=1,
        dropout: float=0.0,
    ):
        super().__init__()

        intra_rnn = DualRnn(
            input_channels,
            hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        inter_rnn = DualRnn(
            input_channels,
            hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.fc1 = nn.Linear(intra_rnn.output_size, input_channels)
        self.fc2 = nn.Linear(inter_rnn.output_size, input_channels)

        self.intra_part = nn.Sequential(
            intra_rnn,
            nn.Linear(intra_rnn.output_size, input_channels)
        )
        self.inter_part = nn.Sequential(
            inter_rnn,
            nn.Linear(inter_rnn.output_size, input_channels)
        )

        self.global_norm1 = GlobalLayerNorm(input_channels)
        self.global_norm2 = GlobalLayerNorm(input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, features, chunk_size, num_chunks = x.size()
        skip_connection = x  
        x = x.permute(0, 3, 2, 1).reshape(batch_size * num_chunks, chunk_size, features)
        x = self.inter_part(x)
        x =  x.reshape(batch_size, num_chunks, chunk_size, features).permute(0, 3, 2, 1) 
        x = self.global_norm1(x)
        skip_connection = skip_connection + x

        x = skip_connection.permute(0, 2, 3, 1).reshape(batch_size * chunk_size, num_chunks, features)
        x = self.intra_part(x)
        x = x.reshape(batch_size, chunk_size, num_chunks, features).permute(0, 3, 1, 2).contiguous()
        x = self.global_norm2(x)
        return skip_connection + x