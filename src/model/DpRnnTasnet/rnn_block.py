from torch import nn
import torch


class DualRnn(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        n_layers: int=1,
        dropout: float =0, 
        bidirectional:bool = True
    ) -> None:
        super().__init__()
        self.output_size = hidden_size + int(bidirectional) * hidden_size + input_size
        self.first_rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.second_rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        result1, _ = self.first_rnn(x)
        result2, _ = self.second_rnn(x)
        return torch.cat((result1 * result2, x), 2)
