from torch import nn

import torch

from src.model.DpRnnTasnet.utils import GlobalLayerNorm
from src.model.DpRnnTasnet.dp_rnn_block import DPRNNBlock

class DpRnnTasNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        num_speakers: int = 2,
        out_channels: int=128,
        hidden_size:int =128,
        chunk_size:int =100,
        num_layers:int =6,
        bidirectional: bool=True,
        rnn_num_layers:int =1,
        dropout:float =0.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.output_channels = in_channels
        self.overlapping_stride = chunk_size // 2
        self.num_speakers = num_speakers
        self.bottleneck_channels = out_channels

        self.tiny_layer = nn.Sequential(GlobalLayerNorm(in_channels), nn.Conv1d(in_channels, out_channels, 1))

        self.dpp_rnn_blocks = nn.Sequential()
        for i in range(num_layers):
            self.dpp_rnn_blocks.add_module(
                f"DPRNNBlock_{i}",
                DPRNNBlock(
                    out_channels,
                    hidden_size,
                    bidirectional=bidirectional,
                    num_layers=rnn_num_layers,
                    dropout=dropout,
                )
            )

        self.source_separator = nn.Sequential(nn.PReLU(), nn.Conv2d(out_channels, num_speakers * out_channels, 1))

        self.out_layer = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.layer_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())
        self.mask_layer = nn.Conv1d(out_channels, in_channels, 1, bias=False)


    def forward(self, mix_audio: torch.Tensor, **batch) -> None:
        mix_audio = mix_audio.unsqueeze(1)
        batch, _, n_frames = mix_audio.size()
        mix_audio = self.tiny_layer(mix_audio)
        splitted_to_chunks = nn.functional.unfold(
            mix_audio.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.overlapping_stride, 1),
        )
        n_chunks = splitted_to_chunks.shape[-1]
        splitted_to_chunks = splitted_to_chunks.view(batch, self.bottleneck_channels, self.chunk_size, n_chunks)

        dpp_rnn_blocks_result = self.dpp_rnn_blocks(splitted_to_chunks)
        source_separated = self.source_separator(dpp_rnn_blocks_result)
        source_separated = source_separated.view(batch * self.num_speakers, self.bottleneck_channels, self.chunk_size, n_chunks)
        combined_chunks = nn.functional.fold(
            source_separated.reshape(batch * self.num_speakers, self.bottleneck_channels * self.chunk_size, n_chunks),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.overlapping_stride, 1),
        )
        combined_chunks = combined_chunks.view(batch * self.num_speakers, self.bottleneck_channels, -1)
        mask = self.mask_layer(self.out_layer(combined_chunks) * self.layer_gate(combined_chunks))
        mask = nn.functional.relu(mask)
        result = mask.view(batch, self.num_speakers, self.output_channels, n_frames).squeeze(2)

        return result

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info