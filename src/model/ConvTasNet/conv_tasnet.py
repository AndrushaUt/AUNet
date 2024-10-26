from src.model.ConvTasNet.masking import Mask

from torch import nn
from torch import Tensor
import torch

class ConvTasNet(nn.Module):
    def __init__(
        self, 
        num_speakers: int = 2,
        encoder_kernel_size:int = 16,
        encoder_num_feats: int = 512,
        mask_kernel_size: int = 3,
        mask_num_feats: int = 128,
        mask_num_hidden: int = 512,
        mask_num_layers: int = 8,
        mask_num_stacks: int = 3,
    ):
        super().__init__()

        self.num_speakers = num_speakers
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_num_feats = encoder_num_feats
        self.encoder_stride = self.encoder_kernel_size // 2


        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.encoder_num_feats,
            kernel_size=self.encoder_kernel_size,
            stride=self.encoder_stride,
            padding=self.encoder_stride,
            bias=False,
        )

        self.separation = Mask(
            input_size=self.encoder_num_feats,
            hidden_size=mask_num_hidden,
            num_speakers=self.num_speakers,
            kernel_size=mask_kernel_size,
            num_feats=mask_num_feats,
            num_layers=mask_num_layers,
            num_stacks=mask_num_stacks,
            device="cpu"
        )

        self.decoder = nn.ConvTranspose1d(
            in_channels=self.encoder_num_feats,
            out_channels=1,
            kernel_size=self.encoder_kernel_size,
            stride=self.encoder_stride,
            padding=self.encoder_stride,
            bias=False,
        )

    def pad_tensor(self, tensor: Tensor):
        batch_size, num_channels, num_frames = tensor.shape
        
        is_odd = self.encoder_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.encoder_stride
        num_remainings = num_frames - (is_odd + num_strides * self.encoder_stride)

        if not num_remainings:
            return tensor, 0

        to_pad = torch.zeros(
            batch_size,
            num_channels,
            self.encoder_stride - num_remainings,
            device=self.device,
        )

        return torch.cat([tensor, to_pad], dim=2), self.encoder_stride - num_remainings


    def forward(self, mix_audio: Tensor, **batch):
        mix_audio = mix_audio.unsqueeze(1)
        padded, num_pads = self.pad_tensor(mix_audio)
        batch_size = len(padded)
        encoded = self.encoder(padded)
        masked = self.separation(encoded) * encoded.unsqueeze(1)
        masked = masked.view(batch_size * self.num_speakers, self.encoder_num_feats, -1)
        decoded = self.decoder(masked)
        output = decoded.view(batch_size, self.num_speakers, -1)
        if num_pads:
            output = output[..., :-num_pads]
        return output

    
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
