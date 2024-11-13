from src.model.ConvTasNet.separator import Separator
from src.utils.model_utils import pad_tensor

from torch import nn
from torch import Tensor
import torch

class ConvTasNet(nn.Module):
    def __init__(
        self, 
        num_speakers: int = 2,
        encoder_kernel_size:int = 16,
        encoder_num_feats: int = 512,
        separator_kernel_size: int = 3,
        separator_num_feats: int = 128,
        separator_num_hidden: int = 512,
        separator_num_layers: int = 8,
        separator_num_stacks: int = 3,
        norm_type: str = "group",
    ):
        super().__init__()

        self.num_speakers = num_speakers
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_num_feats = encoder_num_feats
        self.encoder_stride = self.encoder_kernel_size // 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.encoder_num_feats,
            kernel_size=self.encoder_kernel_size,
            stride=self.encoder_stride,
            padding=self.encoder_stride,
            bias=False,
        )

        self.decoder = nn.ConvTranspose1d(
            in_channels=self.encoder_num_feats,
            out_channels=1,
            kernel_size=self.encoder_kernel_size,
            stride=self.encoder_stride,
            padding=self.encoder_stride,
            bias=False,
        )

        self.separation = Separator(
            input_size=self.encoder_num_feats,
            hidden_size=separator_num_hidden,
            num_speakers=self.num_speakers,
            kernel_size=separator_kernel_size,
            num_feats=separator_num_feats,
            num_layers=separator_num_layers,
            num_stacks=separator_num_stacks,
            norm_type=norm_type,
        )


    def forward(self, mix_audio: Tensor, **batch):
        mix_audio = mix_audio.unsqueeze(1)
        padded, num_pads = pad_tensor(
            mix_audio, 
            self.encoder_kernel_size, 
            self.encoder_stride, 
            self.device
            )
        batch_size = len(padded)

        encoded = self.encoder(padded)
        masked = self.separation(encoded) * encoded.unsqueeze(1)
        masked = masked.view(batch_size * self.num_speakers, self.encoder_num_feats, -1)
        result = self.decoder(masked).view(batch_size, self.num_speakers, -1)
            
        return result[..., :-num_pads] if num_pads else result

    
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
