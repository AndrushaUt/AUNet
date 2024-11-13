import torch

def pad_tensor(
    tensor: torch.Tensor, 
    encoder_kernel_size: int, 
    encoder_stride: int, 
    device: torch.device,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames = tensor.shape
        
        num_strides = (num_frames - encoder_kernel_size % 2) // encoder_stride
        to_subtract = num_frames - (encoder_kernel_size % 2 + num_strides * encoder_stride)

        if not to_subtract:
            return tensor, 0
        else:
            to_pad = torch.zeros(batch_size, num_channels, encoder_stride - to_subtract, device=device)
            padded = torch.cat([tensor, to_pad], dim=2) # batch_size, num_channels, num_frames + encoder_stride - to_subtract
            return padded, encoder_stride - to_subtract  
