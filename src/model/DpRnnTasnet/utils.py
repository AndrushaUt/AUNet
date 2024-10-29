import torch
from torch import nn

class GlobalLayerNorm(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(input_channels), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(input_channels), requires_grad=True)
    
    def forward(self, x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        x = self._global_norm(x, epsilon)

        return x

    def _standardize_input(self, x: torch.Tensor, axes: list[int], epsilon: float):
        mean = x.mean(dim=axes, keepdim=True)
        var = torch.var(x, dim=axes, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + epsilon)

    def _global_norm(self, x, epsilon: float):
        axes_to_standardize = list(range(1, len(x.shape)))
        return self._standardize_input(x, axes_to_standardize, epsilon)
