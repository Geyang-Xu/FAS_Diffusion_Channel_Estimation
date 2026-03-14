import math
import torch

def make_port_selection_mask(num_ports: int, observed_idx):
    """
    observed_idx: list of selected port indices
    return mask: [1, 1, N] with 1 on observed ports
    """
    mask = torch.zeros(1, 1, num_ports, dtype=torch.float32)
    mask[0, 0, observed_idx] = 1.0
    return mask

def observe_partial_ports(x0, mask, noise_std=0.05):
    """
    x0: [B, 2, N]
    mask: [1, 1, N]
    输出 y_obs 仍放在 [B, 2, N] 上，未观测位置置零，便于计算
    """
    y = mask * x0 + noise_std * torch.randn_like(x0) * mask
    return y

def nmse_db(x_hat, x_true):
    """
    x_hat, x_true: [B, 2, N]
    """
    num = ((x_hat - x_true) ** 2).sum(dim=(1, 2))
    den = (x_true ** 2).sum(dim=(1, 2)) + 1e-12
    nmse = (num / den).mean().item()
    return 10.0 * math.log10(nmse + 1e-12)