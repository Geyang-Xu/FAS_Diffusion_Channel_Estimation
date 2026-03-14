import math
import torch
from torch.utils.data import Dataset

def build_covariance_matrix(num_ports: int, corr_len: float = 4.0, device="cpu"):
    """
    构造位置域相关矩阵 R_ij = exp(-(i-j)^2 / (2*corr_len^2))
    """
    idx = torch.arange(num_ports, dtype=torch.float32, device=device)
    dist2 = (idx[:, None] - idx[None, :]) ** 2
    R = torch.exp(-dist2 / (2.0 * corr_len ** 2))
    return R

def sample_complex_gaussian_channels(num_samples: int, num_ports: int, corr_len: float = 4.0, device="cpu"):
    """
    采样复高斯信道：生成 h ~ CN(0, R)
    返回:
        h_complex: [num_samples, num_ports], complex64
        h_real:    [num_samples, 2, num_ports], float32
    """
    R = build_covariance_matrix(num_ports, corr_len=corr_len, device=device)
    L = torch.linalg.cholesky(R + 1e-6 * torch.eye(num_ports, device=device))

    # Re(h), Im(h) independently ~ N(0, R/2)
    zr = torch.randn(num_samples, num_ports, device=device)
    zi = torch.randn(num_samples, num_ports, device=device)

    hr = (zr @ L.T) / math.sqrt(2.0)
    hi = (zi @ L.T) / math.sqrt(2.0)
    h_complex = hr + 1j * hi

    # [B, 2, N]
    h_real = torch.stack([hr, hi], dim=1)
    return h_complex, h_real

class FASChannelDataset(Dataset):
    def __init__(self, num_samples=20000, num_ports=64, corr_len=5.0, device="cpu"):
        _, h_real = sample_complex_gaussian_channels(
            num_samples=num_samples,
            num_ports=num_ports,
            corr_len=corr_len,
            device=device
        )
        self.data = h_real.cpu()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]