import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#%%
# ============================================================
# 1. 数据集：窄带 FAS 相关复信道
# ============================================================

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


# ============================================================
# 2. 时间编码
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B] integer or float
        return: [B, dim]
        """
        half_dim = self.dim // 2
        device = t.device
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


# ============================================================
# 3. 简单 1D U-Net 风格噪声预测网络
# ============================================================

class ResBlock1D(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Simple1DUNet(nn.Module):
    def __init__(self, in_ch=2, base_ch=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv1d(in_ch, base_ch, kernel_size=3, padding=1)
        self.block1 = ResBlock1D(base_ch, time_dim)
        self.block2 = ResBlock1D(base_ch, time_dim)
        self.block3 = ResBlock1D(base_ch, time_dim)
        self.out_norm = nn.GroupNorm(4, base_ch)
        self.out_conv = nn.Conv1d(base_ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        h = self.block1(h, t_emb)
        h = self.block2(h, t_emb)
        h = self.block3(h, t_emb)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# ============================================================
# 4. DDPM 调度器
# ============================================================

class DiffusionScheduler:
    def __init__(self, T=200, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device)

    def q_sample(self, x0, t, noise=None):
        """
        x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps
        x0: [B, C, N]
        t:  [B]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        ab_t = self.alpha_bars[t].view(-1, 1, 1)
        xt = torch.sqrt(ab_t) * x0 + torch.sqrt(1.0 - ab_t) * noise
        return xt, noise

    def predict_x0(self, xt, t, eps_pred):
        ab_t = self.alpha_bars[t].view(-1, 1, 1)
        return (xt - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(ab_t)

    def p_mean_variance(self, model, xt, t):
        """
        标准 DDPM 反向均值
        """
        eps_pred = model(xt, t)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        ab_t = self.alpha_bars[t].view(-1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            xt - ((1.0 - alpha_t) / torch.sqrt(1.0 - ab_t)) * eps_pred
        )
        var = beta_t
        return mean, var, eps_pred


# ============================================================
# 5. 训练函数
# ============================================================

def train_ddpm(
    model,
    scheduler,
    dataloader,
    epochs=20,
    lr=1e-3,
    device="cpu"
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        total_num = 0

        for x0 in dataloader:
            x0 = x0.to(device)  # [B, 2, N]
            bsz = x0.shape[0]
            t = scheduler.sample_timesteps(bsz)
            xt, noise = scheduler.q_sample(x0, t)

            noise_pred = model(xt, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bsz
            total_num += bsz

        print(f"Epoch {epoch+1:03d} | Loss = {total_loss / total_num:.6f}")


# ============================================================
# 6. 观测算子：FAS 少量端口采样
# ============================================================

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


# ============================================================
# 7. 条件后验采样：从部分端口恢复全孔径 CSI
# ============================================================

@torch.no_grad()
def reconstruct_from_partial_observation(
    model,
    scheduler,
    y_obs,
    mask,
    noise_std=0.05,
    lambda_scale=0.5,
    device="cpu"
):
    """
    y_obs: [B, 2, N]，仅观测位置有值
    mask:  [1, 1, N]，观测位置为1
    """
    model.eval()
    B, C, N = y_obs.shape
    xt = torch.randn(B, C, N, device=device)

    sigma_y2 = noise_std ** 2
    mask = mask.to(device)

    for step in reversed(range(scheduler.T)):
        t = torch.full((B,), step, device=device, dtype=torch.long)

        mean, var, eps_pred = scheduler.p_mean_variance(model, xt, t)
        x0_hat = scheduler.predict_x0(xt, t, eps_pred)

        # 线性观测: y = A x0 + w
        # 这里 A 对应端口选择，因此 A^T(y - A x0_hat) == mask * (y_obs - mask*x0_hat)
        residual = y_obs - mask * x0_hat
        likelihood_grad = (mask * residual) / sigma_y2

        # 可选：加一个随 t 衰减的步长
        ab_t = scheduler.alpha_bars[t].view(-1, 1, 1)
        lam_t = lambda_scale * torch.sqrt(ab_t)

        corrected_mean = mean + lam_t * likelihood_grad

        if step > 0:
            z = torch.randn_like(xt)
            xt = corrected_mean + torch.sqrt(var) * z
        else:
            xt = corrected_mean

    # 最后再做一次 x0 估计
    t0 = torch.zeros(B, device=device, dtype=torch.long)
    eps_pred = model(xt, t0)
    x0_est = scheduler.predict_x0(xt, t0, eps_pred)

    return x0_est


# ============================================================
# 8. 评价指标
# ============================================================

def nmse_db(x_hat, x_true):
    """
    x_hat, x_true: [B, 2, N]
    """
    num = ((x_hat - x_true) ** 2).sum(dim=(1, 2))
    den = (x_true ** 2).sum(dim=(1, 2)) + 1e-12
    nmse = (num / den).mean().item()
    return 10.0 * math.log10(nmse + 1e-12)


# ============================================================
# 9. 主程序
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # 超参数
    # ----------------------------
    num_ports = 64
    corr_len = 5.0
    train_samples = 20000
    test_samples = 200
    batch_size = 128

    T = 200
    epochs = 20
    lr = 1e-3

    noise_std = 0.05
    observed_ratio = 0.25  # 只观测 25% 端口

    # ----------------------------
    # 数据
    # ----------------------------
    train_set = FASChannelDataset(
        num_samples=train_samples,
        num_ports=num_ports,
        corr_len=corr_len,
        device=device
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    _, test_x = sample_complex_gaussian_channels(
        num_samples=test_samples,
        num_ports=num_ports,
        corr_len=corr_len,
        device=device
    )

    # ----------------------------
    # 模型
    # ----------------------------
    model = Simple1DUNet(in_ch=2, base_ch=64, time_dim=128).to(device)
    scheduler = DiffusionScheduler(T=T, device=device)

    # ----------------------------
    # 训练
    # ----------------------------
    train_ddpm(
        model=model,
        scheduler=scheduler,
        dataloader=train_loader,
        epochs=epochs,
        lr=lr,
        device=device
    )

    # ----------------------------
    # 构造部分端口观测
    # ----------------------------
    num_obs = int(num_ports * observed_ratio)
    observed_idx = np.sort(np.random.choice(num_ports, size=num_obs, replace=False))
    mask = make_port_selection_mask(num_ports, observed_idx).to(device)

    test_x = test_x.to(device)
    y_obs = observe_partial_ports(test_x, mask, noise_std=noise_std)

    # baseline：直接把未观测位置置0
    x_zero_fill = y_obs.clone()
    zero_fill_nmse = nmse_db(x_zero_fill, test_x)

    # 扩散重构
    x_est = reconstruct_from_partial_observation(
        model=model,
        scheduler=scheduler,
        y_obs=y_obs,
        mask=mask,
        noise_std=noise_std,
        lambda_scale=0.2,
        device=device
    )
    diff_nmse = nmse_db(x_est, test_x)

    print("=" * 60)
    print(f"Observed ratio      : {observed_ratio:.2f}")
    print(f"Zero-fill NMSE (dB) : {zero_fill_nmse:.3f}")
    print(f"Diffusion NMSE (dB) : {diff_nmse:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
#%%