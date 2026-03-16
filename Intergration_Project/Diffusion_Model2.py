import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. 随机种子
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. 数据生成：相关复高斯信道
# ============================================================

def build_spatial_correlation(num_ports: int, corr_len: float, device="cpu"):
    idx = torch.arange(num_ports, device=device, dtype=torch.float32)
    dist2 = (idx[:, None] - idx[None, :]) ** 2
    R = torch.exp(-dist2 / (2.0 * corr_len ** 2))
    return R


def sample_complex_gaussian_channels(
    num_samples: int,
    num_ports: int,
    corr_len: float,
    device="cpu"
):
    """
    生成相关复高斯信道:
        h ~ CN(0, R)
    返回:
        h_complex: [B, N] complex
        x_realimag: [B, 2, N] float
    """
    R = build_spatial_correlation(num_ports, corr_len, device=device)
    eps = 1e-6 * torch.eye(num_ports, device=device)
    L = torch.linalg.cholesky(R + eps)

    z_real = torch.randn(num_samples, num_ports, device=device)
    z_imag = torch.randn(num_samples, num_ports, device=device)

    h_real = z_real @ L.T / math.sqrt(2.0)
    h_imag = z_imag @ L.T / math.sqrt(2.0)

    h_complex = torch.complex(h_real, h_imag)
    x_realimag = torch.stack([h_real, h_imag], dim=1)  # [B, 2, N]
    return h_complex, x_realimag


class FASChannelDataset(Dataset):
    def __init__(self, num_samples, num_ports, corr_len, device="cpu"):
        super().__init__()
        _, self.x = sample_complex_gaussian_channels(
            num_samples=num_samples,
            num_ports=num_ports,
            corr_len=corr_len,
            device=device
        )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


# ============================================================
# 2. 时间步嵌入
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B]
        return: [B, dim]
        """
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ============================================================
# 3. 1D ResBlock / 去噪网络
# ============================================================

class ResBlock1D(nn.Module):
    def __init__(self, ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, ch)
        )

        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x, t_emb):
        """
        x: [B, C, N]
        t_emb: [B, time_dim]
        """
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Simple1DUNet(nn.Module):
    def __init__(self, in_ch=2, base_ch=96, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv1d(in_ch, base_ch, kernel_size=3, padding=1)

        self.block1 = ResBlock1D(base_ch, time_dim)
        self.block2 = ResBlock1D(base_ch, time_dim)
        self.block3 = ResBlock1D(base_ch, time_dim)
        self.block4 = ResBlock1D(base_ch, time_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv1d(base_ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        x: [B, 2, N]
        t: [B]
        """
        t_emb = self.time_embed(t)
        h = self.in_conv(x)
        h = self.block1(h, t_emb)
        h = self.block2(h, t_emb)
        h = self.block3(h, t_emb)
        h = self.block4(h, t_emb)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# ============================================================
# 4. Diffusion Scheduler
# ============================================================

class DiffusionScheduler:
    def __init__(self, T=200, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        alpha_bars_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_bars[:-1]], dim=0
        )
        self.alpha_bars_prev = alpha_bars_prev

        # posterior variance: q(x_{t-1} | x_t, x_0)
        self.posterior_var = self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)
        self.posterior_var = torch.clamp(self.posterior_var, min=1e-8)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device).long()

    def q_sample(self, x0, t, noise=None):
        """
        x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) eps
        """
        if noise is None:
            noise = torch.randn_like(x0)

        ab_t = self.alpha_bars[t].view(-1, 1, 1)
        xt = torch.sqrt(ab_t) * x0 + torch.sqrt(1.0 - ab_t) * noise
        return xt, noise

    def predict_x0(self, xt, t, eps_pred):
        ab_t = self.alpha_bars[t].view(-1, 1, 1)
        x0_hat = (xt - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(torch.clamp(ab_t, min=1e-8))
        return x0_hat

    def p_mean_variance(self, model, xt, t):
        """
        用更标准的 posterior mean / variance:
            q(x_{t-1} | x_t, x_0_hat)
        """
        eps_pred = model(xt, t)
        x0_hat = self.predict_x0(xt, t, eps_pred)

        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        ab_t = self.alpha_bars[t].view(-1, 1, 1)
        ab_prev_t = self.alpha_bars_prev[t].view(-1, 1, 1)
        var = self.posterior_var[t].view(-1, 1, 1)

        coef1 = torch.sqrt(ab_prev_t) * beta_t / torch.clamp(1.0 - ab_t, min=1e-8)
        coef2 = torch.sqrt(alpha_t) * (1.0 - ab_prev_t) / torch.clamp(1.0 - ab_t, min=1e-8)

        mean = coef1 * x0_hat + coef2 * xt
        return mean, var, eps_pred


# ============================================================
# 5. EMA
# ============================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if k in msd:
                if msd[k].dtype.is_floating_point:
                    v.copy_(self.decay * v + (1.0 - self.decay) * msd[k])
                else:
                    v.copy_(msd[k])

    def get_model(self):
        return self.shadow


# ============================================================
# 6. 训练函数
# ============================================================

def train_ddpm(
    model,
    scheduler,
    dataloader,
    epochs=30,
    lr=5e-4,
    grad_clip=1.0,
    device="cpu"
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = EMA(model, decay=0.999)

    for epoch in range(epochs):
        model.train()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update(model)

            total_loss += loss.item() * bsz
            total_num += bsz

        lr_scheduler.step()
        print(f"Epoch {epoch+1:03d} | Loss = {total_loss / total_num:.6f}")

    return ema.get_model()


# ============================================================
# 7. 观测算子：FAS 少量端口采样
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
# 8. 条件后验采样：从部分端口恢复全孔径 CSI
# ============================================================

@torch.no_grad()
def reconstruct_from_partial_observation(
    model,
    scheduler,
    y_obs,
    mask,
    noise_std=0.05,
    lambda_scale=0.02,
    grad_clip=10.0,
    x0_clip=4.0,
    xt_clip=8.0,
    device="cpu"
):
    """
    y_obs: [B, 2, N]，仅观测位置有值
    mask:  [1, 1, N]，观测位置为1
    """
    model.eval()

    y_obs = y_obs.to(device)
    mask = mask.to(device)

    B, C, N = y_obs.shape
    xt = torch.randn(B, C, N, device=device)

    sigma_y2 = max(noise_std ** 2, 1e-8)
    obs_ratio = float(mask.mean().item()) if mask.mean().item() > 0 else 1.0

    for step in reversed(range(scheduler.T)):
        t = torch.full((B,), step, device=device, dtype=torch.long)

        # 标准反向一步
        mean, var, eps_pred = scheduler.p_mean_variance(model, xt, t)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=xt_clip, neginf=-xt_clip)
        var = torch.nan_to_num(var, nan=1e-8, posinf=1.0, neginf=1e-8)
        var = torch.clamp(var, min=1e-8)

        # 估计 x0
        x0_hat = scheduler.predict_x0(xt, t, eps_pred)
        x0_hat = torch.nan_to_num(x0_hat, nan=0.0, posinf=x0_clip, neginf=-x0_clip)
        x0_hat = torch.clamp(x0_hat, -x0_clip, x0_clip)

        # 观测一致性项
        residual = y_obs - mask * x0_hat
        residual = torch.nan_to_num(residual, nan=0.0, posinf=xt_clip, neginf=-xt_clip)

        likelihood_grad = (mask * residual) / sigma_y2
        likelihood_grad = likelihood_grad / obs_ratio
        likelihood_grad = torch.nan_to_num(
            likelihood_grad, nan=0.0, posinf=grad_clip, neginf=-grad_clip
        )
        likelihood_grad = torch.clamp(likelihood_grad, -grad_clip, grad_clip)

        # 时间步调度
        ab_t = scheduler.alpha_bars[t].view(-1, 1, 1)
        lam_t = lambda_scale * torch.sqrt(torch.clamp(ab_t, min=1e-8))

        corrected_mean = mean + lam_t * likelihood_grad
        corrected_mean = torch.nan_to_num(
            corrected_mean, nan=0.0, posinf=xt_clip, neginf=-xt_clip
        )
        corrected_mean = torch.clamp(corrected_mean, -xt_clip, xt_clip)

        if step > 0:
            z = torch.randn_like(xt)
            xt = corrected_mean + torch.sqrt(var) * z
            xt = torch.nan_to_num(xt, nan=0.0, posinf=xt_clip, neginf=-xt_clip)
            xt = torch.clamp(xt, -xt_clip, xt_clip)
        else:
            xt = corrected_mean

    # 最后一次 x0 估计
    t0 = torch.zeros(B, device=device, dtype=torch.long)
    eps_pred = model(xt, t0)
    x0_est = scheduler.predict_x0(xt, t0, eps_pred)
    x0_est = torch.nan_to_num(x0_est, nan=0.0, posinf=x0_clip, neginf=-x0_clip)
    x0_est = torch.clamp(x0_est, -x0_clip, x0_clip)

    return x0_est


# ============================================================
# 9. 评价指标
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
# 10. 主程序
# ============================================================

def main():
    set_seed(42)
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
    epochs = 30
    lr = 5e-4

    noise_std = 0.05
    observed_ratio = 0.25  # 只观测 25% 端口

    # posterior correction 参数
    lambda_scale = 0.02
    post_grad_clip = 10.0
    x0_clip = 4.0
    xt_clip = 8.0

    # ----------------------------
    # 数据
    # ----------------------------
    train_set = FASChannelDataset(
        num_samples=train_samples,
        num_ports=num_ports,
        corr_len=corr_len,
        device=device
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    _, test_x = sample_complex_gaussian_channels(
        num_samples=test_samples,
        num_ports=num_ports,
        corr_len=corr_len,
        device=device
    )

    # ----------------------------
    # 模型
    # ----------------------------
    model = Simple1DUNet(in_ch=2, base_ch=96, time_dim=128).to(device)
    scheduler = DiffusionScheduler(T=T, device=device)

    # ----------------------------
    # 训练
    # ----------------------------
    ema_model = train_ddpm(
        model=model,
        scheduler=scheduler,
        dataloader=train_loader,
        epochs=epochs,
        lr=lr,
        grad_clip=1.0,
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

    # 扩散重构（用 EMA 模型）
    x_est = reconstruct_from_partial_observation(
        model=ema_model,
        scheduler=scheduler,
        y_obs=y_obs,
        mask=mask,
        noise_std=noise_std,
        lambda_scale=lambda_scale,
        grad_clip=post_grad_clip,
        x0_clip=x0_clip,
        xt_clip=xt_clip,
        device=device
    )
    diff_nmse = nmse_db(x_est, test_x)

    print("x_est has nan:", torch.isnan(x_est).any().item())
    print("x_est has inf:", torch.isinf(x_est).any().item())
    print("x_est abs max:", x_est.abs().max().item())

    print("=" * 60)
    print(f"Observed ratio      : {observed_ratio:.2f}")
    print(f"Zero-fill NMSE (dB) : {zero_fill_nmse:.3f}")
    print(f"Diffusion NMSE (dB) : {diff_nmse:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
