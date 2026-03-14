import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
#%%
# 导入拆分出的自定义模块
from dataset import FASChannelDataset, sample_complex_gaussian_channels
from model import Simple1DUNet
from diffusion import DiffusionScheduler, reconstruct_from_partial_observation
from utils import make_port_selection_mask, observe_partial_ports, nmse_db

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