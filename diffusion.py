import torch

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