# FAS_Diffusion_Channel_Estimation
Diffusion-based channel estimation for narrowband fluid antenna systems with partial port observations.
# 1D Diffusion Model for FAS Channel Estimation

本项目实现了一个基于 1D 扩散模型（Diffusion Model, DDPM）的流体天线系统（FAS）信道估计与重构算法。通过训练一个 1D U-Net 网络，模型能够从仅有的部分观测端口数据中，高精度地恢复出完整的全孔径信道状态信息（CSI）。

## 📂 项目结构

项目采用模块化设计，主要包含以下文件：

- `dataset.py`：数据处理模块。负责构建位置域相关矩阵，并采样生成用于训练和测试的复高斯信道数据。
- `model.py`：神经网络模块。定义了基于 1D 卷积和正弦时间编码（Sinusoidal Time Embedding）的 U-Net 架构（包含残差块 `ResBlock1D`）。
- `diffusion.py`：扩散模型核心模块。包含 DDPM 的加噪调度器（Scheduler）以及用于条件后验采样的重构算法。
- `utils.py`：工具与评估模块。包含观测算子（生成随机采样掩码）以及 NMSE（归一化均方误差）计算函数。
- `main.py`：程序主入口。负责整合数据、模型与调度器，执行训练循环并在测试集上对比“补零（Zero-fill）”与“扩散重构（Diffusion）”的性能。
- `requirements.txt`：项目基础依赖项。

## ⚙️ 环境配置

本项目依赖于基础的深度学习计算库。为了充分发挥 RTX 4070 的 Tensor Core 硬件加速性能，强烈推荐使用支持 CUDA 12.1 的 PyTorch 版本。

**1. 安装基础依赖：**
```bash
pip install -r requirements.txt