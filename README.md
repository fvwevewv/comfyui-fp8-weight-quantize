# ComfyUI FP8 Weight Quantization & Mixed-Precision Loaders

为 ComfyUI 开发的权重量化与混合精度加载节点,理论上支持所有基于Unet和DiT的模型(仅量化Linear)。提供基于原生的量化(comfy.quant_ops与comfy_kitchen)、基于 BitsAndBytes 的量化,以及使用随机舍入（Stochastic Rounding）,scale,Hadamard旋转(int8)方法旨在降低模型推理时的显存开销并优化访存效率。

---

## 核心特性

- **原生混合精度加载 (Native Mixed Precision)**：基于 ComfyUI 原生 `comfy.quant_ops` 及 `comfy_kitchen` C++ / CUDA 后端，支持多精度就地量化与动态直接加载。
- **随机舍入 (Stochastic Rounding)**：针对原生 FP8/FP4 格式实现随机舍入，通过注入确定性伪随机噪声降低量化带来的精度损失，在极低精度下维持原有模型的细节还原度。
- **哈达玛旋转 (Hadamard Transform)**：针对 INT8 (W8A16) 精度实现正交变换，通过打散权重与激活值中的通道极值（Outliers），抑制量化导致的偏色与噪声，提高量化后模型的数值稳定性。
- **敏感层保护机制**：自动检测并旁路（Bypass）`adaln`、`modulation`、`norm`、`embed` 以及部分早期敏感网络层（如 Safe Blocks），避免敏感参数被过度量化导致推理异常。
- **UNet 内存排布优化**：针对标准 UNet 架构自动启用 `channels_last` (NHWC) 排布，减少访存带宽开销并提升张量计算效率。

---

## 安装方法

1. 进入 ComfyUI 的自定义节点目录：
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. 克隆本仓库：
   ```bash
   git clone https://github.com/fvwevewv/comfyui-fp8-weight-quantize.git
   ```
3. 重启 ComfyUI 即可。

> [!NOTE]
> - 原生 `nvfp4` 或 `mxfp8` 精度需要较新版本的 ComfyUI 且正确安装了 `comfy_kitchen` 。
> - 使用 `bnb-nf4` 后量化需要ComfyUI已安装 `bitsandbytes` 包。

---

### 1. 💾 FP8 Checkpoint Loader (混合精度加载器)

该节点在 Checkpoint 加载的 `state_dict` 阶段进行元数据注入与就地量化，由 ComfyUI 原生的 `MixedPrecisionOps` 接管，从而优化运行效率。

* **参数配置**：
  - `ckpt_name`：选择需要加载的 Checkpoint 或 Diffusion 模型。
  - `dtype` (精度选项)：
    - `float8_e4m3fn`：原生 FP8 E4M3 精度。
      - **Ada Lovelace (RTX 40系) 及以上显卡**：提供完整硬件级 FP8 加速计算支持。
      - **Ampere (RTX 30系) 显卡**：**不支持硬件级 FP8 加速计算，但支持 FP8 格式的存储和读取（在推理阶段会自动转换为 FP16/BF16 进行计算，可有效节省显存但无法获得加速效益）**。
      - **Turing (RTX 20系) 及以前显卡**：不支持此精度。
    - `nvfp4`：ComfyUI 原生 Block-wise 4-bit 浮点格式，无需安装 bitsandbytes。
      - **Blackwell (RTX 50系) 及以上显卡**：支持原生 nvfp4 硬件加速。
      - **更早架构显卡**：在不支持的旧硬件上运行会退化为软件模拟反量化与计算，会降低生成图像质量并可能影响速度，非必要不建议在不支持的硬件上选择此项。
    - `mxfp8`：ComfyUI 原生 Block-wise 8-bit 浮点格式，具备较好的数值抗噪性能。
    - `bnb-nf4`：基于 BitsAndBytes 的 NF4 4-bit 格式。载入后自动就地量化，适用于所有 CUDA 设备以压缩显存。
  - `stochastic_rounding` (`True` / `False`)：是否启用随机舍入。开启后将在 GPU 上进行随机噪声注入，改善低比特（FP8/FP4）量化下的暗部细节与画质。
  - `skip_sensitive` (`True` / `False`)：敏感层保护。开启后自动保持核心骨架层为原始精度。
  - `allow_compile` (`False` / `True`)：是否启用 `torch.compile` 兼容模式。

---

### 2. INT8 Weight Quantize (INT8 权重量化器)

针对 **INT8 (W8A16)** 精度设计的后处理量化节点，可串联在任何模型加载器后，对模型进行动态量化改写。

* **参数配置**：
  - `model`：接入已加载的 MODEL。
  - `backend` (推理后端)：
    - `PyTorch (torch._int_mm)`：使用 PyTorch 原生的 INT8 矩阵乘法，兼容所有主流 CUDA 环境，无需安装额外库。
    - `Triton`：使用 Triton 实现的自定义 GEMM 内核进行 INT8 乘法，推理速度更快，但需要comfyui安装 `triton` 并仅支持 NVIDIA GPU。
  - `skip_sensitive` (`True` / `False`)：保护敏感层不被量化。
  - `allow_compile` (`False` / `True`)：启用 `torch.compile` 兼容模式。
* **技术原理（哈达玛旋转）**：
  - **Hadamard 旋转保护**：在将权重量化为 INT8 之前，通过对权重施加哈达玛矩阵正交变换，将通道中集中的激活值极值（Outliers）均匀分布到各个维度，从而在数缓解 W8A16 量化固有的数值溢出及偏色、底噪问题。

---

## 性能与实测数据

以下为实际生成测试数据：

**测试配置**：测试分辨率 `1024 * 1536`，挂载 **3 个 LoRA** 并应用 **SageAttention** 优化补丁。

### SDXL (UNet 架构)
- **精度模式**：`float8_e4m3fn``int8+triton`
- **推理速度**：从 **`1.48 it/s`** 提升至 **`1.99 it/s(fp8)1.76(int8+triton)`**（推理性能最高提升约 **34.5%**）
- **推理显存 (VRAM)**：**`5.3 GB`**

### Anima (DiT 架构)
- **精度模式**：`float8_e4m3fn``int8+triton`
- **推理速度**：从 **`2.0 s/it`** 缩短至 **`1.68 s/it(fp8)1.47s/it(int8+triton)`**（推理耗时最大减少约 **16%**）
- **推理显存 (VRAM)**：**`5.2 GB`**
---

## 硬件与环境推荐

| 条件 | 推荐配置 | 说明 |
| :--- | :--- | :--- |
| **PyTorch 版本** | `>= 2.1` | 原生 FP8 与 Triton 算子需要高版本 PyTorch。 |
| **CUDA 版本** | `>= 11.8` | 确保随机舍入与 Triton 编译内核正常运行。 |
| **NVIDIA 显卡 (FP8)** | `Ada Lovelace (RTX 40系)` 或更新 | 原生 FP8 硬件加速需要 Compute Capability >= 8.9；**Ampere (RTX 30系) 仅支持 FP8 存储与加载（运行时会转换为 FP16/BF16 计算），无硬件加速。** |
| **NVIDIA 显卡 (nvfp4)** | `Blackwell (RTX 50系)` 或更新 | 原生 nvfp4 硬件加速仅 Blackwell 架构支持；旧架构显卡运行会退化为软件模拟反量化，从而显著降低生成图像的质量。 |
| **bitsandbytes** | 仅使用 `bnb-nf4` 时需要 | 传统 4-bit 量化加速，支持主流 CUDA 设备。 |

---

## 许可

本插件遵循 **GPL-3.0 license** 开源许可。部分实现与算法设计参考了 ComfyUI 官方、BitsAndBytes 项目与 Webui Forge。
