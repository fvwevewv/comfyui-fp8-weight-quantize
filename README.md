# ComfyUI FP8 Weight Quantize

将 SDXL UNet 的 `Linear` / `Conv2d` 层权重以 **fp8** 格式存储在显存中，
推理时自动 upcast 回 fp16/bf16 计算，**显存节省约 40~50%，质量几乎无损**。

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/fvwevewv/comfyui-fp8-weight-quantize.git
```

重启 ComfyUI 即可。

---

## 节点说明

### FP8 Weight Quantize（显存优化）

| 参数 | 选项 | 说明 |
|---|---|---|
| `model` | — | 接入 Load Checkpoint 的 MODEL 输出 |
| `dtype` | `float8_e4m3fn` / `float8_e5m2` | 推荐 e4m3fn，精度更好 |
| `target` | `Linear only` / `Linear + Conv2d` | 量化范围 |
| `skip_norm_layers` | `True` / `False` | 跳过 LayerNorm/GroupNorm 层（推荐 True） |
| `torch_compile` | `True` / `False` | 将 cast+matmul 融合为单个 CUDA kernel |
| `channels_last` | `True` / `False` | 将 Conv2d 转为 NHWC 布局，加速 SDXL Conv 层 |

**推荐工作流：**
```
Load Checkpoint → FP8 Weight Quantize → KSampler
```

---

## 环境要求

| 条件 | 要求 |
|---|---|
| PyTorch | >= 2.1 |
| 显卡 | NVIDIA Ampere (RTX 30系) 及以上 |
| CUDA | >= 11.8 |

---

## 技术原理

1. 将 Linear/Conv2d 层的 weight 以 fp8 存储（`module._fp8_weight`，非 `nn.Parameter`）
2. 绕开 ComfyUI 参数管理系统（不搬运 fp8 Parameter）
3. `module.weight` 替换为 1 元素 dummy Parameter 保持兼容
4. 首次 forward 将 `_fp8_weight` 搬到 GPU 并原地更新
5. 每次 forward 临时 upcast → 计算 → 释放，不缓存
6. `channels_last` 优化 Conv2d 内存布局（SDXL 推荐）
