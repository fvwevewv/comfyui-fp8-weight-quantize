"""
ComfyUI FP8 Weight Quantization Node

将 UNet 的 Linear/Conv2d 层权重以 fp8 格式存储在显存中，
推理时自动 upcast 回 fp16/bf16 计算，大幅降低显存占用且质量几乎无损。

核心设计决策：
  真实 fp8 权重存为 module._fp8_weight（普通 Python 属性），绕开 ComfyUI
  的参数管理系统（ComfyUI 的 model.to(device) 无法正确搬运 fp8 Parameter）。
  module.weight 替换为一个 1 元素 dummy Parameter（几乎不占显存），
  确保 ComfyUI 其他代码访问 module.weight 时不报 AttributeError。

  首次 forward 将 _fp8_weight 搬到 GPU（fp8 格式，省显存），
  之后每次 forward 临时 upcast → 计算 → 释放，不缓存 upcast，
  保住 fp8 省下的 ~50% 显存。fp8 转 fp16 是纯 GPU 内部操作，开销极小。

要求：
  - PyTorch >= 2.1
  - NVIDIA Ampere (RTX 30系) 或更新显卡（老卡可运行但速度慢）
"""

import torch
import torch.nn.functional as F
from comfy.model_patcher import ModelPatcher

# ─────────────────────────────────────────────
# 运行时检查
# ─────────────────────────────────────────────
FP8_E4M3_AVAILABLE = hasattr(torch, "float8_e4m3fn")
FP8_E5M2_AVAILABLE = hasattr(torch, "float8_e5m2")
COMPILE_AVAILABLE = hasattr(torch, "compile")

_DTYPE_MAP = {}
if FP8_E4M3_AVAILABLE:
    _DTYPE_MAP["float8_e4m3fn (推荐，精度更好)"] = torch.float8_e4m3fn
if FP8_E5M2_AVAILABLE:
    _DTYPE_MAP["float8_e5m2 (范围更大)"] = torch.float8_e5m2


def _get_cuda_capability():
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()


def _check_fp8_support():
    if not _DTYPE_MAP:
        raise RuntimeError(
            "[FP8 Quantize] 当前 PyTorch 不支持 fp8，请升级到 PyTorch >= 2.1"
        )


# ─────────────────────────────────────────────
# 核心 forward patch
#
# 关键设计：
#   权重存为 module._fp8_weight（普通属性，非 nn.Parameter），
#   绕开 ComfyUI 参数管理系统（不搬运 fp8 Parameter）。
#
#   首次 forward 时将 _fp8_weight 搬到 GPU 并原地更新，
#   后续只做 fp8→fp16 临时 upcast（GPU 内部，用完释放，不缓存）。
#   不缓存 upcast 结果以保住 fp8 省下的显存。
#
#   module.bias 每次实时读取，不闭包捕获，避免过期引用。
# ─────────────────────────────────────────────

def _make_linear_forward(module: torch.nn.Linear):

    def forward(x: torch.Tensor) -> torch.Tensor:
        fp8_weight = module._fp8_weight
        if fp8_weight.device != x.device:
            module._fp8_weight = fp8_weight.to(device=x.device)
            fp8_weight = module._fp8_weight

        w = fp8_weight.to(dtype=x.dtype)

        b = module.bias
        if b is not None:
            b = b.to(device=w.device, dtype=w.dtype)

        return F.linear(x, w, b)

    return forward


def _make_conv2d_forward(module: torch.nn.Conv2d):

    def forward(x: torch.Tensor) -> torch.Tensor:
        fp8_weight = module._fp8_weight
        if fp8_weight.device != x.device:
            module._fp8_weight = fp8_weight.to(device=x.device)
            fp8_weight = module._fp8_weight

        w = fp8_weight.to(dtype=x.dtype)

        b = module.bias
        if b is not None:
            b = b.to(device=w.device, dtype=w.dtype)

        return F.conv2d(
            x, w, b,
            module.stride, module.padding, module.dilation, module.groups,
        )

    return forward


# ─────────────────────────────────────────────
# 量化主逻辑
# ─────────────────────────────────────────────

def quantize_model_to_fp8(
    unet: torch.nn.Module,
    fp8_dtype: torch.dtype,
    include_conv: bool = False,
    skip_norm_layers: bool = True,
    use_compile: bool = False,
    channels_last: bool = False,
):
    stats = {
        "quantized_linear": 0,
        "quantized_conv": 0,
        "skipped_norm": 0,
        "compiled": False,
    }

    NORM_TYPES = (
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
    )

    norm_weight_ids: set = set()
    if skip_norm_layers:
        for mod in unet.modules():
            if isinstance(mod, NORM_TYPES) and mod.weight is not None:
                norm_weight_ids.add(id(mod.weight))

    for name, module in unet.named_modules():

        if isinstance(module, torch.nn.Linear):
            if id(module.weight) in norm_weight_ids:
                stats["skipped_norm"] += 1
                continue

            weight_data = module.weight.detach()

            if weight_data.dtype == fp8_dtype:
                continue

            module._fp8_weight = weight_data.to(fp8_dtype)
            module.forward = _make_linear_forward(module)

            module._parameters["weight"] = torch.nn.Parameter(
                torch.zeros(1, device=weight_data.device, dtype=weight_data.dtype),
                requires_grad=False,
            )

            del weight_data
            stats["quantized_linear"] += 1

        elif include_conv and isinstance(module, torch.nn.Conv2d):
            if id(module.weight) in norm_weight_ids:
                stats["skipped_norm"] += 1
                continue

            weight_data = module.weight.detach()

            if weight_data.dtype == fp8_dtype:
                continue

            module._fp8_weight = weight_data.to(fp8_dtype)

            module._parameters["weight"] = torch.nn.Parameter(
                torch.zeros(1, device=weight_data.device, dtype=weight_data.dtype),
                requires_grad=False,
            )

            module.forward = _make_conv2d_forward(module)
            del weight_data
            stats["quantized_conv"] += 1

    if channels_last:
        unet.to(memory_format=torch.channels_last)

    # 量化完成后强制释放被替换掉的旧 fp16 权重显存
    torch.cuda.empty_cache()

    compiled_unet = None
    if use_compile and COMPILE_AVAILABLE:
        print("[FP8 Quantize] torch.compile mode=default（首次生图额外编译时间）...")
        try:
            compiled_unet = torch.compile(unet, mode="default", fullgraph=False)
            stats["compiled"] = True
        except Exception as e:
            print(f"[FP8 Quantize] torch.compile 失败，跳过：{e}")

    return stats, compiled_unet


# ─────────────────────────────────────────────
# ComfyUI 节点：应用 fp8 量化
# ─────────────────────────────────────────────

class Fp8WeightQuantizeNode:

    @classmethod
    def INPUT_TYPES(cls):
        dtype_choices = list(_DTYPE_MAP.keys()) if _DTYPE_MAP else ["[PyTorch < 2.1，不支持 fp8]"]
        compile_choices = ["False", "True（首次生图额外编译时间）"] if COMPILE_AVAILABLE else ["False"]

        return {
            "required": {
                "model": ("MODEL",),
                "dtype": (dtype_choices,),
                "target": (
                    ["Linear only", "Linear + Conv2d"],
                    {"default": "Linear only"},
                ),
                "skip_norm_layers": (
                    ["False", "True"],
                    {"default": "True"},
                ),
                "torch_compile": (
                    compile_choices,
                    {"default": "False"},
                ),
                "channels_last": (
                    ["False", "True（加速 SDXL Conv 层）"],
                    {"default": "False"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/quantization"
    DESCRIPTION = (
        "以 fp8 格式存储 UNet 权重，减少约 40~50% 显存占用，质量几乎无损。\n"
        "需要 PyTorch >= 2.1 | 推荐 RTX 30系及以上显卡。"
    )

    def apply(self, model: ModelPatcher, dtype: str, target: str,
              skip_norm_layers: str, torch_compile: str,
              channels_last: str):
        _check_fp8_support()

        fp8_dtype = _DTYPE_MAP[dtype]
        include_conv = "Conv2d" in target
        skip_norm = "True" in skip_norm_layers
        use_compile = "True" in torch_compile
        use_channels_last = "True" in channels_last

        cc = _get_cuda_capability()
        print(
            f"[FP8 Quantize] GPU sm{cc[0]}.{cc[1]} | "
            f"dtype={dtype} | conv={include_conv} | compile={use_compile}"
        )

        m = model.clone()
        unet = m.model.diffusion_model

        stats, compiled_unet = quantize_model_to_fp8(
            unet, fp8_dtype, include_conv, skip_norm,
            use_compile=use_compile,
            channels_last=use_channels_last,
        )

        if compiled_unet is not None:
            m.model.diffusion_model = compiled_unet

        print(
            f"[FP8 Quantize] 完成：\n"
            f"  Linear 量化 : {stats['quantized_linear']} 层\n"
            f"  Conv2d 量化 : {stats['quantized_conv']} 层\n"
            f"  跳过 Norm   : {stats['skipped_norm']} 层\n"
            f"  torch.compile: {'已启用' if stats['compiled'] else '未启用'}"
        )

        return (m,)


# ─────────────────────────────────────────────
# ComfyUI 节点：显示显存信息
# ─────────────────────────────────────────────

class VramInfoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    CATEGORY = "model_patches/quantization"
    DESCRIPTION = "显示当前 GPU 显存占用（MB），方便对比量化前后效果。"

    def get_info(self):
        if not torch.cuda.is_available():
            return ("CUDA 不可用",)

        lines = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 2
            reserved = torch.cuda.memory_reserved(i) / 1024 ** 2
            total = props.total_memory / 1024 ** 2
            lines.append(
                f"GPU {i} [{props.name}]\n"
                f"  已分配：{allocated:.0f} MB\n"
                f"  已预留：{reserved:.0f} MB\n"
                f"  总显存：{total:.0f} MB\n"
                f"  空闲  ：{total - reserved:.0f} MB"
            )

        info = "\n".join(lines)
        print(f"[VRAM Info]\n{info}")
        return (info,)


# ─────────────────────────────────────────────
# 注册
# ─────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Fp8WeightQuantize": Fp8WeightQuantizeNode,
    "VramInfo": VramInfoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fp8WeightQuantize": "FP8 Weight Quantize (VRAM Saver)",
    "VramInfo": "VRAM Info",
}
