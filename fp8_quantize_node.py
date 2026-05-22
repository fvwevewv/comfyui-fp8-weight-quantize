"""
ComfyUI Weight Quantization Node

将 UNet/DiT 的 Linear 层权重量化为低精度格式存储，
推理时自动 upcast 计算，大幅降低显存占用。

支持精度：float8_e4m3fn / int8 / nvfp4 / mxfp8 / bnb-nf4
兼容所有使用 nn.Linear 的模型架构。
可通过外部 torch.compile 节点进一步加速。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from comfy.model_patcher import ModelPatcher
from folder_paths import get_filename_list
import comfy.float
import comfy.model_management

# ── 优化版 GPU 加速随机舍入与量化内核 (Monkey Patch) ──

def optimized_calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    # 使用 torch.exp2 避免极其缓慢的 CPU/GPU 幂运算 (2.0 ** tensor)
    two_pow = torch.exp2(exponent - EXPONENT_BIAS)
    const_scalar = 2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)
    
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / two_pow - 1.0) * (2**MANTISSA_BITS),
        abs_x / const_scalar
    )

    # 使用 rand_like 避免显式传入形状和设备参数所带来的 CPU 额外开销
    noise = torch.rand_like(mantissa_scaled, generator=generator)
    mantissa_scaled += noise
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)


def optimized_manual_stochastic_round_to_float8(x, dtype, generator=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    # 指数计算
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    normal_mask = ~(exponent == 0)

    # 在 GPU 上完全向量化进行随机尾数舍入
    mantissa = optimized_calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    two_pow = torch.exp2(exponent - EXPONENT_BIAS)
    const_val = 2.0 ** (-EXPONENT_BIAS + 1)
    
    sign *= torch.where(
        normal_mask,
        two_pow * (1.0 + mantissa),
        const_val * mantissa
    )

    inf = torch.finfo(dtype)
    torch.clamp(sign, min=inf.min, max=inf.max, out=sign)
    return sign


def optimized_stochastic_float_to_fp4_e2m1(x, generator):
    orig_shape = x.shape
    sign = torch.signbit(x).to(torch.uint8)

    # 优化指数逻辑，使用 torch.exp2
    exp = torch.floor(torch.log2(x.abs()) + 1.0).clamp(0, 3)
    noise = torch.rand_like(x, generator=generator)
    x += (noise - 0.5) * torch.exp2(exp - 2.0) * 1.25

    x = x.abs()
    exp = torch.floor(torch.log2(x) + 1.1925).clamp(0, 3)

    # 优化尾数逻辑，使用 torch.exp2
    mantissa = torch.where(
        exp > 0,
        (x / torch.exp2(exp - 1.0) - 1.0) * 2.0,
        (x * 2.0),
        out=x
    ).round().to(torch.uint8)
    del x

    exp = exp.to(torch.uint8)

    fp4 = (sign << 3) | (exp << 1) | mantissa
    del sign, exp, mantissa

    fp4_flat = fp4.view(-1)
    
    # 极速连续内存位打包 (重塑为 -1, 2，规避 GPU 上高开销的非连续跨步切片 fp4_flat[0::2] 与 fp4_flat[1::2])
    fp4_reshaped = fp4_flat.view(-1, 2)
    packed = (fp4_reshaped[:, 0] << 4) | fp4_reshaped[:, 1]
    
    return packed.reshape(list(orig_shape)[:-1] + [-1])


# 动态注入 ComfyUI 全局随机舍入内核 API
comfy.float.calc_mantissa = optimized_calc_mantissa
comfy.float.manual_stochastic_round_to_float8 = optimized_manual_stochastic_round_to_float8
comfy.float.stochastic_float_to_fp4_e2m1 = optimized_stochastic_float_to_fp4_e2m1

print("[WeightQuantize] Highly optimized GPU-accelerated stochastic rounding kernels injected successfully.")

# ── 依赖检测 ──
FP8_E4M3_AVAILABLE = hasattr(torch, "float8_e4m3fn")

try:
    from .operations_triton import (
        _TRITON_AVAILABLE,
        triton_quantize_rowwise,
        triton_int8_linear,
        triton_int8_linear_per_row,
    )
except ImportError:
    _TRITON_AVAILABLE = False

try:
    import bitsandbytes as bnb
    _BNB_IMPORTED = True
except (ImportError, Exception) as e:
    _BNB_IMPORTED = False
    _BNB_IMPORT_ERROR = f"import bitsandbytes 失败: {e}"
    _BNB_AVAILABLE = False

if _BNB_IMPORTED:
    try:
        from bitsandbytes.functional import dequantize_4bit as _bnb_dequantize_4bit
        from bitsandbytes.functional import quantize_4bit as _bnb_quantize_4bit
        from bitsandbytes.nn.modules import Params4bit
        _BNB_AVAILABLE = True
        _BNB_IMPORT_ERROR = None
    except ImportError as _bnb_import_err:
        _BNB_AVAILABLE = False
        _BNB_IMPORT_ERROR = f"ImportError: {_bnb_import_err}\n(Tip: bitsandbytes 版本可能太新，API 已变更)"
    except Exception as _bnb_exc:
        _BNB_AVAILABLE = False
        _BNB_IMPORT_ERROR = f"{type(_bnb_exc).__name__}: {_bnb_exc}"

# ── 精度映射 ──
_DTYPE_MAP = {}
if FP8_E4M3_AVAILABLE:
    _DTYPE_MAP["float8_e4m3fn"] = torch.float8_e4m3fn



_INT8_BACKENDS = [
    "PyTorch (torch._int_mm, 兼容最好)",
    "Triton (自定义GEMM kernel, 更快但需triton)",
]
_INT8_TOOLTIPS = {
    _INT8_BACKENDS[0]: "使用 PyTorch 原生 INT8 矩阵乘法，兼容所有 CUDA 环境，无需额外依赖。",
    _INT8_BACKENDS[1]: "使用 Triton GEMM kernel 加速 INT8 推理。速度更快，但需要安装 triton 包且仅 NVIDIA GPU 可用。",
}

# INPUT_TYPES 中含中文括号的 key 名，提取为常量以便 apply() 精确匹配
_KEY_INT8_BACKEND = "int8_backend（仅 dtype=int8 时生效）"
_KEY_ALLOW_COMPILE = "allow_compile"


try:
    from .quant_rotation import build_hadamard, rotate_activation, rotate_weight, CONVROT_GROUP_SIZE
    _HADAMARD_AVAILABLE = True
except ImportError:
    _HADAMARD_AVAILABLE = False


# ── FP8 Per-Tensor Quantize ──

if not _BNB_AVAILABLE and _BNB_IMPORT_ERROR:
    print(f"[WeightQuantize] bitsandbytes 不可用: {_BNB_IMPORT_ERROR}")

# FP8 格式的最大可表示值，用 dtype 对象为 key（避免 str(dtype) 在不同 PyTorch 版本格式不一致导致 lookup 失败）
_FP8_DTYPE_MAX: dict = {}
if FP8_E4M3_AVAILABLE:
    _FP8_DTYPE_MAX[torch.float8_e4m3fn] = 448.0


def _fp8_quantize_weight(weight_data: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """amax/fp8_max scale，与预量化模型行为一致。
    FP8 e4m3fn 最大可表示值不同：e4m3fn=448。
    """
    fp8_max = _FP8_DTYPE_MAX.get(fp8_dtype, 448.0)  # key 是 dtype 对象，不依赖字符串格式
    w_f = weight_data.float()
    amax = w_f.abs().max()
    scale = (amax / fp8_max).clamp(min=1e-30)
    w_q = (w_f / scale).to(fp8_dtype)
    return w_q.cpu(), torch.tensor(scale.item(), device="cpu", dtype=torch.float32)


# ── INT8 工具 ──

def _int8_quantize_axiswise(w: torch.Tensor, dim: int, seed: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    abs_max = w.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q_f = w.float().div(scale)
    if seed >= 0:
        gen = torch.Generator(device=q_f.device)
        gen.manual_seed(abs(hash(str(seed) + str(w.shape))) % (2**31))
        noise = torch.rand(q_f.shape, device=q_f.device, dtype=torch.float32, generator=gen)
        q = (q_f + noise).floor_().clamp_(-128, 127).to(torch.int8)
    else:
        q = q_f.round_().clamp_(-128, 127).to(torch.int8)
    return q, scale


def _int8_dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


def _int8_forward_dynamic_per_row(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    compute_dtype: torch.dtype,
    module: nn.Module = None,
) -> torch.Tensor:
    # ── Weight-Only INT8 Quantization (W8A16) ──
    # 直接在 GPU 上反量化权重到输入浮点类型，并调用高效的 F.linear
    # 避免了动态激活值量化、对齐填充、以及转置缓存导致的显存残留与 OOM Paging
    w_fp = w.to(device=x.device, dtype=compute_dtype) * w_scale.to(device=x.device, dtype=compute_dtype)
    b = bias.to(device=x.device, dtype=compute_dtype) if bias is not None else None
    return F.linear(x.to(compute_dtype), w_fp, b)


# ── FP8 forward patch ──

def _make_fp8_linear_forward(module: nn.Linear):
    def forward(x: torch.Tensor) -> torch.Tensor:
        wf = getattr(module, 'weight_function', [])
        bf = getattr(module, 'bias_function', [])
        if len(wf) > 0 or len(bf) > 0:
            # Bug fix: use getattr in loops to avoid AttributeError when only one function list is set
            w = module.weight.to(device=x.device, dtype=module._forge_orig_dtype) * module._forge_weight_scale.to(device=x.device)
            for f in wf:
                w = f(w)
            b = module.bias
            if b is not None:
                b = b.to(device=x.device, dtype=w.dtype)
                for ff in bf:
                    b = ff(b)
            return F.linear(x, w, b)

        w = module.weight
        if w.device != x.device:
            module._parameters["weight"] = torch.nn.Parameter(w.to(device=x.device), requires_grad=False)
            w = module.weight

        w = w.to(dtype=module._forge_orig_dtype) * module._forge_weight_scale.to(w.device)

        b = module.bias
        if b is not None:
            b = b.to(device=w.device, dtype=w.dtype)
        return F.linear(x, w, b)
    return forward



# ── INT8 forward patch ──

def _make_int8_linear_forward(module: nn.Linear, use_triton: bool):
    def forward(x: torch.Tensor) -> torch.Tensor:
        wf = getattr(module, 'weight_function', [])
        bf = getattr(module, 'bias_function', [])
        if len(wf) > 0 or len(bf) > 0:
            # Dequantize INT8 → same dtype as input (bf16/fp16)，避免 float32 中间张量
            # module.weight 即 INT8 权重，无需单独的 _forge_quant_weight
            w = module.weight.to(device=x.device, dtype=x.dtype) * module._forge_weight_scale.to(device=x.device, dtype=x.dtype)

            if getattr(module, "_use_hadamard", False):
                # Note: module.weight (INT8) stores the ALREADY-ROTATED weight (W @ H^T).
                # For LoRA, we need the ORIGINAL unrotated W.
                # Since H is symmetric and H@H = I, applying rotate_weight twice = identity.
                # So rotate_weight(dequant, H) = (W @ H^T) @ H^T = W @ I = W  ← unrotates!
                gs = module._hadamard_group_size
                H = build_hadamard(gs, device=w.device, dtype=w.dtype)
                w = rotate_weight(w, H, gs)  # unrotate back to original W

            # Apply LoRA on the (unrotated) original-space weight
            for f in wf:
                w = f(w)

            if getattr(module, "_use_hadamard", False):
                # Re-rotate W_new and also rotate x, so the math is consistent
                H_x = build_hadamard(gs, device=x.device, dtype=x.dtype)
                x = rotate_activation(x, H_x, gs)
                H_w = build_hadamard(gs, device=w.device, dtype=w.dtype)
                w = rotate_weight(w, H_w, gs)  # re-rotate: (W + ΔLoRA) @ H^T

            # w 已经是 x.dtype，无需再转换
            # Bug fix: also handle bias LoRA (was completely missing before)
            b = module.bias
            if b is not None:
                b = b.to(device=x.device, dtype=x.dtype)
                for ff in bf:
                    b = ff(b)
            else:
                b = None
            return F.linear(x, w, b)

        # module.weight IS the INT8 weight (moved to GPU by ComfyUI when needed)
        w = module.weight
        if w.device != x.device:
            # 低显存模式下可能没有提前搬运，自行搞到目标设备
            module._parameters["weight"] = torch.nn.Parameter(w.to(device=x.device), requires_grad=False)
            w = module.weight
        scale = module._forge_weight_scale
        if isinstance(scale, torch.Tensor) and scale.device != x.device:
            module._forge_weight_scale = scale.to(device=x.device)
            scale = module._forge_weight_scale

        if getattr(module, "_use_hadamard", False):
            gs = module._hadamard_group_size
            H = build_hadamard(gs, device=x.device, dtype=x.dtype)
            x = rotate_activation(x, H, gs)

        compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

        if use_triton and _TRITON_AVAILABLE and x.is_cuda:
            b = module.bias.to(device=x.device) if module.bias is not None else None
            return triton_int8_linear_per_row(x, w, scale, b, compute_dtype)
        b = module.bias.to(device=x.device, dtype=compute_dtype) if module.bias is not None else None
        return _int8_forward_dynamic_per_row(x, w, scale, b, compute_dtype, module)
    return forward



# ── BNB (nf4) forward patch ──

if _BNB_AVAILABLE:

    def _make_bnb_linear_forward(module):
        def forward(x: torch.Tensor) -> torch.Tensor:
            w = _bnb_dequantize_4bit(
                module._forge_quant_weight,
                quant_state=module._forge_weight_quant_state,
            )
            w = w.to(device=x.device, dtype=x.dtype)

            # Apply LoRA delta if patched in-place on the dummy weight
            if hasattr(module, "weight") and module.weight is not None:
                w = w + module.weight.to(device=x.device, dtype=x.dtype)

            for f in getattr(module, 'weight_function', []):
                w = f(w)

            b = module.bias
            if b is not None:
                b = b.to(device=x.device, dtype=x.dtype)
                for ff in getattr(module, 'bias_function', []):
                    b = ff(b)
            return F.linear(x, w, b)
        return forward


# ── 量化主逻辑 ──

def _normalize_key(key: str) -> str:
    for prefix in ("diffusion_model.", "model.diffusion_model.", "model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _make_dummy_weight(weight_data: torch.Tensor) -> torch.nn.Parameter:
    # Quality fix: requires_grad must be passed to Parameter, NOT to zeros_like.
    # Passing it to zeros_like only affects the plain tensor; Parameter.__init__
    # defaults to requires_grad=True and would silently re-enable grad tracking.
    return torch.nn.Parameter(
        torch.zeros_like(weight_data),
        requires_grad=False,
    )


# ── _forge_* 属性说明 ──
# 这些属性由本节点动态注入到 nn.Linear 实例上，用于存储量化状态：
#   _node_owned        : bool  — 标记此模块由本节点量化，重复执行时可覆盖
#   _forge_layout_type : str   — FP8 布局名称 ("TensorCoreFP8E4M3Layout" 等)
#   _forge_orig_dtype  : dtype — 量化前的原始权重 dtype（FP8 和 INT8 均使用）
#   _forge_weight_scale: Tensor — 反量化用 scale，FP8=标量，INT8=per-row (N,1)
#   module.weight      : 对于 INT8 路径，直接存储 INT8 量化权重（1 字节/参数）
#   _forge_quant_weight: Tensor — 仅用于 BNB 4-bit 路径（INT8 不再使用）
#   _forge_weight_quant_state: object — BNB 4-bit quant_state
#   _use_hadamard      : bool  — 是否对该层启用了 Hadamard 旋转
#   _hadamard_group_size: int  — Hadamard 分组大小


def _restore_quantized_weight(module: nn.Module) -> "torch.Tensor | None":
    """尝试从各量化格式恢复原始浮点权重。

    返回恢复后的 Tensor，或 None（表示该模块未被量化）。
    副作用：清除已识别的量化状态属性。
    """
    # BNB 4-bit（自定义 _forge 存储路径）
    if _BNB_AVAILABLE and hasattr(module, "_forge_weight_quant_state"):
        qs = getattr(module, "_forge_weight_quant_state", None)
        if qs is not None and hasattr(module, "_forge_quant_weight"):
            w = _bnb_dequantize_4bit(module._forge_quant_weight, quant_state=qs)
            del module._forge_quant_weight
            module._forge_weight_quant_state = None
            return w

    # BNB Params4bit（原生路径）
    if _BNB_AVAILABLE and hasattr(module, "weight") and isinstance(module.weight, Params4bit):
        qs = getattr(module.weight, "quant_state", None)
        return _bnb_dequantize_4bit(module.weight, quant_state=qs)

    # FP8（forge layout 路径）
    if hasattr(module, "_forge_layout_type"):
        w = module.weight
        s = getattr(module, "_forge_weight_scale", None)
        if s is not None:
            w = w.float() * s if (s.dim() == 0 or isinstance(s, float)) else _int8_dequant(w, s)
        module._forge_weight_scale = None
        module._forge_layout_type = None
        module._forge_orig_dtype = None
        return w

    # INT8（module.weight 即 INT8 权重）
    if hasattr(module, "_forge_weight_scale") and module._forge_weight_scale is not None and \
            hasattr(module, "weight") and module.weight is not None and module.weight.dtype == torch.int8:
        orig_dtype = getattr(module, "_forge_orig_dtype", torch.bfloat16)
        w = module.weight
        s = module._forge_weight_scale
        result = _int8_dequant(w.to(s.device), s).to(orig_dtype)
        module._forge_weight_scale = None
        module._forge_orig_dtype = None
        if hasattr(module, "_forge_int8_w_t"):
            delattr(module, "_forge_int8_w_t")
        if hasattr(module, "_forge_int8_w_scale_t"):
            delattr(module, "_forge_int8_w_scale_t")
        return result

    return None


def _restore_original_weight(module: nn.Module) -> torch.Tensor:
    """恢复模块的原始浮点权重，兼容所有量化格式。

    若模块由本节点量化（_node_owned=True），先清除该标记再恢复。
    若模块未被量化，返回当前权重的 detach 副本。
    """
    if getattr(module, "_node_owned", False):
        module._node_owned = False
    w = _restore_quantized_weight(module)
    return w if w is not None else module.weight.detach()


def _quantize_module(
    module: nn.Module,
    weight_data: torch.Tensor,
    dtype: str,
    use_triton: bool,
    module_name: str = "",
):
    seed = abs(hash(module_name)) % (2**31) if module_name else -1

    if dtype in _DTYPE_MAP:
        fp8_dtype = _DTYPE_MAP[dtype]
        gpu_device = comfy.model_management.get_torch_device()
        w_q, scale = _fp8_quantize_weight(weight_data.to(device=gpu_device), fp8_dtype)
        layout_map = {
            "float8_e4m3fn": "TensorCoreFP8E4M3Layout",
        }
        module._forge_layout_type = layout_map.get(dtype, "TensorCoreFP8E4M3Layout")
        module._forge_weight_scale = scale
        module._forge_orig_dtype = weight_data.dtype
        module._parameters["weight"] = torch.nn.Parameter(w_q, requires_grad=False)
        module.forward = _make_fp8_linear_forward(module)
        module._node_owned = True

    elif dtype == "int8":
        gpu_device = comfy.model_management.get_torch_device()
        w = weight_data.to(device=gpu_device).float()
        if _HADAMARD_AVAILABLE and w.shape[1] % CONVROT_GROUP_SIZE == 0:
            try:
                H = build_hadamard(CONVROT_GROUP_SIZE, device=w.device, dtype=w.dtype)
                w = rotate_weight(w, H, CONVROT_GROUP_SIZE)
                module._use_hadamard = True
                module._hadamard_group_size = CONVROT_GROUP_SIZE
            except Exception:
                pass
        w_int8, scale = _int8_quantize_axiswise(w, dim=1, seed=seed)
        module._forge_weight_scale = scale.cpu()
        module._forge_orig_dtype = weight_data.dtype  # 用于还原时恢复原始 dtype
        # 把真实 INT8 存为 module.weight，ComfyUI model.to(device) 时只将 INT8（1 字节/参数）搞到 GPU
        # 之前的 _make_dummy_weight 会创建 bf16 零张量（2 字节/参数），导致显存浪费
        module._parameters["weight"] = torch.nn.Parameter(w_int8.cpu(), requires_grad=False)
        module.forward = _make_int8_linear_forward(module, use_triton)
        module._node_owned = True

    elif dtype == "bnb-nf4":
        if not _BNB_AVAILABLE:
            err_msg = f"[WeightQuantize] {dtype} 需要 bitsandbytes，导入失败: {_BNB_IMPORT_ERROR}"
            raise RuntimeError(err_msg)
        bnb_quant_type = "nf4"
        gpu_device = comfy.model_management.get_torch_device()
        w = weight_data.to(device=gpu_device, dtype=torch.float16)
        w_4bit, quant_state = _bnb_quantize_4bit(
            w, blocksize=64, quant_type=bnb_quant_type,
        )
        module._forge_quant_weight = w_4bit
        module._forge_weight_quant_state = quant_state
        module._parameters["weight"] = _make_dummy_weight(weight_data)
        module.forward = _make_bnb_linear_forward(module)
        module._node_owned = True


_SENSITIVE_NAME_PATTERNS = ("adaln", "modulation", "_norm", "embed", "final_layer", "llm_adapter")

_SAFE_BLOCK_START = (0, 1)  # 首两个 block 跳过量化，保持原始精度


def _is_sensitive(name: str) -> bool:
    n = name.lower()
    if any(p in n for p in _SENSITIVE_NAME_PATTERNS):
        return True
    parts = n.split(".")
    for i in range(len(parts) - 1):
        if parts[i] == "blocks" and parts[i + 1].isdigit() and int(parts[i + 1]) in _SAFE_BLOCK_START:
            return True
    return False


def _is_already_quantized(module: nn.Module) -> bool:
    if getattr(module, "_node_owned", False):
        return False
    if hasattr(module, "_forge_layout_type"):
        return True
    # BNB 路径：_forge_quant_weight 与 _forge_weight_quant_state 均存在
    if hasattr(module, "_forge_quant_weight"):
        return True
    if _BNB_AVAILABLE and hasattr(module, "_forge_weight_quant_state") and module._forge_weight_quant_state is not None:
        return True
    if _BNB_AVAILABLE and isinstance(module.weight, Params4bit):
        return True
    if _DTYPE_MAP:
        for d in _DTYPE_MAP.values():
            if module.weight.dtype == d:
                return True
    if module.weight.dtype == torch.int8:
        return True
    return False


def quantize_model(
    full_model: nn.Module,
    dtype: str,
    skip_sensitive: bool,
    use_triton: bool,
):
    stats = {
        "total_linear": 0,
        "quantized_linear": 0,
        "skipped_already_quantized": 0,
        "skipped_sensitive": 0,
    }

    for name, module in full_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        stats["total_linear"] += 1

        if skip_sensitive and _is_sensitive(name):
            stats["skipped_sensitive"] += 1
            continue

        if _is_already_quantized(module):
            stats["skipped_already_quantized"] += 1
            continue

        weight_data = _restore_original_weight(module)
        _quantize_module(module, weight_data, dtype, use_triton, module_name=name)

        stats["quantized_linear"] += 1

        del weight_data

        # 每量化 50 层主动释放一次 GPU 缓存，防止大模型量化时 OOM
        if stats["quantized_linear"] % 50 == 0:
            comfy.model_management.soft_empty_cache()

    comfy.model_management.soft_empty_cache()
    return stats


# ── ComfyUI 节点 ──

# 主节点已被安全移除，现在项目完全由两个更高能且专注的独立节点驱动：
# 1. INT8 Weight Quantize (Int8WeightQuantizeNode) - 专精 8-bit 量化
# 2. FP8 Checkpoint Loader (Fp8CheckpointLoader) - 专精原生 8-bit/4-bit 混精与随机舍入加载


class Int8WeightQuantizeNode:
    """
    独立 INT8 量化节点，专为 INT8 精度量化而设计，具有以下特性：
    - 支持 PyTorch Native 和 Triton 双底层推理路径。
    - 自动对符合条件的线性层应用 Hadamard 旋转，大幅降低异常值对量化精度的影响。
    - 自动跳过对精度敏感的模块（adaln/modulation/norm/embed/final_layer/llm_adapter 等）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (
                    [
                        "PyTorch (torch._int_mm, 兼容最好)",
                        "Triton (更快但需triton)",
                    ],
                    {"default": "PyTorch (torch._int_mm, 兼容最好)"}
                ),
                "skip_sensitive": (["True", "False"], {"default": "True"}),
                "allow_compile": (["False", "True（torch.compile 兼容模式）"], {"default": "False"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/quantization"
    DESCRIPTION = (
        "[INT8 Weight Quantize] 将模型权重量化为 INT8 格式，大幅降低显存。\n\n"
        "量化后端：\n"
        "  - PyTorch (torch._int_mm) — 原生 INT8 矩阵乘法，兼容所有 GPU，最稳定。\n"
        "  - Triton — 自定义 Triton GEMM kernel，速度极快（需要安装 triton 包且仅 NVIDIA GPU 可用）。\n\n"
        "特性：\n"
        "  - Hadamard 旋转：自动构建并应用旋转矩阵，将极值分布到各个维度，大幅提升 INT8 量化的精细度与还原度。\n"
        "  - 敏感层保护：自动跳过 adaln/modulation/norm/embed/final_layer/llm_adapter 等核心模块，兼顾速度与质量。"
    )

    def apply(self, model: ModelPatcher, backend: str, skip_sensitive: str, allow_compile: str):
        use_triton = "Triton" in backend
        skip = (skip_sensitive == "True")

        print(f"[Int8WeightQuantize] backend={backend} triton={use_triton} skip_sensitive={skip}")

        m = model.clone()
        full_model = m.model

        stats = quantize_model(
            full_model=full_model,
            dtype="int8",
            skip_sensitive=skip,
            use_triton=use_triton,
        )

        print(
            f"[Int8WeightQuantize] Linear: {stats['quantized_linear']}/{stats['total_linear']} | "
            f"跳过(敏感层): {stats['skipped_sensitive']} | "
            f"跳过(已量化): {stats['skipped_already_quantized']}"
        )

        if allow_compile.startswith("True"):
            full_model._forge_allow_compile = True

        return (m,)


def _get_stable_seed(layer_name: str) -> int:
    seed = 0
    for char in layer_name:
        seed = (seed * 31 + ord(char)) & 0xFFFFFFFF
    return max(1, seed % (2**31))


# ── FP8 Checkpoint Loader ──

class Fp8CheckpointLoader:
    """
    加载 checkpoint 并注入量化元数据，让 ComfyUI 自动识别并路由到 MixedPrecisionOps。
    此节点的量化发生在 state_dict 阶段，在模型构建前注入 .comfy_quant 元数据
    并直接按 Layout 量化权重，由 ComfyUI 原生管线接管。
    支持精度：float8_e4m3fn, nvfp4, mxfp8, bnb-nf4 并且支持随机舍入（Stochastic Rounding）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        files = get_filename_list("checkpoints") + get_filename_list("diffusion_models")
        return {
            "required": {
                "ckpt_name": (sorted(set(files)), ),
                "dtype": (["float8_e4m3fn", "nvfp4", "mxfp8", "bnb-nf4"], {"default": "float8_e4m3fn"}),
                "stochastic_rounding": (["True", "False"], {"default": "True"}),
                "skip_sensitive": (["True", "False"], {"default": "True"}),
                "allow_compile": (["False", "True（torch.compile 兼容模式）"], {"default": "False"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "[FP8 Checkpoint Loader] 加载 Checkpoint 并自动注入量化元数据。\n\n"
        "支持精度选项：\n"
        "  - float8_e4m3fn — 原生 FP8 E4M3 精度，具备原生硬件加速 (Ampere+ 架构推荐)。\n"
        "  - nvfp4 — ComfyUI 原生 Block-wise 4-bit 浮点格式 (无需 bitsandbytes)。\n"
        "  - mxfp8 — ComfyUI 原生 Block-wise 8-bit 浮点格式 (无需 bitsandbytes)。\n"
        "  - bnb-nf4 — 传统的 bitsandbytes NF4 量化 (载入后自动量化，极限压缩显存，适用于所有 CUDA)。\n\n"
        "特性：\n"
        "  - 随机舍入 (Stochastic Rounding)：对原生 FP8/FP4 格式有效，基于层名称生成确定性种子，大幅提升生成画质与细节。\n"
        "  - 敏感层保护：自动跳过 adaln/modulation/norm/embed/final_layer 等，兼顾速度与质量。\n"
        "  - 兼容编译：可通过 allow_compile 开启对外部 torch.compile 的支持。"
    )

    def load_checkpoint(self, ckpt_name: str, dtype: str, stochastic_rounding: str, skip_sensitive: str, allow_compile: str = "False"):
        import comfy.utils
        import comfy.model_detection
        import comfy.sd
        import folder_paths

        is_bnb = (dtype == "bnb-nf4")

        # dynamic import & environment validation
        if not is_bnb:
            try:
                from comfy.quant_ops import get_layout_class, QUANT_ALGOS
            except ImportError:
                raise RuntimeError(
                    "[Fp8CheckpointLoader] comfy.quant_ops could not be imported.\n"
                    "This node requires a newer ComfyUI version with Native Mixed Precision support."
                )

            if dtype not in QUANT_ALGOS:
                raise RuntimeError(f"[Fp8CheckpointLoader] Quantization format '{dtype}' is not defined in ComfyUI's QUANT_ALGOS.")

            layout_name = QUANT_ALGOS[dtype]["comfy_tensor_layout"]
            layout_cls = get_layout_class(layout_name)
            if layout_cls is None:
                raise RuntimeError(
                    f"[Fp8CheckpointLoader] The layout '{layout_name}' for precision '{dtype}' is not available in your environment.\n"
                    "Please make sure comfy_kitchen C++ / CUDA extension is installed and properly configured."
                )

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if ckpt_path is None:
            ckpt_path = folder_paths.get_full_path("diffusion_models", ckpt_name)
        if ckpt_path is None:
            raise FileNotFoundError(f"[Fp8CheckpointLoader] Model not found: {ckpt_name}")

        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
        if metadata is None:
            metadata = {}
        prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        skip = (skip_sensitive == "True")
        stochastic_enabled = (stochastic_rounding == "True")

        device = comfy.model_management.get_torch_device()

        if is_bnb:
            print(f"[Fp8CheckpointLoader] Loading model in default format before BNB quantization...")
        else:
            print(f"[Fp8CheckpointLoader] Quantization processing device: {device}")

            stats = {"total_linear": 0, "quantized": 0, "skipped_sensitive": 0}
            quant_layers = {}
            original_shapes = {}

            for k in list(sd.keys()):
                if not k.startswith(prefix) or not k.endswith(".weight"):
                    continue
                w = sd[k]
                if w.ndim != 2:
                    continue

                stats["total_linear"] += 1

                layer_name = k[len(prefix):-len(".weight")]
                if skip and _is_sensitive(layer_name):
                    stats["skipped_sensitive"] += 1
                    continue

                w_device = w.to(device)
                w_f = w_device.float()
                seed = _get_stable_seed(layer_name) if stochastic_enabled else 0

                # Quantize using Layout class on device (GPU)
                if dtype == "float8_e4m3fn":
                    fp8_dtype = QUANT_ALGOS[dtype]["storage_t"]
                    fmax = torch.finfo(fp8_dtype).max
                    scale = (w_f.abs().amax() / fmax).clamp(min=1e-30)
                    
                    qdata, params = layout_cls.quantize(w_f, scale=scale, stochastic_rounding=seed)
                    sd[k] = qdata.cpu()
                    sd[k[:-len(".weight")] + ".weight_scale"] = scale.cpu().to(dtype=torch.float32)

                elif dtype == "nvfp4":
                    original_shapes[k] = w.shape
                    # F8_E4M3_MAX * F4_E2M1_MAX = 448.0 * 6.0 = 2688.0
                    scale = (w_f.abs().amax() / 2688.0).clamp(min=1e-30)
                    
                    qdata, params = layout_cls.quantize(w_f, scale=scale, stochastic_rounding=seed)
                    sd[k] = qdata.cpu()
                    sd[k[:-len(".weight")] + ".weight_scale_2"] = scale.cpu().to(dtype=torch.float32)
                    sd[k[:-len(".weight")] + ".weight_scale"] = params.block_scale.cpu().to(dtype=torch.float8_e4m3fn)

                elif dtype == "mxfp8":
                    qdata, params = layout_cls.quantize(w_f, stochastic_rounding=seed)
                    sd[k] = qdata.cpu()
                    sd[k[:-len(".weight")] + ".weight_scale"] = params.scale.cpu().view(torch.uint8)

                quant_layers[k[:-len(".weight")]] = {"format": dtype}
                stats["quantized"] += 1
                
                # Explicitly garbage collect GPU local variables to free memory
                del w_device, w_f, qdata, params

                if stats["quantized"] % 50 == 0:
                    comfy.model_management.soft_empty_cache()

            comfy.model_management.soft_empty_cache()

            metadata["_quantization_metadata"] = json.dumps({
                "format_version": "1.0",
                "layers": quant_layers,
            })

            print(
                f"[Fp8CheckpointLoader] Precision: {dtype} | Linear: {stats['quantized']}/{stats['total_linear']} | "
                f"跳过(敏感层): {stats['skipped_sensitive']}"
            )

        # Temporarily patch model_config_from_unet during load_state_dict_guess_config
        # only if we have collected original shapes for nvfp4.
        _orig_model_config_from_unet = getattr(comfy.model_detection, "model_config_from_unet", None)
        if _orig_model_config_from_unet is not None and not is_bnb and original_shapes:
            def patched_model_config_from_unet(state_dict, *args, **kwargs):
                restored = {}
                for k, orig_shape in original_shapes.items():
                    if k in state_dict:
                        restored[k] = state_dict[k]
                        state_dict[k] = torch.empty(orig_shape, dtype=torch.float16, device="cpu")
                try:
                    return _orig_model_config_from_unet(state_dict, *args, **kwargs)
                finally:
                    for k, v in restored.items():
                        state_dict[k] = v
            comfy.model_detection.model_config_from_unet = patched_model_config_from_unet

        try:
            out = comfy.sd.load_state_dict_guess_config(
                sd,
                output_vae=False,
                output_clip=False,
                output_clipvision=False,
                embedding_directory=None,
                output_model=True,
                metadata=metadata,
            )
        finally:
            if _orig_model_config_from_unet is not None and not is_bnb and original_shapes:
                comfy.model_detection.model_config_from_unet = _orig_model_config_from_unet

        if out is None or out[0] is None:
            raise RuntimeError(f"[Fp8CheckpointLoader] Failed to load model from {ckpt_name}")

        model_patcher = out[0]

        # Implicitly detect standard UNet model and enable channels last memory format
        is_unet = False
        if hasattr(model_patcher, "model") and hasattr(model_patcher.model, "diffusion_model"):
            try:
                from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
                is_unet = isinstance(model_patcher.model.diffusion_model, UNetModel)
            except Exception:
                pass
            if not is_unet:
                is_unet = model_patcher.model.diffusion_model.__class__.__name__ == "UNetModel"

        if is_unet:
            print("[Fp8CheckpointLoader] Detected standard UNet model, implicitly enabling channels_last memory format for diffusion model.")
            model_patcher.model.diffusion_model.to(memory_format=torch.channels_last)

        # Post-load BNB quantization if requested
        if is_bnb:
            print(f"[Fp8CheckpointLoader] Applying BNB {dtype} quantization to loaded model...")
            stats = quantize_model(
                full_model=model_patcher.model,
                dtype=dtype,
                skip_sensitive=skip,
                use_triton=False,
            )
            print(
                f"[Fp8CheckpointLoader] BNB Quantization finished | "
                f"Linear: {stats['quantized_linear']}/{stats['total_linear']} | "
                f"跳过(敏感层): {stats['skipped_sensitive']} | "
                f"跳过(已量化): {stats['skipped_already_quantized']}"
            )

        if allow_compile.startswith("True") and hasattr(model_patcher, "model"):
            model_patcher.model._forge_allow_compile = True

        model_patcher.cached_patcher_init = (
            self.load_checkpoint,
            (ckpt_name, dtype, stochastic_rounding, skip_sensitive, allow_compile),
        )

        return (model_patcher,)


# ── 注册 ──

NODE_CLASS_MAPPINGS = {
    "Fp8CheckpointLoader": Fp8CheckpointLoader,
    "Int8WeightQuantize": Int8WeightQuantizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fp8CheckpointLoader": "FP8 Checkpoint Loader (MixedPrecision)",
    "Int8WeightQuantize": "INT8 Weight Quantize",
}
