# Group-wise Hadamard rotation for INT8 quantization quality improvement
# From sd-webui-forge-classic-2.24/backend/quant_rotation.py

import torch

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

CONVROT_GROUP_SIZE = 256


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    # Bug fix: math.log(size, 4) uses floating-point and can be imprecise
    # (e.g. math.log(64, 4) may return 2.9999... instead of 3.0)
    # Use integer bit arithmetic: a power of 4 is a power of 2 with an even exponent.
    if size < 4 or (size & (size - 1)) != 0 or (size.bit_length() - 1) % 2 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")

    H4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype, device=device,
    )

    H = H4
    current_size = 4
    while current_size < size:
        H = torch.kron(H, H4)
        current_size *= 4

    H_normalized = H / (size ** 0.5)
    _HADAMARD_CACHE[cache_key] = H_normalized
    return H_normalized


def rotate_weight(weight: torch.Tensor, H: torch.Tensor, group_size: int) -> torch.Tensor:
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by {group_size}")
    n_groups = in_f // group_size
    W_grouped = weight.view(out_f, n_groups, group_size)
    H_t = H.T.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(W_grouped, H_t).reshape(out_f, in_f)


def rotate_activation(x: torch.Tensor, H: torch.Tensor, group_size: int) -> torch.Tensor:
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by {group_size}")
    n_groups = features // group_size
    x_grouped = x.view(*orig_shape[:-1], n_groups, group_size)
    H_dev = H.to(dtype=x.dtype, device=x.device)
    return torch.matmul(x_grouped, H_dev).reshape(orig_shape)
