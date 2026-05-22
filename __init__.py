"""
ComfyUI FP8 Weight Quantization Custom Nodes
将模型权重以 fp8 格式存储，大幅减少显存占用且质量几乎无损。
"""

from .fp8_quantize_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
