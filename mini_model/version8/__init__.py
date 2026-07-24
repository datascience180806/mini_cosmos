"""
Version 8: Biến thể kiến trúc ~4.03B Params (Qwen2.5 / DeepSeek-style Optimization)
Đồng quy mô với Version 5 (4.03B) để so sánh trực tiếp:
- QK-Norm (RMSNorm trên Query & Key) giúp ổn định chú ý FP16
- LayerScale (Tỷ lệ kết nối tắt Residual có trọng số tự học gamma)
- GQA Attention (24 Query Heads / 6 KV Heads) + RoPE Position
- RMSNorm + SwiGLU MLP Block
"""

from .model import Cosmos3ToyModel, Cosmos3Config

__all__ = ["Cosmos3ToyModel", "Cosmos3Config"]
