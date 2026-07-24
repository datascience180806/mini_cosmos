"""
Version 9: Unified Mixture-of-Experts (MoE) World Model Architecture
Xây dựng trên Lõi Version 8 (QK-Norm + LayerScale 4.03B FP16)
kết hợp Mạng Điều Phối MoE Router (Top-2 Experts out of 4 Experts) và Auxiliary Load Balancing Loss.
"""

from .model import Cosmos3ToyModel, Cosmos3Config

__all__ = ["Cosmos3ToyModel", "Cosmos3Config"]
