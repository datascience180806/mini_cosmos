"""
Cosmos 3 Toy Model Package (Version 7)
Ultra-large scale Dual GPU MoT architecture (~14.2 Billion parameters) targeting 2x T4 GPUs (32GB VRAM total).
"""

from .model import Cosmos3ToyModel, Cosmos3Config

__all__ = ["Cosmos3ToyModel", "Cosmos3Config"]
