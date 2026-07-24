"""
Version 0: Kiến trúc gốc Cosmos 3 Nano Baseline (Dense Backbone 8B / 16B Scale)
Khởi tạo cấu trúc khung xương (Architecture Shell) dùng Meta Device Init (0 MB CPU RAM)
để đo VRAM, Latency, Throughput (fps) trực tiếp trên Kaggle Dual T4 GPUs mà không cần load trọng số gốc.
"""

from .model import Cosmos3ToyModel, Cosmos3Config

__all__ = ["Cosmos3ToyModel", "Cosmos3Config"]
