"""
Script Đánh Giá Lõi Dense Base Gốc (Version 8 QK-Norm / Version 5 Base 4.03B)
trên Tập Dữ Liệu Thử Nghiệm Đa Phương Tiện (Pilot Dataset Mixture)
"""

import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from mini_model.version8.model import Cosmos3ToyModel as V8Model, Cosmos3Config as V8Config
from mini_model.version5.model import Cosmos3ToyModel as V5Model, Cosmos3Config as V5Config
from dataset_loader import PilotDatasetLoader


def evaluate_dense_base(version: str = "version8", num_steps: int = 10, batch_size: int = 2, use_fp16: bool = True):
    print("=" * 70)
    print(f"BẮT ĐẦU ĐÁNH GIÁ LÕI DENSE BASE [{version.upper()}] TRÊN PILOT DATASET")
    print("=" * 70)

    # 1. Chọn phiên bản mô hình
    if version.lower() == "version8":
        config = V8Config()
        model_cls = V8Model
        print("[INFO] Models: Version 8 (QK-Norm + LayerScale 4.03B Dense Base)")
    else:
        config = V5Config()
        model_cls = V5Model
        print("[INFO] Models: Version 5 (Base FP16 4.03B Dense Base)")

    num_gpus = torch.cuda.device_count()
    print(f"[INFO] Pytorch CUDA Available: {torch.cuda.is_available()} | GPU Count: {num_gpus}")

    # 2. Khởi tạo mô hình dùng Meta Device Init (0 MB CPU RAM)
    model = model_cls.create_meta_model(config, fp16=use_fp16)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total Dense Base Parameters: {total_params / 1e6:.2f} M ({total_params / 1e9:.2f} B)")

    # 3. Khởi tạo DataLoader & Optimizer
    loader = PilotDatasetLoader(vocab_size=config.vocab_size, latent_dim=config.latent_dim, action_dim=config.action_dim)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    ar_criterion = nn.CrossEntropyLoss()
    dm_criterion = nn.MSELoss()

    dev0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float32

    print("\n[INFO] Dang chay vong lap danh gia Pilot 4-Modality Dataset...")
    
    total_time_ms = 0.0
    ar_losses = []
    dm_losses = []
    cos_sims = []

    for step in range(1, num_steps + 1):
        step_start = time.time()

        # Lay batch gia lap tu dataset loader
        batch = loader.get_pilot_batch(
            batch_size=batch_size,
            seq_len_ar=32,
            seq_len_dm=16,
            device=dev0,
            dtype=dtype
        )

        ar_tokens = batch["ar_tokens"]
        dm_latent = batch["dm_latent"]
        action_vectors = batch["action_vectors"]

        ar_targets = torch.randint(0, config.vocab_size, ar_tokens.shape, device=dev0)
        dm_targets = torch.randn_like(dm_latent)

        optimizer.zero_grad()
        outputs = model(
            ar_tokens=ar_tokens,
            dm_latent=dm_latent,
            action_vectors=action_vectors,
            mode="both"
        )

        ar_logits = outputs["ar_logits"].float()
        dm_pred = outputs["dm_predicted_latent"].float()

        ar_loss = ar_criterion(ar_logits.view(-1, config.vocab_size), ar_targets.view(-1))
        dm_loss = dm_criterion(dm_pred, dm_targets.float())
        
        # Cosine Similarity cho DM latent
        cos_sim = torch.cosine_similarity(dm_pred.view(-1), dm_targets.float().view(-1), dim=0).item()

        total_loss = ar_loss + dm_loss
        total_loss.backward()
        optimizer.step()

        elapsed_ms = (time.time() - step_start) * 1000
        total_time_ms += elapsed_ms

        ar_losses.append(ar_loss.item())
        dm_losses.append(dm_loss.item())
        cos_sims.append(cos_sim)

        print(f" Step [{step:02d}/{num_steps:02d}] | Latency: {elapsed_ms:.2f} ms | AR Loss: {ar_loss.item():.4f} | DM Loss: {dm_loss.item():.4f} | Cos Sim: {cos_sim:.4f}")

    avg_latency = total_time_ms / num_steps
    avg_throughput = (batch_size * 1000.0) / avg_latency
    avg_ar_loss = sum(ar_losses) / num_steps
    avg_dm_loss = sum(dm_losses) / num_steps
    avg_cos_sim = sum(cos_sims) / num_steps

    print("\n" + "=" * 70)
    print(" KẾT QUẢ ĐÁNH GIÁ DENSE BASE TRÊN PILOT DATASET")
    print("=" * 70)
    print(f"  • Model Version         : {version.upper()}")
    print(f"  • Total Parameters      : {total_params / 1e9:.2f} B")
    print(f"  • Average Latency       : {avg_latency:.2f} ms")
    print(f"  • Average Throughput    : {avg_throughput:.2f} fps")
    print(f"  • Average AR Loss       : {avg_ar_loss:.4f}")
    print(f"  • Average DM MSE Loss   : {avg_dm_loss:.4f}")
    print(f"  • Average Cosine Sim    : {avg_cos_sim:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dense Base on Pilot Dataset")
    parser.add_argument("--version", type=str, default="version8", choices=["version5", "version8"], help="Dense Base Version")
    parser.add_argument("--steps", type=int, default=10, help="Number of pilot evaluation steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision")

    args = parser.parse_args()
    evaluate_dense_base(version=args.version, num_steps=args.steps, batch_size=args.batch_size, use_fp16=args.fp16)
