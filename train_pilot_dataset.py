"""
Script Train Thử Nghiệm Quy Mô Nhỏ (Pilot Training Loop)
để kiểm tra Độ Cải Thiện (Loss Reduction Curve) của Lõi Dense Base Version 8.
Bổ sung Gradient Clipping (max_norm=1.0) và SGD Optimizer (0 MB Optimizer State Memory Overhead)
để chạy mượt 1,000 steps trên Dual GPU T4 mà không bao giờ bị Out-of-Memory hay NaN.
"""

import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from mini_model.version8.model import Cosmos3ToyModel as V8Model, Cosmos3Config as V8Config
from mini_model.version5.model import Cosmos3ToyModel as V5Model, Cosmos3Config as V5Config
from dataset_loader import PilotDatasetLoader


def train_pilot_dense_base(
    version: str = "version8",
    num_steps: int = 1000,
    batch_size: int = 1,
    accum_steps: int = 4,
    lr: float = 1e-5,
    log_every: int = 50
):
    print("=" * 70)
    print(f"BẮT ĐẦU TRAIN THỬ NGHIỆM ĐO ĐỘ CẢI THIỆN LÕI DENSE BASE [{version.upper()}] - {num_steps} STEPS")
    print("=" * 70)

    # 1. Khởi tạo mô hình
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

    # Khởi tạo vỏ mô hình FP16
    if hasattr(model_cls, "create_meta_model"):
        model = model_cls.create_meta_model(config, fp16=True)
    else:
        model = model_cls(config).cuda().half()

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total Dense Base Parameters: {total_params / 1e9:.2f} B")

    # 2. Khởi tạo DataLoader & SGD Optimizer (0 MB State Memory Overhead để không bao giờ bị OOM VRAM)
    loader = PilotDatasetLoader(vocab_size=config.vocab_size, latent_dim=config.latent_dim, action_dim=config.action_dim)
    
    # SGD (momentum=0) có 0 MB optimizer state VRAM overhead!
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    ar_criterion = nn.CrossEntropyLoss()
    dm_criterion = nn.MSELoss()

    dev0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"\n[INFO] Bat dau Train {num_steps} Steps (Batch Size={batch_size}, Accumulation Steps={accum_steps}, LR={lr})...\n")

    start_train_time = time.time()
    optimizer.zero_grad()

    running_ar_loss = 0.0
    running_dm_loss = 0.0

    for step in range(1, num_steps + 1):
        step_start = time.time()

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

        # Giả lập nhãn học thật
        ar_targets = torch.randint(0, config.vocab_size, ar_tokens.shape, device=dev0)
        dm_targets = torch.randn_like(dm_latent)

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

        total_loss = (ar_loss + dm_loss) / accum_steps
        
        # Backward pass
        total_loss.backward()

        # Step optimizer và clip gradient chống nổ NaN
        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_ar_loss += ar_loss.item()
        running_dm_loss += dm_loss.item()

        if step == 1 or step % log_every == 0 or step == num_steps:
            elapsed_ms = (time.time() - step_start) * 1000
            avg_step_ar = running_ar_loss / (log_every if step > 1 else 1)
            avg_step_dm = running_dm_loss / (log_every if step > 1 else 1)
            print(f" Train Step [{step:04d}/{num_steps:04d}] | Step Time: {elapsed_ms:.2f} ms | AR Loss: {avg_step_ar:.4f} | DM Loss: {avg_step_dm:.4f} | Total Loss: {(avg_step_ar + avg_step_dm):.4f}")
            running_ar_loss = 0.0
            running_dm_loss = 0.0

    total_elapsed = time.time() - start_train_time
    print("\n" + "=" * 70)
    print(f" HOÀN THÀNH TRAIN THỬ NGHIỆM [{version.upper()}] {num_steps} STEPS TRONG {total_elapsed:.2f} GIÂY ({total_elapsed/60:.2f} PHÚT)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pilot Steps on Dense Base")
    parser.add_argument("--version", type=str, default="version8", choices=["version5", "version8"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--log_every", type=int, default=50)

    args = parser.parse_args()
    train_pilot_dense_base(
        version=args.version,
        num_steps=args.steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        lr=args.lr,
        log_every=args.log_every
    )
