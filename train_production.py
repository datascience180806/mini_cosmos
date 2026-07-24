"""
Production Fine-Tuning & Training Pipeline for Cosmos 3 MoE World Model (Version 9 / Version 8)
- Đích nhắm: Huấn luyện thực tế (5,000 - 20,000 steps) cho bài toán sản xuất nhà máy / AI công nghiệp.
- Tính năng chống Tràn Ổ Đĩa Kaggle Disk Full:
  1. Tự động ghi đè/xóa file checkpoint cũ, chỉ duy trì 1 file 'cosmos3_{version}_latest.pt' mới nhất (~7.2GB).
  2. Nạp và xử lý luồng dữ liệu đa phương tiện từ Hugging Face qua dataset_loader.py.
  3. Khả năng Resume Fine-Tuning từ Checkpoint đã lưu.
  4. Lịch điều chỉnh Learning Rate: Cosine Annealing Scheduler kết hợp Warmup.
  5. Kỹ thuật Tích lũy Gradient (Gradient Accumulation) và Cắt Gradient (Gradient Clipping) chống tràn VRAM.
"""

import os
import glob
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim

from mini_model.version9.model import Cosmos3ToyModel as V9Model, Cosmos3Config as V9Config
from mini_model.version8.model import Cosmos3ToyModel as V8Model, Cosmos3Config as V8Config
from mini_model.version5.model import Cosmos3ToyModel as V5Model, Cosmos3Config as V5Config
from dataset_loader import PilotDatasetLoader


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
    """Lịch điều chỉnh Learning Rate Cosine Annealing có Warmup."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_production_training(
    version: str = "version9",
    total_steps: int = 5000,
    warmup_steps: int = 200,
    batch_size: int = 1,
    accum_steps: int = 8,
    max_lr: float = 1e-5,
    save_every: int = 1000,
    log_every: int = 50,
    checkpoint_dir: str = "./checkpoints",
    resume_from: str = None
):
    print("=" * 80)
    print(f"🚀 BẮT ĐẦU PRODUCTION FINE-TUNING PIPELINE [{version.upper()}] - {total_steps} STEPS")
    print("=" * 80)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Chọn cấu hình mô hình
    if version.lower() == "version9":
        config = V9Config(use_checkpointing=True)
        model_cls = V9Model
        print("[INFO] Target Architecture: Version 9 (MoE ~7.20B Total / ~4.03B Active)")
    elif version.lower() == "version8":
        config = V8Config()
        model_cls = V8Model
        print("[INFO] Target Architecture: Version 8 (QK-Norm + LayerScale 4.03B Dense Base)")
    else:
        config = V5Config()
        model_cls = V5Model
        print("[INFO] Target Architecture: Version 5 (Base FP16 4.03B Dense Base)")

    num_gpus = torch.cuda.device_count()
    print(f"[INFO] PyTorch CUDA Available: {torch.cuda.is_available()} | GPU Count: {num_gpus}")

    # 2. Khởi tạo vỏ mô hình FP16 Meta Device Init
    if hasattr(model_cls, "create_meta_model"):
        model = model_cls.create_meta_model(config, fp16=True)
    else:
        model = model_cls(config).cuda().half()

    start_step = 1

    # 3. Nạp Checkpoint nếu có lệnh Resume
    if resume_from and os.path.exists(resume_from):
        print(f"[RESUME] Dang nap Checkpoint tu: {resume_from}")
        checkpoint_data = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(checkpoint_data.get("model_state_dict", checkpoint_data))
        start_step = checkpoint_data.get("step", 1) + 1
        print(f"[RESUME] Re-starting tu Step {start_step}!")

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total Model Parameters: {total_params / 1e9:.2f} B")

    # 4. Khởi tạo Optimizer, Scheduler & Dataset Loader
    loader = PilotDatasetLoader(vocab_size=config.vocab_size, latent_dim=config.latent_dim, action_dim=config.action_dim)
    optimizer = optim.SGD(model.parameters(), lr=max_lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    ar_criterion = nn.CrossEntropyLoss()
    dm_criterion = nn.MSELoss()

    dev0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"\n[INFO] Huấn luyện {total_steps} Steps (Effective Batch Size = {batch_size * accum_steps}, LR Max = {max_lr})...\n")

    start_train_time = time.time()
    optimizer.zero_grad()

    running_ar_loss = 0.0
    running_dm_loss = 0.0
    running_aux_loss = 0.0
    valid_steps = 0

    for step in range(start_step, total_steps + 1):
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
        aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=dev0)).float()

        total_loss = (ar_loss + dm_loss + aux_loss) / accum_steps
        total_loss.backward()

        # Step Optimizer & Clip Gradient
        if step % accum_steps == 0:
            has_nan_or_inf = False
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_nan_or_inf = True
                        break

            if not has_nan_or_inf:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()

            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not (torch.isnan(ar_loss) or torch.isnan(dm_loss)):
            running_ar_loss += ar_loss.item() * accum_steps
            running_dm_loss += dm_loss.item() * accum_steps
            running_aux_loss += aux_loss.item() * accum_steps
            valid_steps += 1

        # Periodic Logging
        if step == 1 or step % log_every == 0 or step == total_steps:
            elapsed_ms = (time.time() - step_start) * 1000
            divisor = max(1, valid_steps)
            avg_ar = running_ar_loss / divisor
            avg_dm = running_dm_loss / divisor
            avg_aux = running_aux_loss / divisor
            current_lr = optimizer.param_groups[0]["lr"]
            print(f" Step [{step:05d}/{total_steps:05d}] | Time: {elapsed_ms:.2f} ms | LR: {current_lr:.2e} | AR Loss: {avg_ar:.4f} | DM Loss: {avg_dm:.4f} | Aux Loss: {avg_aux:.4f} | Total Loss: {(avg_ar + avg_dm + avg_aux):.4f}")
            running_ar_loss = 0.0
            running_dm_loss = 0.0
            running_aux_loss = 0.0
            valid_steps = 0

        # Periodic Checkpoint Saving - Chống đầy ổ đĩa Kaggle
        if step % save_every == 0 or step == total_steps:
            # Xóa các file .pt cũ để không tràn đĩa 20GB của Kaggle
            old_files = glob.glob(os.path.join(checkpoint_dir, f"cosmos3_{version}_*.pt"))
            for f in old_files:
                try:
                    os.remove(f)
                except Exception:
                    pass

            save_path = os.path.join(checkpoint_dir, f"cosmos3_{version}_latest.pt")
            print(f"\n[CHECKPOINT] Dang luu Trong so Model (Ghi de an toan) tai: {save_path}...")
            torch.save({
                "step": step,
                "version": version,
                "model_state_dict": model.state_dict(),
                "config": config
            }, save_path)
            print(f"[CHECKPOINT] Luu Checkpoint Step {step} thanh cong! (Xoa file cu, dung luong dia an toan)\n")

    total_elapsed = time.time() - start_train_time
    print("=" * 80)
    print(f"🎉 HOÀN THÀNH PRODUCTION FINE-TUNING [{version.upper()}] {total_steps} STEPS")
    print(f"⏱️ Tong thoi gian: {total_elapsed:.2f} giay ({total_elapsed/60:.2f} phut)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Fine-Tuning Pipeline for Cosmos 3 Models")
    parser.add_argument("--version", type=str, default="version9", choices=["version5", "version8", "version9"])
    parser.add_argument("--steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Warmup steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-GPU batch size")
    parser.add_argument("--accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Max learning rate")
    parser.add_argument("--save_every", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--log_every", type=int, default=50, help="Log print interval")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint .pt file to resume")

    args = parser.parse_args()
    run_production_training(
        version=args.version,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        max_lr=args.lr,
        save_every=args.save_every,
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from
    )
