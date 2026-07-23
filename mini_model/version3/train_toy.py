"""
Kịch bản huấn luyện thử nghiệm cho Version 3 (GQA + RoPE MoT Model).
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mini_model.version3.model import Cosmos3ToyModel, Cosmos3Config


def train_toy_step():
    print("=" * 70)
    print("BẮT ĐẦU KIỂM THỬ HUẤN LUYỆN COSMOS 3 VERSION 3 MODEL (GQA + RoPE)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiet bi dang su dung: {device}")

    config = Cosmos3Config()
    model = Cosmos3ToyModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] So luong tham so mo hinh Version 3: {total_params / 1e6:.2f}M params")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    ar_loss_fn = nn.CrossEntropyLoss()
    dm_loss_fn = nn.MSELoss()

    batch_size = 4
    seq_len_ar = 32
    seq_len_dm = 16

    print("\n[INFO] Bat dau vong lap huan luyen thu nghiem (5 steps)...")
    model.train()

    for step in range(1, 6):
        start_time = time.time()
        optimizer.zero_grad()

        dummy_ar_input = torch.randint(0, config.vocab_size, (batch_size, seq_len_ar), device=device)
        dummy_ar_target = torch.randint(0, config.vocab_size, (batch_size, seq_len_ar), device=device)
        
        dummy_dm_latent = torch.randn(batch_size, seq_len_dm, config.latent_dim, device=device)
        dummy_dm_clean_target = torch.randn(batch_size, seq_len_dm, config.latent_dim, device=device)
        dummy_action_vectors = torch.randn(batch_size, seq_len_dm, config.action_dim, device=device)

        outputs = model(
            ar_tokens=dummy_ar_input,
            dm_latent=dummy_dm_latent,
            action_vectors=dummy_action_vectors,
            mode="both"
        )

        ar_logits = outputs["ar_logits"]
        loss_ar = ar_loss_fn(ar_logits.view(-1, config.vocab_size), dummy_ar_target.view(-1))

        dm_pred = outputs["dm_predicted_latent"]
        loss_dm = dm_loss_fn(dm_pred, dummy_dm_clean_target)

        total_loss = loss_ar + loss_dm

        total_loss.backward()
        optimizer.step()

        elapsed = (time.time() - start_time) * 1000

        if device.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            mem_info = f"| VRAM: {vram_mb:.1f}MB"
        else:
            mem_info = "| CPU Mode"

        print(f"Step {step}/5 | Total Loss: {total_loss.item():.4f} (AR Loss: {loss_ar.item():.4f}, DM Loss: {loss_dm.item():.4f}) | Thoi gian: {elapsed:.1f}ms {mem_info}")

    print("\n[SUCCESS] Hoan thanh 5 steps huan luyen thử nghiem cho Version 3!")


if __name__ == "__main__":
    train_toy_step()
