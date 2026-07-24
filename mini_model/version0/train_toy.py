"""
Toy training script for mini_cosmos Version 0 (Cosmos 3 Nano Baseline Architecture Shell ~7.24B Params).
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from mini_model.version0.model import Cosmos3ToyModel, Cosmos3Config


def run_toy_training():
    print("=" * 60)
    print("RUNNING MINI_COSMOS VERSION 0 TOY TRAINING LOOP (~7.24B Params)")
    print("=" * 60)

    config = Cosmos3Config()
    model = Cosmos3ToyModel.create_meta_model(config, fp16=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    ar_criterion = nn.CrossEntropyLoss()
    dm_criterion = nn.MSELoss()

    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    seq_len_ar = 16
    seq_len_dm = 8
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("[INFO] Starting 5 toy training iterations...")
    for step in range(1, 6):
        start_time = time.time()
        
        ar_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len_ar), device=device)
        ar_targets = torch.randint(0, config.vocab_size, (batch_size, seq_len_ar), device=device)
        dm_latent = torch.randn(batch_size, seq_len_dm, config.latent_dim, device=device, dtype=dtype)
        dm_targets = torch.randn(batch_size, seq_len_dm, config.latent_dim, device=device, dtype=dtype)

        optimizer.zero_grad()
        outputs = model(ar_tokens=ar_tokens, dm_latent=dm_latent, mode="both")

        ar_loss = ar_criterion(outputs["ar_logits"].view(-1, config.vocab_size).float(), ar_targets.view(-1))
        dm_loss = dm_criterion(outputs["dm_predicted_latent"].float(), dm_targets.float())
        total_loss = ar_loss + dm_loss

        total_loss.backward()
        optimizer.step()

        elapsed = (time.time() - start_time) * 1000
        print(f" Step [{step}/5] | Total Loss: {total_loss.item():.4f} (AR Loss: {ar_loss.item():.4f}, DM Loss: {dm_loss.item():.4f}) | Step Time: {elapsed:.2f} ms")

    print("\n[SUCCESS] Version 0 Toy Training completed cleanly!")
    print("=" * 60)


if __name__ == "__main__":
    run_toy_training()
