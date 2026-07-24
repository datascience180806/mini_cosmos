"""
Toy inference script for mini_cosmos Version 8 (~4.03B Params QK-Norm + LayerScale Architecture).
"""

import time
import torch
from mini_model.version8.model import Cosmos3ToyModel, Cosmos3Config


def run_toy_inference():
    print("=" * 60)
    print("RUNNING MINI_COSMOS VERSION 8 TOY INFERENCE (~4.03B Params QK-Norm Scale)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    config = Cosmos3Config()
    model = Cosmos3ToyModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total model parameters: {total_params / 1e6:.2f} M ({total_params / 1e9:.2f} B)")

    model.eval()

    batch_size = 2
    seq_len_ar = 32
    seq_len_dm = 16

    ar_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len_ar), device=device)
    dm_latent = torch.randn(batch_size, seq_len_dm, config.latent_dim, device=device)
    action_vectors = torch.randn(batch_size, seq_len_dm, config.action_dim, device=device)

    start = time.time()
    with torch.no_grad():
        outputs = model(
            ar_tokens=ar_tokens,
            dm_latent=dm_latent,
            action_vectors=action_vectors,
            mode="both"
        )
    elapsed = (time.time() - start) * 1000

    print(f"\n[OUTPUT SUMMARY]")
    print(f"  • Forward pass time     : {elapsed:.2f} ms")
    if "ar_logits" in outputs:
        print(f"  • AR Logits shape       : {outputs['ar_logits'].shape}")
    if "dm_predicted_latent" in outputs:
        print(f"  • DM Latent Pred shape  : {outputs['dm_predicted_latent'].shape}")
    print("=" * 60)


if __name__ == "__main__":
    run_toy_inference()
