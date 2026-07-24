"""
Script Suy Luận Đơn Giản (Inference Toy) cho Version 9 MoE World Model
"""

import torch
import time
from mini_model.version9.model import Cosmos3ToyModel, Cosmos3Config

def run_inference_v9():
    print("=" * 70)
    print(" BẮT ĐẦU CHẠY SUY LUẬN VERSION 9 (MoE WORLD MODEL ARCHITECTURE)")
    print("=" * 70)

    config = Cosmos3Config()
    model = Cosmos3ToyModel.create_meta_model(config, fp16=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total Parameters: {total_params / 1e9:.2f} B")
    print(f"[INFO] Active Parameters per token: ~4.03 B (Top-2 Experts active)")

    dev0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Giả lập dữ liệu đầu vào đa phương tiện
    batch_size = 2
    seq_len_ar = 32
    seq_len_dm = 16

    ar_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len_ar), device=dev0)
    dm_latent = torch.randn(batch_size, seq_len_dm, config.latent_dim, device=dev0, dtype=dtype)
    action_vectors = torch.randn(batch_size, seq_len_dm, config.action_dim, device=dev0, dtype=dtype)

    with torch.no_grad():
        start_time = time.time()
        outputs = model(
            ar_tokens=ar_tokens,
            dm_latent=dm_latent,
            action_vectors=action_vectors,
            mode="both"
        )
        elapsed_ms = (time.time() - start_time) * 1000

    print(f"\n[OUTPUT] AR Logits Shape            : {outputs['ar_logits'].shape}")
    print(f"[OUTPUT] DM Predicted Latent Shape   : {outputs['dm_predicted_latent'].shape}")
    print(f"[OUTPUT] Router Aux Loss Value       : {outputs['aux_loss'].item():.6f}")
    print(f"[METRIC] Forward Latency             : {elapsed_ms:.2f} ms")
    print("=" * 70)

if __name__ == "__main__":
    run_inference_v9()
