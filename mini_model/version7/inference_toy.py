"""
Kịch bản chạy thử nghiệm suy luận cho Version 7 (~14.2B parameters, Dual T4 GPUs).
"""

import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mini_model.version7.model import Cosmos3ToyModel, Cosmos3Config


def run_inference_demo():
    print("=" * 70)
    print("BẮT ĐẦU THỬ NGHIỆM SUY LUẬN VERSION 7 DUAL GPU MODEL (~14.2B PARAMS)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    config = Cosmos3Config()
    model = Cosmos3ToyModel(config)
    
    if device.type == "cuda":
        model = model.half().to(device)
        if num_gpus > 1:
            print(f"[INFO] Che do DataParallel tren {num_gpus} GPUs duoc kich hoat!")
            model = torch.nn.DataParallel(model)

    model.eval()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("\n--- 1. CHẠY THỬ REASONER MODE (Version 7) ---")
    prompt_tokens = torch.tensor([[10, 45, 203, 89, 500]], device=device)
    
    with torch.no_grad():
        output_reasoner = model(ar_tokens=prompt_tokens, mode="reasoner")
        ar_logits = output_reasoner["ar_logits"]
        next_token = torch.argmax(ar_logits[:, -1, :], dim=-1)
        
    print(f"-> Input Prompt Tokens: {prompt_tokens.tolist()}")
    print(f"-> Shape Logits: {ar_logits.shape}")
    print(f"-> Predicted Next Token ID: {next_token.item()}")

    print("\n--- 2. CHẠY THỬ GENERATOR MODE (Version 7) ---")
    seq_ar = 8
    seq_dm = 4
    ar_condition = torch.randint(0, config.vocab_size, (1, seq_ar), device=device)
    noisy_latent = torch.randn(1, seq_dm, config.latent_dim, device=device, dtype=dtype)
    action_input = torch.randn(1, seq_dm, config.action_dim, device=device, dtype=dtype)

    with torch.no_grad():
        output_generator = model(
            ar_tokens=ar_condition,
            dm_latent=noisy_latent,
            action_vectors=action_input,
            mode="generator"
        )
        denoised_latent = output_generator["dm_predicted_latent"]

    print(f"-> Noisy Latent Shape: {noisy_latent.shape}")
    print(f"-> Action Vector Shape: {action_input.shape}")
    print(f"-> Predicted Denoised Latent Shape: {denoised_latent.shape}")

    print("\n[SUCCESS] Chay thanh cong Version 7 Dual GPU Inference!")


if __name__ == "__main__":
    run_inference_demo()
