"""
Kịch bản chạy thử nghiệm suy luận (Inference Script) cho hai chế độ vận hành:
1. Reasoner Mode (Dự đoán token suy luận tự hồi quy)
2. Generator Mode (Sinh dữ liệu khuếch tán điều khiển bởi action)
"""

import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mini_model.version1.model import Cosmos3ToyModel, Cosmos3Config


def run_inference_demo():
    print("=" * 70)
    print("BẮT ĐẦU THỬ NGHIỆM SUY LUẬN ĐA CHẾ ĐỘ (REASONER & GENERATOR MODES)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Cosmos3Config()
    model = Cosmos3ToyModel(config).to(device)
    model.eval()

    # ==========================================
    # CHẾ ĐỘ 1: REASONER MODE (AUTOREGRESSIVE)
    # ==========================================
    print("\n--- 1. CHẠY THỬ REASONER MODE (Suy luận chuỗi token) ---")
    prompt_tokens = torch.tensor([[10, 45, 203, 89, 500]], device=device)  # Batch 1, Sequence 5
    
    with torch.no_grad():
        output_reasoner = model(ar_tokens=prompt_tokens, mode="reasoner")
        ar_logits = output_reasoner["ar_logits"]
        next_token = torch.argmax(ar_logits[:, -1, :], dim=-1)
        
    print(f"-> Input Prompt Tokens: {prompt_tokens.tolist()}")
    print(f"-> Shape Logits: {ar_logits.shape}")
    print(f"-> Predicted Next Token ID: {next_token.item()}")

    # ==========================================
    # CHẾ ĐỘ 2: GENERATOR MODE (DIFFUSION DENOISING)
    # ==========================================
    print("\n--- 2. CHẠY THỬ GENERATOR MODE (Khử nhiễu sinh mô phỏng) ---")
    seq_ar = 8
    seq_dm = 4
    ar_condition = torch.randint(0, config.vocab_size, (1, seq_ar), device=device)
    noisy_latent = torch.randn(1, seq_dm, config.latent_dim, device=device)
    action_input = torch.randn(1, seq_dm, config.action_dim, device=device)

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

    print("\n[SUCCESS] Chay thanh cong ca 2 che do Reasoner Mode va Generator Mode!")


if __name__ == "__main__":
    run_inference_demo()
