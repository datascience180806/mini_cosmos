"""
Production Inference & Evaluation Script cho Cosmos 3 Models (Version 9 / Version 8 / Version 5)
- Nạp Checkpoint đã huấn luyện (.pt) từ thư mục ./checkpoints/
- Chạy suy luận sinh dự đoán đa phương tiện (Text AR Logits, Video DM Latents, Action 7-DoF)
- Đánh giá Latency, Throughput và Router Aux Loss trên phần cứng thực tế.
"""

import os
import argparse
import time
import torch

from mini_model.version9.model import Cosmos3ToyModel as V9Model, Cosmos3Config as V9Config
from mini_model.version8.model import Cosmos3ToyModel as V8Model, Cosmos3Config as V8Config
from mini_model.version5.model import Cosmos3ToyModel as V5Model, Cosmos3Config as V5Config
from dataset_loader import PilotDatasetLoader


def run_production_inference(checkpoint_path: str = None, version: str = "version9", num_samples: int = 5):
    print("=" * 80)
    print(f"🔮 BẮT ĐẦU PRODUCTION INFERENCE & EVALUATION [{version.upper()}]")
    print("=" * 80)

    # 1. Chọn cấu hình
    if version.lower() == "version9":
        config = V9Config(use_checkpointing=False)
        model_cls = V9Model
    elif version.lower() == "version8":
        config = V8Config()
        model_cls = V8Model
    else:
        config = V5Config()
        model_cls = V5Model

    num_gpus = torch.cuda.device_count()
    print(f"[INFO] PyTorch CUDA Available: {torch.cuda.is_available()} | GPU Count: {num_gpus}")

    # 2. Khởi tạo mô hình
    if hasattr(model_cls, "create_meta_model"):
        model = model_cls.create_meta_model(config, fp16=True)
    else:
        model = model_cls(config).cuda().half()

    # 3. Nạp Checkpoint nếu có
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT] Dang nap khoi trong so tu: {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint_data.get("model_state_dict", checkpoint_data))
        print("[CHECKPOINT] Nạp Checkpoint thành công!")
    else:
        print("[WARN] Khong tim thấy checkpoint_path. Su dung trong so khoi tao ban dau.")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total Model Parameters: {total_params / 1e9:.2f} B")

    loader = PilotDatasetLoader(vocab_size=config.vocab_size, latent_dim=config.latent_dim, action_dim=config.action_dim)
    prompts = loader.fetch_hue_prompts(num_samples=num_samples)
    
    dev0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("\n[INFO] Bat dau luong Suy Luan khong gian nena (Diffusion + Autoregressive)...\n")

    with torch.no_grad():
        for i, prompt in enumerate(prompts, 1):
            start_t = time.time()

            batch = loader.get_pilot_batch(
                batch_size=1,
                seq_len_ar=32,
                seq_len_dm=16,
                device=dev0,
                dtype=dtype
            )

            outputs = model(
                ar_tokens=batch["ar_tokens"],
                dm_latent=batch["dm_latent"],
                action_vectors=batch["action_vectors"],
                mode="both"
            )

            elapsed_ms = (time.time() - start_t) * 1000

            print(f" Sample [{i}/{num_samples}] | Prompt: '{prompt[:60]}...'")
            print(f"   • AR Logits Output       : {outputs['ar_logits'].shape}")
            print(f"   • DM Predicted Latent    : {outputs['dm_predicted_latent'].shape}")
            if "aux_loss" in outputs:
                print(f"   • Router Aux Loss        : {outputs['aux_loss'].item():.6f}")
            print(f"   • Single-pass Latency    : {elapsed_ms:.2f} ms ({1000.0/elapsed_ms:.2f} fps)")
            print("-" * 60)

    print("=" * 80)
    print("🎉 HOÀN THÀNH PRODUCTION INFERENCE BENCHMARK")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Inference for Cosmos 3 Models")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--version", type=str, default="version9", choices=["version5", "version8", "version9"])
    parser.add_argument("--samples", type=int, default=5, help="Number of evaluation samples")

    args = parser.parse_args()
    run_production_inference(checkpoint_path=args.checkpoint, version=args.version, num_samples=args.samples)
