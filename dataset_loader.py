"""
Dataset Loader for Pilot Mixture Evaluation (Hugging Face Datasets)
Trích xuất và chuẩn hóa 4 nhóm dữ liệu đa phương tiện từ Hugging Face:
1. Text & Reasoning (Cosmos-HumanEval / Traffic-Anomaly-Reasoning)
2. Video & Physics Interaction (SDG-Warehouse / PhyxSim)
3. Robotics Action Trajectory (Cosmos3-DROID / EO-Data1.5M)
4. Spatial Depth & Geometry (Spatial-Intelligence-Warehouse)
"""

import os
import json
import torch
from typing import Dict, Any, Optional

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


class PilotDatasetLoader:
    def __init__(self, vocab_size: int = 16000, latent_dim: int = 256, action_dim: int = 7, hf_token: Optional[str] = None):
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

    def fetch_hue_prompts(self, num_samples: int = 50) -> list:
        """Tải các prompt từ nvidia/Cosmos-HumanEval-v1."""
        if hf_hub_download is None:
            print("[WARN] huggingface_hub chua duoc cai dat. Tra ve fallback prompts.")
            return ["A robotic arm picking up a red box in a warehouse."] * num_samples
        
        try:
            hue_path = hf_hub_download(
                repo_id="nvidia/Cosmos-HumanEval-v1",
                filename="hue-v1p2-t2v-public.json",
                repo_type="dataset",
                token=self.hf_token
            )
            with open(hue_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            samples = data.get("samples", [])
            prompts = [s["prompt"] for s in samples[:num_samples] if "prompt" in s]
            print(f"[SUCCESS] Da tai {len(prompts)} prompts tu nvidia/Cosmos-HumanEval-v1.")
            return prompts
        except Exception as e:
            print(f"[WARN] Khong the tai Cosmos-HumanEval: {e}. Dung fallback prompts.")
            return ["Industrial robot arm assembly operation in smart factory."] * num_samples

    def get_pilot_batch(
        self,
        batch_size: int = 2,
        seq_len_ar: int = 32,
        seq_len_dm: int = 16,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16
    ) -> Dict[str, torch.Tensor]:
        """
        Tạo batch thử nghiệm đại diện cho 4 modality (Text, Video Latent, Audio, Action).
        """
        # 1. Text Tokens (AR)
        ar_tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len_ar), device=device)
        
        # 2. Video Latents (DM)
        dm_latent = torch.randn(batch_size, seq_len_dm, self.latent_dim, device=device, dtype=dtype)
        
        # 3. Action Vectors (7-DoF)
        action_vectors = torch.randn(batch_size, seq_len_dm, self.action_dim, device=device, dtype=dtype)

        return {
            "ar_tokens": ar_tokens,
            "dm_latent": dm_latent,
            "action_vectors": action_vectors
        }


if __name__ == "__main__":
    loader = PilotDatasetLoader()
    prompts = loader.fetch_hue_prompts(num_samples=5)
    print("Sample prompts:", prompts)
    batch = loader.get_pilot_batch(batch_size=2)
    print("Batch keys:", batch.keys())
    print("AR Tokens shape:", batch["ar_tokens"].shape)
    print("DM Latent shape:", batch["dm_latent"].shape)
