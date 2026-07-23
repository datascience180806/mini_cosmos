import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cosmos3Config:
    """
    Cấu hình thông số kỹ thuật cho Version 4 (Scale mạnh lên ~1.31 Billion Parameters)
    - hidden_dim: 2048
    - num_layers: 24 layers
    - num_heads: 16 (num_kv_heads=4, GQA 4:1)
    - vocab_size: 8000
    - latent_dim: 128
    """
    hidden_dim: int = 2048           # Dim không gian nhúng lớn
    num_heads: int = 16              # Số lượng Query Attention Heads
    num_kv_heads: int = 4            # Số lượng Key/Value Attention Heads (GQA 4:1)
    num_layers: int = 24             # 24 Transformer Blocks (Quy mô 1.3B)
    mlp_ratio: float = 3.5           # SwiGLU intermediate dim = 7168
    dropout: float = 0.1
    
    # Kích thước từ vựng & Latent các modality mở rộng
    vocab_size: int = 8000           # Từ vựng rời rạc mở rộng
    latent_dim: int = 128            # Không gian nén VAE độ phân giải cao
    audio_dim: int = 128             # Đặc trưng âm thanh
    action_dim: int = 7              # Véc-tơ hành động (6-DoF + gripper)


class RMSNorm(nn.Module):
    """RMSNorm đơn giản và hiệu năng cao."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) cho Q và K."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


class Cosmos3AttentionMask(nn.Module):
    """Ma trận Phân luồng Chú ý (Attention Mask Matrix)."""
    def __init__(self):
        super().__init__()

    def forward(self, seq_len_ar: int, seq_len_dm: int, device: torch.device) -> torch.Tensor:
        total_len = seq_len_ar + seq_len_dm
        mask = torch.full((total_len, total_len), float("-inf"), device=device)

        # Q_AR x K_AR: Causal Mask
        causal_mask = torch.triu(torch.full((seq_len_ar, seq_len_ar), float("-inf"), device=device), diagonal=1)
        mask[:seq_len_ar, :seq_len_ar] = causal_mask

        # Q_DM x [K_AR, K_DM]: Full Attention
        mask[seq_len_ar:, :] = 0.0

        return mask


class GroupedQueryMultimodalAttention(nn.Module):
    """
    Shared Multimodal Attention với GQA & RoPE cho mô hình 1.3B.
    """
    def __init__(self, config: Cosmos3Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_queries_per_kv = config.num_heads // config.num_kv_heads
        self.head_dim = config.hidden_dim // config.num_heads

        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_dim, bias=False)
        
        self.rope = RotaryPositionEmbedding(self.head_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.hidden_dim)

        return self.out_proj(context)


class SwiGLUMLP(nn.Module):
    """Feed-Forward SwiGLU."""
    def __init__(self, config: Cosmos3Config):
        super().__init__()
        intermediate_dim = int(config.hidden_dim * config.mlp_ratio)
        self.w1 = nn.Linear(config.hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, intermediate_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Cosmos3Block(nn.Module):
    """Khối Transformer Version 4 quy mô 1.3B Params."""
    def __init__(self, config: Cosmos3Config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.attn = GroupedQueryMultimodalAttention(config)
        self.norm2 = RMSNorm(config.hidden_dim)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Cosmos3ToyModel(nn.Module):
    """
    Mô hình Cosmos 3 Version 4 (Large Scale MoT Model ~1.31 Billion parameters):
    - Tăng quy mô tham số hidden_dim=2048, num_layers=24.
    - Grouped-Query Attention (GQA) & Rotary Position Embedding (RoPE).
    - RMSNorm + SwiGLU MLP.
    """
    def __init__(self, config: Cosmos3Config):
        super().__init__()
        self.config = config
        
        self.ar_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.dm_vision_proj = nn.Linear(config.latent_dim, config.hidden_dim)
        self.audio_proj = nn.Linear(config.audio_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)

        self.mask_generator = Cosmos3AttentionMask()
        self.blocks = nn.ModuleList([Cosmos3Block(config) for _ in range(config.num_layers)])
        self.norm_f = RMSNorm(config.hidden_dim)

        self.ar_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.dm_vision_head = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(
        self,
        ar_tokens: torch.Tensor,
        dm_latent: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        action_vectors: Optional[torch.Tensor] = None,
        mode: str = "both"
    ) -> Dict[str, torch.Tensor]:
        
        device = ar_tokens.device
        batch_size, seq_len_ar = ar_tokens.shape

        x_ar = self.ar_embedding(ar_tokens)
        
        dm_embeds = []
        seq_len_dm = 0

        if dm_latent is not None:
            x_dm_vis = self.dm_vision_proj(dm_latent)
            dm_embeds.append(x_dm_vis)
            seq_len_dm += x_dm_vis.shape[1]

        if audio_features is not None:
            x_audio = self.audio_proj(audio_features)
            dm_embeds.append(x_audio)
            seq_len_dm += x_audio.shape[1]

        if action_vectors is not None:
            x_action = self.action_proj(action_vectors)
            dm_embeds.append(x_action)
            seq_len_dm += x_action.shape[1]

        if len(dm_embeds) > 0:
            x_dm = torch.cat(dm_embeds, dim=1)
            x_seq = torch.cat([x_ar, x_dm], dim=1)
        else:
            x_seq = x_ar

        if seq_len_dm > 0:
            attn_mask = self.mask_generator(seq_len_ar, seq_len_dm, device=device)
        else:
            attn_mask = torch.triu(torch.full((seq_len_ar, seq_len_ar), float("-inf"), device=device), diagonal=1)

        h = x_seq
        for block in self.blocks:
            h = block(h, attn_mask=attn_mask)
        
        h = self.norm_f(h)

        outputs = {}

        if mode in ["reasoner", "both"]:
            h_ar = h[:, :seq_len_ar, :]
            outputs["ar_logits"] = self.ar_head(h_ar)

        if mode in ["generator", "both"] and seq_len_dm > 0:
            h_dm = h[:, seq_len_ar:, :]
            outputs["dm_predicted_latent"] = self.dm_vision_head(h_dm[:, :dm_latent.shape[1], :])

        return outputs


if __name__ == "__main__":
    config = Cosmos3Config()
    model = Cosmos3ToyModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[SUCCESS] Khoi tao Cosmos 3 Version 4 Model (~1.31B Params) thanh cong!")
    print(f"-> Tong so luong tham so (Total Parameters): {total_params / 1e6:.2f}M ({total_params / 1e9:.2f}B)")
