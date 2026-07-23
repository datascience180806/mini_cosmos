import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cosmos3Config:
    """
    Cấu hình thông số kỹ thuật cho phiên bản Cosmos 3 Toy Model (Proof-of-Concept)
    Kích thước siêu nhẹ (~50M - 80M params) để chạy thử trên GPU T4 / CPU.
    """
    hidden_dim: int = 512            # Dimension không gian nhúng (d_model)
    num_heads: int = 8               # Số lượng Attention Heads (d_head = 64)
    num_layers: int = 6              # Số lượng khối Transformer
    mlp_ratio: float = 4.0           # Tỉ lệ mở rộng của lớp MLP (intermediate_dim = 2048)
    dropout: float = 0.1
    
    # Kích thước từ vựng & Latent các modality
    vocab_size: int = 1000           # Kích thước từ vựng rời rạc (Text + Vision ViT AR)
    latent_dim: int = 16             # Kích thước không gian nén VAE (Vision DM)
    audio_dim: int = 32              # Kích thước đặc trưng âm thanh
    action_dim: int = 7              # Kích thước véc-tơ hành động (ví dụ: 6-DoF pose + gripper state)


class Cosmos3AttentionMask(nn.Module):
    """
    Tạo Ma trận Phân luồng Chú ý (Attention Mask Matrix) cho Cosmos 3:
    - Vùng Q_AR x K_AR: Causal Mask (dạng tam giác dưới).
    - Vùng Q_AR x K_DM: Masked Zero (Chặn nhiễu DM ảnh hưởng tới logic AR).
    - Vùng Q_DM x K_AR & K_DM: Full Attention (Nhánh sinh truy cập toàn bộ ngữ cảnh AR và DM).
    """
    def __init__(self):
        super().__init__()

    def forward(self, seq_len_ar: int, seq_len_dm: int, device: torch.device) -> torch.Tensor:
        total_len = seq_len_ar + seq_len_dm
        # Khởi tạo ma trận mask kích thước (total_len, total_len) với giá trị ban đầu là -inf (bị che)
        mask = torch.full((total_len, total_len), float("-inf"), device=device)

        # 1. Phân vùng Q_AR x K_AR (Top-Left): Causal Triangular Mask
        causal_mask = torch.triu(torch.full((seq_len_ar, seq_len_ar), float("-inf"), device=device), diagonal=1)
        mask[:seq_len_ar, :seq_len_ar] = causal_mask

        # 2. Phân vùng Q_AR x K_DM (Top-Right): Masked Zero (-inf giữ nguyên, AR không chú ý tới DM)

        # 3. Phân vùng Q_DM x K_AR (Bottom-Left): Full Attention (0.0 = cho phép chú ý)
        mask[seq_len_ar:, :seq_len_ar] = 0.0

        # 4. Phân vùng Q_DM x K_DM (Bottom-Right): Full Attention (0.0 = cho phép chú ý)
        mask[seq_len_ar:, seq_len_ar:] = 0.0

        return mask  # Shape: [total_len, total_len]


class SharedMultimodalAttention(nn.Module):
    """
    Khối Shared Attention dùng chung giữa chuỗi AR và DM với Ma trận Attention Mask tùy chỉnh.
    """
    def __init__(self, config: Cosmos3Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        assert config.hidden_dim % config.num_heads == 0, "hidden_dim phải chia hết cho num_heads"

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Tính Query, Key, Value
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            # Broadcast mask cho tất cả batch và heads: [1, 1, seq_len, seq_len]
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # Shape: [batch_size, num_heads, seq_len, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.hidden_dim)

        return self.out_proj(context)


class MLP(nn.Module):
    """Khối Feed-Forward Network với hàm kích hoạt GELU."""
    def __init__(self, config: Cosmos3Config):
        super().__init__()
        intermediate_dim = int(config.hidden_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_dim, intermediate_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class Cosmos3Block(nn.Module):
    """
    Khối Transformer đồng dạng của Cosmos 3:
    LayerNorm -> SharedMultimodalAttention -> LayerNorm -> MLP (có Residual Connection).
    """
    def __init__(self, config: Cosmos3Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = SharedMultimodalAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class Cosmos3ToyModel(nn.Module):
    """
    Mô hình Cosmos 3 Toy Model (Unified Mixture-of-Transformers):
    - Hợp nhất hai luồng token: AR (rời rạc) và DM (liên tục/khuếch tán).
    - Hỗ trợ 2 chế độ vận hành: 'reasoner' và 'generator'.
    """
    def __init__(self, config: Cosmos3Config):
        super().__init__()
        self.config = config
        
        # 1. Khối Input Projection & Encoders
        self.ar_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.dm_vision_proj = nn.Linear(config.latent_dim, config.hidden_dim)
        self.audio_proj = nn.Linear(config.audio_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        # Positional Embeddings đơn giản
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, config.hidden_dim) * 0.02)

        # 2. Transformer Blocks & Mask Generator
        self.mask_generator = Cosmos3AttentionMask()
        self.blocks = nn.ModuleList([Cosmos3Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_dim)

        # 3. Heads Đầu Ra (Dual Runtime Surfaces)
        self.ar_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.dm_vision_head = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(
        self,
        ar_tokens: torch.Tensor,                                # [batch_size, seq_len_ar] (Token từ vựng rời rạc)
        dm_latent: Optional[torch.Tensor] = None,              # [batch_size, seq_len_dm, latent_dim] (Token khuếch tán nhiễu)
        audio_features: Optional[torch.Tensor] = None,         # [batch_size, seq_len_audio, audio_dim]
        action_vectors: Optional[torch.Tensor] = None,         # [batch_size, seq_len_action, action_dim]
        mode: str = "both"                                      # Chế độ: 'reasoner', 'generator', hoặc 'both'
    ) -> Dict[str, torch.Tensor]:
        
        device = ar_tokens.device
        batch_size, seq_len_ar = ar_tokens.shape

        # Embed chuỗi AR
        x_ar = self.ar_embedding(ar_tokens)
        
        # Embed các chuỗi DM (Vision, Audio, Action)
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

        total_seq_len = x_seq.shape[1]
        x_seq = x_seq + self.pos_embed[:, :total_seq_len, :]

        # Tạo Attention Mask
        if seq_len_dm > 0:
            attn_mask = self.mask_generator(seq_len_ar, seq_len_dm, device=device)
        else:
            attn_mask = torch.triu(torch.full((seq_len_ar, seq_len_ar), float("-inf"), device=device), diagonal=1)

        # Chạy qua các Transformer Blocks
        h = x_seq
        for block in self.blocks:
            h = block(h, attn_mask=attn_mask)
        
        h = self.ln_f(h)

        outputs = {}

        # 4. Trích xuất Output dựa trên Mode
        if mode in ["reasoner", "both"]:
            # Nhánh Reasoner: Dự đoán token ngôn ngữ/thị giác tiếp theo cho chuỗi AR
            h_ar = h[:, :seq_len_ar, :]
            outputs["ar_logits"] = self.ar_head(h_ar)

        if mode in ["generator", "both"] and seq_len_dm > 0:
            # Nhánh Generator: Dự đoán véc-tơ nén (latent) đã được khử nhiễu cho chuỗi DM
            h_dm = h[:, seq_len_ar:, :]
            outputs["dm_predicted_latent"] = self.dm_vision_head(h_dm[:, :dm_latent.shape[1], :])

        return outputs


if __name__ == "__main__":
    # Test thử nhanh mô hình với dữ liệu ngẫu nhiên
    config = Cosmos3Config()
    model = Cosmos3ToyModel(config)
    
    # In tổng số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[SUCCESS] Khoi tao Cosmos 3 Toy Model thanh cong!")
    print(f"-> Tong so luong tham so (Total Parameters): {total_params / 1e6:.2f}M")

    # Bơm dữ liệu giả lập
    b_size = 2
    seq_ar, seq_dm = 16, 8
    ar_in = torch.randint(0, config.vocab_size, (b_size, seq_ar))
    dm_in = torch.randn(b_size, seq_dm, config.latent_dim)

    out = model(ar_in, dm_latent=dm_in, mode="both")
    print(f"-> Shape AR Logits: {out['ar_logits'].shape}")
    print(f"-> Shape DM Predicted Latent: {out['dm_predicted_latent'].shape}")
