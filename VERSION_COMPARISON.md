# BÁO CÁO ĐÁNH GIÁ VÀ SO SÁNH CÁC PHIÊN BẢN (VERSION COMPARISON REPORT)

> **Dự án:** `mini_cosmos` - Unified World Model Architecture Experiments  
> **Mục tiêu:** Đánh giá hiệu năng, mức tiêu thụ tài nguyên và độ chính xác kiến trúc giữa các phiên bản thử nghiệm.

---

## 1. Bảng So Sánh Tổng Quan (Benchmark Matrix)

| Thông Số Benchmark | Version 0 (Cosmos 3 Baseline) | Version 1 (Mini PoC) | Version 2 (Scaled) | Version 3 (GQA+RoPE) | Version 4 (1.34B) | Version 5 (4.03B FP16) | Version 6 (7.24B Max 1xT4) | Version 7 (8.12B Dual T4 FP16) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tổng Số Tham Số (Params)** | **16 B (8B Dense)** | **20.49 M** | **127.99 M** | **140.57 M** | **1,342.30 M** | **4,026.76 M** | **7,244.92 M (7.24B)** | **8,117.37 M (8.12B)** |
| **Số Lượng GPU Sử Dụng** | Datacenter | 1x GPU | 1x GPU | 1x GPU | 1x GPU | 1x GPU T4 | **1x GPU T4 (Max 98.2%)** | **2x GPU T4 (Dual GPU)** |
| **Kiểu Dữ Liệu (Precision)** | FP8 / BF16 | FP32 | FP32 | FP32 | FP32 | **FP16** | **FP16** | **FP16 (Pure GPU Direct Init)** |
| **Mức Tiêu Thụ VRAM Peak** | **~18,000 MB** | **139.78 MB** | **702.38 MB** | **810.40 MB** | **6,424.15 MB** | **8,443.60 MB** | **14,306.24 MB (14.31GB)** | **8,521.79 MB (8.52GB)** |
| **Độ Trễ Forward Pass (Latency)** | N/A | **5.27 ms** | **20.44 ms** | **23.09 ms** | **208.90 ms** | **69.62 ms** | **86.58 ms** | **102.22 ms** |
| **Thông Lượng (Throughput)** | N/A | **758.70 fps** | **195.74 fps** | **173.24 fps** | **19.15 fps** | **28.73 fps** | **11.55 fps** | **19.57 fps** |
| **AR Cross-Entropy Loss** | N/A | `7.1093` | `7.7275` | `7.7695` | `9.0605` | `9.9545` | `10.4755` | `82.4236` |
| **DM Reconstruction MSE Loss** | N/A | `1.2332` | `1.3119` | `1.4106` | `1.3204` | `1.3345` | `1.3359` | `399.7594` |
| **Attention Isolation ($Q_{AR} \times K_{DM} = 0$)** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** |

---

## 2. Chi Tiết Kiến Trúc Khối & Sơ Đồ Riêng Cho Từng Phiên Bản

---

### 2.1. Version 0 (`NVIDIA/Cosmos3-Nano`) - Sơ Đồ Kiến Trúc Chuẩn NVIDIA Gốc
- **Đặc điểm kiến trúc:**
  - Tokenizer 3D Causal VAE nén video 8x8x8 kết hợp Text Tokenizer và Action Encoders.
  - Lõi MoT hợp nhất luồng Autoregressive (AR) và Diffusion (DM).
  - Ma trận Attention Mask quy định chặt chẽ $Q_{AR} \times K_{DM} = -\infty$.

```mermaid
flowchart TD
    subgraph V0_INPUTS["Version 0: Input Encoders"]
        V0_TEXT["Text Prompt Tokenizer"] --> V0_EMB["Text Embedding"]
        V0_VIDEO["Video/Image 3D VAE Tokenizer (8x8x8)"] --> V0_LATENT["VAE Latent Tokens"]
        V0_ACT["Action/Audio Encoder"] --> V0_ACT_TOK["Action Tokens"]
    end

    V0_EMB --> V0_MOT["NVIDIA Cosmos 3 Shared MoT Backbone\n(Dense 8B / 16B Total)"]
    V0_LATENT --> V0_MOT
    V0_ACT_TOK --> V0_MOT

    subgraph V0_SURFACES["Version 0: Dual Runtime Surfaces"]
        V0_MOT --> V0_REASONER["Reasoner Surface (AR Causal)\n- Next Token Prediction"]
        V0_MOT --> V0_GENERATOR["Generator Surface (DM Denoising)\n- Flow Matching Video & Action Generation"]
    end
```

---

### 2.2. Version 1 (`mini_model/version1`) - Sơ Đồ Kiến Trúc PoC Thu Nhỏ (~20.5M Params)
- **Đặc điểm kiến trúc:**
  - Khởi tạo PoC cơ bản với `nn.Embedding` (1000, 512), `nn.Linear` (16/32/7, 512).
  - Sử dụng Learned Absolute Position Embeddings (1024 tokens).
  - Khối Transformer Block ($L=6, d_{model}=512, H=8$): **Standard LayerNorm** + **Standard Multi-Head Attention** + **GELU MLP** (`Linear(512->2048) -> GELU -> Linear(2048->512)`).

```mermaid
flowchart TD
    subgraph V1_INPUTS["Version 1: Token Encoders & Absolute Pos"]
        V1_AR["ar_tokens\n(vocab_size=1000)"] --> V1_EMB["nn.Embedding(1000, 512)"]
        V1_DM["dm_latent\n(latent_dim=16)"] --> V1_PROJ["nn.Linear(16, 512)"]
        V1_ACT["action_vectors\n(action_dim=7)"] --> V1_ACT_PROJ["nn.Linear(7, 512)"]
    end

    V1_EMB --> V1_CAT["Concat Sequence + Learned Absolute Pos Embeddings"]
    V1_PROJ --> V1_CAT
    V1_ACT_PROJ --> V1_CAT

    subgraph V1_BLOCKS["Version 1: Standard Transformer Block (x6 Layers)"]
        V1_CAT --> V1_MASK["Cosmos3AttentionMask Matrix\n- Q_AR x K_DM = -inf"]
        V1_MASK --> V1_LN1["nn.LayerNorm(512)"]
        V1_LN1 --> V1_MHA["SharedMultimodalAttention\n(Standard MHA: H=8, d_head=64)"]
        V1_MHA --> V1_LN2["nn.LayerNorm(512)"]
        V1_LN2 --> V1_GELU["Standard GELU MLP\nLinear(512->2048) -> GELU() -> Linear(2048->512)"]
    end

    V1_GELU --> V1_OUT["nn.LayerNorm(512) -> Output Surfaces (ar_head & dm_vision_head)"]
```

---

### 2.3. Version 2 (`mini_model/version2`) - Sơ Đồ Kiến Trúc Mở Rộng SwiGLU & RMSNorm (~128M Params)
- **Đặc điểm kiến trúc:**
  - Nâng cấp chiều ẩn $d_{model}=1024, L=8, H=16$.
  - Thay thế LayerNorm bằng **RMSNorm** (`x * rsqrt(var + eps) * weight`).
  - Thay thế GELU MLP bằng **SwiGLU FFN**: $W_2(\text{SiLU}(W_1 x) \odot W_3 x)$ với `mlp_ratio=3.5`.

```mermaid
flowchart TD
    subgraph V2_INPUTS["Version 2: Scaled Encoders"]
        V2_AR["ar_tokens (vocab=2000)"] --> V2_EMB["nn.Embedding(2000, 1024)"]
        V2_DM["dm_latent (latent_dim=32)"] --> V2_PROJ["nn.Linear(32, 1024)"]
    end

    V2_EMB --> V2_CAT["Concat Sequence + Pos Embeddings (d_model=1024)"]
    V2_PROJ --> V2_CAT

    subgraph V2_BLOCKS["Version 2: SwiGLU + RMSNorm Block (x8 Layers)"]
        V2_CAT --> V2_MASK["Cosmos3AttentionMask Matrix"]
        V2_MASK --> V2_RMS1["RMSNorm(1024)"]
        V2_RMS1 --> V2_MHA["SharedMultimodalAttention (H=16, d_head=64)"]
        V2_MHA --> V2_RMS2["RMSNorm(1024)"]
        V2_RMS2 --> V2_SWIGLU["SwiGLU FFN Block\nw2( SiLU(w1(x)) * w3(x) )\nw1,w3: Linear(1024->3584), w2: Linear(3584->1024)"]
    end

    V2_SWIGLU --> V2_OUT["RMSNorm(1024) -> Output Surfaces"]
```

---

### 2.4. Version 3 (`mini_model/version3`) - Sơ Đồ Kiến Trúc GQA & Position RoPE (~141M Params)
- **Đặc điểm kiến trúc:**
  - Thay thế vị trí tuyệt đối bằng **Rotary Position Embedding (RoPE)** áp dụng trực tiếp lên $Q$ và $K$.
  - Tích hợp **Grouped-Query Attention (GQA)** với $H_Q=16, H_{KV}=4$ (tỷ lệ nén GQA 4:1).

```mermaid
flowchart TD
    subgraph V3_INPUTS["Version 3: RoPE + GQA Encoders"]
        V3_AR["ar_tokens (vocab=2000)"] --> V3_EMB["nn.Embedding(2000, 1024)"]
        V3_DM["dm_latent (latent_dim=32)"] --> V3_PROJ["nn.Linear(32, 1024)"]
    end

    V3_EMB --> V3_CAT["Concat Sequence"]
    V3_PROJ --> V3_CAT

    subgraph V3_BLOCKS["Version 3: GQA + RoPE Block (x10 Layers)"]
        V3_CAT --> V3_RMS1["RMSNorm(1024)"]
        V3_RMS1 --> V3_ROPE["Apply Rotary Position Embedding (RoPE) to Q & K"]
        V3_ROPE --> V3_GQA["GroupedQueryMultimodalAttention\nH_Q = 16, H_KV = 4 (Ratio 4:1)"]
        V3_GQA --> V3_RMS2["RMSNorm(1024)"]
        V3_RMS2 --> V3_SWIGLU["SwiGLU FFN Block"]
    end

    V3_SWIGLU --> V3_OUT["RMSNorm(1024) -> Dual Outputs"]
```

---

### 2.5. Version 4 (`mini_model/version4`) - Sơ Đồ Kiến Trúc Quy Mô Small LLM (~1.34B Params)
- **Đặc điểm kiến trúc:**
  - Mở rộng quy mô $d_{model}=2048, L=24, H_Q=16, H_{KV}=4$, SwiGLU intermediate dim = 5376.
  - Nâng `vocab_size`=4000, `latent_dim`=64.

```mermaid
flowchart TD
    subgraph V4_INPUTS["Version 4: 1.34B Large Scale Encoders"]
        V4_AR["ar_tokens (vocab=4000)"] --> V4_EMB["nn.Embedding(4000, 2048)"]
        V4_DM["dm_latent (latent_dim=64)"] --> V4_PROJ["nn.Linear(64, 2048)"]
    end

    V4_EMB --> V4_CAT["Concat Sequence (d_model=2048)"]
    V4_PROJ --> V4_CAT

    subgraph V4_BLOCKS["Version 4: 24-Layer MoT Transformer Engine"]
        V4_CAT --> V4_RMS1["RMSNorm(2048)"]
        V4_RMS1 --> V4_GQA["GQA Attention + RoPE (H_Q=16, H_KV=4, d_head=128)"]
        V4_GQA --> V4_RMS2["RMSNorm(2048)"]
        V4_RMS2 --> V4_SWIGLU["SwiGLU FFN (Linear 2048 -> 5376 -> 2048)"]
    end

    V4_SWIGLU --> V4_OUT["RMSNorm(2048) -> Dual Heads"]
```

---

### 2.6. Version 5 (`mini_model/version5`) - Sơ Đồ Kiến Trúc Quy Mô Cosmos 3 Edge FP16 (~4.03B Params)
- **Đặc điểm kiến trúc:**
  - Mở rộng quy mô $d_{model}=3072, L=32, H_Q=24, H_{KV}=6$, SwiGLU intermediate dim = 10752.
  - Vận hành chuẩn kiểu dữ liệu **FP16 Half Precision**.

```mermaid
flowchart TD
    subgraph V5_INPUTS["Version 5: 4.03B FP16 Scale Encoders"]
        V5_AR["ar_tokens (vocab=16000)"] --> V5_EMB["nn.Embedding(16000, 3072)"]
        V5_DM["dm_latent (latent_dim=256)"] --> V5_PROJ["nn.Linear(256, 3072)"]
    end

    V5_EMB --> V5_CAT["Concat Sequence (FP16 Mode, d_model=3072)"]
    V5_PROJ --> V5_CAT

    subgraph V5_BLOCKS["Version 5: 32-Layer FP16 MoT Backbone"]
        V5_CAT --> V5_RMS1["RMSNorm(3072)"]
        V5_RMS1 --> V5_GQA["GQA + RoPE (H_Q=24, H_KV=6, d_head=128)"]
        V5_GQA --> V5_RMS2["RMSNorm(3072)"]
        V5_RMS2 --> V5_SWIGLU["SwiGLU FFN (Linear 3072 -> 10752 -> 3072)"]
    end

    V5_SWIGLU --> V5_OUT["RMSNorm(3072) -> FP16 Dual Heads"]
```

---

### 2.7. Version 6 (`mini_model/version6`) - Sơ Đồ Kiến Trúc Quy Mô Cosmos 3 Nano Dense Backbone (~7.24B Params)
- **Đặc điểm kiến trúc:**
  - Mở rộng quy mô $d_{model}=4096, L=32, H_Q=32, H_{KV}=8$, `vocab_size`=32000.
  - Đạt mốc giới hạn tối đa VRAM trên 1 card GPU T4 (14.31 GB VRAM).

```mermaid
flowchart TD
    subgraph V6_INPUTS["Version 6: 7.24B Single GPU Maximum Scale"]
        V6_AR["ar_tokens (vocab=32000)"] --> V6_EMB["nn.Embedding(32000, 4096)"]
        V6_DM["dm_latent (latent_dim=256)"] --> V6_PROJ["nn.Linear(256, 4096)"]
    end

    V6_EMB --> V6_CAT["Concat Sequence (d_model=4096)"]
    V6_PROJ --> V6_CAT

    subgraph V6_BLOCKS["Version 6: 32-Layer FP16 MoT Engine (1x T4 GPU: 14.31GB VRAM)"]
        V6_CAT --> V6_RMS1["RMSNorm(4096)"]
        V6_RMS1 --> V6_GQA["GQA + RoPE (H_Q=32, H_KV=8, d_head=128)"]
        V6_GQA --> V6_RMS2["RMSNorm(4096)"]
        V6_RMS2 --> V6_SWIGLU["SwiGLU FFN (Linear 4096 -> 14336 -> 4096)"]
    end

    V6_SWIGLU --> V6_OUT["RMSNorm(4096) -> Dual Output Surfaces"]
```

---

### 2.8. Version 7 (`mini_model/version7`) - Sơ Đồ Kiến Trúc Đa GPU Pipeline & Meta Direct Init (~8.12B Params)
- **Đặc điểm kiến trúc:**
  - **Meta Device Initialization (`torch.device('meta')`):** Khởi tạo mô hình 0 MB CPU RAM, sau đó `to_empty()` trực tiếp trên GPU VRAM.
  - **Device Pipeline Parallelism:**
    - `cuda:0` (GPU 0): Nạp `ar_embedding`, `dm_vision_proj`, và Blocks $0 \to 17$ (~8.52 GB VRAM).
    - `cuda:1` (GPU 1): Nạp Blocks $18 \to 35$, sau đó chuyển kết quả $h$ về `cuda:0` để qua `norm_f` và các Output Heads.

```mermaid
flowchart TD
    subgraph V7_INIT["Version 7: Pure FP16 Meta Device Init (0 MB CPU RAM)"]
        V7_META["torch.device('meta') Shell Creation"] --> V7_ALLOC["to_empty() Direct GPU Allocation"]
    end

    subgraph V7_GPU0["GPU 0 (cuda:0 - 8.52 GB VRAM)"]
        V7_ALLOC --> V7_EMB0["ar_embedding & dm_vision_proj"]
        V7_EMB0 --> V7_BLK0["Transformer Blocks 0 -> 17\n(d_model=4096, H_Q=32, H_KV=8)"]
    end

    V7_BLK0 --> |Tensor Transfer cuda:0 -> cuda:1| V7_GPU1

    subgraph V7_GPU1["GPU 1 (cuda:1 - 8.52 GB VRAM)"]
        V7_GPU1_BLK["Transformer Blocks 18 -> 35\n(d_model=4096, H_Q=32, H_KV=8)"]
    end

    V7_GPU1_BLK --> |Tensor Transfer cuda:1 -> cuda:0| V7_HEADS

    subgraph V7_HEADS["GPU 0 Output Surface"]
        V7_HEADS_RUN["RMSNorm(4096) -> ar_head & dm_vision_head"]
    end
```

---

## 3. Quy Trình Chạy Benchmark Cho Các Phiên Bản

Để đo lường bất kỳ phiên bản nào trên Kaggle Notebook:

```bash
# Cập nhật repo từ GitHub
!git pull

# Chạy benchmark cho Version 7 trên Dual GPU T4
!python benchmark.py --version version7 --batch_size 2 --num_runs 50 --fp16 --multi_gpu
```
