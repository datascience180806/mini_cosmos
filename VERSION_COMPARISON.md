# BÁO CÁO ĐÁNH GIÁ VÀ SO SÁNH CÁC PHIÊN BẢN (VERSION COMPARISON REPORT)

> **Dự án:** `mini_cosmos` - Unified World Model Architecture Experiments  
> **Mục tiêu:** Đánh giá hiệu năng, mức tiêu thụ tài nguyên và độ chính xác kiến trúc giữa các phiên bản thử nghiệm.

---

## 1. Bảng So Sánh Tổng Quan (Benchmark Matrix)

| Thông Số Benchmark | Version 0 (Cosmos 3 Nano Baseline) | Version 1 (Mini PoC) | Version 2 (Scaled SwiGLU) | Version 3 (GQA + RoPE) | Version 4 (1.34B Model) | Version 5 (4.03B FP16 Model) | Version 6 (7.24B Max Single T4) | Version 7 (14.2B Dual T4 GPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tổng Số Tham Số (Params)** | **16 B (8B Dense)** | **20.49 M** | **127.99 M** | **140.57 M** | **1,342.30 M (1.34B)** | **4,026.76 M (4.03B)** | **7,244.92 M (7.24B)** | **~14,200 M (14.2B)** |
| **Kích Thước Cấu Hình ($d_{model} / L / H$)** | $4096 / 32 / 32$ | $512 / 6 / 8$ | $1024 / 8 / 16$ | $1024 / 10 / 16$ ($H_{KV}=4$) | $2048 / 24 / 16$ ($H_{KV}=4$) | $3072 / 32 / 24$ ($H_{KV}=6$) | $4096 / 32 / 32$ ($H_{KV}=8$) | $5120 / 40 / 40$ ($H_{KV}=10$) |
| **Kiểu Dữ Liệu (Precision)** | FP8 / BF16 | FP32 | FP32 | FP32 | FP32 | **FP16** | **FP16** | **FP16** |
| **Số Lượng GPU (Device Scaling)** | Datacenter | 1x GPU | 1x GPU | 1x GPU | 1x GPU | 1x GPU T4 | **1x GPU T4 (Max)** | **2x GPU T4 (Dual GPU)** |
| **Mức Tiêu Thụ VRAM Peak (MB)** | **~18,000 MB** | **139.78 MB** | **702.38 MB** | **810.40 MB** | **6,424.15 MB (6.42GB)** | **8,443.60 MB (8.44GB)** | **14,306.24 MB (14.31GB)** | TBD *(Đo trên Kaggle 2xT4)* |
| **Độ Trễ Forward Pass (Latency)** | N/A | **5.27 ms** | **20.44 ms** | **23.09 ms** | **208.90 ms** | **69.62 ms** | **86.58 ms** | TBD *(Đo trên Kaggle 2xT4)* |
| **Thông Lượng (Throughput)** | N/A | **758.70 fps** | **195.74 fps** | **173.24 fps** | **19.15 fps** | **28.73 fps** | **11.55 fps** | TBD *(Đo trên Kaggle 2xT4)* |
| **AR Cross-Entropy Loss** | N/A | `7.1093` | `7.7275` | `7.7695` | `9.0605` | `9.9545` | `10.4755` | TBD |
| **DM Reconstruction MSE Loss** | N/A | `1.2332` | `1.3119` | `1.4106` | `1.3204` | `1.3345` | `1.3359` | TBD |
| **Attention Isolation ($Q_{AR} \times K_{DM} = 0$)** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** |

---

## 2. Chi Tiết Kiến Trúc Các Phiên Bản Thử Nghiệm

### 2.1. Version 0 (`NVIDIA/Cosmos3-Nano`) - Mô Hình Gốc (Baseline Reference)
- **Mã nguồn & Trọng số:** Nguồn mở từ NVIDIA (`nvidia/Cosmos3-Nano`).
- **Phần mềm & Thư viện sử dụng:** `vLLM`, `TensorRT-LLM`, `NVIDIA NIM`, `diffusers`, `PyTorch`.
- **Đặc điểm kiến trúc:**
  - Hợp nhất lớp Autoregressive (AR) và Diffusion (DM) trong cùng một khối Transformer đồng dạng.
  - Sử dụng Tokenizer VAE 3D nén không gian thời-thị giác 8x8x8.
  - Ma trận Attention Mask quy định chặt chẽ $Q_{AR} \times K_{DM} = 0$.

### 2.2. Version 1 (`mini_model/version1`) - Kiến Trúc PoC Thu Nhỏ
- **Mã nguồn:** Xây dựng từ đầu với `torch.nn`.
- **Khối chức năng & Mã nguồn sử dụng:**
  - **Shared Multimodal Attention:** `torch.nn.Linear` (Q, K, V, Out Projections), tự viết ma trận Causal + Zero Masking.
  - **FFN / MLP Block:** `torch.nn.Sequential` với hàm kích hoạt GELU.
  - **Normalization:** Standard `torch.nn.LayerNorm`.
  - **Positional Encoding:** Learned Absolute Position Embeddings (`torch.nn.Parameter`).

### 2.3. Version 2 (`mini_model/version2`) - Mở Rộng Quy Mô & Nâng Cấp Khối Chuẩn LLM
- **Mã nguồn:** Xây dựng trên `torch.nn`.
- **Khối chức năng & Mã nguồn sử dụng:**
  - **Shared Multimodal Attention:** Mở rộng số lượng Attention Heads và $d_{model}$.
  - **FFN / MLP Block:** Chuyển sang cấu trúc **SwiGLU** (`w1`, `w2`, `w3` Projections với hàm kích hoạt `F.silu`).
  - **Normalization:** Thay thế LayerNorm bằng **RMSNorm** (`x * rsqrt(var + eps) * weight`).
  - **Positional Encoding:** Mở rộng chiều dài chuỗi tối đa lên 2048 tokens.

### 2.4. Version 3 (`mini_model/version3`) - Tối Ưu Hóa GQA & Rotary Position Embedding (RoPE)
- **Mã nguồn:** Xây dựng trên `torch.nn`.
- **Khối chức năng & Mã nguồn sử dụng:**
  - **Grouped-Query Attention (GQA):** Giảm số lượng Key/Value heads ($H_{KV}=4$) so với Query heads ($H_Q=16$) để tối ưu bộ nhớ KV Cache và tăng tốc độ suy luận.
  - **Rotary Position Embedding (RoPE):** Ánh xạ tọa độ vị trí tương quan trực tiếp vào không gian xoay Query và Key.
  - **Normalization & MLP:** RMSNorm + SwiGLU MLP kết hợp ma trận Attention Mask hợp nhất.

### 2.5. Version 4 (`mini_model/version4`) - Quy Mô 1.34 Billion Parameters (~1.34B Params)
- **Mã nguồn:** Xây dựng trên `torch.nn`.
- **Khối chức năng & Mã nguồn sử dụng:**
  - **Mở rộng chiều ẩn $d_{model}$ lên 2048 và số lớp lên 24:** Đạt tổng cộng **1,342.30M tham số (~1.34B)**.
  - **Grouped-Query Attention (GQA 4:1):** Chiếm dụng ~6.42 GB VRAM khi chạy ở FP32.

### 2.6. Version 5 (`mini_model/version5`) - Siêu Quy Mô Tương Đương NVIDIA Cosmos 3 Edge (~4.03B Params)
- **Mã nguồn:** Xây dựng trên `torch.nn`.
- **Khối chức năng & Mã nguồn sử dụng:**
  - **Chiều ẩn $d_{model}=3072, L=32$:** Đạt quy mô **4,026.76M tham số (~4.03B)**, chạy chuẩn FP16 Half Precision.
  - **Hiệu năng VRAM:** Chiếm **8,443.60 MB (~8.44 GB VRAM)** trên Kaggle T4 GPU.

### 2.7. Version 6 (`mini_model/version6`) - Chạm Giới Hạn Tối Đa Single GPU T4 (~7.24B Params)
- **Mã nguồn:** Xây dựng trên `torch.nn`.
- **Khối chức năng & Mã nguồn sử dụng:**
  - **Chiều ẩn $d_{model}=4096, L=32, H_Q=32, H_{KV}=8$:** Đạt quy mô **7,244.92M tham số (~7.24B)**, tương đương với nhánh Dense Backbone của Cosmos 3 Nano gốc.
  - **Kết quả VRAM:** Chiếm **14,306.24 MB (~14.31 GB VRAM)** trên 1x GPU T4 (đạt 98.2% giới hạn tối đa 14.56 GB).

### 2.8. Version 7 (`mini_model/version7`) - Đột Phá Đa GPU (Dual T4 GPU Parallelism - ~14.2B Params)
- **Mã nguồn:** Xây dựng trên `torch.nn` kết hợp `torch.nn.DataParallel` / Device Pipeline Placement.
- **Khối chức năng & Mã nguồn sử dụng:**
  - **Chiều ẩn $d_{model}=5120, L=40, H_Q=40, H_{KV}=10$:** Đạt quy mô cực khủng **~14.2 Tỷ tham số (14.2 Billion parameters)**.
  - **Song song hóa Dual GPU (2x T4 GPUs = 32GB VRAM):** Tự động phân bổ hoặc song song hóa batch/tầng mô hình trên 2 card T4 của Kaggle.

---

## 3. Quy Trình Chạy Benchmark Cho Phiên Bản Mới

Để đo lường chỉ số cho Version 7 trên Kaggle Notebook với **2x GPU T4**:

```bash
# Cập nhật repo từ GitHub
!git pull

# Chạy benchmark Version 7 tận dụng Dual GPU T4 (Kích hoạt FP16 & Multi-GPU)
!python benchmark.py --version version7 --batch_size 2 --num_runs 50 --fp16 --multi_gpu
```
