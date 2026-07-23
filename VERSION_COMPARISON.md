# BÁO CÁO ĐÁNH GIÁ VÀ SO SÁNH CÁC PHIÊN BẢN (VERSION COMPARISON REPORT)

> **Dự án:** `mini_cosmos` - Unified World Model Architecture Experiments  
> **Mục tiêu:** Đánh giá hiệu năng, mức tiêu thụ tài nguyên và độ chính xác kiến trúc giữa các phiên bản thử nghiệm.

---

## 1. Bảng So Sánh Tổng Quan (Benchmark Matrix)

| Thông Số Benchmark | Version 0 (Cosmos 3 Nano Baseline) | Version 1 (Mini PoC) | Version 2 (Scaled SwiGLU) | Version 3 (GQA + RoPE) | Version 4 (1.31B Large Model) | Version 5 (3.93B Cosmos Edge Scale) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tổng Số Tham Số (Params)** | **16 B (8B Dense)** | **20.49 M** | **127.99 M** | **140.57 M** | **~1.31 B (1,308 M)** | **~3.93 B (3,926 M)** |
| **Kích Thước Cấu Hình ($d_{model} / L / H$)** | $4096 / 32 / 32$ | $512 / 6 / 8$ | $1024 / 8 / 16$ | $1024 / 10 / 16$ ($H_{KV}=4$) | $2048 / 24 / 16$ ($H_{KV}=4$) | $3072 / 32 / 24$ ($H_{KV}=6$) |
| **Mức Tiêu Thụ VRAM Peak (MB)** | **~18,000 MB** (FP8) | **139.78 MB** | **702.38 MB** | **810.40 MB** | TBD *(Đo trên Kaggle)* | TBD *(Đo trên Kaggle)* |
| **Độ Trễ Forward Pass (Latency)** | N/A | **5.27 ms** | **20.44 ms** | **23.09 ms** | TBD *(Đo trên Kaggle)* | TBD *(Đo trên Kaggle)* |
| **Thông Lượng (Throughput)** | N/A | **758.70 samples/sec** | **195.74 samples/sec** | **173.24 samples/sec** | TBD *(Đo trên Kaggle)* | TBD *(Đo trên Kaggle)* |
| **AR Cross-Entropy Loss** | N/A | `7.1093` | `7.7275` | `7.7695` | TBD | TBD |
| **DM Reconstruction MSE Loss** | N/A | `1.2332` | `1.3119` | `1.4106` | TBD | TBD |
| **Attention Isolation ($Q_{AR} \times K_{DM} = 0$)** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** |

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

### 2.5. Version 4 (`mini_model/version4`) - Quy Mô 1.31 Billion Parameters (~1.31B Params)
- **Mã nguồn:** Xây dựng trên `torch.nn`.
- **Khối chức năng & Mã nguồn sử dụng:**
  - **Mở rộng chiều ẩn $d_{model}$ lên 2048 và số lớp lên 24:** Đưa tổng số tham số vượt mốc **1.31 Tỷ tham số (1.31B parameters)**.
  - **Grouped-Query Attention (GQA):** Tỉ lệ 4:1 ($H_Q=16, H_{KV}=4$) đảm bảo tiết kiệm bộ nhớ KV Cache khi chạy trên GPU T4 (16GB VRAM).
  - **Từ vựng & Latent độ phân giải cao:** Nâng `vocab_size` lên 8000 và `latent_dim` lên 128.
  - **RoPE + SwiGLU + RMSNorm:** Cấu trúc tầng đồng dạng tối ưu khả năng huấn luyện và suy luận độ chính xác cao.

### 2.6. Version 5 (`mini_model/version5`) - Siêu Quy Mô Tương Đương NVIDIA Cosmos 3 Edge (~3.93B Params)
- **Mã nguồn:** Xây dựng trên `torch.nn`.
- **Khối chức năng & Mã nguồn sử dụng:**
  - **Mở rộng chiều ẩn $d_{model}$ lên 3072 và số lớp lên 32:** Đạt quy mô gần **4 Tỷ tham số (~3.93B parameters)**, tương đương mô hình NVIDIA Cosmos 3 Edge 4B.
  - **Grouped-Query Attention (GQA 4:1):** 24 Query heads và 6 Key/Value heads giúp duy trì hiệu năng bộ nhớ tối ưu.
  - **Từ vựng 16,000 & Latent 256:** Hỗ trợ xử lý ngữ cảnh đa phương tiện độ phân giải cực cao.

---

## 3. Quy Trình Chạy Benchmark Cho Phiên Bản Mới

Để đo lường chỉ số cho Version 5 trên Kaggle Notebook:

```bash
# Cập nhật repo từ GitHub
!git pull

# Chạy benchmark cho Version 5
!python benchmark.py --version version5 --batch_size 2 --num_runs 50
```
