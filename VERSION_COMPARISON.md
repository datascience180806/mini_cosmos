# BÁO CÁO ĐÁNH GIÁ VÀ SO SÁNH CÁC PHIÊN BẢN (VERSION COMPARISON REPORT)

> **Dự án:** `mini_cosmos` - Unified World Model Architecture Experiments  
> **Mục tiêu:** Đánh giá hiệu năng, mức tiêu thụ tài nguyên và độ chính xác kiến trúc giữa các phiên bản thử nghiệm.

---

## 1. Bảng So Sánh Tổng Quan (Benchmark Matrix)

| Thông Số Benchmark | Version 1 (Baseline) | Version 2 (Scaled Params) | Version 3 (Planned) |
| :--- | :---: | :---: | :---: |
| **Tổng Số Tham Số (Params)** | **20.49 M** | **~85 M - 100 M** | TBD |
| **Kích Thước Cấu Hình ($d_{model} / L / H$)** | $512 / 6 / 8$ | $1024 / 8 / 16$ | TBD |
| **Mức Tiêu Thụ VRAM Peak (MB)** | **139.78 MB** | TBD *(Đo trên Kaggle)* | TBD |
| **Độ Trễ Forward Pass (Latency)** | **5.27 ms** (Batch 4) | TBD *(Đo trên Kaggle)* | TBD |
| **Thông Lượng (Throughput)** | **758.7 samples/sec** | TBD *(Đo trên Kaggle)* | TBD |
| **AR Cross-Entropy Loss** | `7.1093` | TBD | TBD |
| **DM Reconstruction MSE Loss** | `1.2332` | TBD | TBD |
| **Attention Isolation ($Q_{AR} \times K_{DM} = 0$)** | **PASSED [OK]** | **PASSED [OK]** | TBD |

---

## 2. Chi Tiết Các Phiên Bản Thử Nghiệm

### 2.1. Version 1 (`mini_model/version1`) - Baseline Model (~20.49M Params)
- **Mô tả:** Phiên bản PoC thu nhỏ chuẩn kiến trúc NVIDIA Cosmos 3 Mixture-of-Transformers (MoT).
- **Kết quả thực tế (Kaggle Benchmark):**
  - Total Params: **20.49 M**
  - Peak VRAM: **139.78 MB**
  - Latency: **5.27 ms / batch**
  - Throughput: **758.7 samples/sec**
  - AR Loss: **7.1093** | DM MSE Loss: **1.2332**
  - Attention Isolation Check: **PASSED (100% cách ly nhiễu sinh khỏi nhánh AR)**

### 2.2. Version 2 (`mini_model/version2`) - Scaled Model (~95M Params)
- **Mục tiêu:** Nâng cấp quy mô tham số để đánh giá khả năng mở rộng (Scaling Law) của MoT:
  - Nâng chiều $d_{model}$ từ **$512 \rightarrow 1024$**.
  - Nâng số lớp Transformer từ **$6 \rightarrow 8$ layers**.
  - Nâng số Attention Heads từ **$8 \rightarrow 16$ heads**.
  - Tích hợp thêm **RMSNorm** và **SiLU SwiGLU MLP** để tăng tính ổn định khi mở rộng quy mô.

---

## 3. Quy Trình Chạy Benchmark Cho Phiên Bản Mới

Để đo lường chỉ số cho Version 2 trên Kaggle Notebook:

```bash
# Cập nhật repo từ GitHub
!git pull

# Chạy benchmark cho Version 2
!python benchmark.py --version version2 --batch_size 4 --num_runs 50
```
