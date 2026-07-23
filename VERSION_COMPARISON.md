# BÁO CÁO ĐÁNH GIÁ VÀ SO SÁNH CÁC PHIÊN BẢN (VERSION COMPARISON REPORT)

> **Dự án:** `mini_cosmos` - Unified World Model Architecture Experiments  
> **Mục tiêu:** Đánh giá hiệu năng, mức tiêu thụ tài nguyên và độ chính xác kiến trúc giữa các phiên bản thử nghiệm.

---

## 1. Bảng So Sánh Tổng Quan (Benchmark Matrix)

| Thông Số Benchmark | Version 1 (Baseline) | Version 2 (Planned) | Version 3 (Planned) |
| :--- | :---: | :---: | :---: |
| **Tổng Số Tham Số (Params)** | **~74.5 M** | TBD | TBD |
| **Kích Thước Cấu Hình ($d_{model} / L / H$)** | $512 / 6 / 8$ | TBD | TBD |
| **Mức Tiêu Thụ VRAM Peak (MB)** | **~290 MB** (FP32) | TBD | TBD |
| **Độ Trễ Forward Pass (Latency)** | **~12.5 ms** (Batch 4) | TBD | TBD |
| **Thông Lượng (Throughput)** | **~320 samples/sec** | TBD | TBD |
| **AR Cross-Entropy Loss** | `6.91` (Baseline) | TBD | TBD |
| **DM Reconstruction MSE Loss** | `1.02` (Baseline) | TBD | TBD |
| **Attention Isolation ($Q_{AR} \times K_{DM} = 0$)** | **PASSED [OK]** | TBD | TBD |

---

## 2. Chi Tiết Các Phiên Bản Thử Nghiệm

### 2.1. Version 1 (`mini_model/version1`) - Baseline Model
- **Mô tả:** Phiên bản PoC thu nhỏ chuẩn kiến trúc NVIDIA Cosmos 3 Mixture-of-Transformers (MoT).
- **Điểm nổi bật:**
  - Tích hợp Ma trận Attention Mask hợp nhất ($Q_{AR} \times K_{AR}$ Causal, $Q_{AR} \times K_{DM}$ Masked Zero, $Q_{DM} \times [K_{AR}, K_{DM}]$ Full Attention).
  - Hỗ trợ chạy đa chế độ suy luận: **Reasoner Mode** (AR) và **Generator Mode** (DM + Action).
  - Chạy mượt trên mọi thiết bị (VRAM `< 2GB` hoặc CPU).

### 2.2. Kế Hoạch Cho Các Phiên Bản Tiếp Theo
- **Version 2 (`mini_model/version2`):**
  - Thử nghiệm tích hợp **Rotary Position Embeddings (RoPE)** cho chuỗi AR.
  - Thử nghiệm **Grouped-Query Attention (GQA)** để giảm dung lượng KV Cache và tăng throughput.
- **Version 3 (`mini_model/version3`):**
  - Thử nghiệm **FlashAttention-2** hoặc **Chế độ Lượng tử hóa (FP8 / INT4)**.
  - Tăng nhẹ số lớp $L=12, d_{model}=768$ (~150M params) để đo lường giới hạn scaling trên Kaggle T4 GPU.

---

## 3. Quy Trình Chạy Benchmark Cho Phiên Bản Mới

Để đo lường và cập nhật chỉ số cho phiên bản mới, thực hiện lệnh sau:

```bash
# Đánh giá Version 1
python benchmark.py --version version1 --batch_size 4 --num_runs 50

# Đánh giá Version 2 (sau khi tạo thư mục mini_model/version2)
python benchmark.py --version version2 --batch_size 4 --num_runs 50
```

Các kết quả chỉ số sẽ tự động được ghi lại vào file `benchmark_results.json`.
