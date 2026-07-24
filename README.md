# mini_cosmos: Unified World Model Architecture Experiments

Dự án này là bộ nghiên cứu và thực nghiệm kiến trúc mô hình thu nhỏ (**Mini / Toy Model / Architecture Shells từ 20M đến 8.12B parameters**) của nền tảng **NVIDIA Cosmos 3 (Mixture-of-Transformers - MoT)**.

Mục đích chính của dự án là **nghiên cứu, đo đạc phần cứng (Hardware Profiling Benchmark), tối ưu dung lượng VRAM/Latency và kiểm thử màng chắn chú ý cách ly (Attention Isolation Mask)** trên Kaggle Notebooks (1x GPU T4 hoặc Dual GPU T4) trước khi bước vào giai đoạn huấn luyện chính thức trên dữ liệu thật.

---

## 1. Cấu Trúc Dự Án & Tiến Độ Hiện Tại

```text
cosmos/
├── .gitignore                      # Cấu hình bỏ qua các file tạm / checkpoints
├── README.md                       # Hướng dẫn tổng quan & khởi chạy trên Kaggle / Local
├── VERSION_COMPARISON.md           # Báo cáo so sánh 9 phiên bản (Version 0 đến Version 8)
├── REPORT_4B_COMPARISON.md         # Báo cáo thử nghiệm đối đầu 1-to-1 ở mốc 4B (V5 vs V8)
├── cosmos_architecture.md          # Sơ đồ khối kiến trúc Mermaid của Cosmos 3 (Super, Nano, Edge)
├── benchmark.py                    # Script đo Latency, Peak VRAM, Throughput, AR/DM Loss, Isolation Mask
├── benchmark_results.json          # Kết quả lưu trữ benchmark tự động
├── requirements.txt                # Thư viện phụ thuộc cơ bản (torch, numpy, tqdm)
│
└── mini_model/
    ├── version0/                   # Cosmos 3 Nano Baseline Shell (7.24B Dense, Meta Device Init, Dual GPU)
    ├── version1/                   # Mini PoC Baseline (20.5M Params FP32)
    ├── version2/                   # SwiGLU FFN + RMSNorm (128M Params FP32)
    ├── version3/                   # Grouped-Query Attention (GQA 4:1) + RoPE (141M Params FP32)
    ├── version4/                   # Small LLM Scale (1.34B Params FP32)
    ├── version5/                   # Cosmos 3 Edge FP16 Base (4.03B Params FP16)
    ├── version6/                   # Single-GPU Max Wall (7.24B Params FP16 - 14.31GB VRAM)
    ├── version7/                   # Dual-GPU Pipeline Parallelism + Meta Device Init (8.12B Params FP16)
    └── version8/                   # QK-Norm + LayerScale Architecture (4.03B Params FP16 - Ablation Study)
```

---

## 2. Bảng Tóm Tắt So Sánh Các Phiên Bản (20M - 8.12B)

> [!NOTE]
> Các chỉ số dưới đây được đo đạc dựa trên khung xương kiến trúc chưa qua huấn luyện (Untrained Architecture Shells) nhằm mục đích kiểm tra tài nguyên phần cứng (VRAM, Latency, Throughput), chứ không đại diện cho chỉ số độ chính xác sau khi train.

| Phiên Bản | Quy Mô Tham Số | Định Dạng / Cấu HÌnh GPU | Peak VRAM | Latency (ms) | Throughput (fps) | Tính Năng Nổi Bật |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Version 0** | **7.24B** | FP16 (Dual T4 GPUs) | **7.63 GB/GPU** | **90.53 ms** | **22.09 fps** | Cosmos 3 Nano Dense Meta Shell, Dual GPU Pipeline |
| **Version 1** | **20.5M** | FP32 (Single GPU) | 139.78 MB | 5.27 ms | 758.70 fps | Mini PoC Baseline, Standard GELU & LayerNorm |
| **Version 2** | **128M** | FP32 (Single GPU) | 702.38 MB | 20.44 ms | 195.74 fps | Nâng cấp SwiGLU FFN & RMSNorm |
| **Version 3** | **141M** | FP32 (Single GPU) | 810.40 MB | 23.09 ms | 173.24 fps | Tối ưu Grouped-Query Attention (GQA 4:1) & RoPE |
| **Version 4** | **1.34B** | FP32 (Single GPU) | 6.42 GB | 208.90 ms | 19.15 fps | Thử nghiệm quy mô Small LLM Scale |
| **Version 5** | **4.03B** | FP16 (Dual T4 GPUs) | **8.44 GB** | **69.68 ms** | **28.70 fps** | Quy mô Cosmos 3 Edge FP16 Base |
| **Version 6** | **7.24B** | FP16 (Single T4 GPU) | 14.31 GB | 86.58 ms | 11.55 fps | Chạm trần trần phần cứng 1x T4 (98.2% VRAM) |
| **Version 7** | **8.12B** | FP16 (Dual T4 GPUs) | 8.52 GB/GPU | 102.22 ms | 19.57 fps | Meta Device Init (0 MB CPU RAM), Pipeline 36 layers |
| **Version 8** | **4.03B** | FP16 (Dual T4 GPUs) | **8.55 GB** | **72.63 ms** | **27.54 fps** | **QK-Norm + LayerScale** (Giảm MSE Loss, chống tràn FP16) |

---

## 3. Hướng Dẫn Chạy Benchmark Trên Kaggle / Local

### Bước 1: Chuẩn bị môi trường & Repo
```bash
git clone git@github.com:datascience180806/mini_cosmos.git
cd mini_cosmos
pip install -r requirements.txt
```

### Bước 2: Chạy Benchmark cho từng phiên bản

```bash
# 1. Benchmark Version 5 (Base 4B FP16)
python benchmark.py --version version5 --batch_size 2 --num_runs 50 --fp16

# 2. Benchmark Version 8 (QK-Norm + LayerScale 4B FP16)
python benchmark.py --version version8 --batch_size 2 --num_runs 50 --fp16

# 3. Benchmark Version 0 (Cosmos 3 Nano Meta Shell 7.24B trên 2 GPU)
python benchmark.py --version version0 --batch_size 2 --num_runs 50 --fp16 --multi_gpu
```

---

## 4. Các Tài Liệu Báo Cáo Kỹ Thuật

* 📄 **[VERSION_COMPARISON.md](file:///c:/Users/Admin/Documents/reasearch/cosmos/VERSION_COMPARISON.md):** Báo cáo so sánh chi tiết bảng chỉ số thực nghiệm và 6 bài học kỹ thuật rút ra từ V0 đến V8.
* 📄 **[REPORT_4B_COMPARISON.md](file:///c:/Users/Admin/Documents/reasearch/cosmos/REPORT_4B_COMPARISON.md):** Báo cáo thử nghiệm đối đầu trực tiếp (Ablation Study) ở mốc 4B giữa V5 (Base) và V8 (QK-Norm + LayerScale).
* 📐 **[cosmos_architecture.md](file:///c:/Users/Admin/Documents/reasearch/cosmos/cosmos_architecture.md):** Sơ đồ khối kiến trúc Mermaid chi tiết cho 3 dòng mô hình: **Cosmos 3 Edge (4B)**, **Cosmos 3 Nano (16B)** và **Cosmos 3 Super (64B)**.
