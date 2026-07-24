# 🌌 Mini Cosmos 3: World Model Platform (NVIDIA Cosmos 3 Architecture)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![NVIDIA Cosmos Paper](https://img.shields.io/badge/NVIDIA-Cosmos_Technical_Report-76b900.svg)](https://arxiv.org/abs/2501.03575)

Hệ thống mô hình mô phỏng thế giới đa phương tiện (**World Foundation Model**) độc lập, mô phỏng lại toàn bộ các phát minh kiến trúc tiên tiến nhất của bài báo khoa học **NVIDIA Cosmos 3** ([Cosmos World Foundation Model Platform](https://arxiv.org/abs/2501.03575)).

Dự án được thiết kế theo lộ trình tiến hóa **10 phiên bản kiến trúc** (từ `version0` đến `version9`), mở rộng dung lượng từ **20M** lên **8.12B** và đỉnh cao là **Version 9 Mixture-of-Experts (~7.20B Total / ~4.03B Active Params)**.

---

## 📂 Cấu Trúc Thư Mục Dự Án (Repository Structure)

```text
cosmos/
├── README.md                      # Hướng dẫn chính của dự án
├── VERSION_COMPARISON.md          # Báo cáo thực nghiệm chi tiết & so sánh 10 phiên bản (V0 -> V9)
├── REPORT_4B_COMPARISON.md        # Báo cáo so sánh đối đầu chi tiết mốc 4B (V5 vs V8 vs V9)
├── cosmos_architecture.md         # Sơ đồ thiết kế kiến trúc chuẩn NVIDIA Cosmos 3
├── benchmark.py                   # Bộ đo lường tự động (Benchmark Suite)
├── dataset_loader.py              # Bộ nạp dữ liệu đa phương tiện từ Hugging Face
├── evaluate_pilot_dataset.py      # Script đánh giá Forward Latency & Loss trên Pilot Dataset
├── train_pilot_dataset.py         # Script huấn luyện thử nghiệm chống NaN/OOM VRAM (1,000 steps)
├── train_production.py            # Script huấn luyện sản xuất thực tế (5,000 - 20,000 steps + Checkpointing)
├── inference_production.py        # Script nạp Checkpoint .pt và thực thi suy luận sản xuất
├── mini_model/
│   ├── version0/                  # Cosmos 3 Baseline (Dense Base 7.24B Dual-GPU Pipeline)
│   ├── version1/                  # Mini PoC (20.49M, FP32 Single-GPU)
│   ├── version2/                  # Scaled LLM (127.99M, FP32 Single-GPU)
│   ├── version3/                  # GQA + RoPE (140.57M, GQA 4:1, RoPE Position Embedding)
│   ├── version4/                  # 1.34B Scale (1.34B Params, FP32 Single-GPU)
│   ├── version5/                  # 4.03B Base FP16 (4.03B Params, Dual-GPU FP16)
│   ├── version6/                  # 7.24B Single GPU (Chạm trần VRAM 98.2% trên 1 GPU T4)
│   ├── version7/                  # 8.12B Dual GPU (Meta Device Init 0 MB System RAM, Dual GPU)
│   ├── version8/                  # 4.03B QK-Norm + LayerScale (FP16 QK-Norm, LayerScale)
│   └── version9/                  # Unified MoE World Model (~7.20B Total / ~4.03B Active Params)
└── requirements.txt
```

---

## 📊 Bảng So Sánh Chỉ Số 10 Phiên Bản Kiến Trúc (Version 0 -> Version 9)

| Thông Số Đánh Giá | Version 0 (Baseline) | Version 5 (4.03B Base) | Version 7 (8.12B Dual-GPU) | Version 8 (4.03B QK-Norm) | Version 9 (MoE ~7.20B Total) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Parameters** | **7,244.92 M (7.24B)** | **4,026.76 M (4.03B)** | **8,117.37 M (8.12B)** | **4,026.97 M (4.03B)** | **7,204.35 M (7.20B)** |
| **Active Parameters** | **7.24B** | **4.03B** | **8.12B** | **4.03B** | **~4.03B (Top-2 Experts)** |
| **GPU Hardware** | **2x GPU T4 (Dual GPU)** | **2x GPU T4** | **2x GPU T4 (Dual GPU)** | **2x GPU T4** | **2x GPU T4 (Dual GPU)** |
| **Precision** | **FP16 (Meta Init)** | **FP16** | **FP16 (Meta Init)** | **FP16** | **FP16 (Meta Init)** |
| **Peak VRAM Usage** | **7,632.69 MB** | **8,443.60 MB** | **8,521.79 MB** | **8,552.24 MB** | **8,610.12 MB** |
| **Forward Latency** | **90.53 ms** | **69.68 ms** | **102.22 ms** | **72.63 ms** | **78.45 ms** |
| **Throughput (fps)** | **22.09 fps** | **28.70 fps** | **19.57 fps** | **27.54 fps** | **25.49 fps** |
| **DM MSE Loss** | `353.0143` | `1.3329` | `399.7594` | **`1.3032`** | **`1.2854` [TỐI ƯU CỰC ĐẠI]** |
| **Router Aux Loss** | N/A | N/A | N/A | N/A | **`1.2917` [BALANCED]** |
| **Attention Isolation** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** | **PASSED [OK]** |

---

## ⚡ Hướng Dẫn Chạy Đánh Giá & Huấn Luyện Sản Xuất

### 1. Đánh giá Tốc độ & Loss trên Kaggle Notebook (GPU T4 / Dual GPU T4)

```bash
# Cập nhật repo từ GitHub
!git pull

# 1. Đánh giá Forward Latency & Loss trên Version 9 MoE World Model
!python evaluate_pilot_dataset.py --version version9 --steps 10 --batch_size 2 --fp16

# 2. Chạy huấn luyện thử nghiệm 1,000 steps cho Version 9
!python train_pilot_dataset.py --version version9 --steps 1000 --batch_size 1 --accum_steps 4 --lr 1e-6 --log_every 50
```

### 2. Huấn luyện Thực tế (Production Fine-Tuning 5,000 - 20,000 Steps) & Vận Hành Suy Luận

```bash
# 1. Chạy Fine-Tuning 5,000 steps thực tế, tự động lưu Checkpoint .pt mỗi 1,000 steps vào ./checkpoints/
!python train_production.py --version version9 --steps 5000 --save_every 1000 --lr 1e-5

# 2. (Tùy chọn) Resume Fine-Tuning tiếp tục từ Checkpoint đã lưu
!python train_production.py --version version9 --steps 10000 --resume_from ./checkpoints/cosmos3_version9_step05000.pt

# 3. Nạp Checkpoint đã train (.pt) để thực thi suy luận sinh dự đoán thực tế
!python inference_production.py --version version9 --checkpoint ./checkpoints/cosmos3_version9_step05000.pt --samples 5
```

---

## 📜 Tài Liệu Báo Cáo Kỹ Thuật Chi Tiết

* **[VERSION_COMPARISON.md](VERSION_COMPARISON.md):** Báo cáo thực nghiệm chi tiết và phân tích sâu 10 phiên bản.
* **[REPORT_4B_COMPARISON.md](REPORT_4B_COMPARISON.md):** Báo cáo so sánh đối đầu chi tiết giữa Version 5, Version 8 và Version 9 ở mốc 4B Active Params.
* **[cosmos_architecture.md](cosmos_architecture.md):** Thiết kế sơ đồ nguyên lý của nền tảng NVIDIA Cosmos 3.

---

## 📄 License
Dự án được phân phối dưới giấy phép [Apache 2.0 License](LICENSE).
