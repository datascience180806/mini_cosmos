# Cosmos 3 Toy Model (Proof-of-Concept for Kaggle / Local)

Dự án này chứa mã nguồn thực nghiệm phiên bản thu nhỏ (**Mini / Toy Model ~50M-100M parameters**) của kiến trúc **NVIDIA Cosmos 3 (Mixture-of-Transformers - MoT)**. 

Mục đích chính của dự án là **tìm hiểu, kiểm thử luồng chạy và xác minh tính đúng đắn của ma trận Attention Mask hợp nhất (AR + DM)** trên Kaggle Notebooks hoặc môi trường GPU cá nhân với dung lượng VRAM siêu nhẹ (< 2GB - 4GB VRAM).

---

## 1. Cấu Trúc Thư Mục Dự Án

```text
cosmos/
├── .gitignore                      # Cấu hình bỏ qua các file tạm / checkpoints
├── README.md                       # Hướng dẫn chạy trên Kaggle / Local
├── requirements.txt                # Thư viện phụ thuộc cơ bản (torch, numpy, tqdm)
├── report.md                       # Báo cáo lý thuyết chi tiết về kiến trúc Cosmos 3
├── sample_architecture.md          # Sơ đồ thư mục kho mã nguồn NVIDIA Cosmos
│
└── mini_model/
    └── version1/
        ├── __init__.py             # Module init
        ├── model.py                # Định nghĩa kiến trúc Cosmos 3 Toy Model (MoT, Attention Mask, Encoders, Blocks)
        ├── train_toy.py            # Kịch bản huấn luyện thử nghiệm (Dummy Data Forward & Backward Pass)
        └── inference_toy.py        # Kịch bản chạy thử hai chế độ: Reasoner Mode và Generator Mode
```

---

## 2. Hướng Dẫn Chạy Trên Kaggle Notebook

### Bước 1: Cài Đặt Thư Viện
Trên notebook Kaggle (bật GPU T4 hoặc P100 / CPU):

```python
!pip install -r requirements.txt
```

### Bước 2: Chạy thử nghiệm Huấn luyện (Train Step Test)
Kiểm tra luồng Forward, Backward, và tính toán loss kết hợp giữa AR (Cross-Entropy) và DM (Diffusion MSE):

```python
!python mini_model/version1/train_toy.py
```

### Bước 3: Chạy thử nghiệm Suy luận (Inference Dual Modes Test)
Thử nghiệm 2 chế độ vận hành **Reasoner Mode** (dự đoán token văn bản tiếp theo) và **Generator Mode** (sinh dữ liệu liên tục/khử nhiễu):

```python
!python mini_model/version1/inference_toy.py
```

---

## 3. Đặc Điểm Kiến Trúc Toy Model (`mini_model/version1/model.py`)

- **$d_{model}$:** $512$
- **Số lớp ($L$):** $6$ layers
- **Attention Heads:** $8$ heads
- **Attention Mask Matrix:**
  - $Q_{AR} \times K_{AR}$: Causal Self-Attention (dạng tam giác dưới).
  - $Q_{AR} \times K_{DM}$: Masked Zero (0) - Chặn nhiễu sinh ảnh/video ảnh hưởng logic suy luận.
  - $Q_{DM} \times [K_{AR}, K_{DM}]$: Full Attention - Cho phép nhánh sinh tiếp nhận toàn bộ ngữ cảnh AR và tương tác giữa các modality.
- **Dung lượng VRAM tiêu tốn:** `< 2 GB VRAM` khi chạy ngẫu nhiên 1 batch.
