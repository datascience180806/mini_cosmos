# CẤU TRÚC THƯ MỤC COOKBOOKS/COSMOS3

> **Nguồn từ đường dẫn:** [NVIDIA/cosmos/cookbooks/cosmos3](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3)

Thư mục `cookbooks/cosmos3` chứa các tài liệu hướng dẫn, Jupyter Notebooks, ví dụ thực thi và kịch bản huấn luyện/suy luận cho hai chế độ vận hành chính của mô hình **Cosmos 3** (Generator Mode và Reasoner Mode).

---

## 1. Sơ Đồ Cấu Trúc Thư Mục

```text
cookbooks/cosmos3/
├── README.md                            # Hướng dẫn tổng quan về các Cookbook và cách bắt đầu với Cosmos 3
├── cosmos3-model-architecture.png       # Sơ đồ biểu diễn kiến trúc tổng thể của Cosmos 3
│
├── generator/                           # Thư mục tài nguyên cho Generator Mode (Tầng khuếch tán - Diffusion)
│   ├── action/                          # Hướng dẫn và ví dụ về sinh chuỗi hành động điều khiển (Policy / Action)
│   │   ├── README.md                    # Hướng dẫn thực thi Action Generator
│   │   └── assets/                      # Dữ liệu mẫu (tập tin JSON hành động, Parquet, video MP4, ảnh, prompts)
│   │
│   ├── audiovisual/                     # Hướng dẫn sinh và fine-tune video, hình ảnh và âm thanh
│   │   └── README.md                    # Hướng dẫn fine-tuning, nén mô hình (distillation) và inference với SGLang, Diffusers
│   │
│   └── transfer/                        # Hướng dẫn chuyển đổi mô hình và thích ứng miền dữ liệu (Model Transfer)
│       └── README.md
│
└── reasoner/                            # Thư mục tài nguyên cho Reasoner Mode (Tầng tự hồi quy - Autoregressive)
    ├── README.md                        # Hướng dẫn tổng quan và triển khai Reasoner
    ├── reasoner_prompt_guide.md         # Cẩm nang kỹ thuật viết prompt tối ưu cho Reasoner
    ├── run_with_cosmos_framework.ipynb   # Notebook chạy Reasoner với khung cosmos_framework chính thức
    ├── run_with_nim.ipynb                # Notebook thực thi suy luận Reasoner qua NVIDIA NIM microservices
    ├── run_with_transformers.ipynb       # Notebook thực thi Reasoner bằng thư viện HuggingFace Transformers
    ├── run_with_vllm.ipynb               # Notebook chạy suy luận tốc độ cao với framework vLLM
    ├── assets/                          # Tài nguyên và hình ảnh minh họa cho các ví dụ Reasoner
    └── finetune/                        # Hướng dẫn và kịch bản fine-tune mô hình Reasoner trên tập dữ liệu riêng
```

---

## 2. Mô Tả Chi Tiết Các Thành Phần Chính

### 2.1. Thư mục `generator/` (Dành cho Generator Mode)
Tập trung vào các tác vụ sinh đa phương tiện và điều khiển chuỗi hành động:
- **`action/`**: Chứa các ví dụ thực tế nạp dữ liệu định dạng LeRobot (từ các robot DROID, Bridge, AgiBot) để sinh chuỗi hành động điều khiển cho robot và phương tiện.
- **`audiovisual/`**: Cung cấp hướng dẫn sinh video kèm âm thanh đồng bộ, kịch bản tinh chỉnh (fine-tuning) và chắt lọc kiến thức (distillation) để tăng tốc độ sinh.
- **`transfer/`**: Hướng dẫn chuyển giao trọng số và thích ứng mô hình sinh vào các miền dữ liệu công nghiệp cụ thể.

### 2.2. Thư mục `reasoner/` (Dành cho Reasoner Mode)
Tập trung vào các tác vụ nhận thức thị giác, định vị không gian và suy luận logic:
- **Notebooks thực thi đa dạng:** Hỗ trợ người dùng chạy mô hình Reasoner thông qua nhiều framework khác nhau tùy thuộc hạ tầng sẵn có:
  - `run_with_transformers.ipynb`: Phù hợp cho thử nghiệm nhanh với PyTorch/HuggingFace.
  - `run_with_vllm.ipynb`: Phù hợp cho phục vụ suy luận throughput lớn với vLLM.
  - `run_with_nim.ipynb`: Phù hợp cho triển khai microservice chuẩn doanh nghiệp với NVIDIA NIM.
  - `run_with_cosmos_framework.ipynb`: Sử dụng SDK chính thức của NVIDIA Cosmos.
- **`reasoner_prompt_guide.md`**: Hướng dẫn cấu trúc prompt chuẩn để đạt hiệu quả nhận thức vật lý và lập kế hoạch tốt nhất.
- **`finetune/`**: Hướng dẫn tinh chỉnh lớp AR trên dữ liệu chuyên biệt của doanh nghiệp.
