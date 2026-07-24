# KIẾN TRÚC TỔNG QUAN NỀN TẢNG NVIDIA COSMOS 3 (NVIDIA COSMOS 3 ARCHITECTURE GUIDE)

> **Tài liệu:** Tổng hợp sơ đồ khối kiến trúc của 3 dòng mô hình chính thuộc họ **NVIDIA Cosmos 3**: **Cosmos 3 Super (64B)**, **Cosmos 3 Nano (16B)**, và **Cosmos 3 Edge (4B)**.

---

## 1. Bảng So Sánh Tổng Quan 3 Mô Hình Cosmos 3

| Tiêu Chí Kỹ Thuật | **Cosmos 3 Edge (4B)** | **Cosmos 3 Nano (16B)** | **Cosmos 3 Super (64B)** |
| :--- | :---: | :---: | :---: |
| **Quy Mô Tham Số (Params)** | **4 Tỷ (4B Dense)** | **16 Tỷ (8B Dense + MoE Experts)** | **64 Tỷ (MoE Multi-Cluster)** |
| **Môi Trường Triển Khai** | Máy tính nhúng Xe tự hành & Robot | Server nội bộ nhà máy (On-Premise) | Siêu máy tính Trung tâm dữ liệu (Datacenter) |
| **Thiết Bị Phần Cứng Mục Tiêu** | NVIDIA DRIVE Thor / Jetson AGX Orin | 2x - 4x GPU T4 / A100 / H100 | Cụm máy chủ NVIDIA DGX H100 / B200 |
| **Dung Lượng VRAM Tiêu Thụ** | **~8.5 GB đến 12 GB VRAM** | **~16 GB đến 32 GB VRAM** | **~128 GB đến 256 GB VRAM** |
| **Tần Số Điều Khiển (Control Rate)** | **15 Hz - 60 Hz (Thời gian thực)** | **10 Hz - 20 Hz** | Khử nhiễu offline / Batch Processing |
| **Ứng Dụng Chính** | Tự lái xe, điều khiển robot tại chỗ | Quản lý kho hàng, suy luận nhà máy | Huấn luyện mô phỏng Digital Twin 3D |

---

## 2. Sơ Đồ Khối Kiến Trúc Chi Tiết Của 3 Mô Hình

---

### 2.1. Kiến Trúc COSMOS 3 EDGE (~4B Dense - Dành Cho Máy Tính Nhúng Xe Tự Hành & Robot)

* **Đặc điểm:** Kiến trúc thuần **Dense Backbone (4 Tỷ tham số)** nén chuẩn FP16, tối ưu cho các thiết bị Edge yêu cầu phản hồi tức thì dưới $30\text{ ms}$.

```mermaid
flowchart TD
    subgraph EDGE_INPUTS["1. ĐẦU VÀO ĐA CẢM BIẾN (EDGE INPUTS)"]
        E_CAM["Camera HD $360^\circ$ trên xe/robot"] --> E_VAE["Bộ Nén VAE 3D (Thu Nhỏ 8x8x8)"]
        E_TEXT["Câu Lệnh Văn Bản / Biển Báo"] --> E_TXT_EMB["Bộ Nén Chữ Llama-3 (3072 dim)"]
        E_ACT["Trạng Thái Tay Robot / Vô-lăng"] --> E_ACT_PROJ["Linear Projection (3072 dim)"]
    end

    E_VAE --> E_CAT["Gộp Chuỗi Dữ Liệu Đa Phương Tiện (d_model = 3072)"]
    E_TXT_EMB --> E_CAT
    E_ACT_PROJ --> E_CAT

    subgraph EDGE_BACKBONE["2. BỘ NÃO AI LÕI DENSE 4B (32 TẦNG TRANSFORMER)"]
        E_CAT --> E_ROPE["Nhúng Tọa Độ Vị Trí Xoay (RoPE)"]
        E_ROPE --> E_MASK["Cosmos 3 Attention Mask Matrix\n(Chặn rò rỉ nhiễu sinh ảnh vào suy luận chữ)"]
        
        E_MASK --> E_BLOCK["32x Khối Transformer Dense FP16\n- RMSNorm Layer\n- Grouped-Query Attention (GQA 4:1, H_Q=24, H_KV=6)\n- RMSNorm Layer\n- SwiGLU FFN Layer (3072 -> 10752 -> 3072)"]
    end

    E_BLOCK --> E_NORM["Tầng Chuẩn Hóa Cuối RMSNorm (norm_f)"]

    subgraph EDGE_OUTPUTS["3. ĐẦU RA THỜI GIAN THỰC (REAL-TIME OUTPUTS)"]
        E_NORM --> |Tách Chuỗi Văn Bản| E_REASONER["Chế Độ Suy Luận (Reasoner Surface)\nOutput: Lệnh rẽ/phanh, kế hoạch di chuyển"]
        E_NORM --> |Tách Chuỗi Khung Hình| E_GENERATOR["Chế Độ Sinh Hành Động (Generator Surface)\nOutput: Lực vặn mô-tơ 60Hz & Ảnh dự đoán 2 giây tới"]
    end
```

---

### 2.2. Kiến Trúc COSMOS 3 NANO (~16B Total - Dành Cho Máy Chủ Server Nội Bộ Nhà Máy)

* **Đặc điểm:** Kiến trúc kết hợp giữa **Lõi Trung Tâm 8B (Dense Backbone)** và các **Tầng Chuyên Gia MoE (Mixture-of-Experts ~8B)** giúp mở rộng tri thức nhà máy nhưng vẫn duy trì tốc độ xử lý nhanh.

```mermaid
flowchart TD
    subgraph NANO_INPUTS["1. ĐẦU VÀO ĐA PHƯƠNG TIỆN NANO (16B SCALE)"]
        N_CAM["Camera Giám Sát Dây Truyền Nhà Máy"] --> N_VAE["Bộ Nén Video 3D VAE Codec"]
        N_PROMPT["Yêu Cầu Vận Hành / Kế Hoạch Kho Hàng"] --> N_TXT["Text Encoder (T5-XXL / Llama-3 8B)"]
        N_ROBOT["Tín Hiệu Cảm Biến Tay Robot Gắp Hàng"] --> N_ACT["Action Projection (4096 dim)"]
    end

    N_VAE --> N_CAT["Gộp Chuỗi Thông Tin (d_model = 4096)"]
    N_TXT --> N_CAT
    N_ACT --> N_CAT

    subgraph NANO_BACKBONE["2. BỘ NÃO HỖN HỢP CHUYÊN GIA MoE (32 TẦNG)"]
        N_CAT --> N_MASK["Cosmos 3 Attention Mask Matrix"]
        N_MASK --> N_ROPE["Apply Rotary Position Embedding (RoPE)"]
        
        N_ROPE --> N_ROUTER["Mạng Điều Phối Chuyên Gia (MoE Router Network)"]
        
        subgraph NANO_EXPERTS["Các Tầng Chuyên Gia Chuyên Biệt (MoE Expert Layers ~8B)"]
            N_ROUTER --> N_EXP1["Chuyên Gia 1: Vật Lý & Động Lực Học Video"]
            N_ROUTER --> N_EXP2["Chuyên Gia 2: Lập Kế Hoạch & Suy Luận Chữ"]
            N_ROUTER --> N_EXP3["Chuyên Gia 3: Tính Toán Lực Mô-tơ Tay Robot"]
        end
        
        N_EXPERTS --> N_BLOCK["32x Transformer Block với Lõi Dense 8B\n(GQA Attention 32:8 + SwiGLU FFN)"]
    end

    N_BLOCK --> N_NORM["Tầng Chuẩn Hóa RMSNorm"]

    subgraph NANO_OUTPUTS["3. ĐẦU RA ĐA CHẾ ĐỘ NANO"]
        N_NORM --> N_AR_HEAD["Reasoner Head: Trả lời chữ & Kế hoạch kho hàng"]
        N_NORM --> N_DM_HEAD["Generator Head: Khử nhiễu 35 bước -> Video nhà máy & Lệnh gắp hàng"]
    end
```

---

### 2.3. Kiến Trúc COSMOS 3 SUPER (~64B MoE Multi-Cluster - Siêu Máy Tính Datacenter)

* **Đặc điểm:** Mô hình khổng lồ **64 Tỷ tham số** chạy trên cụm Siêu máy tính DGX H100/B200. Sử dụng kỹ thuật song song hóa đa máy chủ (Tensor Parallelism + Pipeline Parallelism) để mô phỏng toàn bộ không gian nhà máy 3D thời gian thực (Digital Twin).

```mermaid
flowchart TD
    subgraph SUPER_INPUTS["1. ĐẦU VÀO SIÊU ĐA DẠNG (SUPER DATACENTER SCALE)"]
        S_DATA1["Dữ liệu Video 4K từ hàng nghìn Camera Nhà Máy"] --> S_ENC1["Bộ Mã Hóa Siêu Cấp Omnimodal Encoders"]
        S_DATA2["Toàn bộ Cơ Sở Dữ Liệu & Quy Trình Công Nghiệp"] --> S_ENC2["LLM Text Encoder 70B"]
        S_DATA3["Hệ Thống Đội Xe Robot Tự Hành (Robot Swarm)"] --> S_ENC3["Multi-Agent Action Encoders"]
    end

    S_ENC1 --> S_BUS["Hệ Thống Cáp Mạng Siêu Tốc NVLink / InfiniBand (Băng thông 900GB/s)"]
    S_ENC2 --> S_BUS
    S_ENC3 --> S_BUS

    subgraph SUPER_CLUSTER["2. CỤM SIÊU MÁY TÍNH 64B MoE (TENSOR & PIPELINE PARALLELISM)"]
        S_BUS --> S_NODE0["Máy Chủ Node 0 (Tầng 0 -> 15): Chuyên Gia Phân Tích Cảnh 3D"]
        S_NODE0 --> S_NODE1["Máy Chủ Node 1 (Tầng 16 -> 31): Chuyên Gia Mô Phỏng Vật Lý"]
        S_NODE1 --> S_NODE2["Máy Chủ Node 2 (Tầng 32 -> 47): Chuyên Gia Ngôn Ngữ & Kế Hoạch"]
        S_NODE2 --> S_NODE3["Máy Chủ Node 3 (Tầng 48 -> 63): Chuyên Gia Khử Nhiễu Video 4K"]
    end

    S_NODE3 --> S_SYNCHRONIZE["Đồng Bộ Hóa Tín Hiệu Cuối (Final Multi-Node RMSNorm)"]

    subgraph SUPER_OUTPUTS["3. ĐẦU RA SIÊU MÔ PHỎNG (OMNIMODAL SIMULATION)"]
        S_SYNCHRONIZE --> S_OMNIVERSE["NVIDIA Omniverse Digital Twin:\n- Tái tạo lại toàn bộ nhà máy 3D thời gian thực"]
        S_SYNCHRONIZE --> S_SWARM["Multi-Robot Swarm Control:\n- Tự động điều khiển hàng trăm Robot cùng lúc"]
    end
```

---

## 3. Tóm Tắt Sự Khác Biệt Cốt Lõi Giữa 3 Mô Hình

1. **Cosmos 3 Edge (4B):**
   * **Bản chất:** Gọn nhẹ, thuần **Dense (không dùng MoE)** để triệt tiêu độ trễ truyền dữ liệu.
   * **Mục đích:** Gắn trực tiếp lên **Xe tự hành & Robot di động** để lái xe và gắp đồ thời gian thực.
2. **Cosmos 3 Nano (16B):**
   * **Bản chất:** Hỗn hợp **Lõi Dense 8B + Các Tầng Chuyên Gia MoE 8B**.
   * **Mục đích:** Đặt tại **Server nội bộ nhà máy** để quản lý kho hàng và giám sát an toàn dây chuyền.
3. **Cosmos 3 Super (64B):**
   * **Bản chất:** Siêu mô hình **MoE 64B** phân bổ qua cụm Siêu máy tính DGX.
   * **Mục đích:** Đặt tại **Trung tâm dữ liệu Cloud** để dựng mô phỏng 3D Digital Twin toàn bộ nhà máy.
