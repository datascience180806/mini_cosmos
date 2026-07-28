"""
🚀 Qwen2-VL-2B-Instruct Inference & Action Recognition Pipeline for HATRec Dataset
Author: Antigravity AI & Research Team
Optimized for 0-OOM on Kaggle Dual T4 / Single T4 GPUs (< 3.0 GB VRAM Footprint)
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np

# Mapping nhãn chuẩn HATRec (7 Task classes)
TASK_MAPPING = {
    0: "Assembling the spring",
    1: "Placing white plastic",
    2: "Screwing-1",
    3: "Inflating valve",
    4: "Placing black plastic",
    5: "Screwing-2",
    6: "Fixing cable"
}

REVERSE_TASK_MAPPING = {
    "spring": 0, "assembling the spring": 0,
    "white plastic": 1, "placing white plastic": 1,
    "screwing-1": 2, "screwing 1": 2, "screwing1": 2,
    "inflating valve": 3, "valve": 3,
    "black plastic": 4, "placing black plastic": 4,
    "screwing-2": 5, "screwing 2": 5, "screwing2": 5,
    "fixing cable": 6, "cable": 6, "fixing the cable": 6
}

def extract_frames_from_video(video_path: str, max_frames: int = 8, target_size=(384, 384)):
    """Trích xuất max_frames khung hình và resize về (384, 384) để kiểm soát VRAM < 3GB"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    frames = []
    
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            # Resize khung hình về 384x384 để giới hạn visual tokens, tránh OOM VRAM
            if target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
    cap.release()
    return frames

def parse_ground_truth(file_path: str):
    """Bóc tách Ground Truth Task ID từ tên file/thư mục (ví dụ: Task_01.mp4 -> Task 1)"""
    name = Path(file_path).name.lower()
    parent = Path(file_path).parent.name.lower()
    
    match = re.search(r'task_?0*([0-6])', name) or re.search(r'task_?0*([0-6])', parent)
    if match:
        return int(match.group(1))
    return None

def parse_predicted_task(text_output: str):
    """Bóc tách nhãn dự đoán từ câu trả lời của Qwen2-VL"""
    text_lower = text_output.lower()
    
    # 1. Tìm theo số nhãn Task (Ví dụ: "Task 2" hoặc "Answer: 2")
    num_match = re.search(r'(?:task|class|answer)\s*[:#-]?\s*([0-6])\b', text_lower)
    if num_match:
        return int(num_match.group(1))
        
    # 2. Tìm theo từ khóa hành động
    for key, task_id in REVERSE_TASK_MAPPING.items():
        if key in text_lower:
            return task_id
            
    return None

def load_qwen2_vl_model(device: str = "cuda:0"):
    """Tải mô hình Qwen2-VL-2B-Instruct với tối ưu hóa FP16"""
    print("⏳ Đang tải mô hình Qwen2-VL-2B-Instruct...")
    start_time = time.time()

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device
    )

    print(f"✅ Tải thành công Qwen2-VL-2B-Instruct trong {time.time() - start_time:.2f}s!")
    return model, processor

def run_qwen2_vl_inference(model, processor, frames, prompt_text: str, device: str = "cuda:0"):
    """Thực hiện suy luận hình ảnh + văn bản với Qwen2-VL (Giới hạn VRAM < 3GB)"""
    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in frames]

    messages = [
        {
            "role": "user",
            "content": [
                *([{"type": "image", "image": img} for img in pil_frames]),
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text_input],
        images=pil_frames,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Xóa giải phóng bộ nhớ PyTorch sau mỗi pass
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

    return output_text

def main():
    parser = argparse.ArgumentParser(description="Run Qwen2-VL-2B on HATRec Dataset")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/real-world-industrial-assembly-action-dataset", help="Path to HATRec dataset")
    parser.add_argument("--max_videos", type=int, default=50, help="Max videos to evaluate (default 50 for quick test)")
    parser.add_argument("--output_json", type=str, default="qwen2_vl_hatrec_results.json", help="Result JSON path")
    args = parser.parse_args()

    # Tìm danh sách tệp video
    data_path = Path(args.data_dir)
    if not data_path.exists():
        data_path = Path("./videos")

    video_files = sorted(list(data_path.rglob("*.mp4")) + list(data_path.rglob("*.avi")))
    if not video_files:
        print(f"❌ KHÔNG TÌM THẤY VIDEO TRONG: {args.data_dir} HOẶC ./videos")
        sys.exit(1)

    print(f"🎬 Tìm thấy tổng cộng {len(video_files)} video. Bắt đầu đánh giá (Tối đa: {args.max_videos} videos)...")
    video_files = video_files[:args.max_videos]

    # Khởi tạo mô hình
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = load_qwen2_vl_model(device=device)

    # Câu Prompt mở đường chuẩn hóa cho 7 thao tác nhà máy
    prompt_text = (
        "Observe this video sequence of an operator at an industrial manual assembly station.\n"
        "Identify which assembly action is being performed from the following 7 classes:\n"
        "0: Assembling the spring\n"
        "1: Placing white plastic\n"
        "2: Screwing-1\n"
        "3: Inflating valve\n"
        "4: Placing black plastic\n"
        "5: Screwing-2\n"
        "6: Fixing cable\n\n"
        "Answer with the exact format: 'Task [ID]: [Task Name]'"
    )

    results = []
    correct_count = 0
    total_eval = 0

    print("\n" + "="*80)
    print("🚀 BẮT ĐẦU CHẠY EVALUATION QWEN2-VL-2B TRÊN HATREC DATASET (OPTIMIZED VRAM)")
    print("="*80 + "\n")

    for idx, video_file in enumerate(video_files, 1):
        torch.cuda.empty_cache()
        gt_task = parse_ground_truth(str(video_file))
        gt_name = TASK_MAPPING.get(gt_task, "Unknown")

        print(f"🎬 [{idx:02d}/{len(video_files)}] Video: '{video_file.name}' | Ground Truth: Task {gt_task} ({gt_name})")

        # Trích xuất khung hình và resize về 384x384
        frames = extract_frames_from_video(str(video_file), max_frames=8, target_size=(384, 384))
        if not frames:
            print("   ⚠️ Không thể đọc khung hình. Bỏ qua.")
            continue

        # Suy luận với Qwen2-VL
        start_t = time.time()
        output_text = run_qwen2_vl_inference(model, processor, frames, prompt_text, device=device)
        latency = time.time() - start_t

        pred_task = parse_predicted_task(output_text)
        pred_name = TASK_MAPPING.get(pred_task, "Unknown")

        is_correct = (pred_task is not None) and (pred_task == gt_task)
        if is_correct:
            correct_count += 1
        total_eval += 1

        print(f"   ⏱️ Latency: {latency:.2f}s | Predicted: Task {pred_task} ({pred_name}) -> {'✅ ĐÚNG' if is_correct else '❌ SAI'}")
        print(f"   🧠 Model Output: {output_text.strip()}\n" + "-"*60)

        results.append({
            "video": video_file.name,
            "path": str(video_file),
            "ground_truth_id": gt_task,
            "ground_truth_name": gt_name,
            "predicted_id": pred_task,
            "predicted_name": pred_name,
            "is_correct": is_correct,
            "latency_seconds": latency,
            "raw_output": output_text
        })

    # Tính toán Accuracy
    acc = (correct_count / total_eval * 100) if total_eval > 0 else 0
    print("\n" + "="*80)
    print(f"📊 BÁO CÁO KẾT QUẢ EVALUATION QWEN2-VL-2B-INSTRUCT:")
    print(f"   • Tổng số Video đánh giá: {total_eval}")
    print(f"   • Số câu trả lời ĐÚNG  : {correct_count}")
    print(f"   • Độ chính xác (Accuracy): {acc:.2f}%")
    print("="*80 + "\n")

    # Lưu file kết quả JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen2-VL-2B-Instruct",
            "accuracy_percent": acc,
            "total_evaluated": total_eval,
            "correct_count": correct_count,
            "details": results
        }, f, indent=2, ensure_ascii=False)

    print(f"💾 Kết quả chi tiết đã được lưu vào: '{args.output_json}'")

if __name__ == "__main__":
    main()
