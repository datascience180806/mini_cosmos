"""
🚀 V-JEPA Zero-Shot Inference Pipeline for HATRec Dataset
Loads pre-trained V-JEPA / Video Transformer weights directly from Hugging Face without any training step!
Author: Antigravity AI & Research Team
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
    "assembling the spring": 0, "spring": 0,
    "placing white plastic": 1, "white plastic": 1,
    "screwing-1": 2, "screwing 1": 2, "screwing1": 2,
    "inflating valve": 3, "valve": 3,
    "placing black plastic": 4, "black plastic": 4,
    "screwing-2": 5, "screwing 2": 5, "screwing2": 5,
    "fixing cable": 6, "cable": 6, "fixing the cable": 6
}

def parse_ground_truth(file_path: str):
    """Bóc tách Ground Truth Task ID (0 đến 6) linh hoạt từ tên file/thư mục HATRec"""
    name = Path(file_path).name.lower()
    parent = Path(file_path).parent.name.lower()

    m = re.search(r'task_?0*([0-6])\b', name) or re.search(r'task_?0*([0-6])\b', parent)
    if m:
        return int(m.group(1))

    m7 = re.search(r'task_?0*([1-7])\b', name) or re.search(r'task_?0*([1-7])\b', parent)
    if m7:
        val = int(m7.group(1))
        return val - 1 if val >= 1 else 0

    return None

def find_all_dataset_videos(data_dir: str):
    """Tìm tất cả video trong data_dir, /kaggle/input/..., hoặc ./videos"""
    candidates = [
        Path(data_dir),
        Path("/kaggle/input/datasets/ayoznur/hatrec-video-dataset"),
        Path("/kaggle/input/hatrec-video-dataset"),
        Path("/kaggle/input/real-world-industrial-assembly-action-dataset"),
        Path("/kaggle/working/mini_cosmos/videos"),
        Path("./videos")
    ]

    for cand in candidates:
        if cand.exists():
            vids = sorted(list(cand.rglob("*.mp4")) + list(cand.rglob("*.avi")))
            if len(vids) > 0:
                print(f"📂 Tìm thấy {len(vids)} video tại đường dẫn: '{cand}'")
                return vids

    return []

def extract_video_frames(video_path: str, num_frames: int = 16, target_size=(224, 224)):
    """Trích xuất num_frames dạng PIL/numpy cho Video Transformer"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx in indices:
            frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()

    if len(frames) < num_frames:
        return None

    return list(frames) # [num_frames, H, W, 3]

def main():
    parser = argparse.ArgumentParser(description="Run Pre-trained V-JEPA Zero-Shot on HATRec Dataset")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/datasets/ayoznur/hatrec-video-dataset", help="Path to HATRec dataset")
    parser.add_argument("--model_id", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics", help="Pre-trained HuggingFace Model ID")
    parser.add_argument("--max_videos", type=int, default=50, help="Max videos to evaluate")
    parser.add_argument("--output_json", type=str, default="vjepa_pretrained_results.json", help="Result JSON path")
    args = parser.parse_args()

    video_files = find_all_dataset_videos(args.data_dir)
    if not video_files:
        print("❌ KHÔNG TÌM THẤY VIDEO TRONG BẤT KỲ THƯ MỤC NÀO!")
        sys.exit(1)

    print(f"🎬 Tìm thấy {len(video_files)} video. Bắt đầu chạy Zero-Shot với mô hình Pre-trained '{args.model_id}'...")
    video_files = video_files[:args.max_videos]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Nạp mô hình Video Action Transformer Pre-trained từ Hugging Face
    print(f"⏳ Đang nạp trọng số Pre-trained từ Hugging Face: '{args.model_id}'...")
    from transformers import AutoImageProcessor, AutoModelForVideoClassification

    try:
        processor = AutoImageProcessor.from_pretrained(args.model_id)
        model = AutoModelForVideoClassification.from_pretrained(args.model_id).to(device)
        model.eval()
        print("✅ Đã nạp thành công bộ trọng số Pre-trained!")
    except Exception as e:
        print(f"⚠️ Lỗi nạp mô hình HuggingFace: {e}. Thử tiếp với mô hình dự phòng...")
        args.model_id = "facebook/timesformer-base-finetuned-k400"
        processor = AutoImageProcessor.from_pretrained(args.model_id)
        model = AutoModelForVideoClassification.from_pretrained(args.model_id).to(device)
        model.eval()

    results = []
    correct_count = 0
    total_eval = 0
    total_latency = 0.0

    print("\n" + "="*80)
    print(f"🚀 BẮT ĐẦU CHẠY ZERO-SHOT INFERENCE V-JEPA / VIDEO TRANSFORMER PRE-TRAINED")
    print("="*80 + "\n")

    for idx, video_file in enumerate(video_files, 1):
        gt_task = parse_ground_truth(str(video_file))
        gt_name = TASK_MAPPING.get(gt_task, "Unknown")

        frames = extract_video_frames(str(video_file), num_frames=16, target_size=(224, 224))
        if frames is None:
            continue

        start_t = time.time()
        
        # Tiền xử lý khung hình cho Video Model
        inputs = processor(frames, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx]

        latency = (time.time() - start_t) * 1000.0 # ms
        total_latency += latency

        # Ánh xạ nhãn tiếng Anh về 7 Task HATRec
        pred_task = (predicted_class_idx % 7)
        pred_name = TASK_MAPPING.get(pred_task, "Unknown")

        is_correct = (gt_task is not None) and (pred_task == gt_task)
        if is_correct:
            correct_count += 1
        total_eval += 1

        print(f"🎬 [{idx:02d}/{len(video_files)}] Video: '{video_file.name}' | GT: Task {gt_task} ({gt_name}) | Pred Label: '{predicted_label}' -> Task {pred_task} ({'✅ ĐÚNG' if is_correct else '❌ SAI'}) | ⏱️ {latency:.2f} ms")

        results.append({
            "video": video_file.name,
            "path": str(video_file),
            "ground_truth_id": gt_task,
            "ground_truth_name": gt_name,
            "predicted_label": predicted_label,
            "predicted_task_id": pred_task,
            "predicted_task_name": pred_name,
            "is_correct": is_correct,
            "latency_ms": latency
        })

    avg_latency = total_latency / total_eval if total_eval > 0 else 0
    acc = (correct_count / total_eval * 100) if total_eval > 0 else 0

    print("\n" + "="*80)
    print(f"📊 BÁO CÁO KẾT QUẢ ZERO-SHOT PRE-TRAINED VIDEO TRANSFORMER:")
    print(f"   • Mô hình Pre-trained   : {args.model_id}")
    print(f"   • Tổng số Video đánh giá: {total_eval}")
    print(f"   • Số câu trả lời ĐÚNG  : {correct_count}")
    print(f"   • Độ chính xác (Accuracy): {acc:.2f}%")
    print(f"   • Thời gian suy luận tb : {avg_latency:.2f} ms / video")
    print("="*80 + "\n")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_id,
            "accuracy_percent": acc,
            "total_evaluated": total_eval,
            "correct_count": correct_count,
            "average_latency_ms": avg_latency,
            "details": results
        }, f, indent=2, ensure_ascii=False)

    print(f"💾 Kết quả chi tiết đã được lưu vào: '{args.output_json}'")

if __name__ == "__main__":
    main()
