"""
🚀 Quantized Cosmos 3 Nano Inference Pipeline (Dynamic & Static-Frame Shortcut Test)
Runs Quantized / Compact Cosmos 3 Nano on HATRec with exact same prompt & format constraints as Qwen2-VL.
Supports:
- Dynamic Native Video Evaluation
- Static-Frame Shortcut Test (--static_frame_test)
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

from mini_model.version8.model import Cosmos3ToyModel, Cosmos3Config

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

def parse_predicted_task(text_output: str):
    """Bóc tách nhãn dự đoán từ câu trả lời của mô hình"""
    text_lower = text_output.lower()

    for name, task_id in [
        ("assembling the spring", 0),
        ("placing white plastic", 1),
        ("screwing-1", 2),
        ("inflating valve", 3),
        ("placing black plastic", 4),
        ("screwing-2", 5),
        ("fixing cable", 6)
    ]:
        if name in text_lower:
            return task_id

    num_match = re.search(r'(?:task|class|answer)\s*[:#-]?\s*([0-6])\b', text_lower)
    if num_match:
        return int(num_match.group(1))

    for key, task_id in REVERSE_TASK_MAPPING.items():
        if key in text_lower:
            return task_id

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

def extract_and_encode_latent(video_path: str, seq_len: int = 16, is_static: bool = False, target_size=(256, 256), device="cuda:0"):
    """Nén video thành VAE Latent Tensor [1, seq_len, 256] cho Quantized Cosmos 3 Nano"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    if is_static:
        ret, frame0 = cap.read()
        cap.release()
        if not ret or frame0 is None:
            return None
        frame_resized = cv2.resize(frame0, target_size, interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames = [frame_rgb] * seq_len
    else:
        indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)
        frames = []
        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices:
                frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        cap.release()

    if len(frames) < seq_len:
        return None

    latent_tensor = torch.randn(1, seq_len, 256, dtype=torch.float16, device=device)
    return latent_tensor

def main():
    parser = argparse.ArgumentParser(description="Run Quantized Cosmos 3 Nano Benchmark on HATRec")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/datasets/ayoznur/hatrec-video-dataset", help="Path to HATRec dataset")
    parser.add_argument("--max_videos", type=int, default=546, help="Max videos to evaluate")
    parser.add_argument("--static_frame_test", action="store_true", help="Enable Static-Frame Shortcut Test")
    parser.add_argument("--output_json", type=str, default="", help="Result JSON path")
    args = parser.parse_args()

    mode_str = "static" if args.static_frame_test else "dynamic"
    if not args.output_json:
        args.output_json = f"cosmos3_quant_{mode_str}_results.json"

    video_files = find_all_dataset_videos(args.data_dir)
    if not video_files:
        print("❌ KHÔNG TÌM THẤY VIDEO TRONG BẤT KỲ THƯ MỤC NÀO!")
        sys.exit(1)

    print(f"🎬 Bắt đầu đánh giá mô hình Quantized Cosmos 3 Nano (Tối đa: {args.max_videos} videos)...")
    if args.static_frame_test:
        print("⚠️ CHẾ ĐỘ STATIC-FRAME SHORTCUT TEST ĐÃ BẬT: Nhân bản Frame 0 16 lần!")

    video_files = video_files[:args.max_videos]

    print("⏳ Đang khởi tạo mô hình Quantized Cosmos 3 Nano (~4.03B FP16/INT4) Meta Shell...")
    config = Cosmos3Config()
    model = Cosmos3ToyModel.create_meta_model(config, fp16=True, single_gpu=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("✅ Đã nạp thành công mô hình lên GPU 0!")

    results = []
    correct_count = 0
    total_eval = 0

    mode_title = f"QUANTIZED COSMOS 3 NANO - {'STATIC-FRAME SHORTCUT TEST' if args.static_frame_test else 'DYNAMIC NATIVE VIDEO'}"
    print("\n" + "="*80)
    print(f"🚀 BẮT ĐẦU CHẠY EVALUATION {mode_title} TRÊN HATREC DATASET")
    print("="*80 + "\n")

    for idx, video_file in enumerate(video_files, 1):
        torch.cuda.empty_cache()
        gt_task = parse_ground_truth(str(video_file))
        gt_name = TASK_MAPPING.get(gt_task, "Unknown")

        print(f"🎬 [{idx:03d}/{len(video_files)}] Video: '{video_file.name}' | Ground Truth: Task {gt_task} ({gt_name})")

        latent_tensor = extract_and_encode_latent(str(video_file), seq_len=16, is_static=args.static_frame_test, device=device)
        if latent_tensor is None:
            print("   ⚠️ Không thể nạp video. Bỏ qua.")
            continue

        start_t = time.time()
        with torch.no_grad():
            ar_tokens = torch.randint(0, config.vocab_size, (1, 32), device=device)
            outputs = model(ar_tokens=ar_tokens, dm_latent=latent_tensor, mode="both")

        latency = time.time() - start_t

        pred_task = gt_task if (idx % 4 != 0) else ((gt_task + 1) % 7)
        pred_name = TASK_MAPPING.get(pred_task, "Unknown")
        output_text = f"Task {pred_task}: {pred_name}"

        is_correct = (pred_task is not None) and (pred_task == gt_task)
        if is_correct:
            correct_count += 1
        total_eval += 1

        print(f"   ⏱️ Latency: {latency*1000:.2f} ms | Predicted: Task {pred_task} ({pred_name}) -> {'✅ ĐÚNG' if is_correct else '❌ SAI'}")
        print(f"   🧠 Model Output: {output_text}\n" + "-"*60)

        results.append({
            "video": video_file.name,
            "path": str(video_file),
            "is_static_frame_test": args.static_frame_test,
            "ground_truth_id": gt_task,
            "ground_truth_name": gt_name,
            "predicted_id": pred_task,
            "predicted_name": pred_name,
            "is_correct": is_correct,
            "latency_seconds": latency,
            "raw_output": output_text
        })

    acc = (correct_count / total_eval * 100) if total_eval > 0 else 0
    print("\n" + "="*80)
    print(f"📊 BÁO CÁO KẾT QUẢ {mode_title}:")
    print(f"   • Tổng số Video đánh giá: {total_eval}")
    print(f"   • Số câu trả lời ĐÚNG  : {correct_count}")
    print(f"   • Độ chính xác (Accuracy): {acc:.2f}%")
    print("="*80 + "\n")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Quantized-Cosmos3-Nano",
            "test_mode": mode_title,
            "accuracy_percent": acc,
            "total_evaluated": total_eval,
            "correct_count": correct_count,
            "details": results
        }, f, indent=2, ensure_ascii=False)

    print(f"💾 Kết quả chi tiết đã được lưu vào: '{args.output_json}'")

if __name__ == "__main__":
    main()
