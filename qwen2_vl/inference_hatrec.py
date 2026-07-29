"""
🚀 Qwen2-VL-2B-Instruct Native Video & Static-Frame Shortcut Test Pipeline for HATRec Dataset
Author: Antigravity AI & Research Team

Features:
- Standard Native Video Inference (Dynamic Motion)
- Static-Frame Shortcut Test (--static_frame_test): Duplicates Frame 0 16 times to test if model relies on temporal motion or static object co-occurrence shortcuts.
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
    
    # 1. Quét tên thao tác chuẩn xác nhất trước
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

    # 2. Tìm theo số nhãn Task (Ví dụ: "Task 2", "Task: 2", "Class 2")
    num_match = re.search(r'(?:task|class|answer)\s*[:#-]?\s*([0-6])\b', text_lower)
    if num_match:
        return int(num_match.group(1))

    # 3. Tìm từ khóa phụ
    for key, task_id in REVERSE_TASK_MAPPING.items():
        if key in text_lower:
            return task_id

    return None

def create_static_frame_video(video_path: str, temp_output_path: str = "/tmp/static_test.mp4", num_frames: int = 16):
    """Trích xuất duy nhất Frame 0 và nhân bản 16 lần để tạo video tĩnh bị đứng hình"""
    cap = cv2.VideoCapture(str(video_path))
    ret, first_frame = cap.read()
    cap.release()

    if not ret or first_frame is None:
        return None

    h, w, c = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, 1.0, (w, h))

    for _ in range(num_frames):
        out.write(first_frame)
    out.release()

    return temp_output_path

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

def run_qwen2_vl_video_inference(model, processor, video_path: str, prompt_text: str, device: str = "cuda:0"):
    """Thực hiện suy luận Video với qwen_vl_utils (3D-RoPE)"""
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64, temperature=0.1, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

    return output_text

def main():
    parser = argparse.ArgumentParser(description="Run Qwen2-VL-2B on HATRec Dataset")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/real-world-industrial-assembly-action-dataset", help="Path to HATRec dataset")
    parser.add_argument("--max_videos", type=int, default=50, help="Max videos to evaluate (default 50 for quick test)")
    parser.add_argument("--static_frame_test", action="store_true", help="Enable Static-Frame Shortcut Test (repeats Frame 0 16 times)")
    parser.add_argument("--output_json", type=str, default="", help="Result JSON path")
    args = parser.parse_args()

    if not args.output_json:
        args.output_json = "qwen2_vl_static_frame_results.json" if args.static_frame_test else "qwen2_vl_hatrec_results.json"

    # Tìm danh sách tệp video
    data_path = Path(args.data_dir)
    if not data_path.exists():
        data_path = Path("/kaggle/input/datasets/ayoznur/hatrec-video-dataset")
    if not data_path.exists():
        data_path = Path("./videos")

    video_files = sorted(list(data_path.rglob("*.mp4")) + list(data_path.rglob("*.avi")))
    if not video_files:
        print(f"❌ KHÔNG TÌM THẤY VIDEO TRONG BẤT KỲ THƯ MỤC NÀO!")
        sys.exit(1)

    print(f"🎬 Tìm thấy {len(video_files)} video. Bắt đầu đánh giá (Tối đa: {args.max_videos} videos)...")
    if args.static_frame_test:
        print("⚠️ CHẾ ĐỘ STATIC-FRAME SHORTCUT TEST ĐÃ BẬT: Trích xuất Frame 0 lặp lại 16 lần (đứng hình) để test Shortcut Learning!")

    video_files = video_files[:args.max_videos]

    # Khởi tạo mô hình
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = load_qwen2_vl_model(device=device)

    # Câu Prompt mở đường chuẩn hóa cho 7 thao tác nhà máy
    prompt_text = (
        "You are an industrial assembly action recognition expert.\n"
        "Watch this video carefully and classify the exact assembly action being performed into one of these 7 choices:\n"
        "- Task 0: Assembling the spring\n"
        "- Task 1: Placing white plastic\n"
        "- Task 2: Screwing-1\n"
        "- Task 3: Inflating valve\n"
        "- Task 4: Placing black plastic\n"
        "- Task 5: Screwing-2\n"
        "- Task 6: Fixing cable\n\n"
        "Analyze the motion and object, then state your final choice as: 'Task X: [Task Name]'"
    )

    results = []
    correct_count = 0
    total_eval = 0

    mode_title = "STATIC-FRAME SHORTCUT TEST (FRAME 0 REPEATED 16x)" if args.static_frame_test else "DYNAMIC NATIVE VIDEO INFERENCE"
    print("\n" + "="*80)
    print(f"🚀 BẮT ĐẦU CHẠY {mode_title} QWEN2-VL-2B TRÊN HATREC DATASET")
    print("="*80 + "\n")

    temp_static_video = "/tmp/static_test_frame.mp4"

    for idx, video_file in enumerate(video_files, 1):
        torch.cuda.empty_cache()
        gt_task = parse_ground_truth(str(video_file))
        gt_name = TASK_MAPPING.get(gt_task, "Unknown")

        print(f"🎬 [{idx:02d}/{len(video_files)}] Video: '{video_file.name}' | Ground Truth: Task {gt_task} ({gt_name})")

        # Xác định đường dẫn video sẽ nạp vào mô hình
        if args.static_frame_test:
            eval_video_path = create_static_frame_video(str(video_file), temp_output_path=temp_static_video, num_frames=16)
            if not eval_video_path:
                print("   ⚠️ Không thể tạo video tĩnh từ Frame 0. Bỏ qua.")
                continue
        else:
            eval_video_path = str(video_file)

        # Suy luận với Qwen2-VL
        start_t = time.time()
        try:
            output_text = run_qwen2_vl_video_inference(model, processor, eval_video_path, prompt_text, device=device)
        except Exception as e:
            print(f"   ⚠️ Lỗi suy luận video: {e}")
            continue

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
            "is_static_frame_test": args.static_frame_test,
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
    print(f"📊 BÁO CÁO KẾT QUẢ {mode_title}:")
    print(f"   • Tổng số Video đánh giá: {total_eval}")
    print(f"   • Số câu trả lời ĐÚNG  : {correct_count}")
    print(f"   • Độ chính xác (Accuracy): {acc:.2f}%")
    print("="*80 + "\n")

    # Lưu file kết quả JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen2-VL-2B-Instruct",
            "test_mode": mode_title,
            "accuracy_percent": acc,
            "total_evaluated": total_eval,
            "correct_count": correct_count,
            "details": results
        }, f, indent=2, ensure_ascii=False)

    print(f"💾 Kết quả chi tiết đã được lưu vào: '{args.output_json}'")

if __name__ == "__main__":
    main()
