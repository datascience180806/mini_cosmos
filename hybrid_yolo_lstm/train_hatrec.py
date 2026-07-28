"""
🚀 Scientific Train/Test Split (70-30) Training Script for Hybrid YOLOv8 + LSTM on HATRec
Prevents Data Leakage by evaluating strictly on unseen video cycles!
Author: Antigravity AI & Research Team
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np

from hybrid_yolo_lstm.model import HybridYOLOv8LSTM
from hybrid_yolo_lstm.inference_hatrec import parse_ground_truth, extract_and_preprocess_video, find_all_dataset_videos

class HATRecDataset(Dataset):
    """PyTorch Dataset Loader for HATRec Industrial Assembly Videos"""
    def __init__(self, data_dir: str, seq_len: int = 16):
        all_videos = find_all_dataset_videos(data_dir)
        self.items = []
        for v in all_videos:
            gt = parse_ground_truth(str(v))
            if gt is not None:
                self.items.append((str(v), gt))

        self.seq_len = seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        v_path, label = self.items[idx]
        tensor = extract_and_preprocess_video(v_path, seq_len=self.seq_len, target_size=(128, 128))
        if tensor is None:
            tensor = torch.zeros((1, self.seq_len, 3, 128, 128), dtype=torch.float32)
        return tensor.squeeze(0), label

def train_hybrid_model():
    parser = argparse.ArgumentParser(description="Train/Val Hybrid YOLOv8 + LSTM without Data Leakage")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/datasets/ayoznur/hatrec-video-dataset", help="Path to HATRec dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default 15)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="hybrid_hatrec_split.pth", help="Model checkpoint output path")
    args = parser.parse_args()

    print("🚀 BẮT ĐẦU HUẤN LUYỆN CHUẨN KHOA HỌC (TRAIN/TEST SPLIT 70%-30%) ĐỂ CHỐNG DATA LEAKAGE...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    full_dataset = HATRecDataset(args.data_dir, seq_len=16)
    total_len = len(full_dataset)
    if total_len == 0:
        print("❌ Không tìm thấy video hợp lệ để train!")
        sys.exit(1)

    # Chia 70% Train - 30% Test (Validation) độc lập chống Data Leakage
    train_size = int(0.70 * total_len)
    val_size = total_len - train_size

    generator = torch.Generator().manual_seed(42) # Cố định seed cho tính lặp lại
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"📦 Tổng mẫu: {total_len} | Tập Train (70%): {len(train_dataset)} samples | Tập Test (30%): {len(val_dataset)} samples (Chưa từng nhìn thấy!)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = HybridYOLOv8LSTM(num_classes=7, seq_len=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # 1. TRONG VÒNG TRAIN
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(videos)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * videos.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # 2. VÒNG VALIDATION (TEST TRÊN VIDEO MỚI CHƯA TỪNG THẤY)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                logits = model(videos)
                loss = criterion(logits, labels)

                val_loss += loss.item() * videos.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        t_loss = train_loss / train_total if train_total else 0
        t_acc = (train_correct / train_total * 100.0) if train_total else 0
        v_loss = val_loss / val_total if val_total else 0
        v_acc = (val_correct / val_total * 100.0) if val_total else 0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {t_loss:.4f} - Train Acc: {t_acc:.2f}% || 🎯 TEST ACC (Unseen Videos): {v_acc:.2f}%")

    total_time = time.time() - start_train_time
    print(f"\n🎉 HOÀN THÀNH TRAIN & VAL CHỐNG DATA LEAKAGE TRONG {total_time:.2f} GIÂY!")
    
    torch.save(model.state_dict(), args.save_path)
    print(f"💾 Trọng số chuẩn đã lưu thành công vào: '{args.save_path}'")

if __name__ == "__main__":
    train_hybrid_model()
