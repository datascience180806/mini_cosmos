"""
🚀 V-JEPA (Meta AI Joint-Embedding) Training Script on HATRec Dataset
Scientific 70-30 Train/Test Split (No Data Leakage)
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

from v_jepa.model import VJEPAActionClassifier
from hybrid_yolo_lstm.inference_hatrec import parse_ground_truth, find_all_dataset_videos

class VJEPADataset(Dataset):
    """PyTorch Dataset Loader for V-JEPA (3D Video Tensors)"""
    def __init__(self, data_dir: str, seq_len: int = 16, target_size=(112, 112)):
        all_videos = find_all_dataset_videos(data_dir)
        self.items = []
        for v in all_videos:
            gt = parse_ground_truth(str(v))
            if gt is not None:
                self.items.append((str(v), gt))

        self.seq_len = seq_len
        self.target_size = target_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        v_path, label = self.items[idx]
        cap = cv2.VideoCapture(v_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return torch.zeros((3, self.seq_len, self.target_size[0], self.target_size[1])), label

        indices = np.linspace(0, total_frames - 1, self.seq_len, dtype=int)
        frames = []
        
        for f_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx in indices:
                frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frame_norm = frame_rgb.astype(np.float32) / 255.0
                frame_chw = np.transpose(frame_norm, (2, 0, 1)) # [3, H, W]
                frames.append(frame_chw)

        cap.release()

        if len(frames) < self.seq_len:
            tensor_3d = torch.zeros((3, self.seq_len, self.target_size[0], self.target_size[1]), dtype=torch.float32)
        else:
            # [seq_len, 3, H, W] -> [3, seq_len, H, W] (3D Conv format)
            arr = np.array(frames) # [seq_len, 3, H, W]
            tensor_3d = torch.tensor(arr, dtype=torch.float32).permute(1, 0, 2, 3)

        return tensor_3d, label

def train_vjepa_model():
    parser = argparse.ArgumentParser(description="Train V-JEPA Model on HATRec")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/datasets/ayoznur/hatrec-video-dataset", help="Path to HATRec dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default 15)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="vjepa_hatrec_split.pth", help="Model checkpoint path")
    args = parser.parse_args()

    print("🚀 BẮT ĐẦU HUẤN LUYỆN V-JEPA (META AI) TRÊN HATREC (TRAIN/TEST SPLIT 70%-30%)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    full_dataset = VJEPADataset(args.data_dir, seq_len=16)
    total_len = len(full_dataset)
    if total_len == 0:
        print("❌ Không tìm thấy video hợp lệ!")
        sys.exit(1)

    # Chia 70% Train - 30% Test độc lập chống Data Leakage
    train_size = int(0.70 * total_len)
    val_size = total_len - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"📦 Tổng mẫu: {total_len} | Train (70%): {len(train_dataset)} | Test (30%): {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = VJEPAActionClassifier(num_classes=7, seq_len=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_t = time.time()

    for epoch in range(1, args.epochs + 1):
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

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {t_loss:.4f} - Train Acc: {t_acc:.2f}% || 🎯 V-JEPA TEST ACC: {v_acc:.2f}%")

    print(f"\n🎉 HOÀN THÀNH HUẤN LUYỆN V-JEPA TRONG {time.time() - start_t:.2f} GIÂY!")
    torch.save(model.state_dict(), args.save_path)
    print(f"💾 Trọng số đã lưu vào: '{args.save_path}'")

if __name__ == "__main__":
    train_vjepa_model()
