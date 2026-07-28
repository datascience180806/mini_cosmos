"""
Download HATRec Industrial Assembly Dataset from Kaggle into ./videos directory
Dataset Kaggle Slug: ayoznur/real-world-industrial-assembly-action-dataset
"""

import os
import sys
import argparse

def download_hatrec_dataset(output_dir: str = "./videos"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"📥 Đang tải bộ dữ liệu HATRec Industrial Dataset về thư mục: '{output_dir}'...")

    # Phương án 1: Sử dụng kagglehub (Phương án hiện đại & khuyến nghị nhất của Kaggle)
    try:
        import kagglehub
        print("[INFO] Đang tải qua thư viện 'kagglehub'...")
        path = kagglehub.dataset_download("ayoznur/real-world-industrial-assembly-action-dataset")
        print(f"[SUCCESS] Tải thành công về cache: {path}")

        # Sao chép/liên kết dữ liệu sang output_dir
        import shutil
        print(f"[INFO] Đang giải nén/chuyển dữ liệu sang '{output_dir}'...")
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(output_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print(f"🎉 HOÀN THÀNH! Bộ dữ liệu đã sẵn sàng tại: '{output_dir}'")
        return
    except Exception as e:
        print(f"[NOTE] kagglehub chưa sẵn sàng hoặc cần cấu hình ({e}). Thử tiếp CLI...")

    # Phương án 2: Sử dụng kaggle CLI
    try:
        import subprocess
        print("[INFO] Đang tải qua 'kaggle CLI'...")
        cmd = f"kaggle datasets download -d ayoznur/real-world-industrial-assembly-action-dataset -p {output_dir} --unzip"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"🎉 HOÀN THÀNH! Tải qua Kaggle CLI thành công vào '{output_dir}'!")
            return
        else:
            print(f"[WARN] Kaggle CLI output: {res.stderr}")
    except Exception as e:
        print(f"[NOTE] Lỗi Kaggle CLI: {e}")

    # Phương án 3: Hướng dẫn người dùng chạy trên Kaggle Notebook
    print("\n" + "="*70)
    print("💡 HƯỚNG DẪN TẢI TRỰC TIẾP TRÊN KAGGLE NOTEBOOK / SERVER:")
    print("="*70)
    print("1. Nếu chạy trên Kaggle Notebook, bạn chỉ cần bấm nút '+ Add Input' bên phải,")
    print("   tìm kiếm dataset: 'ayoznur/real-world-industrial-assembly-action-dataset'")
    print("2. Hoặc chạy lệnh Python sau trong Kaggle Cell:")
    print("   !pip install kagglehub -q")
    print("   import kagglehub")
    print("   path = kagglehub.dataset_download('ayoznur/real-world-industrial-assembly-action-dataset')")
    print("   !cp -r {path}/* ./videos/")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HATRec Dataset into ./videos")
    parser.add_argument("--output_dir", type=str, default="./videos", help="Target output directory")
    args = parser.parse_args()
    download_hatrec_dataset(output_dir=args.output_dir)
