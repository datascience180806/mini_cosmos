"""
Download HATRec Industrial Assembly Dataset from Kaggle into ./videos directory
Dataset Kaggle Slug: ayoznur/real-world-industrial-assembly-action-dataset
"""

import os
import sys
import argparse

def download_hatrec_dataset(output_dir: str = "./videos"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Dang tai bo du lieu HATRec Industrial Dataset ve thu muc: '{output_dir}'...")

    # Phương án 1: Sử dụng kagglehub (Phương án hiện đại & khuyến nghị nhất của Kaggle)
    try:
        import kagglehub
        print("[INFO] Dang tai qua thu vien 'kagglehub'...")
        path = kagglehub.dataset_download("ayoznur/real-world-industrial-assembly-action-dataset")
        print(f"[SUCCESS] Tai thanh cong ve cache: {path}")

        # Sao chép/liên kết dữ liệu sang output_dir
        import shutil
        print(f"[INFO] Dang giai nen/chuyen du lieu sang '{output_dir}'...")
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(output_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print(f"[SUCCESS] HOAN THANH! Bo du lieu da san sang tai: '{output_dir}'")
        return
    except Exception as e:
        print(f"[NOTE] kagglehub chua san sang hoac can cau hinh ({e}). Thu tiep CLI...")

    # Phương án 2: Sử dụng kaggle CLI
    try:
        import subprocess
        print("[INFO] Dang tai qua 'kaggle CLI'...")
        cmd = f"kaggle datasets download -d ayoznur/real-world-industrial-assembly-action-dataset -p {output_dir} --unzip"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[SUCCESS] HOAN THANH! Tai qua Kaggle CLI thanh cong vao '{output_dir}'!")
            return
        else:
            print(f"[WARN] Kaggle CLI output: {res.stderr}")
    except Exception as e:
        print(f"[NOTE] Loi Kaggle CLI: {e}")

    # Phương án 3: Hướng dẫn người dùng chạy trên Kaggle Notebook
    print("\n" + "="*70)
    print("HUONG DAN TAI TRUC TIEP TREN KAGGLE NOTEBOOK / SERVER:")
    print("="*70)
    print("1. Neu chay tren Kaggle Notebook, ban chi can bam nut '+ Add Input' ben phai,")
    print("   tim kiem dataset: 'ayoznur/real-world-industrial-assembly-action-dataset'")
    print("2. Hoac chay lenh Python sau trong Kaggle Cell:")
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
