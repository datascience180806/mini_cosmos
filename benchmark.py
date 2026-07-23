"""
Benchmark suite for mini_cosmos model versions.
Measures parameters, peak VRAM, forward pass latency, loss metrics, and attention mask isolation.

Usage:
    python benchmark.py --version version5 --fp16
"""

import argparse
import importlib
import json
import os
import time
from typing import Dict, Any

import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark mini_cosmos model versions")
    parser.add_argument("--version", type=str, default="version1", help="Version name (e.g. version1)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for benchmarking")
    parser.add_argument("--seq_len_ar", type=int, default=32, help="Sequence length for AR tokens")
    parser.add_argument("--seq_len_dm", type=int, default=16, help="Sequence length for DM tokens")
    parser.add_argument("--num_runs", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--fp16", action="store_true", help="Run benchmark in FP16 half-precision mode")
    parser.add_argument("--output_file", type=str, default="benchmark_results.json", help="Path to save metrics")
    return parser.parse_args()


def load_model_version(version_name: str):
    """Dynamically import model and config for the requested version."""
    module_path = f"mini_model.{version_name}.model"
    try:
        module = importlib.import_module(module_path)
        Cosmos3ToyModel = getattr(module, "Cosmos3ToyModel")
        Cosmos3Config = getattr(module, "Cosmos3Config")
        return Cosmos3ToyModel, Cosmos3Config
    except Exception as e:
        raise ImportError(f"Could not load version '{version_name}' from path '{module_path}': {e}")


def measure_attention_mask_isolation(model: nn.Module, config: Any, device: torch.device) -> bool:
    """
    Sanity check to verify that Q_AR x K_DM is strictly masked (-inf / zero attention weights)
    to prevent diffusion noise from interfering with autoregressive reasoning.
    """
    model.eval()
    with torch.no_grad():
        mask_gen = getattr(model, "mask_generator", None)
        if mask_gen is None:
            return False
        
        seq_len_ar = 16
        seq_len_dm = 8
        attn_mask = mask_gen(seq_len_ar, seq_len_dm, device=device)
        
        top_right_block = attn_mask[:seq_len_ar, seq_len_ar:]
        is_isolated = torch.all(torch.isinf(top_right_block) & (top_right_block < 0)).item()
        return is_isolated


def run_benchmark(args) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16

    print("=" * 70)
    print(f"BENCHMARKING MINI_COSMOS VERSION: {args.version}")
    print(f"Device: {device} | Batch Size: {args.batch_size} | Iterations: {args.num_runs}")
    print("=" * 70)

    Cosmos3ToyModel, Cosmos3Config = load_model_version(args.version)
    config = Cosmos3Config()

    # Create model and handle potential CUDA OutOfMemory via FP16 fallback
    model = Cosmos3ToyModel(config)
    
    if device.type == "cuda":
        try:
            if use_fp16:
                print("[INFO] Che do FP16 (Half Precision) duoc kich hoat.")
                model = model.half().to(device)
            else:
                model = model.to(device)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            print(f"[WARNING] Tràn bộ nhớ VRAM khi load FP32! Tự động chuyển sang FP16 Half Precision Mode...")
            torch.cuda.empty_cache()
            use_fp16 = True
            model = model.half().to(device)
    else:
        model = model.to(device)

    # 1. Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Attention Isolation Sanity Check
    isolation_passed = measure_attention_mask_isolation(model, config, device)

    # Prepare dummy synthetic test data with matching dtype
    dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32

    ar_input = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len_ar), device=device)
    ar_target = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len_ar), device=device)
    dm_input = torch.randn(args.batch_size, args.seq_len_dm, config.latent_dim, device=device, dtype=dtype)
    dm_target = torch.randn(args.batch_size, args.seq_len_dm, config.latent_dim, device=device, dtype=dtype)
    action_input = torch.randn(args.batch_size, args.seq_len_dm, config.action_dim, device=device, dtype=dtype)

    # Warmup runs
    for _ in range(5):
        _ = model(ar_tokens=ar_input, dm_latent=dm_input, action_vectors=action_input, mode="both")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # 3. Latency & Throughput Measurement
    start_time = time.time()
    for _ in range(args.num_runs):
        with torch.no_grad():
            _ = model(ar_tokens=ar_input, dm_latent=dm_input, action_vectors=action_input, mode="both")
        if device.type == "cuda":
            torch.cuda.synchronize()
            
    total_elapsed = time.time() - start_time
    avg_latency_ms = (total_elapsed / args.num_runs) * 1000
    fps_throughput = (args.batch_size * args.num_runs) / total_elapsed

    # 4. Memory Footprint
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        bytes_per_param = 2 if use_fp16 else 4
        peak_vram_mb = (total_params * bytes_per_param) / (1024 * 1024)

    # 5. Quality & Loss Evaluation on Synthetic Test Batch
    model.eval()
    with torch.no_grad():
        out = model(ar_tokens=ar_input, dm_latent=dm_input, action_vectors=action_input, mode="both")
        
        ar_logits = out["ar_logits"]
        ar_loss = nn.CrossEntropyLoss()(ar_logits.view(-1, config.vocab_size).float(), ar_target.view(-1)).item()
        preds = torch.argmax(ar_logits, dim=-1)
        ar_acc = (preds == ar_target).float().mean().item() * 100

        dm_pred = out["dm_predicted_latent"]
        dm_mse = nn.MSELoss()(dm_pred.float(), dm_target.float()).item()
        dm_cos_sim = nn.CosineSimilarity(dim=-1)(dm_pred.float(), dm_target.float()).mean().item()

    metrics = {
        "version": args.version,
        "precision": "FP16" if use_fp16 else "FP32",
        "total_parameters_M": round(total_params / 1e6, 2),
        "trainable_parameters_M": round(trainable_params / 1e6, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "throughput_samples_per_sec": round(fps_throughput, 2),
        "ar_loss": round(ar_loss, 4),
        "ar_accuracy_percent": round(ar_acc, 2),
        "dm_mse_loss": round(dm_mse, 4),
        "dm_cosine_similarity": round(dm_cos_sim, 4),
        "attention_mask_isolation_verified": isolation_passed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print("\n" + "=" * 50)
    print("BENCHMARK METRICS SUMMARY")
    print("=" * 50)
    print(f"  • Model Version           : {metrics['version']} ({metrics['precision']})")
    print(f"  • Total Parameters        : {metrics['total_parameters_M']} M")
    print(f"  • Peak VRAM Usage         : {metrics['peak_vram_mb']} MB")
    print(f"  • Avg Latency (Batch {args.batch_size}) : {metrics['avg_latency_ms']} ms")
    print(f"  • Throughput              : {metrics['throughput_samples_per_sec']} samples/sec")
    print(f"  • AR Loss / Top-1 Acc     : {metrics['ar_loss']} / {metrics['ar_accuracy_percent']}%")
    print(f"  • DM Loss (MSE) / Cos Sim : {metrics['dm_mse_loss']} / {metrics['dm_cosine_similarity']}")
    print(f"  • Attention Mask Isolation: {'PASSED [OK]' if isolation_passed else 'FAILED [X]'}")
    print("=" * 50)

    results_all = {}
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                results_all = json.load(f)
        except Exception:
            results_all = {}
            
    results_all[args.version] = metrics

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=4)
        
    print(f"\n[SUCCESS] Benchmark metrics saved to '{args.output_file}'.")
    return metrics


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
