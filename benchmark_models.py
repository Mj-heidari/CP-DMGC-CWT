"""
Model Benchmarking Script for EEG Seizure Prediction Models

Measures:
- Inference time on GPU and CPU
- Number of parameters
- FLOPs (floating point operations)
- Memory usage

Usage:
    python benchmark_models.py --batch_size 32 --n_runs 100
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.provider import get_builder

# Try to import FLOPs calculation tools
try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    print("Warning: thop not installed. Install with: pip install thop")
    HAS_THOP = False

try:
    from torchinfo import summary
    HAS_TORCHINFO = True
except ImportError:
    print("Warning: torchinfo not installed. Install with: pip install torchinfo")
    HAS_TORCHINFO = False


class ModelBenchmark:
    """Benchmark a single model"""
    
    def __init__(self, model_name: str, input_shape: Tuple[int, int, int] = (1, 18, 640)):
        """
        Args:
            model_name: Name of the model (e.g., 'EEGWaveNet')
            input_shape: Input shape (batch_size, channels, time_points)
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.builder = get_builder(model=model_name)
        
    def count_parameters(self, model: torch.nn.Module) -> Tuple[int, int]:
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def measure_flops(self, model: torch.nn.Module, device: str = 'cpu') -> Tuple[float, float]:
        """Measure FLOPs and MACs using thop"""
        if not HAS_THOP:
            return None, None
        
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape).to(device)
        
        try:
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops, params = clever_format([flops, params], "%.3f")
            return flops, params
        except Exception as e:
            print(f"  Warning: Could not calculate FLOPs for {self.model_name}: {e}")
            return None, None
    
    def measure_inference_time(
        self, 
        model: torch.nn.Module, 
        device: str = 'cpu',
        n_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Measure inference time with warmup"""
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape).to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
        
        # Synchronize for accurate timing (especially important for GPU)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                
                # Synchronize after inference
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to milliseconds
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    def benchmark(self, n_runs: int = 100, batch_size: int = 32) -> Dict:
        """Run full benchmark"""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {self.model_name}")
        print(f"{'='*60}")
        
        # Update input shape with batch size
        self.input_shape = (batch_size, self.input_shape[1], self.input_shape[2])
        
        results = {
            'model_name': self.model_name,
            'batch_size': batch_size,
            'input_shape': str(self.input_shape)
        }
        
        # Build model
        print("Building model...")
        model = self.builder()
        
        # Count parameters
        total_params, trainable_params = self.count_parameters(model)
        results['total_params'] = total_params
        results['trainable_params'] = trainable_params
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # CPU benchmark
        print("\nCPU Benchmark:")
        cpu_times = self.measure_inference_time(model, device='cpu', n_runs=n_runs)
        results['cpu_mean_ms'] = cpu_times['mean']
        results['cpu_std_ms'] = cpu_times['std']
        results['cpu_min_ms'] = cpu_times['min']
        results['cpu_max_ms'] = cpu_times['max']
        results['cpu_median_ms'] = cpu_times['median']
        print(f"  Mean: {cpu_times['mean']:.3f} ± {cpu_times['std']:.3f} ms")
        print(f"  Median: {cpu_times['median']:.3f} ms")
        print(f"  Min/Max: {cpu_times['min']:.3f} / {cpu_times['max']:.3f} ms")
        
        # Calculate throughput (samples per second)
        cpu_throughput = (batch_size * 1000) / cpu_times['mean']
        results['cpu_throughput_samples_per_sec'] = cpu_throughput
        print(f"  Throughput: {cpu_throughput:.2f} samples/sec")
        
        # GPU benchmark (if available)
        if torch.cuda.is_available():
            print("\nGPU Benchmark:")
            model_gpu = self.builder()  # Fresh model for GPU
            gpu_times = self.measure_inference_time(model_gpu, device='cuda', n_runs=n_runs)
            results['gpu_mean_ms'] = gpu_times['mean']
            results['gpu_std_ms'] = gpu_times['std']
            results['gpu_min_ms'] = gpu_times['min']
            results['gpu_max_ms'] = gpu_times['max']
            results['gpu_median_ms'] = gpu_times['median']
            print(f"  Mean: {gpu_times['mean']:.3f} ± {gpu_times['std']:.3f} ms")
            print(f"  Median: {gpu_times['median']:.3f} ms")
            print(f"  Min/Max: {gpu_times['min']:.3f} / {gpu_times['max']:.3f} ms")
            
            # GPU throughput
            gpu_throughput = (batch_size * 1000) / gpu_times['mean']
            results['gpu_throughput_samples_per_sec'] = gpu_throughput
            print(f"  Throughput: {gpu_throughput:.2f} samples/sec")
            
            # Speedup
            speedup = cpu_times['mean'] / gpu_times['mean']
            results['gpu_speedup'] = speedup
            print(f"  GPU Speedup: {speedup:.2f}x")
            
            # GPU memory usage
            try:
                torch.cuda.reset_peak_memory_stats()
                model_gpu = model_gpu.cuda()
                dummy_input = torch.randn(self.input_shape).cuda()
                with torch.no_grad():
                    _ = model_gpu(dummy_input)
                gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                results['gpu_memory_mb'] = gpu_memory_mb
                print(f"  GPU Memory: {gpu_memory_mb:.2f} MB")
            except Exception as e:
                print(f"  Could not measure GPU memory: {e}")
        else:
            print("\nGPU not available")
            results['gpu_mean_ms'] = None
            results['gpu_speedup'] = None
        
        # FLOPs (using CPU model, single sample)
        print("\nComputational Complexity:")
        single_input_shape = (1, self.input_shape[1], self.input_shape[2])
        self.input_shape = single_input_shape
        model_flops = self.builder()
        flops, params_str = self.measure_flops(model_flops, device='cpu')
        results['flops'] = flops
        results['params_formatted'] = params_str
        if flops:
            print(f"  FLOPs: {flops}")
            print(f"  Params: {params_str}")
        
        # Model summary (if torchinfo available)
        if HAS_TORCHINFO:
            print("\nModel Summary:")
            try:
                summary(model, input_size=single_input_shape, verbose=0)
            except Exception as e:
                print(f"  Could not generate summary: {e}")
        
        return results


def benchmark_all_models(
    models: List[str],
    batch_sizes: List[int] = [1, 16, 32],
    n_runs: int = 100,
    output_dir: str = 'benchmark_results'
) -> pd.DataFrame:
    """Benchmark multiple models with different batch sizes"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = []
    
    for model_name in models:
        try:
            for batch_size in batch_sizes:
                print(f"\n{'#'*60}")
                print(f"Model: {model_name} | Batch Size: {batch_size}")
                print(f"{'#'*60}")
                
                benchmarker = ModelBenchmark(model_name, input_shape=(batch_size, 18, 640))
                results = benchmarker.benchmark(n_runs=n_runs, batch_size=batch_size)
                all_results.append(results)
                
        except Exception as e:
            print(f"\n❌ Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f'benchmark_results_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Create summary table
    summary_cols = [
        'model_name', 'batch_size', 'total_params', 
        'cpu_mean_ms', 'gpu_mean_ms', 'gpu_speedup',
        'cpu_throughput_samples_per_sec', 'flops'
    ]
    summary_cols = [col for col in summary_cols if col in df.columns]
    print(df[summary_cols].to_string(index=False))
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Benchmark EEG seizure prediction models')
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['EEGWaveNet', 'EEGWaveNet-tiny', 'MB_dMGC_CWTFFNet', 'CE-stSENet'],
        help='Models to benchmark'
    )
    
    parser.add_argument(
        '--batch_sizes',
        type=int,
        nargs='+',
        default=[1, 16, 32],
        help='Batch sizes to test'
    )
    
    parser.add_argument(
        '--n_runs',
        type=int,
        default=100,
        help='Number of inference runs for timing'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='benchmark_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("EEG Model Benchmark")
    print("="*80)
    print(f"Models: {args.models}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Number of runs: {args.n_runs}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    # Run benchmarks
    df = benchmark_all_models(
        models=args.models,
        batch_sizes=args.batch_sizes,
        n_runs=args.n_runs,
        output_dir=args.output_dir
    )
    
    print("\n✨ Benchmarking complete!")


if __name__ == "__main__":
    main()