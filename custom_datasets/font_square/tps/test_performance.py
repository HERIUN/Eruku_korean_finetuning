#!/usr/bin/env python3
"""
Performance test script for the TPS optimization.

This script compares the performance of the original numpy implementation
with the optimized C++ implementation.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# Import both implementations
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import functions
import functions_optimized
from functions_optimized import is_cpp_available, benchmark_performance

# Original TPS implementation
class TPSOriginal:
    def __init__(self, control_points, target_points, lambda_=0., solver='exact'):
        self.control_points = control_points
        self.coefficient = functions.find_coefficients(
            control_points, target_points, lambda_, solver)
    
    def __call__(self, source_points):
        return functions.transform(source_points, self.control_points, self.coefficient)

# Optimized TPS implementation  
class TPSOptimized:
    def __init__(self, control_points, target_points, lambda_=0., solver='exact'):
        self.control_points = control_points
        self.coefficient = functions_optimized.find_coefficients(
            control_points, target_points, lambda_, solver)
    
    def __call__(self, source_points):
        return functions_optimized.transform(source_points, self.control_points, self.coefficient)


def generate_test_data(num_control_points: int, num_source_points: int, 
                      dimensions: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data for TPS benchmarking."""
    np.random.seed(42)  # For reproducible results
    
    # Generate control points
    control_points = np.random.randn(num_control_points, dimensions)
    
    # Generate target points (slightly deformed control points)
    target_points = control_points + 0.1 * np.random.randn(num_control_points, dimensions)
    
    # Generate source points to transform
    source_points = np.random.randn(num_source_points, dimensions)
    
    return control_points, target_points, source_points


def run_comprehensive_benchmark() -> Dict:
    """Run comprehensive benchmarks with different problem sizes."""
    print("=== TPS Performance Benchmark ===")
    print(f"C++ backend available: {is_cpp_available()}")
    print()
    
    # Test different problem sizes
    test_configs = [
        {"control_pts": 10, "source_pts": 100, "name": "Small"},
        {"control_pts": 50, "source_pts": 500, "name": "Medium"},
        {"control_pts": 100, "source_pts": 1000, "name": "Large"},
    ]
    
    if is_cpp_available():
        test_configs.append({"control_pts": 200, "source_pts": 2000, "name": "Extra Large"})
    
    results = []
    
    for config in test_configs:
        print(f"Testing {config['name']} problem size:")
        print(f"  Control points: {config['control_pts']}")
        print(f"  Source points: {config['source_pts']}")
        
        # Generate test data
        control_pts, target_pts, source_pts = generate_test_data(
            config['control_pts'], config['source_pts']
        )
        # Test optimized implementation
        print("  Running optimized implementation...")
        start_time = time.time()
        tps_opt = TPSOptimized(control_pts, target_pts)
        transformed_opt = tps_opt(source_pts)
        opt_time = time.time() - start_time

        # Test original implementation
        print("  Running original implementation...")
        start_time = time.time()
        tps_orig = TPSOriginal(control_pts, target_pts)
        transformed_orig = tps_orig(source_pts)
        orig_time = time.time() - start_time
        
        
        # Verify results are similar (within numerical precision)
        if is_cpp_available():
            max_diff = np.max(np.abs(transformed_orig - transformed_opt))
            print(f"  Maximum difference between implementations: {max_diff:.2e}")
            
            if max_diff > 1e-10:
                print("  ⚠️  Warning: Large difference between implementations!")
        
        speedup = orig_time / opt_time if is_cpp_available() else 1.0
        
        result = {
            'name': config['name'],
            'control_pts': config['control_pts'],
            'source_pts': config['source_pts'],
            'orig_time': orig_time,
            'opt_time': opt_time,
            'speedup': speedup
        }
        
        results.append(result)
        
        print(f"  Original time: {orig_time:.4f}s")
        print(f"  Optimized time: {opt_time:.4f}s")
        if is_cpp_available():
            print(f"  Speedup: {speedup:.2f}x")
        print()
    
    return results


def plot_benchmark_results(results: List[Dict]):
    """Plot benchmark results."""
    if not results or not is_cpp_available():
        print("Skipping plotting - no C++ backend or no results")
        return
    
    names = [r['name'] for r in results]
    orig_times = [r['orig_time'] for r in results]
    opt_times = [r['opt_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot execution times
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, orig_times, width, label='Original (NumPy)', alpha=0.8)
    ax1.bar(x + width/2, opt_times, width, label='Optimized (C++)', alpha=0.8)
    ax1.set_xlabel('Problem Size')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('TPS Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot speedups
    ax2.bar(names, speedups, alpha=0.8, color='green')
    ax2.set_xlabel('Problem Size')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Performance Improvement (C++ vs NumPy)')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No improvement')
    ax2.legend()
    
    # Add speedup values on bars
    for i, speedup in enumerate(speedups):
        ax2.text(i, speedup + 0.1, f'{speedup:.1f}x', ha='center')
    
    plt.tight_layout()
    plt.savefig('tps_benchmark_results.png', dpi=150, bbox_inches='tight')
    print("Benchmark plot saved as 'tps_benchmark_results.png'")
    plt.show()


def detailed_function_benchmark():
    """Benchmark individual functions."""
    if not is_cpp_available():
        print("C++ backend not available, skipping detailed benchmark")
        return
    
    print("=== Detailed Function Benchmark ===")
    
    # Test with medium-sized data
    control_pts, target_pts, source_pts = generate_test_data(50, 500)
    
    result = benchmark_performance(control_pts, target_pts, source_pts, iterations=5)
    
    print(f"Iterations: {result['iterations']}")
    print(f"NumPy total time: {result['numpy_time']:.4f}s")
    print(f"NumPy average time: {result['numpy_avg_time']:.4f}s")
    
    if result['cpp_available']:
        print(f"C++ total time: {result['cpp_time']:.4f}s")
        print(f"C++ average time: {result['cpp_avg_time']:.4f}s")
        print(f"Overall speedup: {result['speedup']:.2f}x")


if __name__ == "__main__":
    print("Starting TPS performance evaluation...")
    print()
    
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()
    
    # Run detailed function benchmark
    detailed_function_benchmark()
    
    # Create visualizations if possible
    try:
        plot_benchmark_results(results)
    except ImportError:
        print("Matplotlib not available, skipping plots")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    # Summary
    print("=== Summary ===")
    if is_cpp_available() and results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"Average speedup across all tests: {avg_speedup:.2f}x")
        print("✓ Optimization successful!")
    else:
        print("C++ backend not available. Build the extension to see performance improvements.")
    
    print("\nBenchmark completed!") 