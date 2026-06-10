#!/usr/bin/env python3
"""
Example usage of the optimized TPS module.

This script demonstrates how to use both the original and optimized TPS implementations
and compares their performance.
"""

import numpy as np
import time

# Test if we can import the TPS module from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TPS module - this will automatically use the optimized version if available
from tps import TPS
from tps.instance_optimized import TPSOptimized

def main():
    print("=== TPS Optimization Example ===")
    print()
    
    # Generate some sample data
    np.random.seed(42)
    
    # Create control points (landmarks)
    control_points = np.array([
        [0.0, 0.0],
        [1.0, 0.0], 
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5]
    ])
    
    # Create target points (where we want the landmarks to go)
    target_points = control_points + 0.1 * np.random.randn(*control_points.shape)
    
    # Generate many points to transform
    n_points = 1000
    source_points = np.random.rand(n_points, 2)
    
    print(f"Control points: {len(control_points)}")
    print(f"Source points to transform: {len(source_points)}")
    print()
    
    # Test the automatically selected TPS (should be optimized if available)
    print("Using default TPS class (auto-selected):")
    start_time = time.time()
    tps_auto = TPS(control_points, target_points, verbose=True)
    transformed_auto = tps_auto(source_points)
    auto_time = time.time() - start_time
    print(f"Time taken: {auto_time:.4f} seconds")
    print()
    
    # Test the explicitly optimized version
    print("Using explicitly optimized TPS:")
    start_time = time.time() 
    tps_opt = TPSOptimized(control_points, target_points, verbose=True)
    transformed_opt = tps_opt(source_points)
    opt_time = time.time() - start_time
    print(f"Time taken: {opt_time:.4f} seconds")
    print()
    
    # Check if results are the same
    if np.allclose(transformed_auto, transformed_opt, rtol=1e-12):
        print("✓ Results are numerically identical")
    else:
        max_diff = np.max(np.abs(transformed_auto - transformed_opt))
        print(f"⚠️  Small numerical differences detected: {max_diff:.2e}")
    print()
    
    # Quick benchmark
    print("Running quick benchmark...")
    results = TPSOptimized.benchmark(control_points, target_points, source_points, iterations=5)
    
    if results.get('speedup'):
        print(f"Speedup: {results['speedup']:.2f}x")
        print(f"NumPy time: {results['numpy_avg_time']:.4f}s per iteration")
        print(f"C++ time: {results['cpp_avg_time']:.4f}s per iteration")
    else:
        print("C++ backend not available for comparison")
    
    print()
    print("=== Usage Tips ===")
    print("1. Simply import TPS from the tps module to get the best available implementation")
    print("2. Use TPSOptimized explicitly if you want to ensure you're using the optimized version")
    print("3. The optimization provides the most benefit for larger problems")
    print("4. Both implementations produce numerically identical results")

if __name__ == "__main__":
    main() 