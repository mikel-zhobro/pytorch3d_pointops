"""
Example comparing CUDA kernel performance vs Python implementations.

This example demonstrates:
1. Performance comparison between CUDA and Python implementations
2. Memory efficiency of CUDA operations
3. Scalability analysis across different point cloud sizes
"""

import torch
import numpy as np
import time
from typing import List, Tuple

from pytorch3d_pointops.structures.point_structure import Pointclouds
from pytorch3d_pointops.functions.sample_farthest_points import (
    sample_farthest_points,
    sample_farthest_points_naive,
)
from pytorch3d_pointops.functions.knn import knn_points
from pytorch3d_pointops.functions.ball_query import ball_query


def python_knn_naive(
    points1: torch.Tensor, points2: torch.Tensor, K: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Naive Python implementation of KNN for comparison."""
    N1, P1, D = points1.shape
    N2, P2, _ = points2.shape

    # Compute pairwise distances
    dists = torch.cdist(points1, points2, p=2)  # (N, P1, P2)

    # Find K nearest neighbors
    knn_dists, knn_idx = torch.topk(dists, K, dim=2, largest=False, sorted=True)

    return knn_dists, knn_idx


def python_ball_query_naive(
    points1: torch.Tensor, points2: torch.Tensor, radius: float, K: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Naive Python implementation of ball query for comparison."""
    N1, P1, D = points1.shape
    N2, P2, _ = points2.shape

    # Compute pairwise distances
    dists = torch.cdist(points1, points2, p=2)  # (N, P1, P2)

    # Find points within radius
    within_radius = dists <= radius

    # Initialize results
    result_dists = torch.full(
        (N1, P1, K), float("inf"), dtype=points1.dtype, device=points1.device
    )
    result_idx = torch.full((N1, P1, K), -1, dtype=torch.long, device=points1.device)

    for n in range(N1):
        for p in range(P1):
            valid_mask = within_radius[n, p]
            valid_dists = dists[n, p][valid_mask]
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_dists) > 0:
                # Sort by distance and take top K
                sorted_dists, sort_idx = torch.sort(valid_dists)
                num_neighbors = min(K, len(sorted_dists))

                result_dists[n, p, :num_neighbors] = sorted_dists[:num_neighbors]
                result_idx[n, p, :num_neighbors] = valid_indices[
                    sort_idx[:num_neighbors]
                ]

    return (
        result_dists**2,
        result_idx,
    )  # Return squared distances to match CUDA implementation


def create_test_pointclouds(sizes: List[int]) -> Pointclouds:
    """Create point clouds of different sizes for testing."""
    torch.manual_seed(42)

    points_list = []
    for size in sizes:
        # Create random point cloud
        points = torch.randn(size, 3) * 2.0
        points_list.append(points)

    return Pointclouds(points=points_list)


def benchmark_function(func, *args, num_runs: int = 10, warmup: int = 3, **kwargs):
    """Benchmark a function with multiple runs."""
    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)

    # Synchronize GPU if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Actual timing
    start_time = time.time()
    for _ in range(num_runs):
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs

    return result, avg_time


def example_knn_performance_comparison():
    """Compare KNN performance between CUDA and Python implementations."""
    print("=== KNN Performance Comparison ===")

    # Test different point cloud sizes
    sizes = [100, 500, 1000, 2000, 10000, 32000]
    K = 16

    print(f"Comparing KNN performance (K={K}):")
    print(
        f"{'Size':<8} {'CUDA (ms)':<12} {'Python (ms)':<14} {'Speedup':<10} {'Memory (MB)':<12}"
    )
    print("-" * 70)

    for size in sizes:
        # Create test data
        points = torch.randn(1, size, 3)
        if torch.cuda.is_available():
            points_gpu = points.cuda()
        else:
            points_gpu = points

        # Test CUDA implementation
        try:
            _, cuda_time = benchmark_function(
                knn_points, points_gpu, points_gpu, K=K, return_nn=False
            )
            cuda_time_ms = cuda_time * 1000
        except Exception as e:
            cuda_time_ms = float("inf")
            print(f"CUDA error for size {size}: {e}")

        # Test Python implementation
        _, python_time = benchmark_function(python_knn_naive, points, points, K)
        python_time_ms = python_time * 1000

        # Calculate speedup
        speedup = python_time_ms / cuda_time_ms if cuda_time_ms != float("inf") else 0

        # Estimate memory usage
        memory_mb = (points.numel() * points.element_size() * 2) / (1024 * 1024)

        print(
            f"{size:<8} {cuda_time_ms:<12.2f} {python_time_ms:<14.2f} {speedup:<10.1f}x {memory_mb:<12.2f}"
        )

    print()


def example_ball_query_performance_comparison():
    """Compare ball query performance between CUDA and Python implementations."""
    print("=== Ball Query Performance Comparison ===")

    # Test parameters
    sizes = [
        100,
        500,
        1000,
        10000,
    ]  # Smaller sizes for ball query due to O(n²) complexity
    radius = 0.5
    K = 20

    print(f"Comparing Ball Query performance (radius={radius}, K={K}):")
    print(f"{'Size':<8} {'CUDA (ms)':<12} {'Python (ms)':<14} {'Speedup':<10}")
    print("-" * 55)

    for size in sizes:
        # Create test data
        points = torch.randn(1, size, 3)
        if torch.cuda.is_available():
            points_gpu = points.cuda()
        else:
            points_gpu = points

        # Test CUDA implementation
        try:
            _, cuda_time = benchmark_function(
                ball_query, points_gpu, points_gpu, K=K, radius=radius, return_nn=False
            )
            cuda_time_ms = cuda_time * 1000
        except Exception as e:
            cuda_time_ms = float("inf")
            print(f"CUDA error for size {size}: {e}")

        # Test Python implementation
        _, python_time = benchmark_function(
            python_ball_query_naive, points, points, radius, K
        )
        python_time_ms = python_time * 1000

        # Calculate speedup
        speedup = python_time_ms / cuda_time_ms if cuda_time_ms != float("inf") else 0

        print(
            f"{size:<8} {cuda_time_ms:<12.2f} {python_time_ms:<14.2f} {speedup:<10.1f}x"
        )

    print()


def example_fps_performance_comparison():
    """Compare FPS performance between optimized CUDA and naive implementations."""
    print("=== Farthest Point Sampling Performance Comparison ===")

    # Test different point cloud sizes
    sizes = [500, 1000, 2000, 5000]
    K_ratio = 0.1  # Sample 10% of points

    print(f"Comparing FPS performance (sampling {K_ratio * 100:.0f}% of points):")
    print(f"{'Size':<8} {'K':<6} {'CUDA (ms)':<12} {'Naive (ms)':<12} {'Speedup':<10}")
    print("-" * 60)

    for size in sizes:
        K = int(size * K_ratio)

        # Create test data
        points = torch.randn(1, size, 3)
        if torch.cuda.is_available():
            points_gpu = points.cuda()
        else:
            points_gpu = points

        # Test CUDA implementation
        try:
            _, cuda_time = benchmark_function(
                sample_farthest_points, points_gpu, K=K, random_start_point=False
            )
            cuda_time_ms = cuda_time * 1000
        except Exception as e:
            cuda_time_ms = float("inf")
            print(f"CUDA error for size {size}: {e}")

        # Test naive implementation
        _, naive_time = benchmark_function(
            sample_farthest_points_naive, points, K=K, random_start_point=False
        )
        naive_time_ms = naive_time * 1000

        # Calculate speedup
        speedup = naive_time_ms / cuda_time_ms if cuda_time_ms != float("inf") else 0

        print(
            f"{size:<8} {K:<6} {cuda_time_ms:<12.2f} {naive_time_ms:<12.2f} {speedup:<10.1f}x"
        )

    print()


def example_memory_efficiency():
    """Demonstrate memory efficiency of CUDA operations."""
    print("=== Memory Efficiency Analysis ===")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory analysis")
        return

    sizes = [1000, 5000, 10000, 32000, 64000]

    print("Memory usage comparison for KNN operations:")
    print(
        f"{'Size':<8} {'Input (MB)':<12} {'Output (MB)':<12} {'Peak (MB)':<12} {'Efficiency':<12}"
    )
    print("-" * 70)

    for size in sizes:
        K = 32

        # Create test data
        points = torch.randn(1, size, 3).cuda()

        # Measure memory before operation
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)

        # Perform KNN operation
        result = knn_points(points, points, K=K, return_nn=False)  # noqa: F841

        # Measure memory after operation
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        final_memory = torch.cuda.memory_allocated() / (1024 * 1024)

        # Calculate memory usage
        input_mb = points.numel() * points.element_size() / (1024 * 1024)
        output_mb = final_memory - initial_memory
        peak_mb = peak_memory - initial_memory
        efficiency = output_mb / peak_mb if peak_mb > 0 else 0

        print(
            f"{size:<8} {input_mb:<12.2f} {output_mb:<12.2f} {peak_mb:<12.2f} {efficiency:<12.2f}"
        )

    print()


def example_scalability_analysis():
    """Analyze how performance scales with input size."""
    print("=== Scalability Analysis ===")

    # Test exponentially increasing sizes
    base_sizes = [100, 200, 500, 1000, 2000, 10000]

    print("KNN scalability (theoretical O(n log n) vs actual):")
    print(f"{'Size':<8} {'Time (ms)':<12} {'Time/n':<12} {'Time/(n log n)':<15}")
    print("-" * 55)

    times = []
    for size in base_sizes:
        points = torch.randn(1, size, 3)
        if torch.cuda.is_available():
            points = points.cuda()

        # Benchmark KNN
        try:
            _, time_taken = benchmark_function(
                knn_points, points, points, K=16, return_nn=False, num_runs=5
            )
            time_ms = time_taken * 1000
            times.append(time_ms)

            # Calculate complexity metrics
            time_per_n = time_ms / size
            time_per_nlogn = time_ms / (size * np.log(size))

            print(
                f"{size:<8} {time_ms:<12.2f} {time_per_n:<12.4f} {time_per_nlogn:<15.6f}"
            )
        except Exception as e:
            print(f"{size:<8} Error: {e}")

    # Analyze growth rate
    if len(times) >= 2:
        growth_rates = []
        for i in range(1, len(times)):
            size_ratio = base_sizes[i] / base_sizes[i - 1]
            time_ratio = times[i] / times[i - 1]
            growth_rate = np.log(time_ratio) / np.log(size_ratio)
            growth_rates.append(growth_rate)

        avg_growth = np.mean(growth_rates)
        print(f"\nAverage empirical complexity: O(n^{avg_growth:.2f})")
        print("Theoretical complexity: O(n log n) ≈ O(n^1.0 to n^1.3)")

    print()


def example_batch_processing_efficiency():
    """Compare efficiency of batch vs individual processing."""
    print("=== Batch Processing Efficiency ===")

    # Test parameters
    individual_size = 500
    batch_sizes = [1, 2, 4, 8, 16, 32]
    K = 16

    print("Comparing batch vs individual KNN processing:")
    print(
        f"{'Batch Size':<12} {'Total Time (ms)':<16} {'Time per Cloud (ms)':<20} {'Efficiency':<12}"
    )
    print("-" * 70)

    # Baseline: single cloud
    points_single = torch.randn(1, individual_size, 3)
    if torch.cuda.is_available():
        points_single = points_single.cuda()

    _, baseline_time = benchmark_function(
        knn_points, points_single, points_single, K=K, return_nn=False
    )
    baseline_ms = baseline_time * 1000

    for batch_size in batch_sizes:
        # Create batch of point clouds
        points_batch = torch.randn(batch_size, individual_size, 3)
        if torch.cuda.is_available():
            points_batch = points_batch.cuda()

        # Create lengths tensor
        lengths = torch.full((batch_size,), individual_size, dtype=torch.long)
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        # Benchmark batched operation
        try:
            _, batch_time = benchmark_function(
                knn_points,
                points_batch,
                points_batch,
                lengths1=lengths,
                lengths2=lengths,
                K=K,
                return_nn=False,
            )
            batch_ms = batch_time * 1000
            time_per_cloud = batch_ms / batch_size
            efficiency = baseline_ms / time_per_cloud

            print(
                f"{batch_size:<12} {batch_ms:<16.2f} {time_per_cloud:<20.2f} {efficiency:<12.2f}x"
            )
        except Exception as e:
            print(f"{batch_size:<12} Error: {e}")

    print()


def main():
    """Run all performance comparison examples."""
    print("CUDA vs Python Performance Comparison")
    print("=" * 50)
    print()

    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(
            f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("CUDA not available - running CPU comparisons only")

    print()

    example_knn_performance_comparison()
    example_ball_query_performance_comparison()
    example_fps_performance_comparison()

    if torch.cuda.is_available():
        example_memory_efficiency()
        example_scalability_analysis()
        example_batch_processing_efficiency()

    print("Performance analysis completed!")
    print("\nKey takeaways:")
    print("1. CUDA implementations are significantly faster for large point clouds")
    print("2. Memory efficiency is optimized for GPU operations")
    print("3. Batch processing provides additional performance benefits")
    print("4. Speedup increases with problem size due to GPU parallelization")


if __name__ == "__main__":
    main()
