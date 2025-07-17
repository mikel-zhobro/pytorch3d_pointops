"""
Example of using Farthest Point Sampling (FPS) on Pointclouds.

This example demonstrates how to:
1. Create Pointclouds from various input formats
2. Apply FPS using both optimized and naive implementations
3. Visualize the results
4. Handle batched point clouds with different numbers of points
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch3d_pointops.structures.point_structure import Pointclouds, all_close
from pytorch3d_pointops.functions.sample_farthest_points import (
    sample_farthest_points,
    sample_farthest_points_naive,
    masked_gather,
)


def create_sample_pointclouds():
    """Create sample point clouds for demonstration."""
    # Generate random 3D points for multiple point clouds
    torch.manual_seed(42)

    # Create 3 point clouds with different numbers of points
    points_list = []

    # Point cloud 1: Random sphere
    n1 = 1000
    theta = torch.rand(n1) * 2 * np.pi
    phi = torch.rand(n1) * np.pi
    r = torch.rand(n1) * 0.5 + 0.5
    x1 = r * torch.sin(phi) * torch.cos(theta)
    y1 = r * torch.sin(phi) * torch.sin(theta)
    z1 = r * torch.cos(phi)
    points1 = torch.stack([x1, y1, z1], dim=1)
    points_list.append(points1)

    # Point cloud 2: Random cube
    n2 = 800
    points2 = torch.rand(n2, 3) * 2 - 1  # [-1, 1] cube
    points_list.append(points2)

    # Point cloud 3: Random cylinder
    n3 = 1200
    theta = torch.rand(n3) * 2 * np.pi
    r = torch.rand(n3)
    z = torch.rand(n3) * 2 - 1
    x3 = r * torch.cos(theta)
    y3 = r * torch.sin(theta)
    z3 = z
    points3 = torch.stack([x3, y3, z3], dim=1)
    points_list.append(points3)

    # Create features (colors) for each point cloud
    features_dict = {
        "colors": [
            torch.rand(n1, 3),  # Random colors for sphere
            torch.rand(n2, 3),  # Random colors for cube
            torch.rand(n3, 3),  # Random colors for cylinder
        ]
    }

    return Pointclouds(points=points_list, features=features_dict)


def example_basic_fps():
    """Basic example of FPS on a single point cloud."""
    print("=== Basic FPS Example ===")

    # Create a simple point cloud
    torch.manual_seed(123)
    points = torch.rand(500, 3) * 2 - 1  # 500 random points in [-1, 1]^3

    # Convert to batch format (N=1, P=500, D=3)
    points_batch = points.unsqueeze(0)

    # Sample 50 points using FPS
    K = 50
    sampled_points, sampled_indices = sample_farthest_points(
        points_batch, K=K, random_start_point=True
    )

    print(f"Original points shape: {points_batch.shape}")
    print(f"Sampled points shape: {sampled_points.shape}")
    print(f"Sampled indices shape: {sampled_indices.shape}")
    print(f"Number of sampled points: {K}")
    print()


def example_batch_fps():
    """Example of FPS on batched point clouds with different sizes."""
    print("=== Batch FPS Example ===")

    # Create point clouds
    pointclouds = create_sample_pointclouds()

    # Get padded representation for FPS
    points_padded = pointclouds.points_padded()
    lengths = pointclouds.num_points_per_cloud()

    print(f"Batch size: {len(pointclouds)}")
    print(f"Points per cloud: {lengths.tolist()}")
    print(f"Padded points shape: {points_padded.shape}")

    # Sample different numbers of points from each cloud
    K_values = [100, 80, 150]  # Different K for each cloud

    # Apply FPS
    sampled_points, sampled_indices = sample_farthest_points(
        points_padded, lengths=lengths, K=K_values, random_start_point=True
    )

    print(f"Sampled points shape: {sampled_points.shape}")
    print(f"K values: {K_values}")
    print()


def example_fps_comparison():
    """Compare optimized vs naive FPS implementations."""
    print("=== FPS Implementation Comparison ===")

    # Create a moderately sized point cloud for timing
    torch.manual_seed(456)
    n_points = 2000
    points = torch.rand(1, n_points, 3)
    K = 200

    # Time the optimized version
    import time

    start_time = time.time()
    sampled_opt, indices_opt = sample_farthest_points(
        points, K=K, random_start_point=False
    )
    opt_time = time.time() - start_time

    # Time the naive version
    start_time = time.time()
    sampled_naive, indices_naive = sample_farthest_points_naive(
        points, K=K, random_start_point=False
    )
    naive_time = time.time() - start_time

    print(f"Optimized FPS time: {opt_time:.4f} seconds")
    print(f"Naive FPS time: {naive_time:.4f} seconds")
    print(f"Speedup: {naive_time/opt_time:.2f}x")

    # Check if results are similar (they should be identical with same random seed)
    indices_match = torch.equal(indices_opt, indices_naive)
    print(f"Indices match: {indices_match}")
    print()


def example_fps_with_features():
    """Example showing how to use FPS with point cloud features."""
    print("=== FPS with Features Example ===")

    # Create point clouds with features
    pointclouds = create_sample_pointclouds()

    # Get the first point cloud for demonstration
    points = pointclouds.points_list()[0]  # Shape: (n_points, 3)
    colors = pointclouds.get_features_list("colors")[0]  # Shape: (n_points, 3)

    # Convert to batch format
    points_batch = points.unsqueeze(0)

    # Apply FPS
    K = 100
    sampled_points, sampled_indices = sample_farthest_points(
        points_batch, K=K, random_start_point=True
    )

    # Extract corresponding features using the sampled indices
    sampled_colors = colors[sampled_indices[0]]  # Remove batch dimension and index
    sampled_colors_g = masked_gather(colors.unsqueeze(0), sampled_indices)

    print(f"Original points: {points.shape}")
    print(f"Original colors: {colors.shape}")
    print(f"Sampled points: {sampled_points[0].shape}")
    print(f"Sampled colors: {sampled_colors.shape}")
    print(f"Sampled colors (gathered): {sampled_colors_g[0].shape}")
    print(torch.allclose(sampled_colors, sampled_colors_g[0], atol=1e-6))
    print()


def visualize_fps_results():
    """Visualize FPS sampling results."""
    print("=== Visualization Example ===")

    # Create a simple 2D point cloud for easy visualization
    torch.manual_seed(789)
    n_points = 300

    # Create points in a circle
    theta = torch.linspace(0, 2 * np.pi, n_points // 2)
    r1 = torch.ones_like(theta) * 0.8
    r2 = torch.ones_like(theta) * 0.4

    x = torch.cat([r1 * torch.cos(theta), r2 * torch.cos(theta)])
    y = torch.cat([r1 * torch.sin(theta), r2 * torch.sin(theta)])
    # z = torch.zeros_like(x)

    points = torch.stack([x, y], dim=1).unsqueeze(0)

    # Apply FPS
    K = 50
    sampled_points, sampled_indices = sample_farthest_points_naive(
        points, K=K, random_start_point=False
    )

    # Create visualization
    fig = plt.figure(figsize=(12, 5))

    # Original points
    ax1 = fig.add_subplot(121)
    ax1.scatter(points[0, :, 0], points[0, :, 1], alpha=0.5, s=10, c="lightblue")
    ax1.set_title(f"Original Points (n={n_points})")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Sampled points
    ax2 = fig.add_subplot(122)
    ax2.scatter(
        points[0, :, 0],
        points[0, :, 1],
        alpha=0.3,
        s=10,
        c="lightblue",
        label="Original",
    )
    ax2.scatter(
        sampled_points[0, :, 0],
        sampled_points[0, :, 1],
        s=50,
        c="red",
        label=f"FPS Sampled (K={K})",
    )
    ax2.set_title("FPS Sampling Result")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("fps_example.png", dpi=150, bbox_inches="tight")
    print("Visualization saved as 'fps_example.png'")
    print()


def example_fps_with_pointclouds_class():
    """Example showing how to integrate FPS with the Pointclouds class."""
    print("=== FPS with Pointclouds Class Integration ===")

    # Create point clouds
    pointclouds = create_sample_pointclouds()

    # Apply FPS to get subsampled point clouds
    points_padded = pointclouds.points_padded()
    features_padded = pointclouds.features_padded()
    lengths = pointclouds.num_points_per_cloud()

    K = 100  # Same K for all clouds
    sampled_points, sampled_indices = sample_farthest_points(
        points_padded, lengths=lengths, K=K, random_start_point=True
    )

    # Create new Pointclouds object with sampled points via masked_gather
    sampled_features_dict_gathered = {
        k: masked_gather(v, sampled_indices) for k, v in features_padded.items()
    }

    # Create new Pointclouds object with sampled points (manually)
    sampled_points_list = []
    sampled_features_dict = {"colors": []}

    for i in range(len(pointclouds)):
        # Get sampled points for this cloud
        valid_mask = sampled_indices[i] != -1  # Remove padding
        cloud_sampled_points = sampled_points[i][valid_mask]
        sampled_points_list.append(cloud_sampled_points)

        # Get corresponding features
        original_colors = pointclouds.get_features_list("colors")[i]
        valid_indices = sampled_indices[i][valid_mask]
        sampled_colors = original_colors[valid_indices]
        sampled_features_dict["colors"].append(sampled_colors)

    # Create new Pointclouds object
    subsampled_pointclouds = Pointclouds(
        points=sampled_points_list, features=sampled_features_dict
    )
    subsampled_pcd_gathered = Pointclouds(
        points=sampled_points, features=None  # sampled_features_dict_gathered
    )

    print("Original pointclouds:")
    print(f"  Number of clouds: {len(pointclouds)}")
    print(f"  Points per cloud: {pointclouds.num_points_per_cloud().tolist()}")

    print("Subsampled pointclouds (MANUAL INDEXED):")
    print(f"  Number of clouds: {len(subsampled_pointclouds)}")
    print(
        f"  Points per cloud: {subsampled_pointclouds.num_points_per_cloud().tolist()}"
    )
    print("Subsampled pointclouds (GATHERED):")
    print(f"  Number of clouds: {len(subsampled_pcd_gathered)}")
    print(
        f"  Points per cloud: {subsampled_pcd_gathered.num_points_per_cloud().tolist()}"
    )
    print(
        "Pointclouds are equal:",
        all_close(subsampled_pointclouds, subsampled_pcd_gathered, verbose=True),
    )
    # Verify that gathered features match manually sampled features
    for k in sampled_features_dict_gathered.keys():
        gathered_features = sampled_features_dict_gathered[k]
        manual_features = torch.stack(sampled_features_dict[k])
        print(f"Feature '{k}':")
        print(f"  Gathered shape: {gathered_features.shape}")
        print(f"  Manual shape: {manual_features.shape}")
        print(f"  Features match: {torch.allclose(gathered_features, manual_features)}")
    print()


def main():
    """Run all examples."""
    print("Farthest Point Sampling (FPS) Examples")
    print("=" * 50)
    print()

    # Run all examples
    example_basic_fps()
    example_batch_fps()
    example_fps_comparison()
    example_fps_with_features()
    example_fps_with_pointclouds_class()

    # Only run visualization if matplotlib is available
    try:
        visualize_fps_results()
    except ImportError:
        print("Matplotlib not available, skipping visualization example")

    print("All examples completed!")


if __name__ == "__main__":
    main()
