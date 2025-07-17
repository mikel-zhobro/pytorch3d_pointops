"""
Example of using Ball Query on Pointclouds.

This example demonstrates how to:
1. Find neighbors within a radius using ball query
2. Compare ball query vs KNN for local neighborhood search
"""

import torch
import numpy as np

from pytorch3d_pointops.structures.point_structure import Pointclouds
from pytorch3d_pointops.functions.ball_query import ball_query
from pytorch3d_pointops.functions.knn import knn_points


def create_sample_pointclouds():
    """Create sample point clouds for demonstration."""
    torch.manual_seed(42)

    # Create 2 point clouds with different densities
    points_list = []

    # Point cloud 1: Dense sphere
    n1 = 2000
    theta = torch.rand(n1) * 2 * np.pi
    phi = torch.rand(n1) * np.pi
    r = torch.rand(n1) * 0.5 + 0.5
    x1 = r * torch.sin(phi) * torch.cos(theta)
    y1 = r * torch.sin(phi) * torch.sin(theta)
    z1 = r * torch.cos(phi)
    points1 = torch.stack([x1, y1, z1], dim=1)
    points_list.append(points1)

    # Point cloud 2: Sparse cube
    n2 = 500
    points2 = torch.rand(n2, 3) * 4 - 2  # [-2, 2] cube (sparser)
    points_list.append(points2)

    # Create features (colors)
    features_dict = {
        "colors": [
            torch.rand(n1, 3),  # Random colors for sphere
            torch.rand(n2, 3),  # Random colors for cube
        ]
    }

    return Pointclouds(points=points_list, features=features_dict)


def example_basic_ball_query():
    """Basic example of ball query for neighbor finding."""
    print("=== Basic Ball Query Example ===")

    # Create point clouds
    pointclouds = create_sample_pointclouds()

    # Get padded representation
    points_padded = pointclouds.points_padded()
    lengths = pointclouds.num_points_per_cloud()

    print(f"Batch size: {len(pointclouds)}")
    print(f"Points per cloud: {lengths.tolist()}")
    print(f"Padded points shape: {points_padded.shape}")

    # Define query parameters
    radius = 0.3
    K = 50  # Maximum neighbors to find within radius

    # Perform ball query (query points are the same as target points)
    ball_results = ball_query(
        p1=points_padded,
        p2=points_padded,
        lengths1=lengths,
        lengths2=lengths,
        K=K,
        radius=radius,
        return_nn=True,
    )

    print("Ball query results:")
    print(f"  Distances shape: {ball_results.dists.shape}")
    print(f"  Indices shape: {ball_results.idx.shape}")
    print(f"  Neighbors shape: {ball_results.knn.shape}")
    print(f"  Radius threshold: {radius}")
    print(f"  Max neighbors per query: {K}")

    # Analyze results for first cloud
    first_cloud_dists = ball_results.dists[0]
    first_cloud_idx = ball_results.idx[0]

    # Count valid neighbors (non-padded)
    valid_mask = first_cloud_idx != -1
    neighbors_per_point = valid_mask.sum(dim=1)

    print("\nFirst cloud analysis:")
    print(f"  Average neighbors per point: {neighbors_per_point.float().mean():.2f}")
    print(f"  Min neighbors: {neighbors_per_point.min()}")
    print(f"  Max neighbors: {neighbors_per_point.max()}")

    # Check that all distances are within radius
    valid_dists = first_cloud_dists[valid_mask]
    sqrt_dists = torch.sqrt(valid_dists)
    within_radius = (sqrt_dists <= radius).all()
    print(f"  All distances within radius: {within_radius}")
    print(f"  Max distance found: {sqrt_dists.max():.4f}")
    print()


def example_ball_query_vs_knn():
    """Compare ball query vs KNN for local neighborhood search."""
    print("=== Ball Query vs KNN Comparison ===")

    # Create a simple test case with known structure
    torch.manual_seed(123)

    # Create a regular grid of points
    x = torch.linspace(-1, 1, 10)
    y = torch.linspace(-1, 1, 10)
    z = torch.linspace(-1, 1, 10)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")

    grid_points = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1
    )

    # Create a single point cloud
    points_batch = grid_points.unsqueeze(0)  # Shape: (1, 1000, 3)

    print(f"Test grid points shape: {points_batch.shape}")

    # Define parameters
    radius = 0.25  # Should capture immediate neighbors in grid
    K_ball = 30
    K_knn = 10

    # Perform ball query
    import time

    start_time = time.time()
    ball_results = ball_query(
        p1=points_batch, p2=points_batch, K=K_ball, radius=radius, return_nn=False
    )
    ball_time = time.time() - start_time

    # Perform KNN
    start_time = time.time()
    knn_results = knn_points(p1=points_batch, p2=points_batch, K=K_knn, return_nn=False)
    knn_time = time.time() - start_time

    print(f"Ball query time: {ball_time:.4f} seconds")
    print(f"KNN time: {knn_time:.4f} seconds")

    # Analyze neighbor counts
    ball_valid_mask = ball_results.idx[0] != -1
    ball_neighbors_per_point = ball_valid_mask.sum(dim=1)

    knn_valid_mask = knn_results.idx[0] != -1
    knn_neighbors_per_point = knn_valid_mask.sum(dim=1)  # noqa: F841

    print("\nNeighbor statistics:")
    print(f"Ball query (radius={radius}):")
    print(
        f"  Average neighbors per point: {ball_neighbors_per_point.float().mean():.2f}"
    )
    print(f"  Min neighbors: {ball_neighbors_per_point.min()}")
    print(f"  Max neighbors: {ball_neighbors_per_point.max()}")

    print(f"KNN (K={K_knn}):")
    print(f"  Neighbors per point: {K_knn} (fixed)")

    # Compare distances
    ball_dists = ball_results.dists[0]
    knn_dists = knn_results.dists[0]

    # Get valid distances
    ball_valid_dists = ball_dists[ball_valid_mask]
    knn_valid_dists = knn_dists[knn_valid_mask]

    print("\nDistance comparison:")
    print(f"Ball query max distance: {torch.sqrt(ball_valid_dists.max()):.4f}")
    print(f"KNN max distance: {torch.sqrt(knn_valid_dists.max()):.4f}")
    print(f"KNN mean distance: {torch.sqrt(knn_valid_dists.mean()):.4f}")

    # Check overlap: how many KNN neighbors are also in ball query results
    overlap_count = 0
    total_comparisons = 0

    for point_idx in range(min(100, points_batch.shape[1])):  # Sample first 100 points
        ball_neighbors = ball_results.idx[0, point_idx]
        ball_neighbors = ball_neighbors[ball_neighbors != -1]

        knn_neighbors = knn_results.idx[0, point_idx]
        knn_neighbors = knn_neighbors[knn_neighbors != -1]

        if len(knn_neighbors) > 0:
            overlap = len(set(ball_neighbors.tolist()) & set(knn_neighbors.tolist()))
            overlap_count += overlap
            total_comparisons += len(knn_neighbors)

    if total_comparisons > 0:
        overlap_percentage = (overlap_count / total_comparisons) * 100
        print(f"KNN neighbors also found in ball query: {overlap_percentage:.1f}%")

    print()


def main():
    """Run all examples."""
    print("Ball Query Examples")
    print("=" * 40)
    print()

    example_basic_ball_query()
    example_ball_query_vs_knn()

    print("All examples completed!")


if __name__ == "__main__":
    main()
