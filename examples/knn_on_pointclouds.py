"""
Example of using K-Nearest Neighbors (KNN) on Pointclouds.

This example demonstrates how to:
1. Find K nearest neighbors between point clouds
2. Use KNN for feature gathering and interpolation
"""

import torch
import numpy as np

from pytorch3d_pointops.structures.point_structure import Pointclouds
from pytorch3d_pointops.functions.knn import knn_points, knn_gather


def create_sample_pointclouds():
    """Create sample point clouds for demonstration."""
    torch.manual_seed(42)

    # Create 2 point clouds
    points_list = []

    # Point cloud 1: Dense sphere
    n1 = 1500
    theta = torch.rand(n1) * 2 * np.pi
    phi = torch.rand(n1) * np.pi
    r = torch.rand(n1) * 0.8 + 0.2
    x1 = r * torch.sin(phi) * torch.cos(theta)
    y1 = r * torch.sin(phi) * torch.sin(theta)
    z1 = r * torch.cos(phi)
    points1 = torch.stack([x1, y1, z1], dim=1)
    points_list.append(points1)

    # Point cloud 2: Sparse ellipsoid
    n2 = 800
    theta = torch.rand(n2) * 2 * np.pi
    phi = torch.rand(n2) * np.pi
    r = torch.rand(n2) * 0.6 + 0.4
    x2 = r * torch.sin(phi) * torch.cos(theta) * 1.5  # Stretched in x
    y2 = r * torch.sin(phi) * torch.sin(theta) * 0.8  # Compressed in y
    z2 = r * torch.cos(phi)
    points2 = torch.stack([x2, y2, z2], dim=1)
    points_list.append(points2)

    # Create features (normals and colors)
    features_dict = {
        "normals": [
            torch.nn.functional.normalize(points1, p=2, dim=1),  # Sphere normals
            torch.nn.functional.normalize(
                torch.stack([x2 / 2.25, y2 / 0.64, z2], dim=1), p=2, dim=1
            ),  # Ellipsoid normals
        ],
        "colors": [
            torch.rand(n1, 3),  # Random colors for sphere
            torch.rand(n2, 3),  # Random colors for ellipsoid
        ],
    }

    return Pointclouds(points=points_list, features=features_dict)


def example_basic_knn():
    """Basic example of KNN for nearest neighbor search."""
    print("=== Basic KNN Example ===")

    # Create point clouds
    pointclouds = create_sample_pointclouds()

    # Get padded representation
    points_padded = pointclouds.points_padded()
    lengths = pointclouds.num_points_per_cloud()

    print(f"Batch size: {len(pointclouds)}")
    print(f"Points per cloud: {lengths.tolist()}")
    print(f"Padded points shape: {points_padded.shape}")

    # Find K nearest neighbors within each cloud
    K = 10

    # Self-KNN: find neighbors within the same point cloud
    knn_results = knn_points(
        p1=points_padded,
        p2=points_padded,
        lengths1=lengths,
        lengths2=lengths,
        K=K,
        return_nn=True,
    )

    print("KNN results:")
    print(f"  Distances shape: {knn_results.dists.shape}")
    print(f"  Indices shape: {knn_results.idx.shape}")
    print(f"  Neighbors shape: {knn_results.knn.shape}")
    print(f"  K neighbors per query: {K}")

    # Analyze distance statistics
    valid_mask = knn_results.idx != -1
    valid_dists = knn_results.dists[valid_mask]
    sqrt_dists = torch.sqrt(valid_dists)

    print("\nDistance statistics:")
    print(f"  Mean distance to neighbors: {sqrt_dists.mean():.4f}")
    print(f"  Std distance to neighbors: {sqrt_dists.std():.4f}")
    print(f"  Min distance: {sqrt_dists.min():.4f}")
    print(f"  Max distance: {sqrt_dists.max():.4f}")

    # Check nearest neighbor (should be the point itself with distance 0)
    first_neighbor_dists = knn_results.dists[:, :, 0]  # First neighbor for all points
    self_distances = first_neighbor_dists[
        first_neighbor_dists >= 0
    ]  # Valid points only

    print(f"  Self-distance (should be ~0): {self_distances.mean():.6f}")
    print()


def example_knn_feature_interpolation():
    """Example of using KNN for feature interpolation between point clouds."""
    print("=== KNN Feature Interpolation Example ===")

    # Create point clouds
    pointclouds = create_sample_pointclouds()

    # Get individual clouds
    points_list = pointclouds.points_list()
    normals_list = pointclouds.get_features_list("normals")
    colors_list = pointclouds.get_features_list("colors")

    # Use first cloud as query points, second as target
    query_points = points_list[0][:200]  # Subsample for demo
    target_points = points_list[1]
    target_normals = normals_list[1]
    target_colors = colors_list[1]

    print(f"Query points shape: {query_points.shape}")
    print(f"Target points shape: {target_points.shape}")
    print(f"Target normals shape: {target_normals.shape}")
    print(f"Target colors shape: {target_colors.shape}")

    # Convert to batch format for KNN
    query_batch = query_points.unsqueeze(0)
    target_batch = target_points.unsqueeze(0)

    # Find K nearest neighbors in target cloud for each query point
    K = 5
    knn_results = knn_points(
        p1=query_batch,
        p2=target_batch,
        K=K,
        return_nn=False,  # We'll gather features manually
    )

    print("\nKNN cross-cloud search:")
    print(f"  Found {K} neighbors for each of {query_points.shape[0]} query points")

    # Gather features using KNN indices
    # Method 1: Using knn_gather function
    gathered_normals = knn_gather(target_normals.unsqueeze(0), knn_results.idx)[0]
    gathered_colors = knn_gather(target_colors.unsqueeze(0), knn_results.idx)[0]

    print(f"  Gathered normals shape: {gathered_normals.shape}")
    print(f"  Gathered colors shape: {gathered_colors.shape}")

    # Interpolate features using distance-based weights
    distances = knn_results.dists[0]  # Remove batch dimension

    # Compute weights (inverse distance weighting)
    epsilon = 1e-8  # Avoid division by zero
    weights = 1.0 / (torch.sqrt(distances) + epsilon)
    weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize

    # Weighted interpolation of normals
    interpolated_normals = (gathered_normals * weights.unsqueeze(-1)).sum(dim=1)
    interpolated_normals = torch.nn.functional.normalize(
        interpolated_normals, p=2, dim=1
    )

    # Weighted interpolation of colors
    interpolated_colors = (gathered_colors * weights.unsqueeze(-1)).sum(dim=1)

    print(f"  Interpolated normals shape: {interpolated_normals.shape}")
    print(f"  Interpolated colors shape: {interpolated_colors.shape}")

    # Quality metrics
    # Check if interpolated normals are unit vectors
    normal_lengths = torch.norm(interpolated_normals, p=2, dim=1)
    print("\nQuality metrics:")
    print("  Normal vector lengths (should be ~1.0):")
    print(f"    Mean: {normal_lengths.mean():.6f}")
    print(f"    Std: {normal_lengths.std():.6f}")

    # Check color range
    print("  Interpolated color range:")
    print(f"    Min: {interpolated_colors.min():.4f}")
    print(f"    Max: {interpolated_colors.max():.4f}")
    print(f"    Mean: {interpolated_colors.mean():.4f}")

    # Distance statistics for cross-cloud KNN
    valid_dists = distances[distances >= 0]
    sqrt_dists = torch.sqrt(valid_dists)
    print("  Cross-cloud neighbor distances:")
    print(f"    Mean: {sqrt_dists.mean():.4f}")
    print(f"    Min: {sqrt_dists.min():.4f}")
    print(f"    Max: {sqrt_dists.max():.4f}")
    print()


def main():
    """Run all examples."""
    print("K-Nearest Neighbors (KNN) Examples")
    print("=" * 45)
    print()

    example_basic_knn()
    example_knn_feature_interpolation()

    print("All examples completed!")


if __name__ == "__main__":
    main()
