"""
Example of using Utility functions on Pointclouds.

This example demonstrates how to:
1. Compute point covariances for understanding local geometry
2. Use weighted mean calculations and masked operations
"""

import torch
import numpy as np

from pytorch3d_pointops.structures.point_structure import Pointclouds
from pytorch3d_pointops.functions.utils import (
    get_point_covariances,
    wmean,
    masked_gather,
)
from pytorch3d_pointops.functions.knn import knn_points


def create_sample_pointclouds():
    """Create sample point clouds with different geometric properties."""
    torch.manual_seed(42)

    points_list = []

    # Point cloud 1: Sphere (isotropic covariance)
    n1 = 1000
    theta = torch.rand(n1) * 2 * np.pi
    phi = torch.rand(n1) * np.pi
    r = torch.rand(n1) * 0.8 + 0.2
    x1 = r * torch.sin(phi) * torch.cos(theta)
    y1 = r * torch.sin(phi) * torch.sin(theta)
    z1 = r * torch.cos(phi)
    points1 = torch.stack([x1, y1, z1], dim=1)
    points_list.append(points1)

    # Point cloud 2: Ellipsoid (anisotropic covariance)
    n2 = 800
    theta = torch.rand(n2) * 2 * np.pi
    phi = torch.rand(n2) * np.pi
    r = torch.rand(n2) * 0.6 + 0.4
    x2 = r * torch.sin(phi) * torch.cos(theta) * 4.0  # Stretched in x
    y2 = r * torch.sin(phi) * torch.sin(theta) * 0.5  # Compressed in y
    z2 = r * torch.cos(phi) * 1.0  # Normal in z
    points2 = torch.stack([x2, y2, z2], dim=1)
    points_list.append(points2)

    # Create features
    features_dict = {
        "weights": [
            torch.rand(n1, 1) * 0.5 + 0.5,  # Weights between 0.5 and 1.0
            torch.rand(n2, 1) * 0.8 + 0.2,  # Weights between 0.2 and 1.0
        ],
        "values": [
            torch.randn(n1, 4),  # 4D feature vectors
            torch.randn(n2, 4),  # 4D feature vectors
        ],
        "categories": [
            torch.randint(0, 5, (n1, 1)).float(),  # 5 categories as column vector
            torch.randint(0, 3, (n2, 1)).float(),  # 3 categories as column vector
        ],
    }

    return Pointclouds(points=points_list, features=features_dict)


def example_point_covariances():
    """Example of computing point covariances for local geometry analysis."""
    print("=== Point Covariances Example ===")

    # Create point clouds
    pointclouds = create_sample_pointclouds()

    # Get padded representation
    points_padded = pointclouds.points_padded()
    lengths = pointclouds.num_points_per_cloud()

    print("Point clouds:")
    print(f"  Batch size: {len(pointclouds)}")
    print(f"  Points per cloud: {lengths.tolist()}")
    print("  Expected covariance patterns:")
    print("    Cloud 0 (sphere): isotropic (similar eigenvalues)")
    print("    Cloud 1 (ellipsoid): anisotropic (different eigenvalues)")

    # Find neighbors for covariance computation
    K = 16  # Number of neighbors for local covariance

    # Compute point covariances
    covariances, k_nearest_neighbors = get_point_covariances(points_padded, lengths, K)

    print("\nCovariance computation:")
    print(f"  Neighbors per point: {K}")
    print(f"  Covariances shape: {covariances.shape}")  # Should be (N, P, 3, 3)

    # Analyze covariances for each cloud
    for cloud_idx in range(len(pointclouds)):
        n_points = lengths[cloud_idx]
        cloud_covariances = covariances[cloud_idx, :n_points]  # Remove padding

        # Compute eigenvalues to understand local geometry
        eigenvals, _ = torch.linalg.eigh(cloud_covariances)
        eigenvals = torch.sort(eigenvals, dim=-1, descending=True)[
            0
        ]  # Sort by magnitude

        # Compute shape descriptors
        lambda1, lambda2, lambda3 = eigenvals[:, 0], eigenvals[:, 1], eigenvals[:, 2]

        # Linearity: (λ1 - λ2) / λ1
        linearity = (lambda1 - lambda2) / (lambda1 + 1e-8)

        # Planarity: (λ2 - λ3) / λ1
        planarity = (lambda2 - lambda3) / (lambda1 + 1e-8)

        # Sphericity: λ3 / λ1
        sphericity = lambda3 / (lambda1 + 1e-8)

        print(f"\nCloud {cloud_idx} geometry analysis:")
        print(f"  Valid points: {n_points}")
        print("  Eigenvalue ratios (λ1:λ2:λ3):")
        print(
            f"    Mean: {lambda1.mean():.3f}:{lambda2.mean():.3f}:{lambda3.mean():.3f}"
        )
        print("  Shape descriptors (mean ± std):")
        print(f"    Linearity: {linearity.mean():.3f} ± {linearity.std():.3f}")
        print(f"    Planarity: {planarity.mean():.3f} ± {planarity.std():.3f}")
        print(f"    Sphericity: {sphericity.mean():.3f} ± {sphericity.std():.3f}")

        # Isotropy measure: ratio of smallest to largest eigenvalue
        isotropy = lambda3 / (lambda1 + 1e-8)
        print(f"    Isotropy: {isotropy.mean():.3f} ± {isotropy.std():.3f}")

    print()


def example_weighted_operations():
    """Example of weighted mean and masked gathering operations."""
    print("=== Weighted Operations Example ===")

    # Create point clouds
    pointclouds = create_sample_pointclouds()

    # Get data for demonstration
    points_list = pointclouds.points_list()
    weights_list = pointclouds.get_features_list("weights")
    values_list = pointclouds.get_features_list("values")

    print("Data shapes:")
    for i, (points, weights, values) in enumerate(
        zip(points_list, weights_list, values_list)
    ):
        print(
            f"  Cloud {i}: points={points.shape}, weights={weights.shape}, values={values.shape}"
        )

    # Example 1: Weighted mean of 3D points
    print("\nWeighted mean computation:")
    for i, (points, weights) in enumerate(zip(points_list, weights_list)):
        # Regular mean
        regular_mean = points.mean(dim=0)

        # Weighted mean using wmean function
        weighted_mean_result = wmean(points, weights.squeeze(), dim=0, keepdim=False)

        print(f"  Cloud {i}:")
        print(
            f"    Regular mean: [{regular_mean[0].item():.3f}, {regular_mean[1].item():.3f}, {regular_mean[2].item():.3f}]"
        )
        print(
            f"    Weighted mean: [{weighted_mean_result[0].item():.3f}, {weighted_mean_result[1].item():.3f}, {weighted_mean_result[2].item():.3f}]"
        )

        # Compute center of mass using weights
        total_weight = weights.sum()
        weighted_center = (points * weights).sum(dim=0) / total_weight
        print(
            f"    Manual calculation: [{weighted_center[0].item():.3f}, {weighted_center[1].item():.3f}, {weighted_center[2].item():.3f}]"
        )

        # Check if wmean matches manual calculation
        matches = torch.allclose(weighted_mean_result, weighted_center, atol=1e-5)
        print(f"    wmean matches manual: {matches}")

    # Example 2: Masked gathering with KNN
    print("\nMasked gathering example:")

    # Use first cloud for demonstration
    points = points_list[0]
    values = values_list[0]

    # Convert to batch format
    points_batch = points.unsqueeze(0)
    values_batch = values.unsqueeze(0)

    # Find nearest neighbors
    K = 8
    knn_results = knn_points(points_batch, points_batch, K=K, return_nn=False)
    neighbor_indices = knn_results.idx

    print(f"  Original values shape: {values_batch.shape}")
    print(f"  Neighbor indices shape: {neighbor_indices.shape}")

    # Gather neighbor values using masked_gather
    gathered_values = masked_gather(values_batch, neighbor_indices)

    print(f"  Gathered values shape: {gathered_values.shape}")
    print(f"  Values per point: {K} neighbors")

    # Compute statistics of gathered neighborhoods
    valid_mask = neighbor_indices != -1
    valid_gathered = gathered_values[
        valid_mask.unsqueeze(-1).expand_as(gathered_values)
    ]
    valid_gathered = valid_gathered.reshape(-1, values_batch.shape[-1])

    print("  Neighborhood statistics:")
    print(f"    Mean gathered value: {valid_gathered.mean(dim=0)}")
    print(f"    Std gathered value: {valid_gathered.std(dim=0)}")

    # Compare with original statistics
    original_mean = values_batch.mean(dim=1)
    original_std = values_batch.std(dim=1)

    print("  Original statistics:")
    print(f"    Mean original value: {original_mean[0]}")
    print(f"    Std original value: {original_std[0]}")

    # Verify gathering: manually check a few points
    sample_point_idx = 10
    sample_neighbors = neighbor_indices[0, sample_point_idx]
    valid_neighbors = sample_neighbors[sample_neighbors != -1]

    manual_gathered = values[valid_neighbors]
    auto_gathered = gathered_values[0, sample_point_idx, : len(valid_neighbors)]

    gathering_matches = torch.allclose(manual_gathered, auto_gathered, atol=1e-6)
    print(f"  Manual vs masked_gather match (sample point): {gathering_matches}")

    print()


def main():
    """Run all examples."""
    print("Utility Functions Examples")
    print("=" * 40)
    print()

    example_point_covariances()
    example_weighted_operations()

    print("All examples completed!")


if __name__ == "__main__":
    main()
