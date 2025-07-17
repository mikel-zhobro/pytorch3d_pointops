"""
Example of using Packed to Padded tensor conversions on Pointclouds.

This example demonstrates how to:
1. Convert between packed and padded representations
2. Use packed format for efficient processing of variable-length point clouds
"""

import torch
import numpy as np

from pytorch3d_pointops.structures.point_structure import Pointclouds
from pytorch3d_pointops.functions.packed_to_padded import (
    packed_to_padded,
    padded_to_packed,
)


def create_variable_pointclouds():
    """Create point clouds with very different sizes."""
    torch.manual_seed(42)

    points_list = []

    # Point cloud 1: Small cloud (100 points)
    n1 = 100
    points1 = torch.randn(n1, 3) * 0.5
    points_list.append(points1)

    # Point cloud 2: Medium cloud (500 points)
    n2 = 500
    theta = torch.rand(n2) * 2 * np.pi
    r = torch.rand(n2) * 2
    z = torch.rand(n2) * 2 - 1
    x2 = r * torch.cos(theta)
    y2 = r * torch.sin(theta)
    z2 = z
    points2 = torch.stack([x2, y2, z2], dim=1)
    points_list.append(points2)

    # Point cloud 3: Large cloud (2000 points)
    n3 = 2000
    points3 = torch.randn(n3, 3) * 1.5
    points_list.append(points3)

    # Point cloud 4: Very small cloud (25 points)
    n4 = 25
    points4 = torch.randn(n4, 3) * 0.2
    points_list.append(points4)

    # Create features
    features_dict = {
        "intensities": [
            torch.rand(n1, 1),
            torch.rand(n2, 1),
            torch.rand(n3, 1),
            torch.rand(n4, 1),
        ],
        "labels": [
            torch.randint(0, 5, (n1, 1)).float(),
            torch.randint(0, 5, (n2, 1)).float(),
            torch.randint(0, 5, (n3, 1)).float(),
            torch.randint(0, 5, (n4, 1)).float(),
        ],
    }

    return Pointclouds(points=points_list, features=features_dict)


def example_packed_padded_conversion():
    """Example of converting between packed and padded representations."""
    print("=== Packed to Padded Conversion Example ===")

    # Create variable-size point clouds
    pointclouds = create_variable_pointclouds()

    # Get packed representation (concatenated)
    points_packed = pointclouds.points_packed()
    intensities_packed = pointclouds.get_features_packed("intensities")

    # Get lengths for each cloud
    lengths = pointclouds.num_points_per_cloud()

    print("Original point clouds:")
    print(f"  Number of clouds: {len(pointclouds)}")
    print(f"  Points per cloud: {lengths.tolist()}")
    print(f"  Total points: {lengths.sum()}")

    print("\nPacked representation:")
    print(f"  Packed points shape: {points_packed.shape}")
    print(f"  Packed intensities shape: {intensities_packed.shape}")

    # Convert packed to padded
    max_size = lengths.max().item()  # Maximum number of points in any cloud
    # Compute first indices for each cloud in the packed tensor
    first_idxs = torch.cat([torch.tensor([0]), lengths.cumsum(0)[:-1]])
    points_padded = packed_to_padded(points_packed, first_idxs, max_size)
    intensities_padded = packed_to_padded(intensities_packed, first_idxs, max_size)

    print("\nPadded representation:")
    print(f"  Padded points shape: {points_padded.shape}")
    print(f"  Padded intensities shape: {intensities_padded.shape}")
    print(f"  Padding size: {points_padded.shape[1] - lengths.max()}")

    # Convert back to packed
    total_points = lengths.sum().item()
    points_repacked = padded_to_packed(points_padded, first_idxs, total_points)
    intensities_repacked = padded_to_packed(
        intensities_padded, first_idxs, total_points
    )

    print("\nRepacked representation:")
    print(f"  Repacked points shape: {points_repacked.shape}")
    print(f"  Repacked intensities shape: {intensities_repacked.shape}")

    # Verify round-trip conversion
    points_match = torch.allclose(points_packed, points_repacked, atol=1e-6)
    intensities_match = torch.allclose(
        intensities_packed, intensities_repacked, atol=1e-6
    )

    print("\nRound-trip verification:")
    print(f"  Points match: {points_match}")
    print(f"  Intensities match: {intensities_match}")

    # Analyze padding efficiency
    total_elements = points_padded.numel()
    valid_elements = points_packed.numel()
    padding_ratio = (total_elements - valid_elements) / total_elements

    print("\nPadding efficiency:")
    print(f"  Total padded elements: {total_elements}")
    print(f"  Valid elements: {valid_elements}")
    print(f"  Padding ratio: {padding_ratio:.2%}")
    print()


def example_memory_efficiency_comparison():
    """Compare memory usage between packed and padded representations."""
    print("=== Memory Efficiency Comparison ===")

    # Create point clouds with extreme size differences
    torch.manual_seed(123)

    sizes = [50, 100, 5000, 10, 1000, 20, 3000]  # Very variable sizes
    points_list = []

    for size in sizes:
        points = torch.randn(size, 3)
        points_list.append(points)

    features_dict = {
        "features": [torch.randn(size, 16) for size in sizes]  # 16D features
    }

    pointclouds = Pointclouds(points=points_list, features=features_dict)
    lengths = pointclouds.num_points_per_cloud()

    print(f"Point cloud sizes: {sizes}")
    print(f"Total points: {sum(sizes)}")
    print(
        f"Size variance: min={min(sizes)}, max={max(sizes)}, ratio={max(sizes)/min(sizes):.1f}x"
    )

    # Memory usage for packed representation
    points_packed = pointclouds.points_packed()
    features_packed = pointclouds.get_features_packed("features")

    packed_points_memory = points_packed.numel() * points_packed.element_size()
    packed_features_memory = features_packed.numel() * features_packed.element_size()
    packed_total_memory = packed_points_memory + packed_features_memory

    # Memory usage for padded representation
    max_size = lengths.max().item()
    first_idxs = torch.cat([torch.tensor([0]), lengths.cumsum(0)[:-1]])
    points_padded = packed_to_padded(points_packed, first_idxs, max_size)
    features_padded = packed_to_padded(features_packed, first_idxs, max_size)

    padded_points_memory = points_padded.numel() * points_padded.element_size()
    padded_features_memory = features_padded.numel() * features_padded.element_size()
    padded_total_memory = padded_points_memory + padded_features_memory

    print("\nMemory usage (bytes):")
    print("  Packed format:")
    print(f"    Points: {packed_points_memory:,}")
    print(f"    Features: {packed_features_memory:,}")
    print(f"    Total: {packed_total_memory:,}")

    print("  Padded format:")
    print(f"    Points: {padded_points_memory:,}")
    print(f"    Features: {padded_features_memory:,}")
    print(f"    Total: {padded_total_memory:,}")

    memory_overhead = (padded_total_memory - packed_total_memory) / packed_total_memory
    space_efficiency = packed_total_memory / padded_total_memory

    print("\nEfficiency metrics:")
    print(f"  Memory overhead: {memory_overhead:.1%}")
    print(f"  Space efficiency: {space_efficiency:.1%}")
    print(f"  Wasted space: {padded_total_memory - packed_total_memory:,} bytes")

    # Timing comparison for operations
    import time

    # Time packed operations
    start_time = time.time()
    for _ in range(100):
        # Simulate some operations on packed data
        _ = points_packed.mean(dim=0)
        _ = features_packed.std(dim=0)
    packed_time = time.time() - start_time

    # Time padded operations
    start_time = time.time()
    for _ in range(100):
        # Simulate some operations on padded data (need to handle padding)
        mask = torch.arange(points_padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        mask_3d = mask.unsqueeze(-1).expand_as(points_padded)
        masked_points = points_padded[mask_3d].reshape(-1, 3)
        _ = masked_points.mean(dim=0)

        mask_feat = mask.unsqueeze(-1).expand_as(features_padded)
        masked_features = features_padded[mask_feat].reshape(-1, 16)
        _ = masked_features.std(dim=0)
    padded_time = time.time() - start_time

    print("\nOperation timing (100 iterations):")
    print(f"  Packed format: {packed_time:.4f} seconds")
    print(f"  Padded format: {padded_time:.4f} seconds")
    print(f"  Speedup factor: {padded_time/packed_time:.2f}x")
    print()


def main():
    """Run all examples."""
    print("Packed to Padded Conversion Examples")
    print("=" * 50)
    print()

    example_packed_padded_conversion()
    example_memory_efficiency_comparison()

    print("All examples completed!")


if __name__ == "__main__":
    main()
