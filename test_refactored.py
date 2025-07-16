#!/usr/bin/env python3

import torch
from pytorch3d_utils.structures.point_structure import Pointclouds


def test_basic_functionality():
    """Test basic functionality of the refactored Pointclouds class"""

    # Create sample point clouds
    points_list = [
        torch.randn(100, 3),  # First cloud with 100 points
        torch.randn(150, 3),  # Second cloud with 150 points
        torch.randn(80, 3),  # Third cloud with 80 points
    ]

    # Create some features
    normals_list = [
        torch.randn(100, 3),  # Normals for first cloud
        torch.randn(150, 3),  # Normals for second cloud
        torch.randn(80, 3),  # Normals for third cloud
    ]

    colors_list = [
        torch.randn(100, 3),  # RGB colors for first cloud
        torch.randn(150, 3),  # RGB colors for second cloud
        torch.randn(80, 3),  # RGB colors for third cloud
    ]

    # Test 1: Create Pointclouds with dictionary features
    print("Test 1: Creating Pointclouds with dictionary features")
    features_dict = {"normals": normals_list, "colors": colors_list}

    pc = Pointclouds(points=points_list, features=features_dict)
    print(f"Created pointclouds with {len(pc)} clouds")
    print(f"Point counts per cloud: {pc.num_points_per_cloud()}")

    # Test 2: Access features by name
    print("\nTest 2: Accessing features by name")
    normals_retrieved = pc.get_features_list("normals")
    colors_retrieved = pc.get_features_list("colors")
    print(f"Retrieved normals: {len(normals_retrieved)} clouds")
    print(f"Retrieved colors: {len(colors_retrieved)} clouds")

    # Test 3: Access all features
    print("\nTest 3: Accessing all features")
    all_features = pc.features_list()
    print(f"All features: {list(all_features.keys())}")

    # Test 4: Test packed representation
    print("\nTest 4: Testing packed representation")
    points_packed = pc.points_packed()
    normals_packed = pc.get_features_packed("normals")
    colors_packed = pc.get_features_packed("colors")
    print(f"Packed points shape: {points_packed.shape}")
    print(f"Packed normals shape: {normals_packed.shape}")
    print(f"Packed colors shape: {colors_packed.shape}")

    # Test 5: Test padded representation
    print("\nTest 5: Testing padded representation")
    points_padded = pc.points_padded()
    normals_padded = pc.get_features_padded("normals")
    colors_padded = pc.get_features_padded("colors")
    print(f"Padded points shape: {points_padded.shape}")
    print(f"Padded normals shape: {normals_padded.shape}")
    print(f"Padded colors shape: {colors_padded.shape}")

    # Test 6: Test indexing
    print("\nTest 7: Testing indexing")
    pc_subset = pc[0:2]  # Get first two clouds
    print(f"Subset has {len(pc_subset)} clouds")
    subset_features = pc_subset.features_list()
    print(f"Subset features: {list(subset_features.keys())}")

    # Test 8: Test get_cloud method
    print("\nTest 8: Testing get_cloud method")
    cloud_points, cloud_features = pc.get_cloud(0)
    print(f"First cloud points shape: {cloud_points.shape}")
    print(f"First cloud features: {list(cloud_features.keys())}")
    for feature_name, feature_tensor in cloud_features.items():
        print(f"  {feature_name}: {feature_tensor.shape}")

    print("\nAll tests passed successfully!")


def test_empty_features():
    """Test with no features"""
    print("\n" + "=" * 50)
    print("Testing with no features")

    points_list = [
        torch.randn(50, 3),
        torch.randn(75, 3),
    ]

    pc = Pointclouds(points=points_list)
    print(f"Created pointclouds with {len(pc)} clouds, no features")

    # Should return empty dictionary
    all_features = pc.features_list()
    print(f"Features dict: {all_features}")

    # Should return None for specific features
    normals = pc.get_features_list("normals")
    print(f"Normals: {normals}")

    print("Empty features test passed!")


def test_update_padded():
    """Test the updated update_padded method"""
    print("Testing update_padded method...")

    # Create initial point clouds
    points_list = [torch.randn(50, 3), torch.randn(60, 3), torch.randn(40, 3)]

    # Create initial features
    normals_list = [torch.randn(50, 3), torch.randn(60, 3), torch.randn(40, 3)]

    colors_list = [torch.randn(50, 3), torch.randn(60, 3), torch.randn(40, 3)]

    initial_features = {"normals": normals_list, "colors": colors_list}

    # Create initial pointclouds
    pc = Pointclouds(points=points_list, features=initial_features)
    print(f"Initial pointclouds: {len(pc)} clouds")
    print(f"Initial features: {list(pc.features_list().keys())}")

    # Get current padded representations
    old_points_padded = pc.points_padded()
    old_features_padded = pc.features_padded()
    print(f"Original points shape: {old_points_padded.shape}")
    print(f"Original normals shape: {old_features_padded['normals'].shape}")
    print(f"Original colors shape: {old_features_padded['colors'].shape}")

    # Test 1: Update points only
    print("\nTest 1: Update points only")
    new_points = old_points_padded * 2.0  # Scale points by 2

    updated_pc = pc.update_padded(new_points)
    print(f"Updated pointclouds: {len(updated_pc)} clouds")

    # Check that points changed but features remained the same
    updated_points = updated_pc.points_padded()
    updated_features = updated_pc.features_padded()

    print(f"Points changed: {not torch.allclose(old_points_padded, updated_points)}")
    print(
        f"Features preserved: {torch.allclose(old_features_padded['normals'], updated_features['normals'])}"
    )
    print(
        f"Features preserved: {torch.allclose(old_features_padded['colors'], updated_features['colors'])}"
    )

    # Test 2: Update points and features
    print("\nTest 2: Update points and features")
    new_normals = old_features_padded["normals"] * 0.5
    new_features = {"normals": new_normals}

    updated_pc2 = pc.update_padded(new_points, new_features_padded=new_features)
    updated_features2 = updated_pc2.features_padded()

    print(
        f"Normals changed: {not torch.allclose(old_features_padded['normals'], updated_features2['normals'])}"
    )
    # Note: colors should be gone since we only provided normals in the update
    print(f"Colors removed: {'colors' not in updated_features2}")

    print("\nAll update_padded tests passed!")


if __name__ == "__main__":
    test_basic_functionality()
    test_empty_features()
    test_update_padded()
