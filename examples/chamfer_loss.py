#!/usr/bin/env python3

import torch
import sys

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, "/is/sg2/mzhobro/AL_projects/tools/point_ops")

from pytorch3d_pointops.structures.point_structure import Pointclouds
from pytorch3d_pointops.functions.chamfer import chamfer_distance


def test_chamfer_with_features():
    """Test the updated chamfer distance function with feature-based point clouds."""

    # Create test data
    N, P1, P2 = 2, 100, 120
    device = torch.device("cpu")

    # Create random point clouds
    points1 = torch.randn(N, P1, 3, device=device)
    points2 = torch.randn(N, P2, 3, device=device)

    # Create random features (e.g., normals and colors)
    normals1 = torch.randn(N, P1, 3, device=device)
    normals2 = torch.randn(N, P2, 3, device=device)
    colors1 = torch.randn(N, P1, 3, device=device)
    colors2 = torch.randn(N, P2, 3, device=device)

    features1 = {"normals": normals1, "colors": colors1}
    features2 = {"normals": normals2, "colors": colors2}

    # Test 3: With multiple features
    loss_multi, loss_features_multi = chamfer_distance(
        points1,
        points2,
        x_features=features1,
        y_features=features2,
        feature_names=["normals", "colors"],
    )
    print(f"Chamfer loss points only: {loss_multi:.6f}")
    print(f"Feature loss (normals): {loss_features_multi['normals']:.6f}")
    print(f"Feature loss (colors): {loss_features_multi['colors']:.6f}")

    print("\nTesting with Pointclouds objects...")

    # Test 4: With Pointclouds objects
    # Create point list format
    points1_list = [points1[0], points1[1]]
    points2_list = [points2[0], points2[1]]
    normals1_list = [normals1[0], normals1[1]]
    normals2_list = [normals2[0], normals2[1]]
    colors1_list = [colors1[0], colors1[1]]
    colors2_list = [colors2[0], colors2[1]]

    features1_dict = {"normals": normals1_list, "colors": colors1_list}
    features2_dict = {"normals": normals2_list, "colors": colors2_list}

    pc1 = Pointclouds(points=points1_list, features=features1_dict)
    pc2 = Pointclouds(points=points2_list, features=features2_dict)

    loss_pc, loss_features_pc = chamfer_distance(
        pc1, pc2, feature_names=["normals", "colors"]
    )
    print(f"Chamfer loss with Pointclouds (points): {loss_pc:.6f}")
    print(f"Feature loss with Pointclouds (normals): {loss_features_pc['normals']:.6f}")
    print(f"Feature loss with Pointclouds (colors): {loss_features_pc['colors']:.6f}")

    print("\nTesting single directional...")

    # Test 5: Single directional
    loss_single, loss_f_single = chamfer_distance(
        pc1, pc2, feature_names=["normals"], single_directional=True
    )
    print(f"Single directional chamfer loss (points): {loss_single:.6f}")
    print(f"Single directional normals loss (normals): {loss_f_single['normals']:.6f}")
    print(f"Feature losses present: {loss_f_single.keys()}")

    print("\nTesting with multiple features in Pointclouds...")

    # Test 6: Multiple features with Pointclouds
    loss_pc_multi, loss_features_pc_multi = chamfer_distance(
        pc1, pc2, feature_names=["normals", "colors"]
    )
    print(f"Chamfer loss with Pointclouds (points): {loss_pc_multi:.6f}")
    print(f"Feature loss (normals): {loss_features_pc_multi['normals']:.6f}")
    print(f"Feature loss (colors): {loss_features_pc_multi['colors']:.6f}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_chamfer_with_features()
