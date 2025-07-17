# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d_pointops.functions.knn import knn_gather, knn_points
from pytorch3d_pointops.structures.point_structure import Pointclouds


def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: Union[str, None]
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction is not None and point_reduction not in ["mean", "sum", "max"]:
        raise ValueError(
            'point_reduction must be one of ["mean", "sum", "max"] or None'
        )
    if point_reduction is None and batch_reduction is not None:
        raise ValueError("Batch reduction must be None if point_reduction is None")


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    features: Union[torch.Tensor, dict, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded features.
    Otherwise, return the input points (and features) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        features = points.features_padded()  # returns a dict or empty dict
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if features is not None:
            if isinstance(features, dict):
                # Validate each feature tensor in the dict
                for feature_name, feature_tensor in features.items():
                    if feature_tensor is not None and feature_tensor.ndim != 3:
                        raise ValueError(
                            f"Expected {feature_name} to be of shape (N, P, C)"
                        )
            elif torch.is_tensor(features) and features.ndim != 3:
                raise ValueError("Expected features to be of shape (N, P, C)")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, features


def _chamfer_distance_single_direction(
    x,
    y,
    x_lengths,
    y_lengths,
    x_features,
    y_features,
    weights,
    point_reduction: Union[str, None],
    norm: int,
    abs_cosine: bool,
    feature_names: Union[list, None] = None,
):
    # https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/loss/chamfer.py  # noqa: E501
    # assert that feature names are provided in both x and y features
    if feature_names and x_features is not None and y_features is not None:
        for feature_name in feature_names:
            if feature_name not in x_features:
                raise ValueError(f"Feature '{feature_name}' is missing in x_features.")
            if feature_name not in y_features:
                raise ValueError(f"Feature '{feature_name}' is missing in y_features.")

    return_features = (
        x_features is not None
        and y_features is not None
        and feature_names is not None
        and len(feature_names) > 0
    )

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_features_x = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)

    cham_features_x = None
    if return_features:
        # Compute cosine similarity for each specified feature
        cham_features_x = {}
        for feature_name in feature_names:
            x_feature = x_features[feature_name]
            y_feature = y_features[feature_name]

            # Gather the features using the indices and keep only value for k=0
            x_feature_near = knn_gather(y_feature, x_nn.idx, y_lengths)[..., 0, :]

            cosine_sim = F.cosine_similarity(x_feature, x_feature_near, dim=2, eps=1e-6)
            # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
            cosine_sim = (
                torch.abs(cosine_sim) if abs_cosine else cosine_sim
            )  # shape: (N, P1)
            feature_distance = 1 - cosine_sim

            if is_x_heterogeneous:
                feature_distance[x_mask] = 0.0

            if weights is not None:
                feature_distance *= weights.view(N, 1)

            cham_features_x[feature_name] = feature_distance

    if point_reduction == "max":
        assert not return_features
        cham_x = cham_x.max(1).values  # (N,)
    elif point_reduction is not None:
        # Apply point and feature reduction
        # a) Apply the summation (the normalization later)
        cham_x = cham_x.sum(1)  # (N,) - take sum here & normalize later with x_lengths
        if return_features:
            for feature_name in cham_features_x:
                cham_features_x[feature_name] = cham_features_x[feature_name].sum(1)
        # b) Apply the normalization
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_features:
                for feature_name in cham_features_x:
                    cham_features_x[feature_name] /= x_lengths_clamped

    cham_dist = cham_x
    cham_features = cham_features_x
    return cham_dist, cham_features


def _apply_batch_reduction(
    cham_x, cham_features_x, weights, batch_reduction: Union[str, None]
):
    if batch_reduction is None:
        return (cham_x, cham_features_x)
    # batch_reduction == "sum"
    N = cham_x.shape[0]
    cham_x = cham_x.sum()
    if cham_features_x is not None:
        for feature_name in cham_features_x:
            cham_features_x[feature_name] = cham_features_x[feature_name].sum()
    if batch_reduction == "mean":
        if weights is None:
            div = max(N, 1)
        elif weights.sum() == 0.0:
            div = 1
        else:
            div = weights.sum()
        cham_x /= div
        if cham_features_x is not None:
            for feature_name in cham_features_x:
                cham_features_x[feature_name] /= div
    return (cham_x, cham_features_x)


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_features=None,
    y_features=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
    feature_names: Union[list, None] = None,
):
    """
    Chamfer distance between two pointclouds x and y. Taken from:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/loss/chamfer.py  # noqa: E501

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_features: Optional Dict mapping feature names to FloatTensors of shape (N, P1, C)
            or a single FloatTensor of shape (N, P1, C). When x is a Pointclouds object,
            this is automatically retrieved from the pointcloud's features.
        y_features: Optional Dict mapping feature names to FloatTensors of shape (N, P2, C)
            or a single FloatTensor of shape (N, P2, C). When y is a Pointclouds object,
            this is automatically retrieved from the pointcloud's features.
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum", "max"] or None. Using "max" leads to the
            Hausdorff distance.
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.
        single_directional: If False (default), loss comes from both the distance between
            each point in x and its nearest neighbor in y and each point in y and its nearest
            neighbor in x. If True, loss is the distance between each point in x and its
            nearest neighbor in y.
        abs_cosine: If False, loss_features is from one minus the cosine similarity.
            If True (default), loss_features is from one minus the absolute value of the
            cosine similarity, which means that exactly opposite features are considered
            equivalent to exactly matching features, i.e. sign does not matter.
        feature_names: Optional list of feature names to compute cosine similarity for.
            If None, no feature-based loss is computed. Only used when features are provided.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
        - **loss_features**: Dict mapping feature names to tensors giving the reduced
          cosine distance of features between pointclouds in x and pointclouds in y.
          Returns None if features are None or feature_names is None. If point_reduction is None,
          a dict mapping feature names to 2-element tuples of Tensors containing forward and
          backward loss terms shaped (N, P1) and (N, P2) (if single_directional is False) or
          a dict mapping feature names to Tensors containing loss terms shaped (N, P1)
          (if single_directional is True) is returned.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    if point_reduction == "max" and (
        feature_names is not None and len(feature_names) > 0
    ):
        raise ValueError('Features must be None if point_reduction is "max"')

    x, x_lengths, x_features = _handle_pointcloud_input(x, x_lengths, x_features)
    y, y_lengths, y_features = _handle_pointcloud_input(y, y_lengths, y_features)

    cham_x, cham_features_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        x_features,
        y_features,
        weights,
        point_reduction,
        norm,
        abs_cosine,
        feature_names,
    )
    if single_directional:
        loss = cham_x
        loss_features = cham_features_x
    else:
        cham_y, cham_features_y = _chamfer_distance_single_direction(
            y,
            x,
            y_lengths,
            x_lengths,
            y_features,
            x_features,
            weights,
            point_reduction,
            norm,
            abs_cosine,
            feature_names,
        )
        if point_reduction == "max":
            loss = torch.maximum(cham_x, cham_y)
            loss_features = None
        elif point_reduction is not None:
            loss = cham_x + cham_y
            if cham_features_x is not None:
                loss_features = {}
                for feature_name in cham_features_x:
                    if feature_name in cham_features_y:
                        loss_features[feature_name] = (
                            cham_features_x[feature_name]
                            + cham_features_y[feature_name]
                        )
                    else:
                        loss_features[feature_name] = cham_features_x[feature_name]
            else:
                loss_features = None
        else:
            loss = (cham_x, cham_y)
            if cham_features_x is not None:
                loss_features = {}
                for feature_name in cham_features_x:
                    if feature_name in cham_features_y:
                        loss_features[feature_name] = (
                            cham_features_x[feature_name],
                            cham_features_y[feature_name],
                        )
                    else:
                        loss_features[feature_name] = (
                            cham_features_x[feature_name],
                            None,
                        )
            else:
                loss_features = None
    return _apply_batch_reduction(loss, loss_features, weights, batch_reduction)
