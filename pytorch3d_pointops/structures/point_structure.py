# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from . import utils as struct_utils


Device = Union[str, torch.device]


def make_device(device: Device) -> torch.device:
    """
    Makes an actual torch.device object from the device specified as
    either a string or torch.device object. If the device is `cuda` without
    a specific index, the index of the current device is assigned.

    Args:
        device: Device (as str or torch.device)

    Returns:
        A matching torch.device object
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda" and device.index is None:
        # If cuda but with no index, then the current cuda device is indicated.
        # In that case, we fix to that device
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


class Pointclouds:
    """
    This class provides functions for working with batches of 3d point clouds,
    and converting between representations.

    Within Pointclouds, there are three different representations of the data.

    List
       - only used for input as a starting point to convert to other representations.
    Padded
       - has specific batch dimension.
    Packed
       - no batch dimension.
       - has auxiliary variables used to index into the padded representation.

    Example

    Input list of points = [[P_1], [P_2], ... , [P_N]]
    where P_1, ... , P_N are the number of points in each cloud and N is the
    number of clouds.

    # SPHINX IGNORE
     List                      | Padded                  | Packed
    ---------------------------|-------------------------|------------------------
    [[P_1], ... , [P_N]]       | size = (N, max(P_n), 3) |  size = (sum(P_n), 3)
                               |                         |
    Example for locations      |                         |
    or colors:                 |                         |
                               |                         |
    P_1 = 3, P_2 = 4, P_3 = 5  | size = (3, 5, 3)        |  size = (12, 3)
                               |                         |
    List([                     | tensor([                |  tensor([
      [                        |     [                   |    [0.1, 0.3, 0.5],
        [0.1, 0.3, 0.5],       |       [0.1, 0.3, 0.5],  |    [0.5, 0.2, 0.1],
        [0.5, 0.2, 0.1],       |       [0.5, 0.2, 0.1],  |    [0.6, 0.8, 0.7],
        [0.6, 0.8, 0.7]        |       [0.6, 0.8, 0.7],  |    [0.1, 0.3, 0.3],
      ],                       |       [0,    0,    0],  |    [0.6, 0.7, 0.8],
      [                        |       [0,    0,    0]   |    [0.2, 0.3, 0.4],
        [0.1, 0.3, 0.3],       |     ],                  |    [0.1, 0.5, 0.3],
        [0.6, 0.7, 0.8],       |     [                   |    [0.7, 0.3, 0.6],
        [0.2, 0.3, 0.4],       |       [0.1, 0.3, 0.3],  |    [0.2, 0.4, 0.8],
        [0.1, 0.5, 0.3]        |       [0.6, 0.7, 0.8],  |    [0.9, 0.5, 0.2],
      ],                       |       [0.2, 0.3, 0.4],  |    [0.2, 0.3, 0.4],
      [                        |       [0.1, 0.5, 0.3],  |    [0.9, 0.3, 0.8],
        [0.7, 0.3, 0.6],       |       [0,    0,    0]   |  ])
        [0.2, 0.4, 0.8],       |     ],                  |
        [0.9, 0.5, 0.2],       |     [                   |
        [0.2, 0.3, 0.4],       |       [0.7, 0.3, 0.6],  |
        [0.9, 0.3, 0.8],       |       [0.2, 0.4, 0.8],  |
      ]                        |       [0.9, 0.5, 0.2],  |
    ])                         |       [0.2, 0.3, 0.4],  |
                               |       [0.9, 0.3, 0.8]   |
                               |     ]                   |
                               |  ])                     |
    -----------------------------------------------------------------------------

    Auxiliary variables for packed representation

    Name                           |   Size              |  Example from above
    -------------------------------|---------------------|-----------------------
                                   |                     |
    packed_to_cloud_idx            |  size = (sum(P_n))  |   tensor([
                                   |                     |     0, 0, 0, 1, 1, 1,
                                   |                     |     1, 2, 2, 2, 2, 2
                                   |                     |   )]
                                   |                     |   size = (12)
                                   |                     |
    cloud_to_packed_first_idx      |  size = (N)         |   tensor([0, 3, 7])
                                   |                     |   size = (3)
                                   |                     |
    num_points_per_cloud           |  size = (N)         |   tensor([3, 4, 5])
                                   |                     |   size = (3)
                                   |                     |
    padded_to_packed_idx           |  size = (sum(P_n))  |  tensor([
                                   |                     |     0, 1, 2, 5, 6, 7,
                                   |                     |     8, 10, 11, 12, 13,
                                   |                     |     14
                                   |                     |  )]
                                   |                     |  size = (12)
    -----------------------------------------------------------------------------
    # SPHINX IGNORE
    """

    _INTERNAL_TENSORS = [
        "_points_packed",
        "_points_padded",
        "_features_packed",
        "_features_padded",
        "_packed_to_cloud_idx",
        "_cloud_to_packed_first_idx",
        "_num_points_per_cloud",
        "_padded_to_packed_idx",
        "valid",
        "equisized",
    ]

    # -----------------
    # INITIALIZATION
    # -----------------
    def __init__(self, points, features=None) -> None:
        """
        Args:
            points:
                Can be either

                - List where each element is a tensor of shape (num_points, 3)
                  containing the (x, y, z) coordinates of each point.
                - Padded float tensor with shape (num_clouds, num_points, 3).
            features:
                Can be either

                - None
                - Dict where each key is a feature name and value can be either:
                  - List where each element is a tensor of shape (num_points, C)
                    containing the features for the points in the cloud.
                  - Padded float tensor of shape (num_clouds, num_points, C).
                where C is the number of channels in the features.
                For example: {"normals": normals_data, "colors": rgb_data}

        Refer to comments above for descriptions of List and Padded
        representations.
        """
        self.device = torch.device("cpu")

        # Indicates whether the clouds in the list/batch have the same number
        # of points.
        self.equisized = False

        # Boolean indicator for each cloud in the batch.
        # True if cloud has non zero number of points, False otherwise.
        self.valid = None

        self._N = 0  # batch size (number of clouds)
        self._P = 0  # (max) number of points per cloud
        self._C: dict[str, int] = {}

        # List of Tensors of points and features.
        self._points_list = None
        self._features_list = {}  # Dict[str, List[torch.Tensor]]

        # Number of points per cloud.
        self._num_points_per_cloud = None  # N

        # Packed representation.
        self._points_packed = None  # (sum(P_n), 3)
        self._features_packed = {}  # Dict[str, torch.Tensor]

        self._packed_to_cloud_idx = None  # sum(P_n)

        # Index of each cloud's first point in the packed points.
        # Assumes packing is sequential.
        self._cloud_to_packed_first_idx = None  # N

        # Padded representation.
        self._points_padded = None  # (N, max(P_n), 3)
        self._features_padded = {}  # Dict[str, torch.Tensor]

        # Index to convert points from flattened padded to packed.
        self._padded_to_packed_idx = None  # N * max_P

        # Identify type of points.
        if isinstance(points, list):
            self._points_list = points
            self._N = len(self._points_list)
            self.valid = torch.zeros((self._N,), dtype=torch.bool, device=self.device)

            if self._N > 0:
                self.device = self._points_list[0].device
                for p in self._points_list:
                    if len(p) > 0 and (p.dim() != 2 or p.shape[1] != 3):
                        raise ValueError("Clouds in list must be of shape Px3 or empty")
                    if p.device != self.device:
                        raise ValueError("All points must be on the same device")

                num_points_per_cloud = torch.tensor(
                    [len(p) for p in self._points_list], device=self.device
                )
                self._P = int(num_points_per_cloud.max())
                self.valid = torch.tensor(
                    [len(p) > 0 for p in self._points_list],
                    dtype=torch.bool,
                    device=self.device,
                )

                if len(num_points_per_cloud.unique()) == 1:
                    self.equisized = True
                self._num_points_per_cloud = num_points_per_cloud
            else:
                self._num_points_per_cloud = torch.tensor([], dtype=torch.int64)

        elif torch.is_tensor(points):
            if points.dim() != 3 or points.shape[2] != 3:
                raise ValueError("Points tensor has incorrect dimensions.")
            self._points_padded = points
            self._N = self._points_padded.shape[0]
            self._P = self._points_padded.shape[1]
            self.device = self._points_padded.device
            self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)
            self._num_points_per_cloud = torch.tensor(
                [self._P] * self._N, device=self.device
            )
            self.equisized = True
        else:
            raise ValueError(
                "Points must be either a list or a tensor with \
                    shape (batch_size, P, 3) where P is the maximum number of \
                    points in a cloud."
            )

        # parse features
        if features is not None:
            if not isinstance(features, dict):
                raise ValueError(
                    "Features must be a dictionary with feature names as keys"
                )
            for feature_name, feature_data in features.items():
                feature_parsed = self._parse_auxiliary_input(feature_data)
                feature_list, feature_padded, feature_C = feature_parsed
                if feature_list is not None:
                    self._features_list[feature_name] = feature_list
                elif feature_padded is not None:
                    self._features_padded[feature_name] = feature_padded
                else:
                    raise ValueError(
                        "Features must be either a list or a padded tensor with \
                            shape (batch_size, P, C) where P is the maximum number of \
                            points in a cloud and C is the number of channels."
                    )
                self._C[feature_name] = feature_C if feature_C is not None else 0

    def _parse_auxiliary_input(
        self, aux_input
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor], Optional[int]]:
        """
        Interpret the auxiliary inputs (normals, features) given to __init__.

        Args:
            aux_input:
              Can be either

                - List where each element is a tensor of shape (num_points, C)
                  containing the features for the points in the cloud.
                - Padded float tensor of shape (num_clouds, num_points, C).
              For normals, C = 3

        Returns:
            3-element tuple of list, padded, num_channels.
            If aux_input is list, then padded is None. If aux_input is a tensor,
            then list is None.
        """
        if aux_input is None or self._N == 0:
            return None, None, None

        aux_input_C = None

        if isinstance(aux_input, list):
            return self._parse_auxiliary_input_list(aux_input)
        if torch.is_tensor(aux_input):
            if aux_input.dim() != 3:
                raise ValueError("Auxiliary input tensor has incorrect dimensions.")
            if self._N != aux_input.shape[0]:
                raise ValueError("Points and inputs must be the same length.")
            if self._P != aux_input.shape[1]:
                raise ValueError(
                    "Inputs tensor must have the right maximum \
                    number of points in each cloud."
                )
            if aux_input.device != self.device:
                raise ValueError(
                    "All auxiliary inputs must be on the same device as the points."
                )
            aux_input_C = aux_input.shape[2]
            return None, aux_input, aux_input_C
        else:
            raise ValueError(
                "Auxiliary input must be either a list or a tensor with \
                    shape (batch_size, P, C) where P is the maximum number of \
                    points in a cloud."
            )

    def _parse_auxiliary_input_list(
        self, aux_input: list
    ) -> Tuple[Optional[List[torch.Tensor]], None, Optional[int]]:
        """
        Interpret the auxiliary inputs (normals, features) given to __init__,
        if a list.

        Args:
            aux_input:
                - List where each element is a tensor of shape (num_points, C)
                  containing the features for the points in the cloud.
              For normals, C = 3

        Returns:
            3-element tuple of list, padded=None, num_channels.
            If aux_input is list, then padded is None. If aux_input is a tensor,
            then list is None.
        """
        aux_input_C = None
        good_empty = None
        needs_fixing = False

        if len(aux_input) != self._N:
            raise ValueError("Points and auxiliary input must be the same length.")
        for p, d in zip(self._num_points_per_cloud, aux_input):
            valid_but_empty = p == 0 and d is not None and d.ndim == 2
            if p > 0 or valid_but_empty:
                if p != d.shape[0]:
                    raise ValueError(
                        "A cloud has mismatched numbers of points and inputs"
                    )
                if d.dim() != 2:
                    raise ValueError(
                        "A cloud auxiliary input must be of shape PxC or empty"
                    )
                if aux_input_C is None:
                    aux_input_C = d.shape[1]
                elif aux_input_C != d.shape[1]:
                    raise ValueError("The clouds must have the same number of channels")
                if d.device != self.device:
                    raise ValueError(
                        "All auxiliary inputs must be on the same device as the points."
                    )
            else:
                needs_fixing = True

        if aux_input_C is None:
            # We found nothing useful
            return None, None, None

        # If we have empty but "wrong" inputs we want to store "fixed" versions.
        if needs_fixing:
            if good_empty is None:
                good_empty = torch.zeros((0, aux_input_C), device=self.device)
            aux_input_out = []
            for p, d in zip(self._num_points_per_cloud, aux_input):
                valid_but_empty = p == 0 and d is not None and d.ndim == 2
                if p > 0 or valid_but_empty:
                    aux_input_out.append(d)
                else:
                    aux_input_out.append(good_empty)
        else:
            aux_input_out = aux_input

        return aux_input_out, None, aux_input_C

    # -----------------
    # GETTERS LIST
    # -----------------
    def points_list(self) -> List[torch.Tensor]:
        """
        Get the list representation of the points.

        Returns:
            list of tensors of points of shape (P_n, 3).
        """
        if self._points_list is None:
            assert (
                self._points_padded is not None
            ), "points_padded is required to compute points_list."
            points_list = []
            for i in range(self._N):
                points_list.append(
                    self._points_padded[i, : self.num_points_per_cloud()[i]]
                )
            self._points_list = points_list
        return self._points_list

    def get_features_list(self, feature_name: str) -> Optional[List[torch.Tensor]]:
        """
        Get the list representation of the specified feature,
        or None if the feature doesn't exist.

        Args:
            feature_name: Name of the feature to retrieve.

        Returns:
            list of tensors of features of shape (P_n, C).
        """
        if feature_name not in self._features_list:
            if feature_name not in self._features_padded:
                # No features provided so return None
                return None
            # Compute list from padded
            self._features_list[feature_name] = struct_utils.padded_to_list(
                self._features_padded[feature_name],
                self.num_points_per_cloud().tolist(),
            )
        return self._features_list[feature_name]

    def features_list(self) -> dict[str, List[torch.Tensor]]:
        """
        Get the list representation of all features.

        Returns:
            dict mapping feature names to lists of tensors of shape (P_n, C).
        """
        result = {}
        # Get all feature names from both list and padded representations
        all_feature_names = set(self._features_list.keys()) | set(
            self._features_padded.keys()
        )
        for feature_name in all_feature_names:
            feature_list = self.get_features_list(feature_name)
            if feature_list is not None:
                result[feature_name] = feature_list
        return result

    # -----------------
    # GETTERS PACKED
    # -----------------
    # TODO(nikhilar) Improve performance of _compute_packed.
    def _compute_packed(self, refresh: bool = False):
        """
        Computes the packed version from points_list and
        features_list and sets the values of auxiliary tensors.

        Args:
            refresh: Set to True to force recomputation of packed
                representations. Default: False.
        """

        if not (
            refresh
            or any(
                v is None
                for v in [
                    self._points_packed,
                    self._packed_to_cloud_idx,
                    self._cloud_to_packed_first_idx,
                ]
            )
        ):
            return

        # Packed can be calculated from padded or list, so can call the
        # accessor function for the lists.
        points_list = self.points_list()
        features_dict = self.features_list()
        if self.isempty():
            self._points_packed = torch.zeros(
                (0, 3), dtype=torch.float32, device=self.device
            )
            self._packed_to_cloud_idx = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._cloud_to_packed_first_idx = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._features_packed = {}
            return

        points_list_to_packed = struct_utils.list_to_packed(points_list)
        self._points_packed = points_list_to_packed[0]
        if not torch.allclose(self._num_points_per_cloud, points_list_to_packed[1]):
            raise ValueError("Inconsistent list to packed conversion")
        self._cloud_to_packed_first_idx = points_list_to_packed[2]
        self._packed_to_cloud_idx = points_list_to_packed[3]

        # Process all features
        self._features_packed = {}
        for feature_name, feature_list in features_dict.items():
            if feature_list is not None:
                feature_list_to_packed = struct_utils.list_to_packed(feature_list)
                self._features_packed[feature_name] = feature_list_to_packed[0]

    def points_packed(self) -> torch.Tensor:
        """
        Get the packed representation of the points.

        Returns:
            tensor of points of shape (sum(P_n), 3).
        """
        self._compute_packed()
        return self._points_packed

    def get_features_packed(self, feature_name: str) -> Optional[torch.Tensor]:
        """
        Get the packed representation of the specified feature.

        Args:
            feature_name: Name of the feature to retrieve.

        Returns:
            tensor of features of shape (sum(P_n), C),
            or None if the feature doesn't exist.
        """
        self._compute_packed()
        return self._features_packed.get(feature_name)

    def features_packed(self) -> dict[str, torch.Tensor]:
        """
        Get the packed representation of all features.

        Returns:
            dict mapping feature names to tensors of shape (sum(P_n), C).
        """
        self._compute_packed()
        return self._features_packed

    # -----------------
    # GETTERS PADDED
    # -----------------
    def _compute_padded(self, refresh: bool = False):
        """
        Computes the padded version from points_list and features_list.

        Args:
            refresh: whether to force the recalculation.
        """
        if not (refresh or self._points_padded is None):
            return

        # Clear existing padded features
        self._features_padded = {}
        if self.isempty():
            self._points_padded = torch.zeros((self._N, 0, 3), device=self.device)
        else:
            self._points_padded = struct_utils.list_to_padded(
                self.points_list(),
                (self._P, 3),
                pad_value=0.0,
                equisized=self.equisized,
            )
            # Process all features
            features_dict = self.features_list()
            for feature_name, feature_list in features_dict.items():
                if feature_list is not None and len(feature_list) > 0:
                    # Determine the number of channels for this feature
                    feature_C = (
                        feature_list[0].shape[1]
                        if len(feature_list[0].shape) > 1
                        else 1
                    )
                    self._features_padded[feature_name] = struct_utils.list_to_padded(
                        feature_list,
                        (self._P, feature_C),
                        pad_value=0.0,
                        equisized=self.equisized,
                    )

    def points_padded(self) -> torch.Tensor:
        """
        Get the padded representation of the points.

        Returns:
            tensor of points of shape (N, max(P_n), 3).
        """
        self._compute_padded()
        return self._points_padded

    def get_features_padded(self, feature_name: str) -> Optional[torch.Tensor]:
        """
        Get the padded representation of the specified feature,
        or None if the feature doesn't exist.

        Args:
            feature_name: Name of the feature to retrieve.

        Returns:
            tensor of features of shape (N, max(P_n), C).
        """
        self._compute_padded()
        return self._features_padded.get(feature_name)

    def features_padded(self) -> dict[str, torch.Tensor]:
        """
        Get the padded representation of all features.

        Returns:
            dict mapping feature names to tensors of shape (N, max(P_n), C).
        """
        self._compute_padded()
        return self._features_padded

    # -----------------
    # PACKED Methods
    # -----------------
    # to be combined with:
    # - utils.list_to_packed
    # - utils.packed_to_list
    # - utils.padded_to_packed
    # -----------------
    def num_points_per_cloud(self) -> torch.Tensor:
        """
        Return a 1D tensor x with length equal to the number of clouds giving
        the number of points in each cloud.

        Returns:
            1D tensor of sizes.
        """
        return self._num_points_per_cloud

    def packed_to_cloud_idx(self):
        """
        Return a 1D tensor x with length equal to the total number of points.
        packed_to_cloud_idx()[i] gives the index of the cloud which contains
        points_packed()[i].

        Returns:
            1D tensor of indices.
        """
        self._compute_packed()
        return self._packed_to_cloud_idx

    def cloud_to_packed_first_idx(self):
        """
        Return a 1D tensor x with length equal to the number of clouds such that
        the first point of the ith cloud is points_packed[x[i]].

        Returns:
            1D tensor of indices of first items.
        """
        self._compute_packed()
        return self._cloud_to_packed_first_idx

    def padded_to_packed_idx(self):
        """
        Return a 1D tensor x with length equal to the total number of points
        such that points_packed()[i] is element x[i] of the flattened padded
        representation.
        The packed representation can be calculated as follows.

        .. code-block:: python

            p = points_padded().reshape(-1, 3)
            points_packed = p[x]

        Returns:
            1D tensor of indices.
        """
        if self._padded_to_packed_idx is not None:
            return self._padded_to_packed_idx
        if self._N == 0:
            self._padded_to_packed_idx = []
        else:
            self._padded_to_packed_idx = torch.cat(
                [
                    torch.arange(v, dtype=torch.int64, device=self.device) + i * self._P
                    for (i, v) in enumerate(self.num_points_per_cloud())
                ],
                dim=0,
            )
        return self._padded_to_packed_idx

    # -----------------
    # TENSOR OPERATIONS
    # -----------------
    def __len__(self) -> int:
        return self._N

    def __getitem__(
        self,
        index: Union[int, List[int], slice, torch.BoolTensor, torch.LongTensor],
    ) -> "Pointclouds":
        """
        Args:
            index: Specifying the index of the cloud to retrieve.
                Can be an int, slice, list of ints or a boolean tensor.

        Returns:
            Pointclouds object with selected clouds. The tensors are not cloned.
        """
        features = {}
        features_dict = self.features_list()

        if isinstance(index, int):
            points = [self.points_list()[index]]
            for feature_name, feature_list in features_dict.items():
                features[feature_name] = [feature_list[index]]
        elif isinstance(index, slice):
            points = self.points_list()[index]
            for feature_name, feature_list in features_dict.items():
                features[feature_name] = feature_list[index]
        elif isinstance(index, list):
            points = [self.points_list()[i] for i in index]
            for feature_name, feature_list in features_dict.items():
                features[feature_name] = [feature_list[i] for i in index]
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
            # NOTE consider converting index to cpu for efficiency
            if index.dtype == torch.bool:
                # advanced indexing on a single dimension
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            points = [self.points_list()[i] for i in index]
            for feature_name, feature_list in features_dict.items():
                features[feature_name] = [feature_list[i] for i in index]
        else:
            raise IndexError(index)

        return self.__class__(points=points, features=features if features else None)

    def isempty(self) -> bool:
        """
        Checks whether any cloud is valid.

        Returns:
            bool indicating whether there is any data.
        """
        return self._N == 0 or self.valid.eq(False).all()

    def clone(self):
        """
        Deep copy of Pointclouds object. All internal tensors are cloned
        individually.

        Returns:
            new Pointclouds object.
        """
        # instantiate new pointcloud with the representation which is not None
        # (either list or tensor) to save compute.
        new_points, new_features = None, None
        if self._points_list is not None:
            new_points = [v.clone() for v in self.points_list()]
            features_dict = self.features_list()
            if features_dict:
                new_features = {}
                for feature_name, feature_list in features_dict.items():
                    new_features[feature_name] = [f.clone() for f in feature_list]
        elif self._points_padded is not None:
            new_points = self.points_padded().clone()
            features_dict = self.features_padded()
            if features_dict:
                new_features = {}
                for feature_name, feature_tensor in features_dict.items():
                    new_features[feature_name] = feature_tensor.clone()
        other = self.__class__(points=new_points, features=new_features)
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
            elif isinstance(v, dict):
                # Handle dictionary features
                setattr(
                    other,
                    k,
                    {
                        key: val.clone() if torch.is_tensor(val) else val
                        for key, val in v.items()
                    },
                )
        return other

    def detach(self):
        """
        Detach Pointclouds object. All internal tensors are detached
        individually.

        Returns:
            new Pointclouds object.
        """
        # instantiate new pointcloud with the representation which is not None
        # (either list or tensor) to save compute.
        new_points, new_features = None, None
        if self._points_list is not None:
            new_points = [v.detach() for v in self.points_list()]
            features_dict = self.features_list()
            if features_dict:
                new_features = {}
                for feature_name, feature_list in features_dict.items():
                    new_features[feature_name] = [f.detach() for f in feature_list]
        elif self._points_padded is not None:
            new_points = self.points_padded().detach()
            features_dict = self.features_padded()
            if features_dict:
                new_features = {}
                for feature_name, feature_tensor in features_dict.items():
                    new_features[feature_name] = feature_tensor.detach()
        other = self.__class__(points=new_points, features=new_features)
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())
            elif isinstance(v, dict):
                # Handle dictionary features
                setattr(
                    other,
                    k,
                    {
                        key: val.detach() if torch.is_tensor(val) else val
                        for key, val in v.items()
                    },
                )
        return other

    def to(self, device: Device, copy: bool = False):
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
          device: Device (as str or torch.device) for the new tensor.
          copy: Boolean indicator whether or not to clone self. Default False.

        Returns:
          Pointclouds object.
        """
        device_ = make_device(device)

        if not copy and self.device == device_:
            return self

        other = self.clone()
        if self.device == device_:
            return other

        other.device = device_
        if other._N > 0:
            other._points_list = [v.to(device_) for v in other.points_list()]
            # Update all features
            features_dict = other.features_list()
            for feature_name, feature_list in features_dict.items():
                other._features_list[feature_name] = [
                    f.to(device_) for f in feature_list
                ]
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.to(device_))
            elif isinstance(v, dict):
                # Handle dictionary features
                setattr(
                    other,
                    k,
                    {
                        key: val.to(device_) if torch.is_tensor(val) else val
                        for key, val in v.items()
                    },
                )
        return other

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def extend(self, N: int):
        """
        Create new Pointclouds which contains each cloud N times.

        Args:
            N: number of new copies of each cloud.

        Returns:
            new Pointclouds object.
        """
        if not isinstance(N, int):
            raise ValueError("N must be an integer.")
        if N <= 0:
            raise ValueError("N must be > 0.")

        new_points_list, new_features_list = [], None

        for points in self.points_list():
            new_points_list.extend(points.clone() for _ in range(N))

        new_features_list: dict[str, list[torch.Tensor]] = {}
        for feature_name, feature_list in self.features_list().items():
            new_features_list[feature_name] = []
            for features in feature_list:
                new_features_list[feature_name].extend(
                    features.clone() for _ in range(N)
                )

        return self.__class__(points=new_points_list, features=new_features_list)

    def split(self, split_sizes: list):
        """
        Splits Pointclouds object of size N into a list of Pointclouds objects
        of size len(split_sizes), where the i-th Pointclouds object is of size
        split_sizes[i]. Similar to torch.split().

        Args:
            split_sizes: List of integer sizes of Pointclouds objects to be
            returned.

        Returns:
            list[Pointclouds].
        """
        if not all(isinstance(x, int) for x in split_sizes):
            raise ValueError("Value of split_sizes must be a list of integers.")
        cloudlist = []
        curi = 0
        for i in split_sizes:
            cloudlist.append(self[curi : curi + i])
            curi += i
        return cloudlist

    # -----------------
    # GETTERS FOR SINGLE CLOUD (points, features)
    # -----------------
    def get_cloud(self, index: int):
        """
        Get tensors for a single cloud from the list representation.

        Args:
            index: Integer in the range [0, N).

        Returns:
            tuple: (points, features_dict) where:
                points: Tensor of shape (P, 3).
                features_dict: Dict[str, Tensor] mapping feature names to tensors
        """
        if not isinstance(index, int):
            raise ValueError("Cloud index must be an integer.")
        if index < 0 or index > self._N:
            raise ValueError(
                "Cloud index must be in the range [0, N) where \
            N is the number of clouds in the batch."
            )
        points = self.points_list()[index]
        features_dict = {}
        all_features = self.features_list()
        for feature_name, feature_list in all_features.items():
            if feature_list is not None:
                features_dict[feature_name] = feature_list[index]
        return points, features_dict

    # -----------------
    # POINT (XYZ) TRANSLATION/SCALING
    # -----------------
    def offset_(self, offsets_packed):
        """
        Translate the point clouds by an offset. In place operation.

        Args:
            offsets_packed: A Tensor of shape (3,) or the same shape
                as self.points_packed giving offsets to be added to
                all points.

        Returns:
            self.
        """
        points_packed = self.points_packed()
        if offsets_packed.shape == (3,):
            offsets_packed = offsets_packed.expand_as(points_packed)
        if offsets_packed.shape != points_packed.shape:
            raise ValueError("Offsets must have dimension (all_p, 3).")
        self._points_packed = points_packed + offsets_packed
        new_points_list = list(
            self._points_packed.split(self.num_points_per_cloud().tolist(), 0)
        )
        # Note that since _compute_packed() has been executed, points_list
        # cannot be None even if not provided during construction.
        self._points_list = new_points_list
        if self._points_padded is not None:
            for i, points in enumerate(new_points_list):
                if len(points) > 0:
                    self._points_padded[i, : points.shape[0], :] = points
        return self

    def scale_(self, scale):
        """
        Multiply the coordinates of this object by a scalar value.
        - i.e. enlarge/dilate
        In place operation.

        Args:
            scale: A scalar, or a Tensor of shape (N,).

        Returns:
            self.
        """
        if not torch.is_tensor(scale):
            scale = torch.full((len(self),), scale, device=self.device)
        new_points_list = []
        points_list = self.points_list()
        for i, old_points in enumerate(points_list):
            new_points_list.append(scale[i] * old_points)
        self._points_list = new_points_list
        if self._points_packed is not None:
            self._points_packed = torch.cat(new_points_list, dim=0)
        if self._points_padded is not None:
            for i, points in enumerate(new_points_list):
                if len(points) > 0:
                    self._points_padded[i, : points.shape[0], :] = points
        return self

    def update_padded(self, new_points_padded, new_features_padded=None):
        """
        Returns a Pointcloud structure with updated padded tensors and copies of
        the auxiliary tensors. This function allows for an update of
        points_padded and features without having to explicitly
        convert it to the list representation for heterogeneous batches.

        Args:
            new_points_padded: FloatTensor of shape (N, P, 3)
            new_features_padded: (optional) Dict mapping feature names to
                FloatTensors of shape (N, P, C)

        Returns:
            Pointcloud with updated padded representations
        """

        def check_shapes(x, size):
            if x.shape[0] != size[0]:
                raise ValueError("new values must have the same batch dimension.")
            if x.shape[1] != size[1]:
                raise ValueError("new values must have the same number of points.")
            if size[2] is not None:
                if x.shape[2] != size[2]:
                    raise ValueError(
                        "new values must have the same number of channels."
                    )

        check_shapes(new_points_padded, [self._N, self._P, 3])

        # Validate new features if provided
        if new_features_padded is not None:
            if not isinstance(new_features_padded, dict):
                raise ValueError("new_features_padded must be a dictionary")
            for feature_name, feature_tensor in new_features_padded.items():
                check_shapes(feature_tensor, [self._N, self._P, self._C[feature_name]])

        new = self.__class__(
            points=new_points_padded,
            features=new_features_padded,
        )

        # overwrite the equisized flag
        new.equisized = self.equisized

        # copy features if not provided
        if new_features_padded is None:
            # If no features are provided, keep old ones (shallow copy)
            new._features_list = self._features_list
            new._features_padded = self._features_padded
            new._features_packed = self._features_packed

        # copy auxiliary tensors
        copy_tensors = [
            "_packed_to_cloud_idx",
            "_cloud_to_packed_first_idx",
            "_num_points_per_cloud",
            "_padded_to_packed_idx",
            "valid",
        ]
        for k in copy_tensors:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(new, k, v)  # shallow copy

        # update points
        new._points_padded = new_points_padded
        assert new._points_list is None
        assert new._points_packed is None

        # update features if provided
        if new_features_padded is not None:
            new._features_padded = new_features_padded
            new._features_list = {}
            new._features_packed = {}

        return new

    def inside_box(self, box):
        """
        Finds the points inside a 3D box.

        Args:
            box: FloatTensor of shape (2, 3) or (N, 2, 3) where N is the number
                of clouds.
                    box[..., 0, :] gives the min x, y & z.
                    box[..., 1, :] gives the max x, y & z.
        Returns:
            idx: BoolTensor of length sum(P_i) indicating whether the packed points are
                within the input box.
        """
        if box.dim() > 3 or box.dim() < 2:
            raise ValueError("Input box must be of shape (2, 3) or (N, 2, 3).")

        if box.dim() == 3 and box.shape[0] != 1 and box.shape[0] != self._N:
            raise ValueError(
                "Input box dimension is incompatible with pointcloud size."
            )

        if box.dim() == 2:
            box = box[None]

        if (box[..., 0, :] > box[..., 1, :]).any():
            raise ValueError("Input box is invalid: min values larger than max values.")

        points_packed = self.points_packed()
        sumP = points_packed.shape[0]

        if box.shape[0] == 1:
            box = box.expand(sumP, 2, 3)
        elif box.shape[0] == self._N:
            box = box.unbind(0)
            box = [
                b.expand(p, 2, 3) for (b, p) in zip(box, self.num_points_per_cloud())
            ]
            box = torch.cat(box, 0)

        coord_inside = (points_packed >= box[:, 0]) * (points_packed <= box[:, 1])
        return coord_inside.all(dim=-1)


def join_pointclouds_as_batch(pointclouds: Sequence[Pointclouds]) -> Pointclouds:
    """
    Merge a list of Pointclouds objects into a single batched Pointclouds
    object. All pointclouds must be on the same device.

    Args:
        batch: List of Pointclouds objects each with batch dim [b1, b2, ..., bN]
    Returns:
        pointcloud: Poinclouds object with all input pointclouds collated into
            a single object with batch dim = sum(b1, b2, ..., bN)
    """
    if isinstance(pointclouds, Pointclouds) or not isinstance(pointclouds, Sequence):
        raise ValueError("Wrong first argument to join_points_as_batch.")

    device = pointclouds[0].device
    if not all(p.device == device for p in pointclouds):
        raise ValueError("Pointclouds must all be on the same device")

    # Handle points
    points_list = [getattr(p, "points_list")() for p in pointclouds]
    if None in points_list:
        raise ValueError("Pointclouds cannot have their points set to None!")
    points_list = [p for points in points_list for p in points]

    # Handle features dictionary
    all_features_dicts = [p.features_list() for p in pointclouds]

    # Get all unique feature names
    all_feature_names = set()
    for features_dict in all_features_dicts:
        all_feature_names.update(features_dict.keys())

    # Build combined features dictionary
    combined_features = {}
    for feature_name in all_feature_names:
        feature_lists = []
        for features_dict in all_features_dicts:
            if (
                feature_name in features_dict
                and features_dict[feature_name] is not None
            ):
                feature_lists.extend(features_dict[feature_name])
            else:
                # If some pointclouds don't have this feature, we can't combine
                feature_lists = None
                break

        if feature_lists is not None:
            # Check that all features have the same number of channels
            if len(feature_lists) > 0 and any(
                p.shape[1] != feature_lists[0].shape[1] for p in feature_lists[1:]
            ):
                raise ValueError(
                    f"Pointclouds must have the same number of channels for feature '{feature_name}'"
                )
            combined_features[feature_name] = feature_lists

    return Pointclouds(
        points=points_list, features=combined_features if combined_features else None
    )


def join_pointclouds_as_scene(
    pointclouds: Union[Pointclouds, List[Pointclouds]],
) -> Pointclouds:
    """
    Joins a batch of point cloud in the form of a Pointclouds object or a list of Pointclouds
    objects as a single point cloud. If the input is a list, the Pointclouds objects in the
    list must all be on the same device, and they must either all or none have features and
    all or none have normals.

    Args:
        Pointclouds: Pointclouds object that contains a batch of point clouds, or a list of
                    Pointclouds objects.

    Returns:
        new Pointclouds object containing a single point cloud
    """
    if isinstance(pointclouds, list):
        pointclouds = join_pointclouds_as_batch(pointclouds)

    if len(pointclouds) == 1:
        return pointclouds
    points = pointclouds.points_packed()
    features_dict = pointclouds.features_packed()

    # Convert packed features back to per-cloud format
    features_for_scene = {}
    for feature_name, feature_tensor in features_dict.items():
        features_for_scene[feature_name] = feature_tensor[None]

    pointcloud = Pointclouds(
        points=points[None],
        features=features_for_scene if features_for_scene else None,
    )
    return pointcloud


# -----------------
# UTILS
# -----------------
# TODO(nikhilar) Move function to utils file.
def get_bounding_boxes(pointcloud: "Pointclouds") -> torch.Tensor:
    """
    Compute an axis-aligned bounding box for each cloud.

    Returns:
        bboxes: Tensor of shape (N, 3, 2) where bbox[i, j] gives the
        min and max values of cloud i along the jth coordinate axis.
    """
    all_mins, all_maxes = [], []
    for points in pointcloud.points_list():
        cur_mins = points.min(dim=0)[0]  # (3,)
        cur_maxes = points.max(dim=0)[0]  # (3,)
        all_mins.append(cur_mins)
        all_maxes.append(cur_maxes)
    all_mins = torch.stack(all_mins, dim=0)  # (N, 3)
    all_maxes = torch.stack(all_maxes, dim=0)  # (N, 3)
    bboxes = torch.stack([all_mins, all_maxes], dim=2)
    return bboxes


# TODO(nikhilar) Move out of place operator to a utils file.
def offset(pointcloud: "Pointclouds", offsets_packed: torch.Tensor) -> "Pointclouds":
    """
    Out of place offset.

    Args:
        offsets_packed: A Tensor of the same shape as self.points_packed
            giving offsets to be added to all points.
    Returns:
        new Pointclouds object.
    """
    new_clouds = pointcloud.clone()
    return new_clouds.offset_(offsets_packed)


def scale(
    pointcloud: "Pointclouds", scale: Union[float, torch.Tensor]
) -> "Pointclouds":
    """
    Out of place scale_.

    Args:
        scale: A scalar, or a Tensor of shape (N,).

    Returns:
        new Pointclouds object.
    """
    new_clouds = pointcloud.clone()
    return new_clouds.scale_(scale)


def subsample(
    pointclouds: Pointclouds, max_points: Union[int, Sequence[int]]
) -> "Pointclouds":
    """
    Subsample each cloud so that it has at most max_points points.

    Args:
        max_points: maximum number of points in each cloud.

    Returns:
        new Pointclouds object, or self if nothing to be done.
    """
    if isinstance(max_points, int):
        max_points = [max_points] * len(pointclouds)
    elif len(max_points) != len(pointclouds):
        raise ValueError("wrong number of max_points supplied")
    if all(
        int(n_points) <= int(max_)
        for n_points, max_ in zip(pointclouds.num_points_per_cloud(), max_points)
    ):
        return pointclouds

    points_list = []
    all_features_dict = pointclouds.features_list()
    feature_names = list(all_features_dict.keys())
    features_dict = {name: [] for name in feature_names}

    for max_, n_points, points in zip(
        map(int, max_points),
        map(int, pointclouds.num_points_per_cloud()),
        pointclouds.points_list(),
    ):
        if n_points > max_:
            keep_np = np.random.choice(n_points, max_, replace=False)
            keep = torch.tensor(keep_np, device=points.device, dtype=torch.int64)
            points = points[keep]
            # Apply the same subsampling to all features
            for feature_name in feature_names:
                feature_list = all_features_dict[feature_name]
                if feature_list is not None:
                    feature_idx = len(points_list)  # current cloud index
                    if feature_idx < len(feature_list):
                        features_dict[feature_name].append(
                            feature_list[feature_idx][keep]
                        )
                    else:
                        features_dict[feature_name].append(None)
                else:
                    features_dict[feature_name].append(None)
        else:
            # No subsampling needed, just append the original data
            for feature_name in feature_names:
                feature_list = all_features_dict[feature_name]
                if feature_list is not None:
                    feature_idx = len(points_list)  # current cloud index
                    if feature_idx < len(feature_list):
                        features_dict[feature_name].append(feature_list[feature_idx])
                    else:
                        features_dict[feature_name].append(None)
                else:
                    features_dict[feature_name].append(None)
        points_list.append(points)

    # Filter out empty feature lists
    filtered_features = {}
    for feature_name, feature_list in features_dict.items():
        if any(f is not None for f in feature_list):
            filtered_features[feature_name] = feature_list

    return Pointclouds(
        points=points_list,
        features=filtered_features if filtered_features else None,
    )


def all_close(
    pcd1: Pointclouds, pcd2: Pointclouds, rtol=1e-05, atol=1e-08, verbose=False
) -> bool:
    """
    Check if two Pointclouds objects are close to each other.

    Args:
        pcd1: First Pointclouds object.
        pcd2: Second Pointclouds object.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        bool indicating whether the two point clouds are close.
    """
    if pcd1.device != pcd2.device:
        raise ValueError("Pointclouds must be on the same device.")

    points_all_close = torch.allclose(
        pcd1.points_packed(), pcd2.points_packed(), rtol, atol
    )
    if verbose:
        print("Points all close:", points_all_close)

    # check whether they have the same keys:
    if set(pcd1.features_packed().keys()) != set(pcd2.features_packed().keys()):
        if verbose:
            print(
                "Features keys mismatch:",
                "Keys in pcd1:",
                pcd1.features_packed().keys(),
                "Keys in pcd2:",
                pcd2.features_packed().keys(),
            )
        return False
    features_all_close = {
        name: torch.allclose(
            pcd1.get_features_packed(name),
            pcd2.get_features_packed(name),
            rtol,
            atol,
        )
        for name in pcd1.features_packed().keys()
    }
    if verbose:
        print("Features all close:", features_all_close)

    return points_all_close and all(features_all_close.values())
