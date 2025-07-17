# PyTorch3D Point Operations

Fast CUDA-accelerated point cloud operations based on PyTorch3D.

## Installation
```bash
pip install -e .
```

## Quick Usage

### PointCloud Structure
The only difference to Pytorch3D is that the Pointclouds here doen't have `normals` or `features` attributes.
Instead they hold a dict of tensors called `features` that can contain any number of tensors, including `normals` and `features`. This allows for more flexibility in the data structure.

See [`examples/pointclouds.py`](examples/pointclouds.py) for complete examples.
```python
from pytorch3d_pointops.structures.point_structure import Pointclouds

# Create point clouds with features
points_list = [torch.randn(1000, 3), torch.randn(800, 3)]
features = {
    "normals": [torch.randn(1000, 3), torch.randn(800, 3)],
    "colors": [torch.randn(1000, 3), torch.randn(800, 3)]
}
pointclouds = Pointclouds(points=points_list, features=features)

```

### Core Operations

**Farthest Point Sampling**
See [`examples/fps_on_pointclouds.py`](examples/fps_on_pointclouds.py) for complete examples.
```python
from pytorch3d_pointops.functions.sample_farthest_points import sample_farthest_points

# Sample 128 points using FPS
sampled_points, indices = sample_farthest_points(pointclouds.points_padded(), pointclouds.num_points_per_cloud(), K=128)
```

**K-Nearest Neighbors**
See [`examples/knn_on_pointclouds.py`](examples/knn_on_pointclouds.py) for complete examples.
```python
from pytorch3d_pointops.functions.knn import knn_points, knn_gather

# Find 8 nearest neighbors
knn_result = knn_points(
    p1=pointclouds.points_padded(),
    p2=pointclouds.points_padded(),
    lengths1=pointclouds.num_points_per_cloud(),
    lengths2=pointclouds.num_points_per_cloud(),
    K=K,
)
distances, indices = knn_result.dists, knn_result.idx

# Gather features using KNN indices
gathered_features = knn_gather(features, indices)
```

**Ball Query (Radius Search)**
See [`examples/ball_query_on_pointclouds.py`](examples/ball_query_on_pointclouds.py) for complete examples.
```python
from pytorch3d_pointops.functions.ball_query import ball_query

# Find neighbors within radius 0.5
ball_results = ball_query(
    p1=pointclouds.points_padded(),
    p2=pointclouds.points_padded(),
    lengths1=pointclouds.num_points_per_cloud(),
    lengths2=pointclouds.num_points_per_cloud(),
    K=K,
    radius=radius,
    return_nn=True,
)
```

**Chamfer Distance**
See [`examples/chamfer_loss.py`](examples/chamfer_loss.py) for complete examples.
```python
from pytorch3d_pointops.functions.chamfer import chamfer_distance

# Compute chamfer loss between point clouds
loss, feature_losses = chamfer_distance(
    pointcloud1, pointcloud2,
    feature_names=["normals", "colors"]
)
```

**Packed ↔ Padded Conversion**
See [`examples/packed_to_padded_on_pointclouds.py`](examples/packed_to_padded_on_pointclouds.py) for complete examples.
```python
from pytorch3d_pointops.functions.packed_to_padded import packed_to_padded, padded_to_packed

# Convert between representations for efficient batching
padded, num_points = packed_to_padded(pointclouds.points_packed(), pointclouds.num_points_per_cloud())
packed = padded_to_packed(padded, num_points)
```


