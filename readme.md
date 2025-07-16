
```
torch3d_pointops:
    version: 0.1.0
    description: Taken from torch3d
    url: https://github.com/facebookresearch/pytorch3d/tree/main/pytorch3d/csrc
    items:
        - ball_query
        - knn
        - marching_cubes
        - packed_to_padded_tensor
        - sample_farthest_points
        - sample_pdf: Samples a probability density functions defined by bin edges `bins` and the non-negative per-bin probabilities `weights`.
```



### Pointclouds
The only difference is that the Pointclouds class doesnt have `normals` or `features` attributes, but only a dict of tensors called `features` that can contain any number of tensors, including `normals` and `features`. This allows for more flexibility in the data structure.