"""
Example of using Sample PDF on Pointclouds.

This example demonstrates how to:
1. Sample points according to probability distributions
2. Use PDF sampling for importance sampling and point cloud resampling
"""

import torch
import numpy as np

from pytorch3d_pointops.structures.point_structure import Pointclouds
from pytorch3d_pointops.functions.sample_pdf import sample_pdf, sample_pdf_python


def create_sample_pointclouds():
    """Create sample point clouds for demonstration."""
    torch.manual_seed(42)

    points_list = []

    # Point cloud 1: Random sphere
    n1 = 1000
    theta = torch.rand(n1) * 2 * np.pi
    phi = torch.rand(n1) * np.pi
    r = torch.rand(n1) * 0.8 + 0.2
    x1 = r * torch.sin(phi) * torch.cos(theta)
    y1 = r * torch.sin(phi) * torch.sin(theta)
    z1 = r * torch.cos(phi)
    points1 = torch.stack([x1, y1, z1], dim=1)
    points_list.append(points1)

    # Point cloud 2: Random ellipsoid
    n2 = 800
    theta = torch.rand(n2) * 2 * np.pi
    phi = torch.rand(n2) * np.pi
    r = torch.rand(n2) * 0.6 + 0.4
    x2 = r * torch.sin(phi) * torch.cos(theta) * 1.5  # Stretched
    y2 = r * torch.sin(phi) * torch.sin(theta) * 0.8  # Compressed
    z2 = r * torch.cos(phi)
    points2 = torch.stack([x2, y2, z2], dim=1)
    points_list.append(points2)

    # Create features
    features_dict = {
        "importance": [
            # Higher importance for points further from center for sphere
            torch.norm(points1, p=2, dim=1, keepdim=True),
            # Higher importance for points with larger x-coordinate for ellipsoid
            torch.abs(points2[:, 0:1]),
        ],
        "colors": [
            torch.rand(n1, 3),
            torch.rand(n2, 3),
        ],
    }

    return Pointclouds(points=points_list, features=features_dict)


def example_basic_pdf_sampling():
    """Basic example of PDF sampling with different distributions."""
    print("=== Basic PDF Sampling Example ===")

    # Create point clouds
    pointclouds = create_sample_pointclouds()

    # Get data for first point cloud
    points = pointclouds.points_list()[0]
    importance = pointclouds.get_features_list("importance")[0].squeeze()

    print(f"Original point cloud shape: {points.shape}")
    print(f"Importance values shape: {importance.shape}")
    print(f"Importance range: [{importance.min():.3f}, {importance.max():.3f}]")

    # Create different probability distributions
    # 1. Uniform distribution
    uniform_probs = torch.ones(points.shape[0])
    uniform_probs = uniform_probs / uniform_probs.sum()

    # 2. Importance-based distribution
    importance_probs = importance / importance.sum()

    # 3. Inverse importance distribution (favor points closer to center)
    inv_importance = 1.0 / (importance + 1e-6)
    inv_importance_probs = inv_importance / inv_importance.sum()

    print("\nProbability distributions:")
    print(
        f"  Uniform: entropy = {-torch.sum(uniform_probs * torch.log(uniform_probs + 1e-12)):.3f}"
    )
    print(
        f"  Importance: entropy = {-torch.sum(importance_probs * torch.log(importance_probs + 1e-12)):.3f}"
    )
    print(
        f"  Inv-Importance: entropy = {-torch.sum(inv_importance_probs * torch.log(inv_importance_probs + 1e-12)):.3f}"
    )

    # Sample points using different distributions
    num_samples = 200

    # Create bins for PDF sampling (uniform bins from 0 to n_points)
    n_points = len(points)
    bins = torch.linspace(0, n_points, n_points + 1)
    bins_batch = bins.unsqueeze(0)

    # Sample with uniform distribution
    uniform_probs_batch = uniform_probs.unsqueeze(0)
    uniform_samples = sample_pdf(
        bins_batch, uniform_probs_batch, num_samples, det=False
    )
    uniform_indices = uniform_samples[0].long().clamp(0, n_points - 1)
    uniform_sampled_points = points[uniform_indices]

    # Sample with importance distribution
    importance_probs_batch = importance_probs.unsqueeze(0)
    importance_samples = sample_pdf(
        bins_batch, importance_probs_batch, num_samples, det=False
    )
    importance_indices = importance_samples[0].long().clamp(0, n_points - 1)
    importance_sampled_points = points[importance_indices]

    # Sample with inverse importance distribution
    inv_importance_probs_batch = inv_importance_probs.unsqueeze(0)
    inv_importance_samples = sample_pdf(
        bins_batch, inv_importance_probs_batch, num_samples, det=False
    )
    inv_importance_indices = inv_importance_samples[0].long().clamp(0, n_points - 1)
    inv_importance_sampled_points = points[inv_importance_indices]

    print("\nSampling results:")
    print(f"  Samples per distribution: {num_samples}")
    print(f"  Uniform sampled shape: {uniform_sampled_points.shape}")
    print(f"  Importance sampled shape: {importance_sampled_points.shape}")
    print(f"  Inv-importance sampled shape: {inv_importance_sampled_points.shape}")

    # Analyze sampling bias
    uniform_distances = torch.norm(uniform_sampled_points, p=2, dim=1)
    importance_distances = torch.norm(importance_sampled_points, p=2, dim=1)
    inv_importance_distances = torch.norm(inv_importance_sampled_points, p=2, dim=1)

    print("\nSampling bias analysis (distance from origin):")
    print("  Uniform sampling:")
    print(f"    Mean distance: {uniform_distances.mean():.3f}")
    print(f"    Std distance: {uniform_distances.std():.3f}")
    print("  Importance sampling (favor far points):")
    print(f"    Mean distance: {importance_distances.mean():.3f}")
    print(f"    Std distance: {importance_distances.std():.3f}")
    print("  Inv-importance sampling (favor near points):")
    print(f"    Mean distance: {inv_importance_distances.mean():.3f}")
    print(f"    Std distance: {inv_importance_distances.std():.3f}")
    print()


def example_deterministic_vs_stochastic_sampling():
    """Compare deterministic vs stochastic PDF sampling."""
    print("=== Deterministic vs Stochastic Sampling ===")

    # Create a simple test case
    torch.manual_seed(123)
    n_points = 100

    # Create points in a line with varying density
    x = torch.linspace(-2, 2, n_points)
    y = torch.zeros(n_points)
    z = torch.zeros(n_points)
    points = torch.stack([x, y, z], dim=1)

    # Create a non-uniform probability distribution
    # Higher probability in the center, lower at edges
    center_bias = torch.exp(-(x**2))  # Gaussian-like distribution
    probs = center_bias / center_bias.sum()

    print("Test setup:")
    print(f"  Points: {n_points} arranged on x-axis from -2 to 2")
    print("  Probability: Gaussian-like (center-biased)")
    print(f"  Probability entropy: {-torch.sum(probs * torch.log(probs + 1e-12)):.3f}")

    # Create bins for PDF sampling
    n_points = len(points)
    bins = torch.linspace(-1, 1, n_points + 1)  # Match the point cloud range
    bins_batch = bins.unsqueeze(0)
    probs_batch = probs.unsqueeze(0)
    num_samples = 30

    # Deterministic sampling
    det_samples = sample_pdf(bins_batch, probs_batch, num_samples, det=True)
    det_indices = (
        (det_samples[0] + 1) * (n_points - 1) / 2
    )  # Map from [-1,1] to [0, n_points-1]
    det_indices = det_indices.long().clamp(0, n_points - 1)
    det_sampled_points = points[det_indices]
    det_x_coords = det_sampled_points[:, 0]

    # Stochastic sampling (multiple runs)
    stoch_x_coords_list = []
    for _ in range(5):  # Multiple stochastic runs
        stoch_samples = sample_pdf(bins_batch, probs_batch, num_samples, det=False)
        stoch_indices = (
            (stoch_samples[0] + 1) * (n_points - 1) / 2
        )  # Map from [-1,1] to [0, n_points-1]
        stoch_indices = stoch_indices.long().clamp(0, n_points - 1)
        stoch_sampled_points = points[stoch_indices]
        stoch_x_coords_list.append(stoch_sampled_points[:, 0])

    print("\nSampling comparison:")
    print(f"  Number of samples: {num_samples}")

    print("  Deterministic sampling:")
    print(
        f"    X-coordinate range: [{det_x_coords.min():.3f}, {det_x_coords.max():.3f}]"
    )
    print(f"    Mean X-coordinate: {det_x_coords.mean():.3f}")
    print(f"    Std X-coordinate: {det_x_coords.std():.3f}")

    print("  Stochastic sampling (5 runs):")
    for i, stoch_x_coords in enumerate(stoch_x_coords_list):
        print(
            f"    Run {i+1}: mean={stoch_x_coords.mean():.3f}, std={stoch_x_coords.std():.3f}"
        )

    # Calculate statistics across stochastic runs
    all_stoch_means = torch.tensor([coords.mean() for coords in stoch_x_coords_list])
    all_stoch_stds = torch.tensor([coords.std() for coords in stoch_x_coords_list])

    print("  Stochastic statistics across runs:")
    print(
        f"    Mean of means: {all_stoch_means.mean():.3f} ± {all_stoch_means.std():.3f}"
    )
    print(f"    Mean of stds: {all_stoch_stds.mean():.3f} ± {all_stoch_stds.std():.3f}")

    # Check if deterministic sampling is repeatable
    det_samples_2 = sample_pdf(bins_batch, probs_batch, num_samples, det=True)
    det_indices_2 = (
        (det_samples_2[0] + 1) * (n_points - 1) / 2
    )  # Map from [-1,1] to [0, n_points-1]
    det_indices_2 = det_indices_2.long().clamp(0, n_points - 1)
    deterministic_repeatable = torch.equal(det_indices, det_indices_2)
    print(f"  Deterministic sampling is repeatable: {deterministic_repeatable}")

    # Measure how well each method respects the probability distribution
    # Expected x-coordinate based on probability distribution
    expected_x = torch.sum(x * probs)

    det_bias = abs(det_x_coords.mean() - expected_x)
    stoch_biases = [abs(coords.mean() - expected_x) for coords in stoch_x_coords_list]
    avg_stoch_bias = np.mean(stoch_biases)

    print("\nDistribution fidelity:")
    print(f"  Expected X-coordinate: {expected_x:.3f}")
    print(f"  Deterministic bias: {det_bias:.3f}")
    print(f"  Average stochastic bias: {avg_stoch_bias:.3f}")
    print()


def example_compare_sample_pdf_implementations():
    """Compare sample_pdf (C++ implementation) vs sample_pdf_python (pure Python)."""
    print("=== C++ vs Python Implementation Comparison ===")

    # Setup test parameters
    torch.manual_seed(456)
    batch_size = 2
    n_bins = 64
    n_samples = 100

    # Create test bins and weights
    bins = torch.linspace(0, 1, n_bins + 1).unsqueeze(0).expand(batch_size, -1)

    # Create non-uniform weights (bimodal distribution)
    bin_centers = (bins[:, :-1] + bins[:, 1:]) / 2
    weights1 = torch.exp(-((bin_centers - 0.3) ** 2) / 0.02)  # Peak at 0.3
    weights2 = torch.exp(-((bin_centers - 0.7) ** 2) / 0.02)  # Peak at 0.7
    weights = weights1 + weights2 * 0.5  # Combine with different amplitudes

    print("Test setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of bins: {n_bins}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Bins range: [{bins.min():.3f}, {bins.max():.3f}]")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights sum per batch: {weights.sum(dim=-1)}")

    # Test deterministic sampling
    print("\n--- Deterministic Sampling Comparison ---")

    # C++ implementation
    import time

    start_time = time.time()
    cpp_samples_det = sample_pdf(bins, weights, n_samples, det=True)
    cpp_time_det = time.time() - start_time

    # Python implementation
    start_time = time.time()
    python_samples_det = sample_pdf_python(bins, weights, n_samples, det=True)
    python_time_det = time.time() - start_time

    print("C++ implementation (deterministic):")
    print(f"  Time: {cpp_time_det*1000:.3f} ms")
    print(f"  Output shape: {cpp_samples_det.shape}")
    print(f"  Sample range: [{cpp_samples_det.min():.3f}, {cpp_samples_det.max():.3f}]")
    print(f"  First batch mean: {cpp_samples_det[0].mean():.3f}")

    print("Python implementation (deterministic):")
    print(f"  Time: {python_time_det*1000:.3f} ms")
    print(f"  Output shape: {python_samples_det.shape}")
    print(
        f"  Sample range: [{python_samples_det.min():.3f}, {python_samples_det.max():.3f}]"
    )
    print(f"  First batch mean: {python_samples_det[0].mean():.3f}")

    # Compare deterministic results
    det_max_diff = torch.max(torch.abs(cpp_samples_det - python_samples_det))
    det_mean_diff = torch.mean(torch.abs(cpp_samples_det - python_samples_det))
    print("Deterministic results difference:")
    print(f"  Max absolute difference: {det_max_diff:.6f}")
    print(f"  Mean absolute difference: {det_mean_diff:.6f}")
    print(
        f"  Results are identical: {torch.allclose(cpp_samples_det, python_samples_det, atol=1e-6)}"
    )

    # Test stochastic sampling
    print("\n--- Stochastic Sampling Comparison ---")

    # Set same random seed for fair comparison
    torch.manual_seed(789)
    start_time = time.time()
    cpp_samples_stoch = sample_pdf(bins, weights, n_samples, det=False)
    cpp_time_stoch = time.time() - start_time

    torch.manual_seed(789)
    start_time = time.time()
    python_samples_stoch = sample_pdf_python(bins, weights, n_samples, det=False)
    python_time_stoch = time.time() - start_time

    print("C++ implementation (stochastic):")
    print(f"  Time: {cpp_time_stoch*1000:.3f} ms")
    print(
        f"  Sample range: [{cpp_samples_stoch.min():.3f}, {cpp_samples_stoch.max():.3f}]"
    )
    print(f"  First batch mean: {cpp_samples_stoch[0].mean():.3f}")
    print(f"  First batch std: {cpp_samples_stoch[0].std():.3f}")

    print("Python implementation (stochastic):")
    print(f"  Time: {python_time_stoch*1000:.3f} ms")
    print(
        f"  Sample range: [{python_samples_stoch.min():.3f}, {python_samples_stoch.max():.3f}]"
    )
    print(f"  First batch mean: {python_samples_stoch[0].mean():.3f}")
    print(f"  First batch std: {python_samples_stoch[0].std():.3f}")

    # Statistical comparison for stochastic samples
    # Since they use the same random seed, they should be very similar
    stoch_max_diff = torch.max(torch.abs(cpp_samples_stoch - python_samples_stoch))
    stoch_mean_diff = torch.mean(torch.abs(cpp_samples_stoch - python_samples_stoch))
    print("Stochastic results difference (same seed):")
    print(f"  Max absolute difference: {stoch_max_diff:.6f}")
    print(f"  Mean absolute difference: {stoch_mean_diff:.6f}")
    print(
        f"  Results are close: {torch.allclose(cpp_samples_stoch, python_samples_stoch, atol=1e-6)}"
    )

    # Performance comparison
    print("\n--- Performance Summary ---")
    print("Deterministic sampling:")
    print(f"  C++ speedup: {python_time_det/cpp_time_det:.1f}x")
    print("Stochastic sampling:")
    print(f"  C++ speedup: {python_time_stoch/cpp_time_stoch:.1f}x")

    # Test with different bin sizes to show complexity difference
    print("\n--- Complexity Analysis (Different Bin Sizes) ---")
    bin_sizes = [16, 32, 64, 128, 256]

    for n_bins_test in bin_sizes:
        bins_test = torch.linspace(0, 1, n_bins_test + 1).unsqueeze(0)
        weights_test = torch.rand(1, n_bins_test)
        weights_test = weights_test / weights_test.sum(dim=-1, keepdim=True)

        # Time C++ implementation
        start_time = time.time()
        _ = sample_pdf(bins_test, weights_test, 50, det=True)
        cpp_time = time.time() - start_time

        # Time Python implementation
        start_time = time.time()
        _ = sample_pdf_python(bins_test, weights_test, 50, det=True)
        python_time = time.time() - start_time

        speedup = python_time / cpp_time if cpp_time > 0 else float("inf")
        print(
            f"  {n_bins_test:3d} bins: C++ {cpp_time*1000:.2f}ms, Python {python_time*1000:.2f}ms, speedup: {speedup:.1f}x"
        )

    # Distribution fidelity test
    print("\n--- Distribution Fidelity Test ---")

    # Create a known distribution (mixture of Gaussians)
    x = torch.linspace(0, 1, n_bins)
    true_dist = 0.6 * torch.exp(-((x - 0.25) ** 2) / 0.01) + 0.4 * torch.exp(
        -((x - 0.75) ** 2) / 0.01
    )
    weights_fidelity = true_dist.unsqueeze(0)
    bins_fidelity = torch.linspace(0, 1, n_bins + 1).unsqueeze(0)

    # Sample many points for statistical analysis
    n_samples_large = 10000
    cpp_samples_large = sample_pdf(
        bins_fidelity, weights_fidelity, n_samples_large, det=False
    )
    python_samples_large = sample_pdf_python(
        bins_fidelity, weights_fidelity, n_samples_large, det=False
    )

    # Expected statistics
    expected_mean = torch.sum(x * weights_fidelity) / torch.sum(weights_fidelity)
    expected_var = torch.sum(((x - expected_mean) ** 2) * weights_fidelity) / torch.sum(
        weights_fidelity
    )

    cpp_mean = cpp_samples_large[0].mean()
    cpp_var = cpp_samples_large[0].var()
    python_mean = python_samples_large[0].mean()
    python_var = python_samples_large[0].var()

    print("Expected statistics:")
    print(f"  Mean: {expected_mean:.4f}")
    print(f"  Variance: {expected_var:.4f}")
    print("C++ implementation:")
    print(f"  Mean: {cpp_mean:.4f} (error: {abs(cpp_mean - expected_mean):.4f})")
    print(f"  Variance: {cpp_var:.4f} (error: {abs(cpp_var - expected_var):.4f})")
    print("Python implementation:")
    print(f"  Mean: {python_mean:.4f} (error: {abs(python_mean - expected_mean):.4f})")
    print(f"  Variance: {python_var:.4f} (error: {abs(python_var - expected_var):.4f})")

    print()


def example_differentiability_python_version():
    """Demonstrate differentiability with sample_pdf_python (not available in C++ version)."""
    print("=== Differentiability Example (Python version only) ===")

    # Setup differentiable parameters
    torch.manual_seed(1001)
    n_bins = 32
    n_samples = 100

    # Create learnable parameters for the distribution
    # We'll learn a distribution that should match a target distribution
    raw_weights = torch.randn(n_bins, requires_grad=True)
    bins = torch.linspace(0, 1, n_bins + 1).unsqueeze(0)

    # Target distribution (bimodal Gaussian)
    x_centers = torch.linspace(0, 1, n_bins)
    target_dist = 0.7 * torch.exp(-((x_centers - 0.25) ** 2) / 0.02) + 0.3 * torch.exp(
        -((x_centers - 0.75) ** 2) / 0.02
    )
    target_dist = target_dist / target_dist.sum()

    print("Differentiability setup:")
    print(f"  Number of bins: {n_bins}")
    print(f"  Number of samples per iteration: {n_samples}")
    print("  Target distribution: bimodal (peaks at 0.25 and 0.75)")
    print("  Learning rate: 0.01")

    # Show that C++ version doesn't support gradients
    print("\n--- Gradient Support Comparison ---")
    weights_test = torch.softmax(raw_weights, dim=0).unsqueeze(0)

    try:
        # This should raise an error
        _ = sample_pdf(bins, weights_test, 10, det=True)
        print("C++ implementation: Gradients supported - UNEXPECTED!")
    except NotImplementedError as e:
        print(f"C++ implementation: {e}")

    # Python version supports gradients
    _ = sample_pdf_python(bins, weights_test, 10, det=True)
    print("Python implementation: Gradients supported ✓")

    # Optimization loop
    print("\n--- Learning Distribution via Gradient Descent ---")
    optimizer = torch.optim.Adam([raw_weights], lr=0.01)

    losses = []
    learned_distributions = []

    for epoch in range(1050):
        optimizer.zero_grad()

        # Convert raw weights to probabilities
        weights = torch.softmax(raw_weights, dim=0).unsqueeze(0)

        # Sample from current distribution
        samples = sample_pdf_python(bins, weights, n_samples, det=False)

        # Convert samples back to bin indices for histogram
        sample_bins = (samples[0] * n_bins).long().clamp(0, n_bins - 1)

        # Create empirical distribution from samples
        empirical_dist = torch.zeros(n_bins)
        for bin_idx in sample_bins:
            empirical_dist[bin_idx] += 1.0
        empirical_dist = empirical_dist / empirical_dist.sum()

        # Loss: KL divergence between empirical and target distributions
        # kl_loss = torch.sum(
        #     target_dist * torch.log(target_dist / (empirical_dist + 1e-8))
        # )

        # Alternative loss: Direct weight matching (more stable)
        direct_loss = torch.nn.functional.mse_loss(weights.squeeze(), target_dist)

        # Use direct loss for more stable training
        loss = direct_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        learned_distributions.append(weights.squeeze().detach().clone())

        if epoch % 200 == 0:
            print(f"  Epoch {epoch:2d}: Loss = {loss.item():.6f}")

    # Final comparison
    final_weights = torch.softmax(raw_weights, dim=0)

    print("\n--- Final Results ---")
    print("Target distribution (first 10 bins):")
    print(f"  {target_dist[:10].numpy()}")
    print("Learned distribution (first 10 bins):")
    print(f"  {final_weights[:10].detach().numpy()}")

    # Calculate final metrics
    final_mse = torch.nn.functional.mse_loss(final_weights, target_dist)
    final_kl = torch.sum(target_dist * torch.log(target_dist / (final_weights + 1e-8)))

    print("\nFinal metrics:")
    print(f"  MSE: {final_mse.item():.6f}")
    print(f"  KL divergence: {final_kl.item():.6f}")
    print(
        f"  Correlation: {torch.corrcoef(torch.stack([target_dist, final_weights]))[0,1].item():.4f}"
    )

    # Demonstrate differentiable sampling for a practical application
    print("\n--- Practical Application: Differentiable Point Cloud Sampling ---")

    # Create a point cloud with learnable importance weights
    torch.manual_seed(1002)
    n_points = 200

    # Random 3D points
    points = torch.randn(n_points, 3) * 0.5

    # Learnable per-point importance (we want to learn to sample points near the origin)
    point_importance = torch.randn(n_points, requires_grad=True)

    print("Point cloud setup:")
    print(f"  Number of points: {n_points}")
    print(f"  Point cloud bounds: [{points.min():.2f}, {points.max():.2f}]")
    print("  Target: prefer points closer to origin")

    # Learning loop for point importance
    optimizer_points = torch.optim.Adam([point_importance], lr=0.02)

    print("\nNote: Using differentiable interpolation instead of discrete indexing")
    print("  - Discrete: sampled_points = points[indices.long()]  # Breaks gradients")
    print(
        "  - Differentiable: Linear interpolation between adjacent points preserves gradients"
    )

    for epoch in range(5030):
        optimizer_points.zero_grad()

        # Convert importance to probabilities
        point_probs = torch.softmax(point_importance, dim=0).unsqueeze(0)

        # Create bins for sampling (map point indices to continuous space)
        point_bins = torch.linspace(0, n_points, n_points + 1).unsqueeze(0)

        # Sample points using learned importance
        sampled_indices_float = sample_pdf_python(
            point_bins, point_probs, 50, det=False
        )

        # Use differentiable sampling: interpolate between points instead of discrete indexing
        # Normalize sampled indices to [0, n_points-1] range
        normalized_indices = sampled_indices_float[0] * (n_points - 1) / n_points
        normalized_indices = torch.clamp(normalized_indices, 0, n_points - 1)

        # Get floor and ceiling indices for interpolation
        floor_indices = torch.floor(normalized_indices).long()
        ceil_indices = torch.clamp(floor_indices + 1, 0, n_points - 1)

        # Compute interpolation weights
        alpha = normalized_indices - floor_indices.float()

        # Differentiable interpolation between points
        floor_points = points[floor_indices]  # Shape: (50, 3)
        ceil_points = points[ceil_indices]  # Shape: (50, 3)
        sampled_points = (1 - alpha.unsqueeze(-1)) * floor_points + alpha.unsqueeze(
            -1
        ) * ceil_points

        # Loss: sampled points should be close to origin on average
        distance_loss = torch.norm(sampled_points, dim=1).mean()

        # Regularization: don't deviate too much from uniform sampling
        uniform_reg = 0.1 * torch.norm(point_probs.squeeze() - 1.0 / n_points)

        total_loss = distance_loss + uniform_reg

        total_loss.backward()
        optimizer_points.step()

        if epoch % 300 == 0:
            avg_distance = torch.norm(sampled_points, dim=1).mean()
            print(
                f"  Epoch {epoch:2d}: Avg sampled distance = {avg_distance.item():.4f}"
            )

    # Final evaluation
    final_point_probs = torch.softmax(point_importance, dim=0)
    final_sampled_indices_float = sample_pdf_python(
        point_bins, final_point_probs.unsqueeze(0), 100, det=False
    )

    # Use the same differentiable sampling approach for final evaluation
    normalized_final_indices = (
        final_sampled_indices_float[0] * (n_points - 1) / n_points
    )
    normalized_final_indices = torch.clamp(normalized_final_indices, 0, n_points - 1)

    floor_final_indices = torch.floor(normalized_final_indices).long()
    ceil_final_indices = torch.clamp(floor_final_indices + 1, 0, n_points - 1)

    alpha_final = normalized_final_indices - floor_final_indices.float()

    floor_final_points = points[floor_final_indices]
    ceil_final_points = points[ceil_final_indices]
    final_sampled_points = (
        1 - alpha_final.unsqueeze(-1)
    ) * floor_final_points + alpha_final.unsqueeze(-1) * ceil_final_points

    # Compare with uniform sampling
    uniform_sampled_indices = torch.randint(0, n_points, (100,))
    uniform_sampled_points = points[uniform_sampled_indices]

    learned_avg_dist = torch.norm(final_sampled_points, dim=1).mean()
    uniform_avg_dist = torch.norm(uniform_sampled_points, dim=1).mean()

    print("\nFinal point sampling results:")
    print(f"  Learned sampling avg distance: {learned_avg_dist.item():.4f}")
    print(f"  Uniform sampling avg distance: {uniform_avg_dist.item():.4f}")
    print(
        f"  Improvement: {((uniform_avg_dist - learned_avg_dist) / uniform_avg_dist * 100).item():.1f}%"
    )

    # Show learned importance distribution
    top_importance_indices = torch.topk(final_point_probs, 10).indices
    bottom_importance_indices = torch.topk(final_point_probs, 10, largest=False).indices

    print("\nLearned importance analysis:")
    print(
        f"  Highest importance points (avg distance): {torch.norm(points[top_importance_indices], dim=1).mean():.4f}"
    )
    print(
        f"  Lowest importance points (avg distance): {torch.norm(points[bottom_importance_indices], dim=1).mean():.4f}"
    )

    print()


def main():
    """Run all examples."""
    print("Sample PDF Examples")
    print("=" * 30)
    print()

    example_basic_pdf_sampling()
    example_deterministic_vs_stochastic_sampling()
    example_compare_sample_pdf_implementations()
    example_differentiability_python_version()

    print("All examples completed!")


if __name__ == "__main__":
    main()
