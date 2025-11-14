#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution to Bivector Mapping Utilities
==========================================

Core utilities for RL patent validation:
- Map probability distributions to Cl(3,1) bivectors
- Compute Lambda = ||[B1, B2]|| between distributions
- Calculate standard distance metrics for comparison

Rick Mathews - RL Patent Application
November 14, 2024
"""

import numpy as np
from scipy.stats import norm
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


class BivectorCl31:
    """
    Bivector in Clifford algebra Cl(3,1) with signature (+,-,-,-)

    6 components: [e_01, e_02, e_03, e_23, e_31, e_12]
    """

    def __init__(self, components):
        """
        Initialize bivector.

        Args:
            components: Array of 6 components
        """
        self.B = np.array(components, dtype=float)

    def commutator(self, other):
        """
        Compute [self, other] = self * other - other * self

        For distributions, we use simplified commutator.
        """
        # Simplified: cross-product-like structure
        # Full Clifford algebra commutator is complex
        # For our purposes, we use the 2D subspace (e_01, e_23)

        B1 = self.B
        B2 = other.B

        comm = np.zeros(6)

        # Key components for distribution distinguishability
        # e_01 (mean) and e_23 (std) cross terms
        comm[0] = B1[0] * B2[3] - B1[3] * B2[0]
        comm[3] = B1[3] * B2[0] - B1[0] * B2[3]

        # Additional cross terms
        comm[1] = B1[1] * B2[4] - B1[4] * B2[1]
        comm[2] = B1[2] * B2[5] - B1[5] * B2[2]
        comm[4] = B1[4] * B2[1] - B1[1] * B2[4]
        comm[5] = B1[5] * B2[2] - B1[2] * B2[5]

        return BivectorCl31(comm)

    def norm(self):
        """Frobenius norm of bivector"""
        return np.linalg.norm(self.B)


def gaussian_to_bivector(mu, sigma):
    """
    Map Gaussian distribution to Cl(3,1) bivector.

    Encoding:
    - Mean (μ) → e_01 component (boost-like)
    - Std (σ) → e_23 component (rotation-like)

    Interpretation:
    - e_01: "Displacement" in value space
    - e_23: "Uncertainty" in value space

    Args:
        mu: Mean of Gaussian
        sigma: Standard deviation of Gaussian

    Returns:
        BivectorCl31 object
    """
    B = np.zeros(6)
    B[0] = mu      # e_01 component
    B[3] = sigma   # e_23 component

    return BivectorCl31(B)


def compute_distribution_lambda(dist1, dist2):
    """
    Compute Lambda = ||[B1, B2]|| between two distributions.

    Args:
        dist1: Dictionary with 'mu' and 'sigma'
        dist2: Dictionary with 'mu' and 'sigma'

    Returns:
        Lambda: Non-commutativity metric (scalar)
    """
    # Convert to bivectors
    B1 = gaussian_to_bivector(dist1['mu'], dist1['sigma'])
    B2 = gaussian_to_bivector(dist2['mu'], dist2['sigma'])

    # Compute commutator
    comm = B1.commutator(B2)

    # Return Frobenius norm
    Lambda = comm.norm()

    return Lambda


def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """
    KL divergence D_KL(P1 || P2) for Gaussian distributions.

    D_KL(N(μ1,σ1) || N(μ2,σ2)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2

    Args:
        mu1, sigma1: Parameters of P1
        mu2, sigma2: Parameters of P2

    Returns:
        KL divergence (scalar)
    """
    term1 = np.log(sigma2 / (sigma1 + 1e-10) + 1e-10)
    term2 = (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2 + 1e-10)
    term3 = -0.5

    kl = term1 + term2 + term3
    return kl


def wasserstein_distance_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Wasserstein-2 distance for Gaussian distributions.

    W_2(N(μ1,σ1), N(μ2,σ2)) = sqrt((μ1-μ2)² + (σ1-σ2)²)

    Args:
        mu1, sigma1: Parameters of P1
        mu2, sigma2: Parameters of P2

    Returns:
        Wasserstein distance (scalar)
    """
    w2 = np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)
    return w2


def hellinger_distance_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Hellinger distance for Gaussian distributions.

    H²(N(μ1,σ1), N(μ2,σ2)) = 1 - sqrt(2σ1σ2/(σ1²+σ2²)) * exp(-(μ1-μ2)²/(4(σ1²+σ2²)))

    Args:
        mu1, sigma1: Parameters of P1
        mu2, sigma2: Parameters of P2

    Returns:
        Hellinger distance (scalar, in [0,1])
    """
    sigma_term = 2 * sigma1 * sigma2 / (sigma1**2 + sigma2**2 + 1e-10)
    exp_term = np.exp(-(mu1 - mu2)**2 / (4 * (sigma1**2 + sigma2**2) + 1e-10))

    h_squared = 1 - np.sqrt(sigma_term) * exp_term
    h = np.sqrt(h_squared)

    return h


def bhattacharyya_distance_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Bhattacharyya distance for Gaussian distributions.

    DB = (1/4) * (μ1-μ2)²/(σ1²+σ2²) + (1/2) * log((σ1²+σ2²)/(2σ1σ2))

    Args:
        mu1, sigma1: Parameters of P1
        mu2, sigma2: Parameters of P2

    Returns:
        Bhattacharyya distance (scalar)
    """
    term1 = 0.25 * (mu1 - mu2)**2 / (sigma1**2 + sigma2**2 + 1e-10)
    term2 = 0.5 * np.log((sigma1**2 + sigma2**2) / (2 * sigma1 * sigma2 + 1e-10) + 1e-10)

    db = term1 + term2
    return db


def compute_all_distances(dist1, dist2):
    """
    Compute all distance metrics between two Gaussian distributions.

    Args:
        dist1: Dictionary with 'mu' and 'sigma'
        dist2: Dictionary with 'mu' and 'sigma'

    Returns:
        Dictionary of all distances
    """
    mu1, sigma1 = dist1['mu'], dist1['sigma']
    mu2, sigma2 = dist2['mu'], dist2['sigma']

    distances = {
        'lambda': compute_distribution_lambda(dist1, dist2),
        'kl': kl_divergence_gaussian(mu1, sigma1, mu2, sigma2),
        'wasserstein': wasserstein_distance_gaussian(mu1, sigma1, mu2, sigma2),
        'hellinger': hellinger_distance_gaussian(mu1, sigma1, mu2, sigma2),
        'bhattacharyya': bhattacharyya_distance_gaussian(mu1, sigma1, mu2, sigma2)
    }

    return distances


def compute_overlap_coefficient(dist1, dist2, n_samples=10000):
    """
    Compute empirical overlap coefficient between two distributions.

    OC = ∫ min(p1(x), p2(x)) dx

    Args:
        dist1, dist2: Dictionaries with 'mu' and 'sigma'
        n_samples: Number of points for numerical integration

    Returns:
        Overlap coefficient in [0,1]
    """
    mu1, sigma1 = dist1['mu'], dist1['sigma']
    mu2, sigma2 = dist2['mu'], dist2['sigma']

    # Sample points from both distributions
    min_x = min(mu1 - 4*sigma1, mu2 - 4*sigma2)
    max_x = max(mu1 + 4*sigma1, mu2 + 4*sigma2)

    x = np.linspace(min_x, max_x, n_samples)

    # Evaluate PDFs
    p1 = norm.pdf(x, mu1, sigma1)
    p2 = norm.pdf(x, mu2, sigma2)

    # Overlap = integral of minimum
    overlap = np.trapz(np.minimum(p1, p2), x)

    return overlap


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_bivector_basics():
    """Test basic bivector operations"""
    print("="*80)
    print("TESTING BIVECTOR BASICS")
    print("="*80)
    print()

    # Create two bivectors
    B1 = BivectorCl31([1, 0, 0, 0.5, 0, 0])
    B2 = BivectorCl31([2, 0, 0, 1.0, 0, 0])

    print("B1 components:", B1.B)
    print("B2 components:", B2.B)
    print()

    # Commutator
    comm = B1.commutator(B2)
    print("Commutator [B1, B2]:", comm.B)
    print("Lambda = ||[B1, B2]||:", comm.norm())
    print()


def test_gaussian_mapping():
    """Test Gaussian to bivector mapping"""
    print("="*80)
    print("TESTING GAUSSIAN MAPPING")
    print("="*80)
    print()

    # Two Gaussians
    dist1 = {'mu': 1.0, 'sigma': 0.5}
    dist2 = {'mu': 2.0, 'sigma': 1.0}

    print(f"Distribution 1: N(mu={dist1['mu']}, sigma={dist1['sigma']})")
    print(f"Distribution 2: N(mu={dist2['mu']}, sigma={dist2['sigma']})")
    print()

    # Compute Lambda
    Lambda = compute_distribution_lambda(dist1, dist2)
    print(f"Lambda = {Lambda:.4f}")
    print()

    # Compare to standard metrics
    distances = compute_all_distances(dist1, dist2)

    print("Distance Metrics:")
    print("-"*40)
    for metric, value in distances.items():
        print(f"  {metric:15s}: {value:.4f}")
    print()


def test_identical_distributions():
    """Test that Lambda = 0 for identical distributions"""
    print("="*80)
    print("TESTING IDENTICAL DISTRIBUTIONS")
    print("="*80)
    print()

    dist = {'mu': 1.0, 'sigma': 0.5}

    Lambda = compute_distribution_lambda(dist, dist)
    print(f"Lambda for identical distributions: {Lambda:.10f}")
    print("Expected: 0.0")
    print(f"Result: {'PASS' if abs(Lambda) < 1e-10 else 'FAIL'}")
    print()


def test_varying_mean():
    """Test Lambda vs mean difference"""
    print("="*80)
    print("TESTING VARYING MEAN")
    print("="*80)
    print()

    sigma = 1.0
    mu_base = 0.0
    mu_values = np.linspace(0, 5, 11)

    print(f"Fixed: sigma = {sigma}")
    print(f"Varying: mu from {mu_values[0]} to {mu_values[-1]}")
    print()

    for mu in mu_values:
        dist1 = {'mu': mu_base, 'sigma': sigma}
        dist2 = {'mu': mu, 'sigma': sigma}

        Lambda = compute_distribution_lambda(dist1, dist2)
        kl = kl_divergence_gaussian(mu_base, sigma, mu, sigma)

        print(f"  mu = {mu:4.1f}: Lambda = {Lambda:6.4f}, KL = {kl:6.4f}")
    print()


def test_varying_sigma():
    """Test Lambda vs variance difference"""
    print("="*80)
    print("TESTING VARYING SIGMA")
    print("="*80)
    print()

    mu = 0.0
    sigma_base = 1.0
    sigma_values = np.linspace(0.1, 3.0, 11)

    print(f"Fixed: mu = {mu}")
    print(f"Varying: sigma from {sigma_values[0]} to {sigma_values[-1]}")
    print()

    for sigma in sigma_values:
        dist1 = {'mu': mu, 'sigma': sigma_base}
        dist2 = {'mu': mu, 'sigma': sigma}

        Lambda = compute_distribution_lambda(dist1, dist2)
        kl = kl_divergence_gaussian(mu, sigma_base, mu, sigma)

        print(f"  sigma = {sigma:4.1f}: Lambda = {Lambda:6.4f}, KL = {kl:6.4f}")
    print()


def main():
    """Run all tests"""
    print("\n")
    print("="*80)
    print("DISTRIBUTION BIVECTOR UTILITIES - TEST SUITE")
    print("="*80)
    print()

    test_bivector_basics()
    test_gaussian_mapping()
    test_identical_distributions()
    test_varying_mean()
    test_varying_sigma()

    print("="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
    print()
    print("This module provides core utilities for RL patent validation:")
    print("  - gaussian_to_bivector(mu, sigma)")
    print("  - compute_distribution_lambda(dist1, dist2)")
    print("  - compute_all_distances(dist1, dist2)")
    print()
    print("Next steps:")
    print("  1. Run test_distribution_correlation.py (Day 1)")
    print("  2. Validate Lambda vs KL divergence")
    print("  3. Proceed with Reddit 3-door problem (Day 2)")
    print()


if __name__ == "__main__":
    main()
