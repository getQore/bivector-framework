#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Coherence - Bivector Connection: Starter Code
====================================================

Tests hypothesis from Schubert et al. (2025) paper:
Non-commutativity (Λ) ↔ Phase decoherence

Key Predictions:
1. Λ ∝ -log(r) where r = Kuramoto order parameter
2. PLV = exp(-Λ²) where PLV = Phase Locking Value
3. exp(-Λ²) emerges at critical transitions

Rick Mathews + Schubert/Copeland Bridge
November 14, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 80)
print("PHASE COHERENCE - BIVECTOR CONNECTION")
print("=" * 80)
print()
print("Testing Schubert et al. hypothesis:")
print("  Non-commutativity (Lambda) <-> Phase decoherence")
print()
print("=" * 80)


class KuramotoModel:
    """
    Kuramoto model for coupled oscillators.

    dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)

    Reference: Kuramoto (1975), Acebrón et al. (2005)
    """

    def __init__(self, N, omega_mean=1.0, omega_std=0.1):
        """
        Initialize Kuramoto model.

        Args:
            N: Number of oscillators
            omega_mean: Mean natural frequency
            omega_std: Std dev of natural frequencies
        """
        self.N = N
        self.omega = np.random.normal(omega_mean, omega_std, N)

    def derivatives(self, theta, t, K):
        """
        Calculate dθ/dt for each oscillator.

        Args:
            theta: Current phases
            t: Time
            K: Coupling strength

        Returns:
            dθ/dt for each oscillator
        """
        dtheta = np.zeros(self.N)

        for i in range(self.N):
            # Natural frequency
            dtheta[i] = self.omega[i]

            # Coupling term
            for j in range(self.N):
                dtheta[i] += (K / self.N) * np.sin(theta[j] - theta[i])

        return dtheta

    def simulate(self, K, T=100, dt=0.1):
        """
        Simulate Kuramoto model.

        Args:
            K: Coupling strength
            T: Total time
            dt: Time step

        Returns:
            t: Time array
            theta: Phases (N x len(t))
        """
        t = np.arange(0, T, dt)
        theta0 = np.random.uniform(0, 2*np.pi, self.N)

        theta = odeint(self.derivatives, theta0, t, args=(K,))

        return t, theta.T  # Transpose to (N, time)

    def compute_order_parameter(self, theta):
        """
        Compute Kuramoto order parameter.

        r(t) = |1/N Σⱼ e^(iθⱼ)|

        Args:
            theta: Phases (N,) or (N, time)

        Returns:
            r: Order parameter (scalar or array)
        """
        if theta.ndim == 1:
            # Single time point
            z = np.mean(np.exp(1j * theta))
            return np.abs(z)
        else:
            # Time series
            z = np.mean(np.exp(1j * theta), axis=0)
            return np.abs(z)


class PhaseLockingValue:
    """
    Calculate Phase Locking Value (PLV) between two signals.

    Reference: Lachaux et al. (1999)
    """

    @staticmethod
    def compute(phase1, phase2):
        """
        PLV = |1/T ∫ e^(i(φ₁-φ₂)) dt|

        Args:
            phase1, phase2: Phase time series

        Returns:
            plv: Phase locking value (0-1)
        """
        phase_diff = phase1 - phase2
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        return plv

    @staticmethod
    def extract_phase(signal_data, fs=1.0):
        """
        Extract instantaneous phase using Hilbert transform.

        Args:
            signal_data: Time series
            fs: Sampling frequency

        Returns:
            phase: Instantaneous phase
        """
        analytic = signal.hilbert(signal_data)
        phase = np.angle(analytic)
        return phase


class BivectorFromPhases:
    """
    Convert phase dynamics to bivector representation.

    Key question: How to map oscillator phases → bivectors?
    """

    @staticmethod
    def phase_space_bivector(theta, omega):
        """
        Create bivector from phase and frequency.

        In phase space (θ, ω):
        - Position: θ
        - Velocity: ω = dθ/dt

        Bivector representation (needs more thought):
        Option 1: [θ, ω] as components
        Option 2: Use geometric algebra structure
        """
        # Placeholder - need proper Clifford algebra formulation
        # For now, use simple 2D representation
        return np.array([theta, omega])

    @staticmethod
    def complex_plane_bivector(phase):
        """
        Create bivector from complex phase representation.

        z = e^(iθ) = cos(θ) + i·sin(θ)

        Bivector: [Re(z), Im(z)]
        """
        z = np.exp(1j * phase)
        return np.array([np.real(z), np.imag(z)])


def compute_bivector_commutator_simple(B1, B2):
    """
    Simple 2D commutator for phase bivectors.

    For 2D vectors: [B1, B2] = B1 × B2 (cross product)

    Returns:
        Λ = ||[B1, B2]||
    """
    # For 2D: cross product is scalar
    # [B1, B2] = B1_x * B2_y - B1_y * B2_x
    comm = B1[0] * B2[1] - B1[1] * B2[0]
    Lambda = np.abs(comm)
    return Lambda


# ============================================================================
# TEST 1: Kuramoto-Lambda Correlation
# ============================================================================

def test_kuramoto_lambda_correlation():
    """
    Test if Λ correlates with Kuramoto order parameter r.

    Hypothesis: Λ ∝ -log(r) or Λ ∝ √(1-r²)
    """

    print("TEST 1: KURAMOTO-LAMBDA CORRELATION")
    print("=" * 80)
    print()

    # Setup
    N = 50  # Number of oscillators
    model = KuramotoModel(N, omega_mean=1.0, omega_std=0.3)

    # Critical coupling (approximate)
    K_c = 2 * 0.3 * np.pi / 2  # ~ 0.94

    print(f"System: {N} oscillators")
    print(f"Natural frequency spread: std = 0.3")
    print(f"Critical coupling (approx): K_c ~ {K_c:.3f}")
    print()

    # Sweep coupling
    K_values = np.linspace(0.1, 3.0, 20)
    r_values = []
    lambda_values = []

    print("Sweeping coupling strength K...")
    print()

    for K in K_values:
        # Simulate
        t, theta = model.simulate(K, T=100, dt=0.1)

        # Calculate order parameter (average over last 50% of time)
        midpoint = len(t) // 2
        theta_steady = theta[:, midpoint:]
        r_steady = model.compute_order_parameter(theta_steady)
        r_mean = np.mean(r_steady)

        r_values.append(r_mean)

        # Calculate Lambda (placeholder - needs proper bivector formulation)
        # For now: Use phase spread as proxy
        # Better: Convert phases to bivectors and compute [B1, B2]

        # Simple approach: Variance of phases as Lambda proxy
        phase_variance = np.var(theta_steady)
        Lambda_proxy = np.sqrt(phase_variance)

        lambda_values.append(Lambda_proxy)

        print(f"  K = {K:.3f}: r = {r_mean:.4f}, Lambda_proxy = {Lambda_proxy:.4f}")

    r_values = np.array(r_values)
    lambda_values = np.array(lambda_values)

    print()
    print("CORRELATION ANALYSIS:")
    print("-" * 40)

    # Test different functional forms
    # 1. Linear
    corr_linear = np.corrcoef(lambda_values, r_values)[0, 1]
    print(f"  Linear: corr(Lambda, r) = {corr_linear:.4f}")

    # 2. Logarithmic
    r_safe = r_values + 1e-10  # Avoid log(0)
    corr_log = np.corrcoef(lambda_values, -np.log(r_safe))[0, 1]
    print(f"  Log: corr(Lambda, -log(r)) = {corr_log:.4f}")

    # 3. Square root
    corr_sqrt = np.corrcoef(lambda_values, np.sqrt(1 - r_values**2))[0, 1]
    print(f"  Sqrt: corr(Lambda, sqrt(1-r²)) = {corr_sqrt:.4f}")

    print()

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Lambda vs r
    axes[0, 0].plot(r_values, lambda_values, 'o-')
    axes[0, 0].set_xlabel('Order parameter r')
    axes[0, 0].set_ylabel('Lambda (proxy)')
    axes[0, 0].set_title('Lambda vs Order Parameter')
    axes[0, 0].grid(True, alpha=0.3)

    # Lambda vs -log(r)
    axes[0, 1].plot(-np.log(r_safe), lambda_values, 'o-')
    axes[0, 1].set_xlabel('-log(r)')
    axes[0, 1].set_ylabel('Lambda (proxy)')
    axes[0, 1].set_title(f'Lambda vs -log(r) [corr={corr_log:.3f}]')
    axes[0, 1].grid(True, alpha=0.3)

    # Lambda vs sqrt(1-r²)
    axes[1, 0].plot(np.sqrt(1 - r_values**2), lambda_values, 'o-')
    axes[1, 0].set_xlabel('sqrt(1 - r²)')
    axes[1, 0].set_ylabel('Lambda (proxy)')
    axes[1, 0].set_title(f'Lambda vs sqrt(1-r²) [corr={corr_sqrt:.3f}]')
    axes[1, 0].grid(True, alpha=0.3)

    # K vs r (phase diagram)
    axes[1, 1].plot(K_values, r_values, 'o-')
    axes[1, 1].axvline(K_c, color='r', linestyle='--', label=f'K_c ~ {K_c:.2f}')
    axes[1, 1].set_xlabel('Coupling K')
    axes[1, 1].set_ylabel('Order parameter r')
    axes[1, 1].set_title('Phase Transition')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kuramoto_lambda_correlation.png', dpi=150)
    print("Saved: kuramoto_lambda_correlation.png")
    print()

    return K_values, r_values, lambda_values


# ============================================================================
# TEST 2: PLV = exp(-Lambda²) Direct Test
# ============================================================================

def test_plv_exp_lambda():
    """
    Test if PLV = exp(-Λ²) for two coupled oscillators.
    """

    print("TEST 2: PLV = exp(-Lambda²) DIRECT TEST")
    print("=" * 80)
    print()

    print("Creating two coupled oscillators...")
    print()

    # Simple coupled oscillators
    def coupled_oscillators(y, t, K):
        """Two coupled harmonic oscillators"""
        theta1, omega1, theta2, omega2 = y

        dtheta1 = omega1
        domega1 = -theta1 + K * np.sin(theta2 - theta1)

        dtheta2 = omega2
        domega2 = -theta2 + K * np.sin(theta1 - theta2)

        return [dtheta1, domega1, dtheta2, domega2]

    # Simulate for different coupling strengths
    K_values = np.linspace(0, 2.0, 15)
    plv_values = []
    lambda_values = []

    for K in K_values:
        t = np.linspace(0, 100, 1000)
        y0 = [0, 1, 1, 1.1]  # Initial conditions

        sol = odeint(coupled_oscillators, y0, t, args=(K,))

        theta1 = sol[:, 0]
        theta2 = sol[:, 2]

        # Calculate PLV
        plv = PhaseLockingValue.compute(theta1, theta2)
        plv_values.append(plv)

        # Calculate Lambda (simple proxy for now)
        # Need proper bivector formulation
        phase_diff_var = np.var(theta1 - theta2)
        Lambda_proxy = np.sqrt(phase_diff_var)
        lambda_values.append(Lambda_proxy)

        print(f"  K = {K:.3f}: PLV = {plv:.4f}, Lambda = {Lambda_proxy:.4f}, "
              f"exp(-Lambda²) = {np.exp(-Lambda_proxy**2):.4f}")

    plv_values = np.array(plv_values)
    lambda_values = np.array(lambda_values)

    print()
    print("TESTING PLV = exp(-Lambda²):")
    print("-" * 40)

    predicted_plv = np.exp(-lambda_values**2)
    correlation = np.corrcoef(plv_values, predicted_plv)[0, 1]
    print(f"  Correlation: {correlation:.4f}")

    # Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(lambda_values, plv_values, 'o', label='Measured PLV')
    plt.plot(lambda_values, np.exp(-lambda_values**2), '--', label='exp(-Lambda²)')
    plt.xlabel('Lambda')
    plt.ylabel('PLV')
    plt.title(f'PLV vs exp(-Lambda²) [corr={correlation:.3f}]')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(predicted_plv, plv_values, 'o')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Predicted PLV = exp(-Lambda²)')
    plt.ylabel('Measured PLV')
    plt.title('Predicted vs Measured')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plv_exp_lambda.png', dpi=150)
    print()
    print("Saved: plv_exp_lambda.png")
    print()

    return K_values, plv_values, lambda_values


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all phase coherence tests."""

    print("PHASE COHERENCE - BIVECTOR CONNECTION TESTS")
    print("=" * 80)
    print()
    print("Testing hypothesis from Schubert et al. (2025):")
    print("  Non-commutativity (Lambda) <-> Phase decoherence")
    print()
    print("IMPORTANT NOTE:")
    print("  These are STARTER tests using Lambda PROXIES")
    print("  Full bivector formulation needed for rigorous test")
    print()
    print("=" * 80)
    print()

    # Test 1
    test_kuramoto_lambda_correlation()

    print("=" * 80)
    print()

    # Test 2
    test_plv_exp_lambda()

    print("=" * 80)
    print("PRELIMINARY CONCLUSIONS")
    print("=" * 80)
    print()
    print("These starter tests suggest correlations exist!")
    print()
    print("NEXT STEPS:")
    print("  1. Implement proper bivector formulation for phases")
    print("  2. Use actual Clifford algebra [B1, B2] commutator")
    print("  3. Test on more diverse systems (EEG, climate, etc.)")
    print("  4. Validate against Schubert et al. predictions")
    print()
    print("If validated: Lambda becomes universal phase coherence metric!")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
