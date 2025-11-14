#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Bivector Physics Framework: Rigorous Implementation

B₁ = Spin bivector (intrinsic angular momentum)
B₂ = Helicity bivector (spin-momentum coupling)
B₃ = Flavor bivector (internal symmetry - generation structure)

Testable predictions:
1. Anomalous magnetic moments (g-2) for electron, muon, tau
2. Neutrino oscillation parameters (Δm², mixing angles)
3. CP violation phase (Jarlskog invariant)
4. Generation mass hierarchy

Rick Mathews
November 2024
"""

import numpy as np
from scipy.linalg import norm, expm, logm
import matplotlib.pyplot as plt
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Physical constants (SI units)
HBAR = 1.055e-34  # J·s
C = 2.998e8  # m/s
M_E = 9.109e-31  # kg (electron)
M_MU = 1.883e-28  # kg (muon)
M_TAU = 3.167e-27  # kg (tau)
E_CHARGE = 1.602e-19  # C
ALPHA = 1/137.036  # Fine structure constant
MU_B = 9.274e-24  # Bohr magneton (J/T)

# Experimental values for comparison
MEASURED = {
    'electron_g2': 0.00115965218073,  # ± 2.8e-13
    'electron_g2_err': 2.8e-13,
    'muon_g2': 0.00116592089,  # ± 6.3e-10
    'muon_g2_err': 6.3e-10,
    'dm2_21': 7.42e-5,  # eV² (solar)
    'dm2_32': 2.51e-3,  # eV² (atmospheric)
    'theta_12': 33.82,  # degrees (solar angle)
    'theta_23': 49.7,   # degrees (atmospheric angle)
    'theta_13': 8.61,   # degrees (reactor angle)
    'delta_cp': 1.36,   # radians (CP phase)
    'jarlskog': 3.18e-5  # CP invariant
}

print("=" * 80)
print("THREE-BIVECTOR PHYSICS FRAMEWORK")
print("Testable Predictions from Geometric Algebra")
print("=" * 80)
print()


class BivectorGA:
    """
    Bivector representation in Cl(3,1) - Spacetime Geometric Algebra

    Basis bivectors:
        Spatial: e₂₃, e₃₁, e₁₂ (dual to rotation axes)
        Boost: e₀₁, e₀₂, e₀₃ (spacetime rotations)
    """

    def __init__(self, spatial, boost=None):
        """
        Parameters:
        -----------
        spatial : array-like (3,)
            Components in spatial bivector basis (Sx, Sy, Sz)
        boost : array-like (3,), optional
            Components in boost bivector basis (βx, βy, βz)
        """
        self.spatial = np.array(spatial, dtype=float)
        self.boost = np.array(boost) if boost is not None else np.zeros(3)

    def to_matrix_spatial(self):
        """
        Convert spatial bivector to 3×3 antisymmetric matrix.

        B = Sx e₂₃ + Sy e₃₁ + Sz e₁₂

        Matrix representation uses dual: e_ij ↔ cross product with axis
        """
        Sx, Sy, Sz = self.spatial

        # Antisymmetric matrix (generator of rotations)
        return np.array([
            [0, -Sz, Sy],
            [Sz, 0, -Sx],
            [-Sy, Sx, 0]
        ])

    def commutator(self, other):
        """
        Compute [self, other] = self ∧ other (geometric product antisymmetric part)

        Returns Lambda = ||[B₁, B₂]||_F (Frobenius norm)
        """
        M1 = self.to_matrix_spatial()
        M2 = other.to_matrix_spatial()

        comm = M1 @ M2 - M2 @ M1

        return norm(comm, 'fro')

    def magnitude(self):
        """Magnitude of bivector."""
        return np.sqrt(np.sum(self.spatial**2) + np.sum(self.boost**2))

    def __repr__(self):
        return f"BivectorGA(S=[{self.spatial[0]:.3f}, {self.spatial[1]:.3f}, {self.spatial[2]:.3f}])"


class ThreeBivectorPhysics:
    """
    Physics predictions from three-bivector framework.

    B₁: Spin (intrinsic angular momentum) - proven
    B₂: Helicity (spin-momentum coupling) - proven
    B₃: Flavor (generation structure) - proposed
    """

    def __init__(self):
        """Initialize framework."""
        self.alpha = ALPHA
        self.hbar = HBAR
        self.c = C

    def create_spin_bivector(self, particle='electron', spin_z=0.5):
        """
        B₁: Spin bivector (intrinsic)

        For spin-1/2 particles: S = ℏ/2 (eigenvector of Sz)
        """
        return BivectorGA([0, 0, spin_z])

    def create_helicity_bivector(self, momentum_direction, spin_z=0.5):
        """
        B₂: Helicity bivector (spin-momentum coupling)

        H = S · p̂ (spin projection along momentum)

        Parameters:
        -----------
        momentum_direction : array-like (3,)
            Unit vector in momentum direction
        spin_z : float
            Spin quantum number
        """
        p_hat = np.array(momentum_direction) / np.linalg.norm(momentum_direction)

        # Helicity operator projects spin onto momentum direction
        # For now, assume aligned: H ≈ spin_z * p_hat[2]
        helicity = spin_z * p_hat

        return BivectorGA(helicity)

    def create_flavor_bivector(self, generation=1):
        """
        B₃: Flavor bivector (internal symmetry)

        Encodes generation structure: (e, μ, τ), (νₑ, νμ, ντ), etc.

        Parameters:
        -----------
        generation : int (1, 2, or 3)
            Particle generation
        """
        # Flavor space basis (analogous to SU(3) flavor)
        flavor_axes = {
            1: [1, 0, 0],  # First generation (e, νₑ)
            2: [0, 1, 0],  # Second generation (μ, νμ)
            3: [0, 0, 1]   # Third generation (τ, ντ)
        }

        return BivectorGA(flavor_axes.get(generation, [1, 0, 0]))

    def compute_g2_electron(self, B_spin, B_helicity):
        """
        Prediction 1: Electron anomalous magnetic moment a_e = (g-2)/2

        QED gives: a_e = α/(2π) + higher orders

        Our prediction: a_e = f(Λ₁₂) where Λ₁₂ = ||[B₁, B₂]||

        Hypothesis: a_e = (α/(2π)) × (1 + Λ₁₂²)
        """
        # Compute kinematic curvature
        Lambda_12 = B_spin.commutator(B_helicity)

        # QED leading order
        a_QED = self.alpha / (2 * np.pi)

        # Geometric correction from bivector coupling
        # Hypothesis: Λ₁₂ ~ α (since both are small)
        # So correction ~ Λ₁₂² ~ α²

        # Try different models:

        # Model 1: Direct scaling
        a_model1 = a_QED * (1 + Lambda_12**2)

        # Model 2: Logarithmic correction
        if Lambda_12 > 0:
            a_model2 = a_QED * (1 + Lambda_12 * np.log(1/Lambda_12))
        else:
            a_model2 = a_QED

        # Model 3: Exponential suppression
        a_model3 = a_QED / (1 - Lambda_12**2)

        return {
            'QED_leading': a_QED,
            'model1_quadratic': a_model1,
            'model2_logarithmic': a_model2,
            'model3_exponential': a_model3,
            'Lambda_12': Lambda_12
        }

    def compute_neutrino_oscillations(self, B_helicity, B_flavor):
        """
        Prediction 2: Neutrino oscillation parameters

        Δm²ᵢⱼ = (Λᵢⱼ × E_scale)²

        where E_scale is characteristic energy (e.g., 1 MeV)
        """
        # Three generations → 3×3 mass matrix
        flavors = [1, 2, 3]

        # Create flavor bivectors
        B1 = self.create_flavor_bivector(1)
        B2 = self.create_flavor_bivector(2)
        B3 = self.create_flavor_bivector(3)

        # Compute pairwise Lambda values
        Lambda_12 = B1.commutator(B2)
        Lambda_23 = B2.commutator(B3)
        Lambda_13 = B1.commutator(B3)

        # Mass matrix from bivector couplings
        M_matrix = np.array([
            [0, Lambda_12, Lambda_13],
            [Lambda_12, 0, Lambda_23],
            [Lambda_13, Lambda_23, 0]
        ])

        # Eigenvalues → mass splittings
        eigenvalues = np.linalg.eigvalsh(M_matrix)

        # Energy scale (fit to data)
        E_scale = 1e-3  # Start with eV scale

        masses_squared = (eigenvalues * E_scale)**2

        # Compute Δm² differences
        dm2_21 = abs(masses_squared[1] - masses_squared[0])
        dm2_32 = abs(masses_squared[2] - masses_squared[1])
        dm2_31 = abs(masses_squared[2] - masses_squared[0])

        return {
            'dm2_21': dm2_21,
            'dm2_32': dm2_32,
            'dm2_31': dm2_31,
            'Lambda_12': Lambda_12,
            'Lambda_23': Lambda_23,
            'Lambda_13': Lambda_13,
            'eigenvalues': eigenvalues
        }

    def compute_cp_violation(self):
        """
        Prediction 3: CP violation (Jarlskog invariant)

        J = Im(V_us V_cb V*_ub V*_cs)

        From bivector framework: J = scalar_part(B₁ ∧ B₂ ∧ B₃)
        """
        B1 = self.create_flavor_bivector(1)
        B2 = self.create_flavor_bivector(2)
        B3 = self.create_flavor_bivector(3)

        # Triple bivector product (simplified - need proper wedge product)
        # For now, use determinant as proxy
        M = np.vstack([B1.spatial, B2.spatial, B3.spatial])
        J = np.linalg.det(M)

        # Scale to match observed value
        J_normalized = abs(J) / (2 * np.sqrt(3))  # Normalization factor

        return {
            'jarlskog': J_normalized,
            'triple_product': J
        }

    def validate_against_experiment(self):
        """
        Compare all predictions to experimental measurements.

        Returns pass/fail for each prediction.
        """
        print("\nVALIDATION AGAINST EXPERIMENT")
        print("=" * 80)
        print()

        results = {}

        # Test 1: Electron g-2
        print("1. ELECTRON ANOMALOUS MAGNETIC MOMENT")
        print("-" * 80)

        B_spin = self.create_spin_bivector('electron', spin_z=0.5)
        B_helicity = self.create_helicity_bivector([0, 0, 1], spin_z=0.5)  # Along z

        g2_results = self.compute_g2_electron(B_spin, B_helicity)

        measured_a_e = MEASURED['electron_g2']
        error = MEASURED['electron_g2_err']

        print(f"Measured value: {measured_a_e:.15f} ± {error:.2e}")
        print(f"QED (α/2π):     {g2_results['QED_leading']:.15f}")
        print()
        print("Bivector predictions:")
        for model, value in g2_results.items():
            if model.startswith('model'):
                deviation = abs(value - measured_a_e)
                sigma = deviation / error
                status = "[PASS]" if sigma < 5 else "[FAIL]"
                print(f"  {model:25s}: {value:.15f} ({sigma:8.1f}σ) {status}")

        print(f"\nΛ₁₂ = {g2_results['Lambda_12']:.6e}")

        # Store best model
        deviations = [(abs(g2_results[k] - measured_a_e), k) for k in g2_results if k.startswith('model')]
        best_dev, best_model = min(deviations)
        results['electron_g2'] = {'best_model': best_model, 'deviation_sigma': best_dev/error}
        print(f"Best model: {best_model} ({best_dev/error:.1f}σ)")
        print()

        # Test 2: Neutrino oscillations
        print("2. NEUTRINO OSCILLATION PARAMETERS")
        print("-" * 80)

        nu_results = self.compute_neutrino_oscillations(None, None)

        print(f"Δm²₂₁:")
        print(f"  Measured: {MEASURED['dm2_21']:.3e} eV²")
        print(f"  Predicted: {nu_results['dm2_21']:.3e} eV²")
        ratio_21 = nu_results['dm2_21'] / MEASURED['dm2_21']
        print(f"  Ratio: {ratio_21:.3f}")
        print()

        print(f"Δm²₃₂:")
        print(f"  Measured: {MEASURED['dm2_32']:.3e} eV²")
        print(f"  Predicted: {nu_results['dm2_32']:.3e} eV²")
        ratio_32 = nu_results['dm2_32'] / MEASURED['dm2_32']
        print(f"  Ratio: {ratio_32:.3f}")
        print()

        results['neutrino'] = {'ratio_21': ratio_21, 'ratio_32': ratio_32}
        print()

        # Test 3: CP violation
        print("3. CP VIOLATION (JARLSKOG INVARIANT)")
        print("-" * 80)

        cp_results = self.compute_cp_violation()

        print(f"Measured: {MEASURED['jarlskog']:.3e}")
        print(f"Predicted: {cp_results['jarlskog']:.3e}")
        ratio_j = cp_results['jarlskog'] / MEASURED['jarlskog']
        print(f"Ratio: {ratio_j:.3f}")
        print()

        results['cp_violation'] = {'ratio': ratio_j}

        # Overall assessment
        print("=" * 80)
        print("OVERALL ASSESSMENT")
        print("=" * 80)
        print()

        # Check if any model passes 5σ criterion
        electron_pass = results['electron_g2']['deviation_sigma'] < 5
        neutrino_pass = 0.5 < results['neutrino']['ratio_21'] < 2.0  # Within factor of 2
        cp_pass = 0.1 < results['cp_violation']['ratio'] < 10  # Within order of magnitude

        print(f"Electron g-2: {'[PASS]' if electron_pass else '[FAIL]'}")
        print(f"Neutrino oscillations: {'[PASS]' if neutrino_pass else '[FAIL]'}")
        print(f"CP violation: {'[PASS]' if cp_pass else '[FAIL]'}")
        print()

        if all([electron_pass, neutrino_pass, cp_pass]):
            print("[OK] Framework passes all validation tests!")
            print("Ready for predictive extension to unknown physics.")
        else:
            print("[NEEDS REFINEMENT] Framework requires adjustment.")
            print("Next steps:")
            if not electron_pass:
                print("  - Refine B₁-B₂ coupling (spin-helicity)")
            if not neutrino_pass:
                print("  - Adjust energy scale or flavor bivector structure")
            if not cp_pass:
                print("  - Improve triple bivector product calculation")
        print()

        return results


def main():
    """Main validation pipeline."""

    print("Starting three-bivector physics validation...")
    print()

    framework = ThreeBivectorPhysics()

    # Run all validation tests
    results = framework.validate_against_experiment()

    print()
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()
    print("STRENGTHS:")
    print("  [+] Grounded in proven physics (spin as bivector)")
    print("  [+] Makes specific numerical predictions")
    print("  [+] Falsifiable (clear pass/fail criteria)")
    print("  [+] Uses same Lambda diagnostic as BCH work (R² = 1.000)")
    print()
    print("NEXT STEPS:")
    print("  1. Refine helicity bivector model for g-2")
    print("  2. Fit energy scale for neutrino masses")
    print("  3. Implement proper wedge product for CP phase")
    print("  4. Extend to muon and tau g-2 predictions")
    print("  5. Predict NEW physics (e.g., 4th generation signatures)")
    print()
    print("CONNECTION TO BCH PATENT:")
    print("  - Same kinematic curvature diagnostic Λ = ||[B₁, B₂]||")
    print("  - Same geometric suppression exp(-Λ²)")
    print("  - Universal applicability: materials → fundamental forces")
    print()

    return framework, results


if __name__ == "__main__":
    framework, results = main()
