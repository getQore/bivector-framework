#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orthogonality Scaling Test

Rick's hypothesis: Λ(θ) = Λ_max * |sin(θ)|
where θ = angle between bivector "planes"

If this holds → geometric origin confirmed!

Rick Mathews
November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 80)
print("ORTHOGONALITY SCALING TEST")
print("Testing Hypothesis: Lambda(theta) = Lambda_max * |sin(theta)|")
print("=" * 80)
print()


class BivectorRotation:
    """Test angular dependence of commutator."""

    def __init__(self):
        self.beta = 0.1

    def create_bivector_matrix(self, spatial, boost):
        """Create 4x4 antisymmetric matrix from bivector components."""
        M = np.zeros((4, 4))

        # Spatial part
        Sx, Sy, Sz = spatial
        M[1, 2] = Sz
        M[2, 1] = -Sz
        M[0, 2] = Sy
        M[2, 0] = -Sy
        M[0, 1] = Sx
        M[1, 0] = -Sx

        # Boost part
        Bx, By, Bz = boost
        M[0, 1] += Bx
        M[1, 0] -= Bx
        M[0, 2] += By
        M[2, 0] -= By
        M[0, 3] = Bz
        M[3, 0] = -Bz

        return M

    def rotate_spatial_bivector(self, spatial, angle, axis='z'):
        """
        Rotate spatial bivector by angle around axis.

        For spin bivector S_z e_23, rotation around y-axis by θ gives:
        S'_z = S_z * cos(θ) + S_x * sin(θ)
        """
        Sx, Sy, Sz = spatial

        if axis == 'y':
            # Rotation around y-axis
            Sx_new = Sx * np.cos(angle) - Sz * np.sin(angle)
            Sy_new = Sy
            Sz_new = Sx * np.sin(angle) + Sz * np.cos(angle)
        elif axis == 'x':
            # Rotation around x-axis
            Sx_new = Sx
            Sy_new = Sy * np.cos(angle) - Sz * np.sin(angle)
            Sz_new = Sy * np.sin(angle) + Sz * np.cos(angle)
        elif axis == 'z':
            # Rotation around z-axis
            Sx_new = Sx * np.cos(angle) - Sy * np.sin(angle)
            Sy_new = Sx * np.sin(angle) + Sy * np.cos(angle)
            Sz_new = Sz
        else:
            raise ValueError(f"Unknown axis: {axis}")

        return np.array([Sx_new, Sy_new, Sz_new])

    def compute_commutator(self, M1, M2):
        """Compute ||[M1, M2]||_F."""
        comm = M1 @ M2 - M2 @ M1
        return norm(comm, 'fro')

    def test_spin_boost_angle(self):
        """Test [spin, boost] as function of rotation angle."""

        print("Test 1: Spin-Boost Angular Dependence")
        print("-" * 80)
        print()

        # Fixed boost along x
        boost_x = np.array([self.beta, 0, 0])

        # Spin initially along z, rotate around y-axis
        angles = np.linspace(0, np.pi, 100)
        Lambdas = []

        for theta in angles:
            # Rotate spin from z toward x
            spin_rotated = self.rotate_spatial_bivector([0, 0, 0.5], theta, axis='y')

            M_spin = self.create_bivector_matrix(spin_rotated, [0, 0, 0])
            M_boost = self.create_bivector_matrix([0, 0, 0], boost_x)

            Lambda = self.compute_commutator(M_spin, M_boost)
            Lambdas.append(Lambda)

        Lambdas = np.array(Lambdas)

        # Fit to sin(theta)
        Lambda_max = np.max(Lambdas)
        Lambda_predicted = Lambda_max * np.abs(np.sin(angles))

        # Compute R²
        SS_res = np.sum((Lambdas - Lambda_predicted)**2)
        SS_tot = np.sum((Lambdas - np.mean(Lambdas))**2)
        R_squared = 1 - SS_res / SS_tot

        print(f"Lambda_max: {Lambda_max:.6f}")
        print(f"R² for sin(theta) fit: {R_squared:.9f}")
        print()

        if R_squared > 0.999:
            print("[SUCCESS] Lambda(theta) = Lambda_max * |sin(theta)| confirmed!")
        else:
            print(f"[PARTIAL] R² = {R_squared:.6f} (not perfect sin dependence)")

        print()

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(np.degrees(angles), Lambdas, 'b-', linewidth=2, label='Measured Lambda')
        ax1.plot(np.degrees(angles), Lambda_predicted, 'r--', linewidth=2, label=f'Lambda_max * |sin(theta)| (R²={R_squared:.6f})')
        ax1.set_xlabel('Angle (degrees)', fontsize=12)
        ax1.set_ylabel('Lambda', fontsize=12)
        ax1.set_title('Spin-Boost Commutator vs Rotation Angle', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Residuals
        residuals = Lambdas - Lambda_predicted
        ax2.plot(np.degrees(angles), residuals, 'g-', linewidth=2)
        ax2.axhline(0, color='k', linestyle='--', linewidth=1)
        ax2.set_xlabel('Angle (degrees)', fontsize=12)
        ax2.set_ylabel('Residual', fontsize=12)
        ax2.set_title('Fit Residuals', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('C:\\v2_files\\hierarchy_test\\orthogonality_scaling.png', dpi=150, bbox_inches='tight')
        print(f"Saved: orthogonality_scaling.png")
        plt.close()

        return R_squared

    def test_spin_spin_angle(self):
        """Test [spin_z, spin_rotated] angular dependence."""

        print("Test 2: Spin-Spin Angular Dependence")
        print("-" * 80)
        print()

        # Fixed spin along z
        spin_z = np.array([0, 0, 0.5])

        # Second spin rotated from x toward y
        angles = np.linspace(0, np.pi, 100)
        Lambdas = []

        for theta in angles:
            spin_rotated = self.rotate_spatial_bivector([0.5, 0, 0], theta, axis='z')

            M1 = self.create_bivector_matrix(spin_z, [0, 0, 0])
            M2 = self.create_bivector_matrix(spin_rotated, [0, 0, 0])

            Lambda = self.compute_commutator(M1, M2)
            Lambdas.append(Lambda)

        Lambdas = np.array(Lambdas)

        Lambda_max = np.max(Lambdas)
        Lambda_predicted = Lambda_max * np.abs(np.sin(angles))

        SS_res = np.sum((Lambdas - Lambda_predicted)**2)
        SS_tot = np.sum((Lambdas - np.mean(Lambdas))**2)
        R_squared = 1 - SS_res / SS_tot

        print(f"Lambda_max: {Lambda_max:.6f}")
        print(f"R² for sin(theta) fit: {R_squared:.9f}")
        print()

        if R_squared > 0.999:
            print("[SUCCESS] Sin(theta) dependence confirmed!")
        else:
            print(f"[NOTE] R² = {R_squared:.6f}")

        print()

        return R_squared


def main():
    """Run orthogonality tests."""

    tester = BivectorRotation()

    # Test 1: Spin-Boost
    R2_spin_boost = tester.test_spin_boost_angle()

    # Test 2: Spin-Spin
    R2_spin_spin = tester.test_spin_spin_angle()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print(f"Spin-Boost R²: {R2_spin_boost:.9f}")
    print(f"Spin-Spin R²:  {R2_spin_spin:.9f}")
    print()

    if min(R2_spin_boost, R2_spin_spin) > 0.999:
        print("[CONFIRMED] Orthogonality hypothesis Lambda = Lambda_max * |sin(theta)|")
        print()
        print("PHYSICAL MEANING:")
        print("  - Parallel bivectors (theta=0):     Lambda = 0 (conserved)")
        print("  - Orthogonal bivectors (theta=90):  Lambda = max (maximum coupling)")
        print("  - Intermediate angles:               Lambda ~ sin(theta)")
        print()
        print("This is the GEOMETRIC ORIGIN of:")
        print("  - Selection rules (allowed vs forbidden transitions)")
        print("  - Conservation laws (commutation = conservation)")
        print("  - Perturbation theory (coupling strength from geometry)")
        print()
    else:
        print("[PARTIAL] Sin(theta) holds approximately but not exactly")
        print("May need to account for:")
        print("  - Higher-order geometric terms")
        print("  - Bivector blade structure (simple vs composite)")
        print("  - Metric signature effects")

    print()

    return R2_spin_boost, R2_spin_spin


if __name__ == "__main__":
    R2_sb, R2_ss = main()
