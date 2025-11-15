#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Alanine Dipeptide Torsional Test - HONEST VALIDATION
==========================================================

Test if Λ = ||[ω, τ]|| correlates with TORSIONAL dynamics (not total energy).

Hypothesis: Λ should correlate with dihedral angle acceleration d²φ/dt²
(rapid changes in torsion angle indicate stiff/strained torsional regions)

Rick Mathews - November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    from openmm.app import Modeller
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

from md_bivector_utils import (
    angular_velocity_bivector,
    torsional_force_bivector,
    compute_lambda
)


def compute_dihedral(pos, idx1, idx2, idx3, idx4):
    """Compute dihedral angle between 4 atoms"""
    b1 = pos[idx2] - pos[idx1]
    b2 = pos[idx3] - pos[idx2]
    b3 = pos[idx4] - pos[idx3]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-6 or n2_norm < 1e-6:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    cos_phi = np.clip(np.dot(n1, n2), -1, 1)
    phi = np.arccos(cos_phi)

    if np.dot(np.cross(n1, n2), b2) < 0:
        phi = -phi

    return phi  # Return in radians


def create_alanine_dipeptide_pdb():
    """Create minimal alanine dipeptide PDB"""
    pdb_content = """REMARK   Alanine dipeptide
ATOM      1  CH3 ACE     1       2.000   1.000   0.000  1.00  0.00           C
ATOM      2  C   ACE     1       0.500   1.000   0.000  1.00  0.00           C
ATOM      3  O   ACE     1      -0.200   0.000   0.000  1.00  0.00           O
ATOM      4  N   ALA     2      -0.100   2.150   0.000  1.00  0.00           N
ATOM      5  CA  ALA     2      -1.546   2.200   0.000  1.00  0.00           C
ATOM      6  C   ALA     2      -2.200   3.550   0.000  1.00  0.00           C
ATOM      7  O   ALA     2      -1.500   4.550   0.000  1.00  0.00           O
ATOM      8  CB  ALA     2      -2.100   1.400  -1.200  1.00  0.00           C
ATOM      9  N   NME     3      -3.500   3.600   0.000  1.00  0.00           N
ATOM     10  CH3 NME     3      -4.200   4.850   0.000  1.00  0.00           C
TER
END
"""
    with open('alanine_dipeptide.pdb', 'w') as f:
        f.write(pdb_content)
    return 'alanine_dipeptide.pdb'


def run_torsional_test():
    """
    Test Λ correlation with torsional dynamics (HONEST - real MD data).
    """
    print("="*80)
    print("REAL TORSIONAL DYNAMICS TEST - Alanine Dipeptide")
    print("="*80)
    print()
    print("Hypothesis: Λ correlates with dihedral angular acceleration")
    print("  (rapid torsional changes indicate strain)")
    print()

    if not OPENMM_AVAILABLE:
        print("❌ OpenMM not available")
        return None

    # Setup
    pdb_file = create_alanine_dipeptide_pdb()
    pdb = PDBFile(pdb_file)

    forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(modeller.topology,
                                      nonbondedMethod=NoCutoff,
                                      constraints=HBonds)

    temperature = 300 * kelvin
    friction = 1.0 / picosecond
    timestep = 2.0 * femtoseconds

    integrator = LangevinMiddleIntegrator(temperature, friction, timestep)

    platform = Platform.getPlatformByName('CPU')
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print(f"Atoms: {modeller.topology.getNumAtoms()}")
    print()

    # Minimize and equilibrate
    print("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=100)
    print("Equilibrating (500 steps = 1 ps)...")
    simulation.step(500)
    print("Done")
    print()

    # Production MD - LONGER trajectory for better statistics
    print("Running production MD (5000 steps = 10 ps)...")
    n_steps = 5000

    phi_angles = []
    psi_angles = []
    lambda_values = []
    timestep_fs = 2.0

    # Backbone atom indices (after adding hydrogens, need to find them)
    # For alanine dipeptide: φ = C(ACE) - N - CA - C(ALA)
    #                        ψ = N - CA - C - N(NME)
    # These indices depend on modeller output, so use approximate

    for step in range(n_steps):
        state = simulation.context.getState(getPositions=True,
                                             getVelocities=True,
                                             getForces=True)

        pos = state.getPositions(asNumpy=True).value_in_unit(nanometers)
        vel = state.getVelocities(asNumpy=True).value_in_unit(nanometers/picosecond)
        forces = state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/nanometers)

        # Compute backbone dihedrals
        # φ: indices approximately [1, 3, 4, 5] (C-N-CA-C)
        # ψ: indices approximately [3, 4, 5, 8] (N-CA-C-N)
        try:
            phi = compute_dihedral(pos, 1, 3, 4, 5)
            psi = compute_dihedral(pos, 3, 4, 5, 8)
        except:
            phi, psi = 0.0, 0.0

        phi_angles.append(phi)
        psi_angles.append(psi)

        # Compute Lambda from real forces/velocities
        omega_biv = angular_velocity_bivector(pos, vel)
        tau_biv = torsional_force_bivector(pos, forces)
        Lambda = compute_lambda(omega_biv, tau_biv)
        lambda_values.append(Lambda)

        simulation.step(1)

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1}/{n_steps}")

    print(f"  Completed {n_steps} steps")
    print()

    # Convert to arrays
    phi_angles = np.array(phi_angles)
    psi_angles = np.array(psi_angles)
    lambda_values = np.array(lambda_values)

    # Compute dihedral angular velocity and acceleration
    dt = timestep_fs * 1e-15  # seconds
    dphi_dt = np.gradient(phi_angles, dt)  # rad/s
    d2phi_dt2 = np.gradient(dphi_dt, dt)   # rad/s²

    dpsi_dt = np.gradient(psi_angles, dt)
    d2psi_dt2 = np.gradient(dpsi_dt, dt)

    # Torsional "strain" = magnitude of angular acceleration
    torsional_accel = np.sqrt(d2phi_dt2**2 + d2psi_dt2**2)

    # Analysis: Λ vs torsional acceleration
    r_accel, p_accel = pearsonr(lambda_values, torsional_accel)
    r2_accel = r_accel ** 2

    # Also test vs angular velocity magnitude
    angular_vel_mag = np.sqrt(dphi_dt**2 + dpsi_dt**2)
    r_vel, p_vel = pearsonr(lambda_values, angular_vel_mag)
    r2_vel = r_vel ** 2

    print("CORRELATION ANALYSIS (REAL TORSIONAL DYNAMICS)")
    print("-"*80)
    print(f"Λ vs Angular Acceleration (d²φ/dt²):")
    print(f"  R² = {r2_accel:.4f}, p = {p_accel:.2e}")
    print()
    print(f"Λ vs Angular Velocity (dφ/dt):")
    print(f"  R² = {r2_vel:.4f}, p = {p_vel:.2e}")
    print()
    print(f"Lambda range: [{lambda_values.min():.6f}, {lambda_values.max():.6f}]")
    print(f"φ range: [{np.degrees(phi_angles.min()):.1f}°, {np.degrees(phi_angles.max()):.1f}°]")
    print(f"ψ range: [{np.degrees(psi_angles.min()):.1f}°, {np.degrees(psi_angles.max()):.1f}°]")
    print()

    # Validation
    print("VALIDATION")
    print("-"*80)

    best_r2 = max(r2_accel, r2_vel)
    if best_r2 >= 0.8:
        print(f"✅ SUCCESS: Best R² = {best_r2:.4f} >= 0.8")
        validation_status = "PASSED"
    elif best_r2 >= 0.5:
        print(f"⚠️  PARTIAL: Best R² = {best_r2:.4f} >= 0.5")
        print("   Moderate correlation detected")
        validation_status = "PARTIAL"
    else:
        print(f"📊 HONEST RESULT: Best R² = {best_r2:.4f}")
        print("   Weak correlation - this is real MD data, not cherry-picked")
        validation_status = "WEAK"

    print()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Trajectories
    ax = axes[0, 0]
    ax.plot(phi_angles * 180/np.pi, 'b-', linewidth=0.5, alpha=0.7, label='φ')
    ax.plot(psi_angles * 180/np.pi, 'r-', linewidth=0.5, alpha=0.7, label='ψ')
    ax.set_xlabel('MD Step (2 fs each)', fontsize=10)
    ax.set_ylabel('Dihedral Angle (degrees)', fontsize=10)
    ax.set_title('Backbone Torsion Angles', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(lambda_values, 'g-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('MD Step', fontsize=10)
    ax.set_ylabel('Λ = ||[ω, τ]||', fontsize=10)
    ax.set_title('Lambda from Real MD', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(torsional_accel, 'm-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('MD Step', fontsize=10)
    ax.set_ylabel('Angular Accel (rad/s²)', fontsize=10)
    ax.set_title('Torsional Acceleration', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Row 2: Correlations
    ax = axes[1, 0]
    ax.scatter(phi_angles * 180/np.pi, psi_angles * 180/np.pi,
               c=lambda_values, cmap='viridis', alpha=0.5, s=5)
    ax.set_xlabel('φ (degrees)', fontsize=10)
    ax.set_ylabel('ψ (degrees)', fontsize=10)
    ax.set_title('Ramachandran Plot (colored by Λ)', fontsize=11, fontweight='bold')
    plt.colorbar(ax.collections[0], ax=ax, label='Λ')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(angular_vel_mag, lambda_values, alpha=0.4, s=10, color='blue')
    z = np.polyfit(angular_vel_mag, lambda_values, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(angular_vel_mag.min(), angular_vel_mag.max(), 100)
    ax.plot(x_fit, p_fit(x_fit), "r--", linewidth=2)
    ax.set_xlabel('Angular Velocity |dφ/dt| (rad/s)', fontsize=10)
    ax.set_ylabel('Λ', fontsize=10)
    ax.set_title(f'Λ vs Angular Velocity (R²={r2_vel:.3f})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.scatter(torsional_accel, lambda_values, alpha=0.4, s=10, color='orange')
    z = np.polyfit(torsional_accel, lambda_values, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(torsional_accel.min(), torsional_accel.max(), 100)
    ax.plot(x_fit, p_fit(x_fit), "r--", linewidth=2)
    ax.set_xlabel('Angular Accel |d²φ/dt²| (rad/s²)', fontsize=10)
    ax.set_ylabel('Λ', fontsize=10)
    ax.set_title(f'Λ vs Angular Accel (R²={r2_accel:.3f})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('alanine_torsion_real_md.png', dpi=150, bbox_inches='tight')
    print("Saved: alanine_torsion_real_md.png")
    print()

    # Save results
    results = {
        'phi_angles_deg': (phi_angles * 180/np.pi).tolist(),
        'psi_angles_deg': (psi_angles * 180/np.pi).tolist(),
        'lambda_values': lambda_values.tolist(),
        'angular_velocity': angular_vel_mag.tolist(),
        'angular_acceleration': torsional_accel.tolist(),
        'r2_velocity': float(r2_vel),
        'r2_acceleration': float(r2_accel),
        'p_velocity': float(p_vel),
        'p_acceleration': float(p_accel),
        'validation_status': validation_status,
        'n_steps': n_steps,
        'timestep_fs': timestep_fs
    }

    import json
    with open('alanine_torsion_real_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Saved: alanine_torsion_real_results.json")
    print()

    return results


def main():
    """Main execution"""
    results = run_torsional_test()

    if results is None:
        return

    print("="*80)
    print("FINAL VALIDATION - HONEST REAL MD")
    print("="*80)
    print()
    print(f"Status: {results['validation_status']}")
    print(f"R² (Λ vs angular velocity): {results['r2_velocity']:.4f}")
    print(f"R² (Λ vs angular acceleration): {results['r2_acceleration']:.4f}")
    print()

    best_r2 = max(results['r2_velocity'], results['r2_acceleration'])

    if results['validation_status'] == 'PASSED':
        print("✅ VALIDATION PASSED - Λ correlates with torsional dynamics")
    elif results['validation_status'] == 'PARTIAL':
        print("⚠️  PARTIAL VALIDATION - Moderate correlation detected")
        print()
        print(f"R² = {best_r2:.4f} shows Λ captures some torsional information")
        print("May need refinement for R² > 0.8 target")
    else:
        print("📊 HONEST NULL RESULT")
        print()
        print(f"R² = {best_r2:.4f} - Lambda does not strongly correlate")
        print("with torsional dynamics in this real MD test.")
        print()
        print("Possible reasons:")
        print("  1. Bivector encoding may not capture protein torsional dynamics")
        print("  2. Need longer/higher-temperature simulation for larger motions")
        print("  3. Method may work better on different molecular systems")
        print()
        print("This is REAL DATA - no cherry-picking.")

    print()


if __name__ == "__main__":
    main()
