#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Alanine Dipeptide MD Test with OpenMM
===========================================

HONEST validation using actual MD simulation - no cherry-picking.

Use alanine dipeptide (well-studied MD test system).

Rick Mathews - November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

from md_bivector_utils import (
    angular_velocity_bivector,
    torsional_force_bivector,
    compute_lambda
)


def create_alanine_dipeptide_pdb():
    """
    Create minimal alanine dipeptide PDB file.

    Ace-Ala-Nme (blocked alanine)
    """
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


def compute_backbone_dihedral(positions, residue_idx=1):
    """
    Compute φ (phi) and ψ (psi) backbone dihedrals.

    For alanine dipeptide (minimal):
    φ = C(i-1) - N(i) - CA(i) - C(i)
    ψ = N(i) - CA(i) - C(i) - N(i+1)

    Simplified: Just compute one representative dihedral
    """
    pos = positions
    n_atoms = len(pos)

    # Simple approach: compute dihedral between first 4 backbone atoms
    if n_atoms >= 4:
        idx = [1, 3, 4, 5]  # C, N, CA, C (approximate)
        return compute_dihedral(pos, *idx)

    return 0.0


def compute_dihedral(pos, idx1, idx2, idx3, idx4):
    """Compute dihedral angle"""
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

    return np.degrees(phi)


def run_alanine_md_test():
    """
    Run real MD on alanine dipeptide and test Λ.

    This is HONEST - uses real OpenMM forces/velocities.
    """
    print("="*80)
    print("REAL ALANINE DIPEPTIDE MD TEST (OpenMM)")
    print("="*80)
    print()

    if not OPENMM_AVAILABLE:
        print("❌ OpenMM not available")
        return None

    # Create PDB file
    print("Creating alanine dipeptide structure...")
    pdb_file = create_alanine_dipeptide_pdb()

    # Load structure
    pdb = PDBFile(pdb_file)

    # Use implicit solvent force field
    forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')

    # Create system
    system = forcefield.createSystem(pdb.topology,
                                      nonbondedMethod=NoCutoff,
                                      constraints=HBonds)

    # Integrator
    temperature = 300 * kelvin
    friction = 1.0 / picosecond
    timestep = 2.0 * femtoseconds

    integrator = LangevinMiddleIntegrator(temperature, friction, timestep)

    # Simulation
    platform = Platform.getPlatformByName('CPU')
    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    print(f"  Atoms: {pdb.topology.getNumAtoms()}")
    print()

    # Minimize
    print("Minimizing energy...")
    try:
        simulation.minimizeEnergy(maxIterations=100)
        print("  Done")
    except Exception as e:
        print(f"  Warning: {e}")
        print("  Continuing anyway...")
    print()

    # Equilibrate
    print("Equilibrating (200 steps)...")
    simulation.step(200)
    print("  Done")
    print()

    # Production MD
    print("Running production MD (2000 steps = 4 ps)...")
    n_steps = 2000

    dihedral_angles = []
    lambda_values = []
    energy_values = []

    for step in range(n_steps):
        state = simulation.context.getState(getPositions=True,
                                             getVelocities=True,
                                             getForces=True,
                                             getEnergy=True)

        pos = state.getPositions(asNumpy=True).value_in_unit(nanometers)
        vel = state.getVelocities(asNumpy=True).value_in_unit(nanometers/picosecond)
        forces = state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/nanometers)
        energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

        # Backbone dihedral
        phi = compute_backbone_dihedral(pos)
        dihedral_angles.append(phi)
        energy_values.append(energy)

        # Compute Lambda from REAL forces/velocities
        omega_biv = angular_velocity_bivector(pos, vel)
        tau_biv = torsional_force_bivector(pos, forces)
        Lambda = compute_lambda(omega_biv, tau_biv)
        lambda_values.append(Lambda)

        # MD step
        simulation.step(1)

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1}/{n_steps}")

    print(f"  Completed {n_steps} steps")
    print()

    # Convert to arrays
    dihedral_angles = np.array(dihedral_angles)
    lambda_values = np.array(lambda_values)
    energy_values = np.array(energy_values)

    # Analysis: Test if Lambda correlates with energy fluctuations
    # (high energy regions = stiff/strained regions)
    energy_std = np.std(energy_values)
    energy_mean = np.mean(energy_values)
    energy_strain = np.abs(energy_values - energy_mean)

    r_energy, p_energy = pearsonr(lambda_values, energy_strain)
    r2_energy = r_energy ** 2

    print("CORRELATION ANALYSIS (REAL MD)")
    print("-"*80)
    print(f"Λ vs Energy Fluctuations:")
    print(f"  R² = {r2_energy:.4f}, p = {p_energy:.2e}")
    print()
    print(f"Lambda range: [{lambda_values.min():.4f}, {lambda_values.max():.4f}]")
    print(f"Energy range: [{energy_values.min():.1f}, {energy_values.max():.1f}] kJ/mol")
    print()

    # Validation
    print("VALIDATION")
    print("-"*80)

    if r2_energy >= 0.5:
        print(f"✅ Moderate correlation: R² = {r2_energy:.4f}")
        validation_status = "PARTIAL"
    else:
        print(f"📊 Weak correlation: R² = {r2_energy:.4f}")
        print("   (This is honest real MD data - not cherry-picked)")
        validation_status = "WEAK"

    print()

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Dihedral trajectory
    ax = axes[0, 0]
    ax.plot(dihedral_angles, 'b-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('MD Step (2 fs each)', fontsize=11)
    ax.set_ylabel('Backbone Dihedral φ (degrees)', fontsize=11)
    ax.set_title('Alanine Dipeptide Backbone Rotation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. Lambda trajectory
    ax = axes[0, 1]
    ax.plot(lambda_values, 'g-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('MD Step', fontsize=11)
    ax.set_ylabel('Λ = ||[ω, τ]|| (Real MD)', fontsize=11)
    ax.set_title('Lambda from Real OpenMM Forces/Velocities', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Energy trajectory
    ax = axes[1, 0]
    ax.plot(energy_values, 'r-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('MD Step', fontsize=11)
    ax.set_ylabel('Potential Energy (kJ/mol)', fontsize=11)
    ax.set_title('Energy Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Λ vs Energy correlation
    ax = axes[1, 1]
    ax.scatter(energy_strain, lambda_values, alpha=0.3, s=5, color='orange')

    z = np.polyfit(energy_strain, lambda_values, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(energy_strain.min(), energy_strain.max(), 100)
    ax.plot(x_fit, p_fit(x_fit), "r--", alpha=0.8, linewidth=2)

    ax.set_xlabel('Energy Fluctuation |E - <E>| (kJ/mol)', fontsize=11)
    ax.set_ylabel('Λ (Real MD)', fontsize=11)
    ax.set_title(f'Λ vs Energy Strain (R² = {r2_energy:.3f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('alanine_openmm_real_md.png', dpi=150, bbox_inches='tight')
    print("Saved: alanine_openmm_real_md.png")
    print()

    # Save results
    results = {
        'dihedral_angles': dihedral_angles.tolist(),
        'lambda_values': lambda_values.tolist(),
        'energy_values': energy_values.tolist(),
        'r2_energy': float(r2_energy),
        'p_energy': float(p_energy),
        'validation_status': validation_status,
        'n_steps': n_steps,
        'timestep_fs': 2.0,
        'temperature_K': 300
    }

    import json
    with open('alanine_openmm_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Saved: alanine_openmm_results.json")
    print()

    return results


def main():
    """Main execution"""
    results = run_alanine_md_test()

    if results is None:
        return

    print("="*80)
    print("SUMMARY - REAL MD VALIDATION")
    print("="*80)
    print()
    print(f"System: Alanine dipeptide (Ace-Ala-Nme)")
    print(f"MD Steps: {results['n_steps']} × {results['timestep_fs']} fs = {results['n_steps']*results['timestep_fs']/1000:.1f} ps")
    print(f"Temperature: {results['temperature_K']} K")
    print()
    print(f"Λ correlation with energy fluctuations: R² = {results['r2_energy']:.4f}")
    print()

    if results['validation_status'] == 'PARTIAL':
        print("✅ Moderate correlation detected")
        print()
        print("Lambda (from real MD forces/velocities) shows correlation")
        print("with molecular strain. Further optimization needed for R² > 0.8.")
    else:
        print("📊 This is the HONEST result from real MD")
        print()
        print("No cherry-picking - Lambda computed from actual OpenMM trajectory.")
        print("Correlation is modest, which tells us:")
        print("  - Bivector encoding may need refinement for proteins")
        print("  - OR: Energy fluctuations ≠ torsional strain (they measure different things)")
        print("  - Next: Try explicit torsional strain metrics")

    print()


if __name__ == "__main__":
    main()
