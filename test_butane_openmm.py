#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Butane MD Test with OpenMM
================================

HONEST validation using actual molecular dynamics simulation.

No cherry-picking: Use real OpenMM forces and velocities from actual trajectory.

Rick Mathews - November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    import openmm.app as app
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    print("OpenMM not available")

from md_bivector_utils import (
    angular_velocity_bivector,
    torsional_force_bivector,
    compute_lambda,
    compute_torsional_strain_butane,
    BivectorCl31
)


def create_butane_system():
    """
    Create butane molecule with OpenMM.

    Returns:
        topology, system, positions
    """
    if not OPENMM_AVAILABLE:
        raise ImportError("OpenMM required for real MD")

    # Create topology
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue('BUT', chain)

    # Add atoms (4 carbons, 10 hydrogens)
    element_C = Element.getBySymbol('C')
    element_H = Element.getBySymbol('H')

    atoms = []
    # Carbons
    for i in range(4):
        atoms.append(topology.addAtom(f'C{i+1}', element_C, residue))
    # Hydrogens (simplified - just add them)
    for i in range(10):
        atoms.append(topology.addAtom(f'H{i+1}', element_H, residue))

    # Add bonds (C-C backbone)
    for i in range(3):
        topology.addBond(atoms[i], atoms[i+1])

    # Add C-H bonds (simplified topology)
    topology.addBond(atoms[0], atoms[4])  # C1-H1
    topology.addBond(atoms[0], atoms[5])  # C1-H2
    topology.addBond(atoms[0], atoms[6])  # C1-H3
    topology.addBond(atoms[1], atoms[7])  # C2-H4
    topology.addBond(atoms[1], atoms[8])  # C2-H5
    topology.addBond(atoms[2], atoms[9])  # C3-H6
    topology.addBond(atoms[2], atoms[10]) # C3-H7
    topology.addBond(atoms[3], atoms[11]) # C4-H8
    topology.addBond(atoms[3], atoms[12]) # C4-H9
    topology.addBond(atoms[3], atoms[13]) # C4-H10

    # Initial positions (simple linear arrangement)
    positions = []
    bond_length = 0.154  # nm (C-C bond)

    # Carbons along x-axis
    for i in range(4):
        positions.append(Vec3(i * bond_length, 0, 0))

    # Hydrogens (rough tetrahedral positions)
    h_bond = 0.109  # nm (C-H bond)
    for i, c_idx in enumerate([0, 0, 0, 1, 1, 2, 2, 3, 3, 3]):
        offset_x = 0.05 * np.cos(2*np.pi*i/10)
        offset_y = h_bond * np.sin(2*np.pi*i/10)
        offset_z = h_bond * np.cos(2*np.pi*i/10)
        pos = positions[c_idx]
        positions.append(Vec3(pos[0] + offset_x, offset_y, offset_z))

    # Create system with force field
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # Use generic force field (simpler)
    system = System()

    # Add particles (atoms)
    for atom in topology.atoms():
        if atom.element.symbol == 'C':
            system.addParticle(12.0)  # Carbon mass (amu)
        else:
            system.addParticle(1.0)   # Hydrogen mass (amu)

    # Add harmonic bond forces
    bond_force = HarmonicBondForce()
    for bond in topology.bonds():
        idx1 = bond[0].index
        idx2 = bond[1].index

        # Bond parameters (simple)
        if bond[0].element.symbol == 'C' and bond[1].element.symbol == 'C':
            length = 0.154 * nanometers  # C-C
            k = 250000.0 * kilojoules_per_mole/nanometers**2
        else:
            length = 0.109 * nanometers  # C-H
            k = 340000.0 * kilojoules_per_mole/nanometers**2

        bond_force.addBond(idx1, idx2, length, k)

    system.addForce(bond_force)

    # Add torsional force (this is the key for butane rotation)
    torsion_force = PeriodicTorsionForce()

    # C1-C2-C3-C4 dihedral (OPLS parameters)
    V1 = 3.4 * kilojoules_per_mole
    V2 = -0.8 * kilojoules_per_mole
    V3 = 6.8 * kilojoules_per_mole

    # Add V1 term
    torsion_force.addTorsion(0, 1, 2, 3, 1, 0.0*radians, V1)
    # Add V2 term
    torsion_force.addTorsion(0, 1, 2, 3, 2, np.pi*radians, -V2)
    # Add V3 term
    torsion_force.addTorsion(0, 1, 2, 3, 3, 0.0*radians, V3)

    system.addForce(torsion_force)

    return topology, system, positions


def compute_dihedral_angle(pos, idx1, idx2, idx3, idx4):
    """Compute dihedral angle between 4 atoms"""
    import numpy as np

    # Vectors
    b1 = pos[idx2] - pos[idx1]
    b2 = pos[idx3] - pos[idx2]
    b3 = pos[idx4] - pos[idx3]

    # Normal vectors to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # Dihedral angle
    cos_phi = np.dot(n1, n2)
    cos_phi = np.clip(cos_phi, -1, 1)
    phi = np.arccos(cos_phi)

    # Sign
    if np.dot(np.cross(n1, n2), b2) < 0:
        phi = -phi

    return np.degrees(phi)


def run_real_md_test():
    """
    Run real MD simulation of butane and test Λ correlation.

    NO CHERRY-PICKING: Use actual OpenMM trajectory.
    """
    print("="*80)
    print("REAL BUTANE MD TEST (OpenMM)")
    print("="*80)
    print()

    if not OPENMM_AVAILABLE:
        print("❌ OpenMM not available. Install with: conda install -c conda-forge openmm")
        return None

    print("Setting up butane system...")
    topology, system, positions = create_butane_system()
    print(f"  Atoms: {topology.getNumAtoms()}")
    print(f"  Bonds: {topology.getNumBonds()}")
    print()

    # Create integrator
    temperature = 300 * kelvin
    friction = 1.0 / picosecond
    timestep = 0.5 * femtoseconds

    integrator = LangevinMiddleIntegrator(temperature, friction, timestep)

    # Create simulation
    platform = Platform.getPlatformByName('CPU')
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    # Minimize energy
    print("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=100)
    print("  Done")
    print()

    # Equilibrate
    print("Equilibrating (100 steps)...")
    simulation.step(100)
    print("  Done")
    print()

    # Production MD: Sample dihedral angles
    print("Running production MD (1000 steps)...")
    print("  Extracting forces and velocities at each step")
    print()

    n_steps = 1000
    dihedral_angles = []
    lambda_values = []
    strain_values = []

    for step in range(n_steps):
        # Get current state
        state = simulation.context.getState(getPositions=True, getVelocities=True, getForces=True)

        # Extract positions, velocities, forces
        pos = state.getPositions(asNumpy=True).value_in_unit(nanometers)
        vel = state.getVelocities(asNumpy=True).value_in_unit(nanometers/picosecond)
        forces = state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/nanometers)

        # Compute dihedral angle (C1-C2-C3-C4)
        phi = compute_dihedral_angle(pos, 0, 1, 2, 3)
        dihedral_angles.append(phi)

        # Compute analytical strain for this angle
        strain = compute_torsional_strain_butane(phi)
        strain_values.append(strain)

        # Compute Lambda from REAL forces and velocities
        omega_biv = angular_velocity_bivector(pos, vel)
        tau_biv = torsional_force_bivector(pos, forces)
        Lambda = compute_lambda(omega_biv, tau_biv)
        lambda_values.append(Lambda)

        # Take MD step
        simulation.step(1)

    print(f"  Collected {len(lambda_values)} frames")
    print()

    # Analysis
    lambda_values = np.array(lambda_values)
    strain_values = np.array(strain_values)
    dihedral_angles = np.array(dihedral_angles)

    # Correlation
    r_strain, p_strain = pearsonr(lambda_values, strain_values)
    r2_strain = r_strain ** 2

    print("CORRELATION ANALYSIS (REAL MD DATA)")
    print("-"*80)
    print(f"Λ vs Torsional Strain |dV/dφ|:")
    print(f"  R² = {r2_strain:.4f}, p = {p_strain:.2e}")
    print()

    # Validation
    print("VALIDATION")
    print("-"*80)

    if r2_strain >= 0.8:
        print(f"✅ SUCCESS: R²(Λ vs strain) = {r2_strain:.4f} >= 0.8")
        validation_status = "PASSED"
    elif r2_strain >= 0.6:
        print(f"⚠️  PARTIAL: R²(Λ vs strain) = {r2_strain:.4f} >= 0.6")
        validation_status = "PARTIAL"
    else:
        print(f"❌ FAILED: R²(Λ vs strain) = {r2_strain:.4f} < 0.6")
        validation_status = "FAILED"

    print()

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Dihedral angle trajectory
    ax = axes[0, 0]
    ax.plot(dihedral_angles, 'b-', linewidth=1, alpha=0.7)
    ax.set_xlabel('MD Step', fontsize=11)
    ax.set_ylabel('Dihedral Angle φ (degrees)', fontsize=11)
    ax.set_title('Butane Dihedral Trajectory (Real MD)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. Lambda trajectory
    ax = axes[0, 1]
    ax.plot(lambda_values, 'g-', linewidth=1, alpha=0.7)
    ax.set_xlabel('MD Step', fontsize=11)
    ax.set_ylabel('Λ = ||[ω, τ]||', fontsize=11)
    ax.set_title('Lambda from Real Forces/Velocities', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Λ vs dihedral
    ax = axes[1, 0]
    scatter = ax.scatter(dihedral_angles, lambda_values, c=np.arange(len(lambda_values)),
                         cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel('Dihedral Angle φ (degrees)', fontsize=11)
    ax.set_ylabel('Λ', fontsize=11)
    ax.set_title('Λ vs Dihedral (colored by time)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='MD Step')
    ax.grid(True, alpha=0.3)

    # 4. Λ vs Strain correlation
    ax = axes[1, 1]
    ax.scatter(strain_values, lambda_values, alpha=0.5, s=20, color='orange')

    # Linear fit
    z = np.polyfit(strain_values, lambda_values, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(strain_values.min(), strain_values.max(), 100)
    ax.plot(x_fit, p(x_fit), "r--", alpha=0.8, linewidth=2, label='Linear fit')

    ax.set_xlabel('Torsional Strain |dV/dφ| (kJ/mol/rad)', fontsize=11)
    ax.set_ylabel('Λ (Real MD)', fontsize=11)
    ax.set_title(f'Λ vs Strain (R² = {r2_strain:.3f})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('butane_openmm_real_md.png', dpi=150, bbox_inches='tight')
    print("Saved: butane_openmm_real_md.png")
    print()

    # Save results
    results = {
        'dihedral_angles': dihedral_angles.tolist(),
        'lambda_values': lambda_values.tolist(),
        'strain_values': strain_values.tolist(),
        'r2_strain': float(r2_strain),
        'p_strain': float(p_strain),
        'validation_status': validation_status,
        'n_steps': n_steps
    }

    import json
    with open('butane_openmm_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Saved: butane_openmm_results.json")
    print()

    return results


def main():
    """Main execution"""
    results = run_real_md_test()

    if results is None:
        return

    print("="*80)
    print("SUMMARY - REAL MD DATA")
    print("="*80)
    print()
    print(f"Status: {results['validation_status']}")
    print(f"R² (Λ vs strain): {results['r2_strain']:.4f}")
    print(f"MD steps: {results['n_steps']}")
    print()

    if results['validation_status'] == 'PASSED':
        print("✅ HONEST VALIDATION PASSED")
        print()
        print("Λ = ||[ω, τ]|| computed from REAL OpenMM forces and velocities")
        print("correlates strongly with torsional strain (R² > 0.8)")
    else:
        print("📊 Honest result from real MD data")
        print()
        print("This is the true correlation - not cherry-picked.")

    print()


if __name__ == "__main__":
    main()
