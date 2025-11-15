#!/usr/bin/env python3
"""
NVE Energy Conservation Test for Adaptive Integrator
====================================================

Test adaptive timestep integrator in microcanonical (NVE) ensemble
to isolate energy drift from thermostat effects.

Goal: Verify energy drift < 1% over 100 ps

Rick Mathews - November 2024
"""

import numpy as np
import matplotlib.pyplot as plt

from openmm import *
from openmm.app import *
from openmm.unit import *

from md_bivector_utils import compute_phi_dot, compute_Q_phi


def compute_dihedral(pos, idx1, idx2, idx3, idx4):
    """Compute dihedral angle (radians)"""
    b1 = pos[idx2] - pos[idx1]
    b2 = pos[idx3] - pos[idx2]
    b3 = pos[idx4] - pos[idx3]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    cos_phi = np.clip(np.dot(n1, n2), -1, 1)
    phi = np.arccos(cos_phi)

    if np.dot(np.cross(n1, n2), b2) < 0:
        phi = -phi

    return phi


def create_butane_nve_system():
    """Create butane system for NVE test"""
    system = System()

    # Particles
    for i in range(4):
        system.addParticle(12.0)  # C
    for i in range(10):
        system.addParticle(1.0)   # H

    # Bonds
    bond_force = HarmonicBondForce()
    bond_force.addBond(0, 1, 1.54*angstroms, 2000.0*kilocalories_per_mole/angstroms**2)
    bond_force.addBond(1, 2, 1.54*angstroms, 2000.0*kilocalories_per_mole/angstroms**2)
    bond_force.addBond(2, 3, 1.54*angstroms, 2000.0*kilocalories_per_mole/angstroms**2)

    # C-H bonds with constraints
    ch_bonds = [(0,4), (0,5), (0,6), (1,7), (1,8), (2,9), (2,10), (3,11), (3,12), (3,13)]
    for c, h in ch_bonds:
        bond_force.addBond(c, h, 1.09*angstroms, 2000.0*kilocalories_per_mole/angstroms**2)
        system.addConstraint(c, h, 1.09*angstroms)

    system.addForce(bond_force)

    # Angles
    angle_force = HarmonicAngleForce()
    angle_force.addAngle(0, 1, 2, 109.5*degrees, 200.0*kilocalories_per_mole/radians**2)
    angle_force.addAngle(1, 2, 3, 109.5*degrees, 200.0*kilocalories_per_mole/radians**2)
    system.addForce(angle_force)

    # Torsion (OPLS) in separate force group
    torsion_force = PeriodicTorsionForce()
    torsion_force.setForceGroup(1)

    V1 = 3.4 * kilocalories_per_mole
    V2 = -0.8 * kilocalories_per_mole
    V3 = 6.8 * kilocalories_per_mole

    torsion_force.addTorsion(0, 1, 2, 3, 1, 0.0, V1/2)
    torsion_force.addTorsion(0, 1, 2, 3, 2, 180.0*degrees, V2/2)
    torsion_force.addTorsion(0, 1, 2, 3, 3, 0.0, V3/2)

    system.addForce(torsion_force)

    # Initial positions
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.54, 0.0, 0.0],
        [2.31, 1.29, 0.0],
        [3.85, 1.29, 0.0],
        [-0.63, -0.63, 0.63],
        [-0.63, 0.63, 0.63],
        [-0.63, 0.0, -0.89],
        [1.54, -0.63, 0.89],
        [1.54, -0.63, -0.89],
        [2.31, 1.92, 0.89],
        [2.31, 1.92, -0.89],
        [4.48, 0.66, 0.63],
        [4.48, 1.92, 0.63],
        [3.85, 1.29, -1.09],
    ]) * angstroms

    return system, positions


def run_fixed_timestep_nve(system, positions, dt_fs=1.0, n_steps=10000):
    """
    Run fixed timestep NVE simulation.

    Args:
        system: OpenMM System
        positions: Initial positions
        dt_fs: Timestep in femtoseconds
        n_steps: Number of steps

    Returns:
        dict with trajectory data
    """
    print(f"Running FIXED timestep NVE (dt={dt_fs} fs, {n_steps} steps)...")

    # Verlet integrator (NVE, no thermostat)
    integrator = VerletIntegrator(dt_fs*femtoseconds)

    platform = Platform.getPlatformByName('CPU')
    context = Context(system, integrator, platform)
    context.setPositions(positions)

    # Minimize
    LocalEnergyMinimizer.minimize(context, 1.0, 200)

    # Set velocities to 300K
    context.setVelocitiesToTemperature(300*kelvin)

    # Trajectory storage
    traj = {
        'time': [],
        'E_total': [],
        'E_potential': [],
        'E_kinetic': [],
        'temperature': []
    }

    # Run
    for step in range(n_steps):
        integrator.step(1)

        if step % 10 == 0:
            state = context.getState(getEnergy=True)
            E_pot = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
            E_kin = state.getKineticEnergy().value_in_unit(kilocalories_per_mole)
            E_tot = E_pot + E_kin

            # Temperature from kinetic energy
            # KE = (3/2) N k_B T, N = 14 atoms
            # T = (2/3) KE / (N k_B)
            k_B_kcal = 0.001987  # kcal/(mol·K)
            T = (2.0/3.0) * E_kin / (14 * k_B_kcal)

            traj['time'].append(step * dt_fs / 1000.0)  # ps
            traj['E_total'].append(E_tot)
            traj['E_potential'].append(E_pot)
            traj['E_kinetic'].append(E_kin)
            traj['temperature'].append(T)

    del context, integrator

    # Convert to arrays
    for key in traj:
        traj[key] = np.array(traj[key])

    return traj


def run_adaptive_timestep_nve(system, positions, dt_base=1.0, k=0.001, n_steps=10000):
    """
    Run adaptive timestep NVE simulation.

    This is a simplified adaptive integrator without the full class overhead.
    Uses Verlet (symplectic) instead of Langevin.

    Args:
        system: OpenMM System
        positions: Initial positions
        dt_base: Base timestep (fs)
        k: Stiffness scaling parameter
        n_steps: Number of steps

    Returns:
        dict with trajectory data
    """
    print(f"Running ADAPTIVE timestep NVE (dt_base={dt_base} fs, k={k}, {n_steps} steps)...")

    # Start with base timestep
    integrator = VerletIntegrator(dt_base*femtoseconds)

    platform = Platform.getPlatformByName('CPU')
    context = Context(system, integrator, platform)
    context.setPositions(positions)

    # Minimize
    LocalEnergyMinimizer.minimize(context, 1.0, 200)

    # Set velocities
    context.setVelocitiesToTemperature(300*kelvin)

    # Get masses
    masses = np.array([system.getParticleMass(i).value_in_unit(dalton) for i in range(system.getNumParticles())])

    # Torsion
    torsion_atoms = (0, 1, 2, 3)

    # Trajectory storage
    traj = {
        'time': [],
        'E_total': [],
        'E_potential': [],
        'E_kinetic': [],
        'temperature': [],
        'dt': [],
        'Lambda_stiff': []
    }

    # Adaptive timestep state
    dt_current = dt_base
    Lambda_smooth = 0.0
    alpha = 0.1  # EMA parameter (conservative smoothing for NVE)
    time_elapsed = 0.0  # ps

    # Run
    for step in range(n_steps):
        # 1) Get state for Lambda computation (no energy needed here)
        state_all = context.getState(getPositions=True, getVelocities=True)
        state_torsion = context.getState(getForces=True, groups={1})

        pos = state_all.getPositions(asNumpy=True).value_in_unit(angstroms)
        vel = state_all.getVelocities(asNumpy=True).value_in_unit(angstroms/picosecond)
        F_torsion = state_torsion.getForces(asNumpy=True).value_in_unit(kilocalories_per_mole/angstroms)

        # 2) Compute Λ_stiff
        r_a, r_b, r_c, r_d = pos[list(torsion_atoms)]
        v_a, v_b, v_c, v_d = vel[list(torsion_atoms)]
        F_a, F_b, F_c, F_d = F_torsion[list(torsion_atoms)]

        try:
            phi_dot = compute_phi_dot(r_a, r_b, r_c, r_d, v_a, v_b, v_c, v_d)
            Q_phi = compute_Q_phi(r_a, r_b, r_c, r_d, F_a, F_b, F_c, F_d)
            Lambda_current = abs(phi_dot * Q_phi)
        except:
            Lambda_current = 0.0

        # Update smoothed Lambda
        Lambda_smooth = alpha * Lambda_current + (1 - alpha) * Lambda_smooth

        # Adaptive timestep (only SHRINK below known-stable dt_base)
        dt_adaptive = dt_base / (1.0 + k * Lambda_smooth)

        # Bounds: never exceed dt_base in NVE
        dt_min = 0.25 * dt_base
        dt_max = dt_base
        dt_adaptive = max(dt_min, min(dt_max, dt_adaptive))

        # Rate limiting (max 10% change - more conservative than 20%)
        max_change = 0.1 * dt_current
        if abs(dt_adaptive - dt_current) > max_change:
            if dt_adaptive > dt_current:
                dt_adaptive = dt_current + max_change
            else:
                dt_adaptive = dt_current - max_change

        # Update timestep
        dt_current = dt_adaptive
        integrator.setStepSize(dt_current*femtoseconds)

        # 3) Take the step
        integrator.step(1)
        time_elapsed += dt_current / 1000.0  # ps

        # 4) Record AFTER the step (every 10 steps)
        if step % 10 == 0:
            state_energy = context.getState(getEnergy=True)
            E_pot = state_energy.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
            E_kin = state_energy.getKineticEnergy().value_in_unit(kilocalories_per_mole)
            E_tot = E_pot + E_kin

            k_B_kcal = 0.001987
            T = (2.0/3.0) * E_kin / (14 * k_B_kcal)

            traj['time'].append(time_elapsed)
            traj['E_total'].append(E_tot)
            traj['E_potential'].append(E_pot)
            traj['E_kinetic'].append(E_kin)
            traj['temperature'].append(T)
            traj['dt'].append(dt_current)
            traj['Lambda_stiff'].append(Lambda_current)

    del context, integrator

    # Convert to arrays
    for key in traj:
        traj[key] = np.array(traj[key])

    return traj


def analyze_energy_conservation(traj_fixed, traj_adaptive, dt_baseline):
    """
    Analyze and compare energy conservation.

    Args:
        traj_fixed: Fixed timestep trajectory
        traj_adaptive: Adaptive timestep trajectory
        dt_baseline: Baseline timestep (fs) used for fixed run

    Returns:
        dict with metrics
    """
    print()
    print("="*80)
    print("ENERGY CONSERVATION ANALYSIS")
    print("="*80)
    print()

    # Fixed timestep
    E_fixed = traj_fixed['E_total']
    E_fixed_initial = E_fixed[0]
    E_fixed_drift = np.abs(E_fixed - E_fixed_initial)
    E_fixed_drift_percent = (E_fixed_drift / np.abs(E_fixed_initial)) * 100

    # Adaptive timestep
    E_adaptive = traj_adaptive['E_total']
    E_adaptive_initial = E_adaptive[0]
    E_adaptive_drift = np.abs(E_adaptive - E_adaptive_initial)
    E_adaptive_drift_percent = (E_adaptive_drift / np.abs(E_adaptive_initial)) * 100

    print(f"FIXED TIMESTEP ({dt_baseline:.2f} fs):")
    print(f"  E_initial:     {E_fixed_initial:.4f} kcal/mol")
    print(f"  E_final:       {E_fixed[-1]:.4f} kcal/mol")
    print(f"  Drift (abs):   {E_fixed_drift[-1]:.4f} kcal/mol")
    print(f"  Drift (%):     {E_fixed_drift_percent[-1]:.4f}%")
    print(f"  Max drift:     {E_fixed_drift_percent.max():.4f}%")
    print(f"  Median drift:  {np.median(E_fixed_drift_percent):.4f}%")
    print()

    print("ADAPTIVE TIMESTEP:")
    print(f"  E_initial:     {E_adaptive_initial:.4f} kcal/mol")
    print(f"  E_final:       {E_adaptive[-1]:.4f} kcal/mol")
    print(f"  Drift (abs):   {E_adaptive_drift[-1]:.4f} kcal/mol")
    print(f"  Drift (%):     {E_adaptive_drift_percent[-1]:.4f}%")
    print(f"  Max drift:     {E_adaptive_drift_percent.max():.4f}%")
    print(f"  Median drift:  {np.median(E_adaptive_drift_percent):.4f}%")
    print(f"  Average dt:    {traj_adaptive['dt'].mean():.3f} fs")
    print(f"  Min/Max dt:    {traj_adaptive['dt'].min():.3f} / {traj_adaptive['dt'].max():.3f} fs")
    print(f"  Speedup:       {traj_adaptive['dt'].mean() / dt_baseline:.3f}× vs baseline")
    print()

    # Verdict
    print("VERDICT:")
    if E_fixed_drift_percent[-1] < 1.0:
        print(f"✅ Fixed timestep: {E_fixed_drift_percent[-1]:.4f}%% < 1%% (good)")
    else:
        print(f"⚠️  Fixed timestep: {E_fixed_drift_percent[-1]:.4f}%% > 1%% (poor)")

    if E_adaptive_drift_percent[-1] < 1.0:
        print(f"✅ Adaptive timestep: {E_adaptive_drift_percent[-1]:.4f}%% < 1%% (good)")
    else:
        print(f"⚠️  Adaptive timestep: {E_adaptive_drift_percent[-1]:.4f}%% > 1%% (poor)")

    if E_adaptive_drift_percent[-1] < 2 * E_fixed_drift_percent[-1]:
        print("✅ Adaptive drift comparable to fixed (within 2×)")
    else:
        print(f"❌ Adaptive drift {E_adaptive_drift_percent[-1] / E_fixed_drift_percent[-1]:.1f}× worse than fixed")

    print()

    return {
        'fixed_drift_percent': E_fixed_drift_percent[-1],
        'adaptive_drift_percent': E_adaptive_drift_percent[-1],
        'adaptive_speedup': traj_adaptive['dt'].mean() / dt_baseline
    }


def plot_energy_conservation(traj_fixed, traj_adaptive, dt_baseline):
    """Plot energy conservation comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Total energy vs time
    ax = axes[0, 0]
    ax.plot(traj_fixed['time'], traj_fixed['E_total'], 'b-', label=f'Fixed ({dt_baseline:.2f} fs)', linewidth=1.5)
    ax.plot(traj_adaptive['time'], traj_adaptive['E_total'], 'r-', label='Adaptive', linewidth=1.5)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Total Energy (kcal/mol)', fontsize=12)
    ax.set_title('Energy Conservation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Energy drift (%)
    ax = axes[0, 1]
    E_fixed_drift = np.abs(traj_fixed['E_total'] - traj_fixed['E_total'][0]) / np.abs(traj_fixed['E_total'][0]) * 100
    E_adaptive_drift = np.abs(traj_adaptive['E_total'] - traj_adaptive['E_total'][0]) / np.abs(traj_adaptive['E_total'][0]) * 100

    ax.plot(traj_fixed['time'], E_fixed_drift, 'b-', label='Fixed', linewidth=1.5)
    ax.plot(traj_adaptive['time'], E_adaptive_drift, 'r-', label='Adaptive', linewidth=1.5)
    ax.axhline(1.0, color='g', linestyle='--', alpha=0.5, label='1% threshold')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Energy Drift (%)', fontsize=12)
    ax.set_title('Relative Energy Drift', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Adaptive timestep and Lambda
    ax = axes[1, 0]
    ax2 = ax.twinx()

    ax.plot(traj_adaptive['time'], traj_adaptive['dt'], 'b-', linewidth=1.5, label='Δt')
    ax2.plot(traj_adaptive['time'], traj_adaptive['Lambda_stiff'], 'r-', linewidth=1.5, alpha=0.7, label='Λ_stiff')

    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Timestep (fs)', color='blue', fontsize=12)
    ax2.set_ylabel('Λ_stiff', color='red', fontsize=12)
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.set_title('Adaptive Timestep Control', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Temperature
    ax = axes[1, 1]
    ax.plot(traj_fixed['time'], traj_fixed['temperature'], 'b-', label='Fixed', linewidth=1.5)
    ax.plot(traj_adaptive['time'], traj_adaptive['temperature'], 'r-', label='Adaptive', linewidth=1.5)
    ax.axhline(300, color='g', linestyle='--', alpha=0.5, label='Target (300K)')
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title('Temperature Fluctuations (NVE)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nve_energy_conservation_test.png', dpi=150, bbox_inches='tight')
    print("Plot saved: nve_energy_conservation_test.png")
    print()


if __name__ == "__main__":
    print("="*80)
    print("NVE ENERGY CONSERVATION TEST")
    print("="*80)
    print()

    # Create system
    system, positions = create_butane_nve_system()

    # Run fixed timestep (establish safe baseline first)
    dt_baseline = 0.5  # fs - conservative baseline for stiff system
    traj_fixed = run_fixed_timestep_nve(system, positions, dt_fs=dt_baseline, n_steps=10000)

    # Run adaptive timestep (only shrink from baseline)
    traj_adaptive = run_adaptive_timestep_nve(system, positions, dt_base=dt_baseline, k=0.001, n_steps=10000)

    # Analyze
    metrics = analyze_energy_conservation(traj_fixed, traj_adaptive, dt_baseline)

    # Plot
    plot_energy_conservation(traj_fixed, traj_adaptive, dt_baseline)

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Fixed timestep drift:    {metrics['fixed_drift_percent']:.4f}%%")
    print(f"Adaptive timestep drift: {metrics['adaptive_drift_percent']:.4f}%%")
    print(f"Adaptive speedup:        {metrics['adaptive_speedup']:.3f}×")
    print()

    if metrics['adaptive_drift_percent'] < 1.0:
        print("✅ ENERGY CONSERVATION VALIDATED")
        print("   Adaptive integrator maintains <1% energy drift in NVE")
    else:
        print("❌ ENERGY DRIFT TOO HIGH")
        print(f"   {metrics['adaptive_drift_percent']:.2f}% > 1% threshold")
        print("   Need further improvements (symplectic scheme, smaller k, etc.)")
    print()
