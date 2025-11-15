#!/usr/bin/env python3
"""
NVE K-Parameter Sweep for Adaptive Integrator
==============================================

Systematically test different k values to find optimal stability/speedup tradeoff.

Goal: Find k where:
  - Energy drift < 0.5% over 10 ps
  - Speedup > 1.2× (meaningful adaptation)

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


def run_fixed_timestep_nve(system, positions, dt_fs=0.5, n_steps=10000):
    """Run fixed timestep NVE simulation (baseline)."""
    integrator = VerletIntegrator(dt_fs*femtoseconds)
    platform = Platform.getPlatformByName('CPU')
    context = Context(system, integrator, platform)
    context.setPositions(positions)

    LocalEnergyMinimizer.minimize(context, 1.0, 200)
    context.setVelocitiesToTemperature(300*kelvin)

    traj = {
        'time': [],
        'E_total': [],
        'E_potential': [],
        'E_kinetic': [],
        'temperature': []
    }

    for step in range(n_steps):
        integrator.step(1)

        if step % 10 == 0:
            state = context.getState(getEnergy=True)
            E_pot = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
            E_kin = state.getKineticEnergy().value_in_unit(kilocalories_per_mole)
            E_tot = E_pot + E_kin

            k_B_kcal = 0.001987
            T = (2.0/3.0) * E_kin / (14 * k_B_kcal)

            traj['time'].append(step * dt_fs / 1000.0)
            traj['E_total'].append(E_tot)
            traj['E_potential'].append(E_pot)
            traj['E_kinetic'].append(E_kin)
            traj['temperature'].append(T)

    del context, integrator

    for key in traj:
        traj[key] = np.array(traj[key])

    return traj


def run_adaptive_timestep_nve(system, positions, dt_base=0.5, k=0.001,
                               alpha=0.1, n_steps=10000):
    """
    Run adaptive timestep NVE simulation.

    Args:
        system: OpenMM System
        positions: Initial positions
        dt_base: Base timestep (fs) - never exceeded
        k: Stiffness scaling parameter
        alpha: EMA smoothing parameter
        n_steps: Number of steps

    Returns:
        dict with trajectory data
    """
    integrator = VerletIntegrator(dt_base*femtoseconds)
    platform = Platform.getPlatformByName('CPU')
    context = Context(system, integrator, platform)
    context.setPositions(positions)

    LocalEnergyMinimizer.minimize(context, 1.0, 200)
    context.setVelocitiesToTemperature(300*kelvin)

    masses = np.array([system.getParticleMass(i).value_in_unit(dalton)
                       for i in range(system.getNumParticles())])
    torsion_atoms = (0, 1, 2, 3)

    traj = {
        'time': [],
        'E_total': [],
        'E_potential': [],
        'E_kinetic': [],
        'temperature': [],
        'dt': [],
        'Lambda_stiff': []
    }

    dt_current = dt_base
    Lambda_smooth = 0.0
    time_elapsed = 0.0

    for step in range(n_steps):
        # 1) Get state for Lambda computation
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

        # Adaptive timestep (only SHRINK below dt_base)
        dt_adaptive = dt_base / (1.0 + k * Lambda_smooth)

        # Bounds: never exceed dt_base
        dt_min = 0.25 * dt_base
        dt_max = dt_base
        dt_adaptive = max(dt_min, min(dt_max, dt_adaptive))

        # Rate limiting (max 10% change per step)
        max_change = 0.1 * dt_current
        if abs(dt_adaptive - dt_current) > max_change:
            if dt_adaptive > dt_current:
                dt_adaptive = dt_current + max_change
            else:
                dt_adaptive = dt_current - max_change

        dt_current = dt_adaptive
        integrator.setStepSize(dt_current*femtoseconds)

        # 3) Take the step
        integrator.step(1)
        time_elapsed += dt_current / 1000.0

        # 4) Record AFTER the step
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

    for key in traj:
        traj[key] = np.array(traj[key])

    return traj


def analyze_k_value(traj_fixed, traj_adaptive, k_value, dt_baseline):
    """Analyze results for a single k value."""
    E_fixed = traj_fixed['E_total']
    E_fixed_initial = E_fixed[0]
    E_fixed_drift_percent = (np.abs(E_fixed - E_fixed_initial) / np.abs(E_fixed_initial)) * 100

    E_adaptive = traj_adaptive['E_total']
    E_adaptive_initial = E_adaptive[0]
    E_adaptive_drift_percent = (np.abs(E_adaptive - E_adaptive_initial) / np.abs(E_adaptive_initial)) * 100

    return {
        'k': k_value,
        'drift_final': E_adaptive_drift_percent[-1],
        'drift_median': np.median(E_adaptive_drift_percent),
        'drift_max': E_adaptive_drift_percent.max(),
        'dt_mean': traj_adaptive['dt'].mean(),
        'dt_min': traj_adaptive['dt'].min(),
        'dt_max': traj_adaptive['dt'].max(),
        'speedup': traj_adaptive['dt'].mean() / dt_baseline,
        'Lambda_mean': traj_adaptive['Lambda_stiff'].mean(),
        'Lambda_max': traj_adaptive['Lambda_stiff'].max(),
    }


def plot_k_sweep(results, dt_baseline):
    """Plot k-sweep results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    k_vals = [r['k'] for r in results]

    # Plot 1: Final drift vs k
    ax = axes[0, 0]
    ax.plot(k_vals, [r['drift_final'] for r in results], 'o-', linewidth=2, markersize=8)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='0.5% target')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='1.0% limit')
    ax.set_xlabel('k parameter', fontsize=12)
    ax.set_ylabel('Final Drift (%)', fontsize=12)
    ax.set_title('Energy Drift vs k', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Median drift vs k
    ax = axes[0, 1]
    ax.plot(k_vals, [r['drift_median'] for r in results], 'o-', linewidth=2, markersize=8, color='green')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('k parameter', fontsize=12)
    ax.set_ylabel('Median Drift (%)', fontsize=12)
    ax.set_title('Median Drift vs k', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 3: Speedup vs k
    ax = axes[0, 2]
    speedups = [r['speedup'] for r in results]
    ax.plot(k_vals, speedups, 'o-', linewidth=2, markersize=8, color='purple')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')
    ax.axhline(1.2, color='green', linestyle='--', alpha=0.5, label='1.2× target')
    ax.set_xlabel('k parameter', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title('Speedup vs k', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: dt range vs k
    ax = axes[1, 0]
    for r in results:
        ax.barh(r['k'], r['dt_max'] - r['dt_min'], left=r['dt_min'], alpha=0.6, height=0.0002)
    ax.axvline(dt_baseline, color='red', linestyle='--', alpha=0.7, label=f'Baseline ({dt_baseline} fs)')
    ax.set_ylabel('k parameter', fontsize=12)
    ax.set_xlabel('Timestep (fs)', fontsize=12)
    ax.set_title('Timestep Range vs k', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Drift vs speedup (pareto frontier)
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i, r in enumerate(results):
        ax.scatter(r['speedup'], r['drift_final'], s=150, alpha=0.7,
                  color=colors[i], label=f"k={r['k']:.4f}")
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(1.2, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Speedup Factor', fontsize=12)
    ax.set_ylabel('Final Drift (%)', fontsize=12)
    ax.set_title('Drift vs Speedup (Pareto)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 6: Mean Lambda vs k
    ax = axes[1, 2]
    ax.plot(k_vals, [r['Lambda_mean'] for r in results], 'o-', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('k parameter', fontsize=12)
    ax.set_ylabel('Mean Λ_stiff', fontsize=12)
    ax.set_title('Mean Λ_stiff vs k', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nve_k_sweep_analysis.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: nve_k_sweep_analysis.png\n")


if __name__ == "__main__":
    print("="*80)
    print("NVE K-PARAMETER SWEEP")
    print("="*80)
    print()

    # Create system
    system, positions = create_butane_nve_system()

    # Baseline parameters
    dt_baseline = 0.5  # fs
    alpha = 0.1
    n_steps = 10000

    # Run fixed baseline once
    print(f"Running FIXED baseline (dt={dt_baseline} fs)...")
    traj_fixed = run_fixed_timestep_nve(system, positions, dt_fs=dt_baseline, n_steps=n_steps)
    print(f"  Done. Final E = {traj_fixed['E_total'][-1]:.4f} kcal/mol\n")

    # K values to test
    k_values = [0.001, 0.002, 0.005, 0.01, 0.02]

    results = []

    for k in k_values:
        print(f"Testing k = {k:.4f}...")
        traj_adaptive = run_adaptive_timestep_nve(
            system, positions,
            dt_base=dt_baseline,
            k=k,
            alpha=alpha,
            n_steps=n_steps
        )

        metrics = analyze_k_value(traj_fixed, traj_adaptive, k, dt_baseline)
        results.append(metrics)

        print(f"  Drift (final):  {metrics['drift_final']:.4f}%")
        print(f"  Drift (median): {metrics['drift_median']:.4f}%")
        print(f"  Mean dt:        {metrics['dt_mean']:.4f} fs")
        print(f"  dt range:       [{metrics['dt_min']:.4f}, {metrics['dt_max']:.4f}] fs")
        print(f"  Speedup:        {metrics['speedup']:.4f}×")
        print(f"  Mean Λ_stiff:   {metrics['Lambda_mean']:.2f}")
        print()

    # Summary table
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    print(f"{'k':<10} {'Drift (%)':>12} {'Median (%)':>12} {'Mean dt (fs)':>14} {'Speedup':>10} {'Status':>12}")
    print("-"*80)

    for r in results:
        status = "✅ GOOD" if r['drift_final'] < 0.5 and r['speedup'] >= 1.0 else "⚠️  CHECK"
        if r['drift_final'] > 1.0:
            status = "❌ POOR"
        print(f"{r['k']:<10.4f} {r['drift_final']:>12.4f} {r['drift_median']:>12.4f} "
              f"{r['dt_mean']:>14.4f} {r['speedup']:>10.4f}× {status:>12}")

    print()

    # Find optimal k
    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()

    # Find k with best speedup while maintaining drift < 0.5%
    good_results = [r for r in results if r['drift_final'] < 0.5]

    if good_results:
        best = max(good_results, key=lambda r: r['speedup'])
        print(f"✅ OPTIMAL k = {best['k']:.4f}")
        print()
        print(f"   Drift:   {best['drift_final']:.4f}% (< 0.5% target)")
        print(f"   Speedup: {best['speedup']:.4f}× vs baseline")
        print(f"   Mean dt: {best['dt_mean']:.4f} fs")
        print()
        print(f"This k balances stability and performance for production use.")
    else:
        print("⚠️  No k value achieved < 0.5% drift target.")
        print("   Consider:")
        print("   - Using smaller k values")
        print("   - Reducing dt_baseline further")
        print("   - Increasing alpha (smoother EMA)")

    print()

    # Generate plots
    plot_k_sweep(results, dt_baseline)

    print("="*80)
    print("K-SWEEP COMPLETE")
    print("="*80)
    print()
