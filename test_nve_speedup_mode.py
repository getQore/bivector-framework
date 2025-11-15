#!/usr/bin/env python3
"""
NVE Speedup Mode Test: dt_base = 1.0 fs
========================================

Test if adaptive timestep from 1.0 fs baseline can:
1. Match stability of fixed 0.5 fs (< 0.5% drift)
2. Achieve speedup by averaging 0.7-0.8 fs

Target: 1.4-1.6× speedup vs stable 0.5 fs baseline

Rick Mathews - November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from openmm import *
from openmm.app import *
from openmm.unit import *
from md_bivector_utils import compute_phi_dot, compute_Q_phi


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
    """Run fixed timestep NVE simulation."""
    integrator = VerletIntegrator(dt_fs*femtoseconds)
    platform = Platform.getPlatformByName('CPU')
    context = Context(system, integrator, platform)
    context.setPositions(positions)

    LocalEnergyMinimizer.minimize(context, 1.0, 200)
    context.setVelocitiesToTemperature(300*kelvin)

    traj = {'time': [], 'E_total': [], 'E_potential': [], 'E_kinetic': [], 'temperature': []}

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


def run_adaptive_timestep_nve(system, positions, dt_base=1.0, k=0.001, alpha=0.1, n_steps=10000):
    """Run adaptive timestep NVE simulation."""
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
        'time': [], 'E_total': [], 'E_potential': [], 'E_kinetic': [],
        'temperature': [], 'dt': [], 'Lambda_stiff': []
    }

    dt_current = dt_base
    Lambda_smooth = 0.0
    time_elapsed = 0.0

    for step in range(n_steps):
        # 1) Get state for Lambda
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

        Lambda_smooth = alpha * Lambda_current + (1 - alpha) * Lambda_smooth

        # Adaptive timestep (only SHRINK)
        dt_adaptive = dt_base / (1.0 + k * Lambda_smooth)
        dt_min = 0.25 * dt_base
        dt_max = dt_base
        dt_adaptive = max(dt_min, min(dt_max, dt_adaptive))

        # Rate limiting
        max_change = 0.1 * dt_current
        if abs(dt_adaptive - dt_current) > max_change:
            dt_adaptive = dt_current + max_change if dt_adaptive > dt_current else dt_current - max_change

        dt_current = dt_adaptive
        integrator.setStepSize(dt_current*femtoseconds)

        # 3) Take step
        integrator.step(1)
        time_elapsed += dt_current / 1000.0

        # 4) Record
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


if __name__ == "__main__":
    print("="*80)
    print("NVE SPEEDUP MODE TEST (dt_base = 1.0 fs)")
    print("="*80)
    print()

    system, positions = create_butane_nve_system()

    # Gold standard: fixed 0.5 fs
    print("Running GOLD STANDARD: Fixed 0.5 fs...")
    traj_gold = run_fixed_timestep_nve(system, positions, dt_fs=0.5, n_steps=10000)
    E_gold = traj_gold['E_total']
    drift_gold = (np.abs(E_gold - E_gold[0]) / np.abs(E_gold[0])) * 100
    print(f"  Drift: {drift_gold[-1]:.4f}% (median: {np.median(drift_gold):.4f}%)")
    print()

    # Fixed 1.0 fs (potentially unstable?)
    print("Running FIXED 1.0 fs (test baseline stability)...")
    traj_fixed_1fs = run_fixed_timestep_nve(system, positions, dt_fs=1.0, n_steps=10000)
    E_fixed = traj_fixed_1fs['E_total']
    drift_fixed = (np.abs(E_fixed - E_fixed[0]) / np.abs(E_fixed[0])) * 100
    print(f"  Drift: {drift_fixed[-1]:.4f}% (median: {np.median(drift_fixed):.4f}%)")
    print()

    # Adaptive 1.0 fs with different k values
    k_values = [0.001, 0.002, 0.005, 0.01]
    results = []

    for k in k_values:
        print(f"Running ADAPTIVE dt_base=1.0 fs, k={k:.4f}...")
        traj_adapt = run_adaptive_timestep_nve(system, positions, dt_base=1.0, k=k, n_steps=10000)
        E_adapt = traj_adapt['E_total']
        drift_adapt = (np.abs(E_adapt - E_adapt[0]) / np.abs(E_adapt[0])) * 100

        result = {
            'k': k,
            'drift_final': drift_adapt[-1],
            'drift_median': np.median(drift_adapt),
            'drift_max': drift_adapt.max(),
            'dt_mean': traj_adapt['dt'].mean(),
            'dt_min': traj_adapt['dt'].min(),
            'speedup_vs_gold': traj_adapt['dt'].mean() / 0.5,
            'traj': traj_adapt
        }
        results.append(result)

        print(f"  Drift: {result['drift_final']:.4f}% (median: {result['drift_median']:.4f}%)")
        print(f"  Mean dt: {result['dt_mean']:.4f} fs (range: [{result['dt_min']:.4f}, 1.000] fs)")
        print(f"  Speedup vs gold: {result['speedup_vs_gold']:.3f}×")
        print()

    # Summary
    print("="*80)
    print("SUMMARY: Speedup Mode Analysis")
    print("="*80)
    print()
    print(f"{'Method':<30} {'Drift (%)':>12} {'Median (%)':>12} {'Speedup':>10}")
    print("-"*80)
    print(f"{'Fixed 0.5 fs (GOLD)':30} {drift_gold[-1]:>12.4f} {np.median(drift_gold):>12.4f} {'1.00×':>10}")
    print(f"{'Fixed 1.0 fs':30} {drift_fixed[-1]:>12.4f} {np.median(drift_fixed):>12.4f} {'2.00×':>10}")

    for r in results:
        method = f"Adaptive 1.0 fs (k={r['k']:.3f})"
        print(f"{method:30} {r['drift_final']:>12.4f} {r['drift_median']:>12.4f} {r['speedup_vs_gold']:>10.3f}×")

    print()
    print("="*80)
    print("VERDICT")
    print("="*80)
    print()

    # Check if fixed 1.0 fs is acceptable
    if drift_fixed[-1] < 1.0:
        print("✅ Fixed 1.0 fs is stable (<1% drift)")
        print(f"   Can use as baseline for comparison")
    else:
        print("⚠️  Fixed 1.0 fs has high drift ({drift_fixed[-1]:.2f}%)")
        print("   1.0 fs baseline may be too aggressive for this system")
    print()

    # Find best adaptive result
    good_adaptive = [r for r in results if r['drift_final'] < 0.5 and r['speedup_vs_gold'] > 1.2]

    if good_adaptive:
        best = max(good_adaptive, key=lambda r: r['speedup_vs_gold'])
        print(f"✅ SPEEDUP ACHIEVED with k={best['k']:.4f}")
        print()
        print(f"   Drift:   {best['drift_final']:.4f}% (< 0.5% target)")
        print(f"   Speedup: {best['speedup_vs_gold']:.3f}× vs stable 0.5 fs baseline")
        print(f"   Mean dt: {best['dt_mean']:.4f} fs")
        print()
        print(f"   This achieves meaningful speedup while maintaining excellent stability!")
        print()
    else:
        # Check for any with drift < 1%
        acceptable = [r for r in results if r['drift_final'] < 1.0]
        if acceptable:
            best = max(acceptable, key=lambda r: r['speedup_vs_gold'])
            print(f"⚠️  Best compromise: k={best['k']:.4f}")
            print()
            print(f"   Drift:   {best['drift_final']:.4f}% (< 1% acceptable)")
            print(f"   Speedup: {best['speedup_vs_gold']:.3f}× vs 0.5 fs")
            print()
            print("   Consider: Lower k or dt_base for better stability")
        else:
            print("❌ No configuration achieved <1% drift")
            print("   1.0 fs baseline may be too large for this stiff system")

    print()
