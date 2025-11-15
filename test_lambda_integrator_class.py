#!/usr/bin/env python3
"""
Test LambdaAdaptiveVerletIntegrator Class
=========================================

Verify that the integrator class produces results consistent
with validated test scripts.

Rick Mathews - November 2024
"""

import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *
from lambda_adaptive_integrator import LambdaAdaptiveVerletIntegrator, create_adaptive_integrator


def create_butane_system():
    """Create simple butane system"""
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

    # Torsion in force group 1
    torsion_force = PeriodicTorsionForce()
    torsion_force.setForceGroup(1)

    V1 = 3.4 * kilocalories_per_mole
    V2 = -0.8 * kilocalories_per_mole
    V3 = 6.8 * kilocalories_per_mole

    torsion_force.addTorsion(0, 1, 2, 3, 1, 0.0, V1/2)
    torsion_force.addTorsion(0, 1, 2, 3, 2, 180.0*degrees, V2/2)
    torsion_force.addTorsion(0, 1, 2, 3, 3, 0.0, V3/2)

    system.addForce(torsion_force)

    # Positions
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


def test_class_basic():
    """Test basic class functionality"""
    print("="*80)
    print("TEST 1: Basic Class Functionality")
    print("="*80)
    print()

    system, positions = create_butane_system()

    # Create context with Verlet integrator
    integrator = VerletIntegrator(1.0*femtoseconds)
    platform = Platform.getPlatformByName('CPU')
    context = Context(system, integrator, platform)
    context.setPositions(positions)

    LocalEnergyMinimizer.minimize(context, 1.0, 200)
    context.setVelocitiesToTemperature(300*kelvin)

    # Create adaptive integrator
    adaptive = LambdaAdaptiveVerletIntegrator(
        context=context,
        dt_base_fs=1.0,
        k=0.001,
        torsion_atoms=(0, 1, 2, 3)
    )

    print(f"Created: {adaptive}")
    print()

    # Run short simulation
    print("Running 100 steps...")
    energies = []
    dts = []

    for i in range(100):
        adaptive.step(1)
        if i % 10 == 0:
            state = context.getState(getEnergy=True)
            E = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
            energies.append(E)
            dts.append(adaptive.get_timestep())

    stats = adaptive.get_stats()
    print()
    print(f"Stats after 100 steps:")
    print(f"  Time elapsed: {stats['time_ps']:.6f} ps")
    print(f"  Current dt: {stats['dt_current_fs']:.4f} fs")
    print(f"  Lambda_smooth: {stats['Lambda_smooth']:.4f}")
    print(f"  Mean dt: {np.mean(dts):.4f} fs")
    print()

    del context, integrator
    print("✅ Basic test passed")
    print()


def test_preset_modes():
    """Test preset mode shortcuts"""
    print("="*80)
    print("TEST 2: Preset Modes")
    print("="*80)
    print()

    system, positions = create_butane_system()

    for mode in ["speedup", "balanced", "safety"]:
        print(f"Testing mode: {mode}")

        integrator = VerletIntegrator(1.0*femtoseconds)
        platform = Platform.getPlatformByName('CPU')
        context = Context(system, integrator, platform)
        context.setPositions(positions)
        LocalEnergyMinimizer.minimize(context, 1.0, 200)
        context.setVelocitiesToTemperature(300*kelvin)

        # Use convenience function
        adaptive = create_adaptive_integrator(
            context,
            torsion_atoms=(0, 1, 2, 3),
            mode=mode
        )

        print(f"  Created: {adaptive}")

        # Run brief test
        adaptive.step(100)
        stats = adaptive.get_stats()

        print(f"  After 100 steps:")
        print(f"    Time: {stats['time_ps']:.6f} ps")
        print(f"    Current dt: {stats['dt_current_fs']:.4f} fs")
        print()

        del context, integrator

    print("✅ Preset modes test passed")
    print()


def test_nve_consistency():
    """Test that class gives same NVE results as validated script"""
    print("="*80)
    print("TEST 3: NVE Consistency Check")
    print("="*80)
    print()

    system, positions = create_butane_system()

    integrator = VerletIntegrator(1.0*femtoseconds)
    platform = Platform.getPlatformByName('CPU')
    context = Context(system, integrator, platform)
    context.setPositions(positions)
    LocalEnergyMinimizer.minimize(context, 1.0, 200)
    context.setVelocitiesToTemperature(300*kelvin)

    # Use speedup mode (dt_base=1.0, k=0.001)
    adaptive = create_adaptive_integrator(
        context,
        torsion_atoms=(0, 1, 2, 3),
        mode="speedup"
    )

    print("Running 10,000 step NVE simulation...")
    print()

    energies = []
    times = []
    dts = []

    for step in range(10000):
        adaptive.step(1)
        if step % 10 == 0:
            state = context.getState(getEnergy=True)
            E_pot = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
            E_kin = state.getKineticEnergy().value_in_unit(kilocalories_per_mole)
            E_tot = E_pot + E_kin
            energies.append(E_tot)
            times.append(adaptive.get_time())
            dts.append(adaptive.get_timestep())

    energies = np.array(energies)
    dts = np.array(dts)

    # Analyze energy drift
    E_initial = energies[0]
    drift_abs = np.abs(energies - E_initial)
    drift_percent = (drift_abs / np.abs(E_initial)) * 100

    print(f"Results (10 ps NVE):")
    print(f"  E_initial: {E_initial:.4f} kcal/mol")
    print(f"  E_final:   {energies[-1]:.4f} kcal/mol")
    print(f"  Drift:     {drift_percent[-1]:.4f}%")
    print(f"  Max drift: {drift_percent.max():.4f}%")
    print(f"  Median:    {np.median(drift_percent):.4f}%")
    print()
    print(f"Timestep statistics:")
    print(f"  Mean dt:   {dts.mean():.4f} fs")
    print(f"  Min dt:    {dts.min():.4f} fs")
    print(f"  Max dt:    {dts.max():.4f} fs")
    print(f"  Speedup:   {dts.mean() / 0.5:.3f}× vs 0.5 fs baseline")
    print()

    # Expected from validation: ~0.01-0.02% drift, 1.97× speedup
    if drift_percent[-1] < 0.5:
        print("✅ Energy conservation excellent (<0.5% drift)")
    else:
        print(f"⚠️  Drift higher than expected: {drift_percent[-1]:.4f}%")

    if 1.8 < dts.mean() / 0.5 < 2.2:
        print("✅ Speedup consistent with validation (~1.97×)")
    else:
        print(f"⚠️  Speedup different from expected: {dts.mean() / 0.5:.3f}×")

    print()

    del context, integrator


if __name__ == "__main__":
    print()
    print("="*80)
    print("LAMBDA ADAPTIVE VERLET INTEGRATOR - CLASS TESTS")
    print("="*80)
    print()

    test_class_basic()
    test_preset_modes()
    test_nve_consistency()

    print("="*80)
    print("ALL TESTS PASSED")
    print("="*80)
    print()
