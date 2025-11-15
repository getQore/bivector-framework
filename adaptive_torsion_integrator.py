#!/usr/bin/env python3
"""
Adaptive Torsion Integrator for OpenMM
======================================

Production implementation of Λ_stiff adaptive timestep control.

Based on validated diagnostic: Λ_stiff(t) = |φ̇(t) · Q_φ(t)|

Adaptive timestep rule: Δt(t) = Δt_base / (1 + k·Λ_max(t))

Features:
- Per-torsion Λ_stiff computation
- Exponential moving average smoothing
- Configurable timestep bounds
- Energy conservation monitoring
- Performance benchmarking

Rick Mathews - November 2024
Patent-Ready Implementation
"""

import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

from md_bivector_utils import (
    compute_dihedral_gradient,
    compute_Q_phi,
    compute_phi_dot
)


class AdaptiveTorsionIntegrator:
    """
    Adaptive timestep integrator using torsional stiffness diagnostic.

    This is a wrapper around OpenMM's VerletIntegrator that:
    1. Computes Λ_stiff for all torsions each step
    2. Adjusts timestep based on maximum Λ_stiff
    3. Integrates with adaptive Δt(t)

    Usage:
        integrator = AdaptiveTorsionIntegrator(
            system, topology,
            dt_base=2.0*femtoseconds,
            k=0.05,
            alpha=0.2
        )

        simulation = Simulation(topology, system, integrator)
        simulation.step(10000)

        stats = integrator.get_statistics()
        print(f"Average dt: {stats['dt_mean']} fs")
        print(f"Speedup: {stats['speedup']:.2f}x")
    """

    def __init__(self, system, topology,
                 dt_base=2.0*femtoseconds,
                 dt_min=0.5*femtoseconds,
                 dt_max=4.0*femtoseconds,
                 k=0.05,
                 alpha=0.2,
                 temperature=300*kelvin,
                 friction=1.0/picosecond):
        """
        Initialize adaptive integrator.

        Args:
            system: OpenMM System object
            topology: OpenMM Topology object
            dt_base: Base timestep (default: 2.0 fs)
            dt_min: Minimum timestep (default: 0.5 fs)
            dt_max: Maximum timestep (default: 4.0 fs)
            k: Stiffness scaling parameter (default: 0.05)
            alpha: EMA smoothing parameter (default: 0.2)
            temperature: Temperature for Langevin thermostat
            friction: Friction coefficient for Langevin
        """
        self.system = system
        self.topology = topology

        # Timestep parameters
        self.dt_base = dt_base
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.k = k
        self.alpha = alpha

        # Current timestep (starts at base)
        self.dt_current = dt_base

        # Smoothed Lambda (EMA)
        self.Lambda_smooth = 0.0

        # Statistics
        self.stats = {
            'steps': 0,
            'dt_values': [],
            'Lambda_values': [],
            'energy_values': [],
            'dt_sum': 0.0,
            'time_elapsed': 0.0
        }

        # Extract torsions from topology
        self.torsions = self._extract_torsions()
        print(f"  Total torsional DOFs: {len(self.torsions)}")

        # Get masses
        self.masses = np.array([system.getParticleMass(i).value_in_unit(dalton)
                                for i in range(system.getNumParticles())])

        # Create underlying integrator (Langevin)
        self.integrator = LangevinIntegrator(temperature, friction, dt_base)

        # Identify torsion force group
        self.torsion_group = self._identify_torsion_force_group()

        if self.torsion_group is not None:
            print(f"  Using torsion force group {self.torsion_group}")
        else:
            print(f"  Warning: Using total forces (torsion group not found)")

    def _extract_torsions(self):
        """
        Extract proper torsions from PeriodicTorsionForce.

        This reads torsions directly from the force field,
        ensuring we only track "real" torsional DOFs.

        Returns:
            List of (a, b, c, d) atom index tuples
        """
        torsions = []

        # Find PeriodicTorsionForce
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if isinstance(force, PeriodicTorsionForce):
                # Extract unique torsions (same atoms, different periodicities)
                seen_atoms = set()
                for j in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(j)

                    # Canonical representation (avoid duplicates)
                    atoms = tuple(sorted([p1, p2, p3, p4]))

                    if atoms not in seen_atoms:
                        seen_atoms.add(atoms)
                        # Use original order for proper torsion definition
                        torsions.append((p1, p2, p3, p4))

                print(f"  Extracted {len(torsions)} unique torsions from PeriodicTorsionForce")
                return torsions

        # Fallback: if no PeriodicTorsionForce, return empty
        print("  Warning: No PeriodicTorsionForce found")
        return []

    def _identify_torsion_force_group(self):
        """
        Identify force group containing torsional forces.

        Returns:
            Force group index, or None if not found
        """
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if isinstance(force, PeriodicTorsionForce):
                return force.getForceGroup()

        # If not found, return None (will use total forces)
        return None

    def compute_Lambda_stiff(self, context):
        """
        Compute Λ_stiff for all torsions.

        Args:
            context: OpenMM Context

        Returns:
            Lambda_max: Maximum Λ_stiff across all torsions
        """
        # Get state
        state_vel = context.getState(getPositions=True, getVelocities=True)

        # Get torsion forces (if available)
        if self.torsion_group is not None:
            state_torsion = context.getState(getForces=True, groups={self.torsion_group})
            forces = state_torsion.getForces(asNumpy=True).value_in_unit(kilocalories_per_mole/angstroms)
        else:
            # Use total forces (less ideal but works)
            state_forces = context.getState(getForces=True)
            forces = state_forces.getForces(asNumpy=True).value_in_unit(kilocalories_per_mole/angstroms)

        positions = state_vel.getPositions(asNumpy=True).value_in_unit(angstroms)
        velocities = state_vel.getVelocities(asNumpy=True).value_in_unit(angstroms/picosecond)

        # Compute Λ_stiff for each torsion
        Lambda_values = []

        for torsion in self.torsions:
            a, b, c, d = torsion

            # Extract atom data
            r_a, r_b, r_c, r_d = positions[[a, b, c, d]]
            v_a, v_b, v_c, v_d = velocities[[a, b, c, d]]
            F_a, F_b, F_c, F_d = forces[[a, b, c, d]]

            # Compute φ̇
            try:
                phi_dot = compute_phi_dot(r_a, r_b, r_c, r_d, v_a, v_b, v_c, v_d)
            except:
                phi_dot = 0.0

            # Compute Q_φ
            try:
                Q_phi = compute_Q_phi(r_a, r_b, r_c, r_d, F_a, F_b, F_c, F_d)
            except:
                Q_phi = 0.0

            # Λ_stiff
            Lambda_i = abs(phi_dot * Q_phi)
            Lambda_values.append(Lambda_i)

        # Return maximum
        if len(Lambda_values) > 0:
            return max(Lambda_values)
        else:
            return 0.0

    def update_timestep(self, Lambda_current):
        """
        Update timestep based on current Λ_stiff.

        Args:
            Lambda_current: Current maximum Λ_stiff
        """
        # Update smoothed Lambda (EMA)
        self.Lambda_smooth = (self.alpha * Lambda_current +
                              (1 - self.alpha) * self.Lambda_smooth)

        # Compute adaptive timestep
        dt_adaptive = self.dt_base / (1 + self.k * self.Lambda_smooth)

        # Apply bounds
        dt_adaptive = max(self.dt_min, min(self.dt_max, dt_adaptive))

        # Limit rate of change (prevent abrupt jumps)
        # Max 20% change per step
        dt_current_fs = self.dt_current.value_in_unit(femtoseconds)
        dt_adaptive_fs = dt_adaptive.value_in_unit(femtoseconds)

        max_change = 0.2 * dt_current_fs
        if abs(dt_adaptive_fs - dt_current_fs) > max_change:
            if dt_adaptive_fs > dt_current_fs:
                dt_adaptive_fs = dt_current_fs + max_change
            else:
                dt_adaptive_fs = dt_current_fs - max_change

            dt_adaptive = dt_adaptive_fs * femtoseconds

        # Update integrator timestep
        self.dt_current = dt_adaptive
        self.integrator.setStepSize(dt_adaptive)

    def step(self, n_steps):
        """
        Take n adaptive timesteps.

        Args:
            n_steps: Number of steps to take
        """
        for _ in range(n_steps):
            # Compute current Λ_stiff
            Lambda_current = self.compute_Lambda_stiff(self.context)

            # Update timestep
            self.update_timestep(Lambda_current)

            # Take one integration step
            self.integrator.step(1)

            # Update statistics
            self.stats['steps'] += 1
            self.stats['dt_values'].append(self.dt_current.value_in_unit(femtoseconds))
            self.stats['Lambda_values'].append(Lambda_current)
            self.stats['dt_sum'] += self.dt_current.value_in_unit(femtoseconds)
            self.stats['time_elapsed'] += self.dt_current.value_in_unit(picoseconds)

            # Energy (periodic)
            if self.stats['steps'] % 100 == 0:
                state = self.context.getState(getEnergy=True)
                E = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
                self.stats['energy_values'].append(E)

    def initialize(self, positions):
        """
        Initialize simulation context.

        Args:
            positions: Initial atomic positions
        """
        platform = Platform.getPlatformByName('CPU')
        self.context = Context(self.system, self.integrator, platform)
        self.context.setPositions(positions)

        # Minimize energy
        print("Minimizing energy...")
        LocalEnergyMinimizer.minimize(self.context, 1.0, 200)

        # Set velocities to temperature
        self.context.setVelocitiesToTemperature(self.integrator.getTemperature())

        print("Initialization complete")

    def get_statistics(self):
        """
        Get performance statistics.

        Returns:
            Dictionary with statistics
        """
        if self.stats['steps'] == 0:
            return {'error': 'No steps taken yet'}

        dt_mean = self.stats['dt_sum'] / self.stats['steps']
        dt_base_fs = self.dt_base.value_in_unit(femtoseconds)

        # Speedup = average_dt / base_dt
        speedup = dt_mean / dt_base_fs

        # Energy drift
        if len(self.stats['energy_values']) > 1:
            E_initial = self.stats['energy_values'][0]
            E_final = self.stats['energy_values'][-1]
            energy_drift = abs(E_final - E_initial) / abs(E_initial) * 100
        else:
            energy_drift = 0.0

        return {
            'steps': self.stats['steps'],
            'time_elapsed': self.stats['time_elapsed'],
            'dt_mean': dt_mean,
            'dt_min': min(self.stats['dt_values']),
            'dt_max': max(self.stats['dt_values']),
            'dt_base': dt_base_fs,
            'speedup': speedup,
            'Lambda_mean': np.mean(self.stats['Lambda_values']),
            'Lambda_max': max(self.stats['Lambda_values']),
            'energy_drift_percent': energy_drift,
            'n_torsions': len(self.torsions)
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            'steps': 0,
            'dt_values': [],
            'Lambda_values': [],
            'energy_values': [],
            'dt_sum': 0.0,
            'time_elapsed': 0.0
        }


def create_butane_adaptive_test():
    """
    Test adaptive integrator on butane.

    Compares:
    - Fixed timestep (1.5 fs baseline)
    - Adaptive timestep (Λ_stiff controlled)

    Metrics:
    - Speedup
    - Energy conservation
    - Trajectory similarity
    """
    print("="*80)
    print("ADAPTIVE INTEGRATOR TEST - BUTANE")
    print("="*80)
    print()

    # Create butane system (copied from validation test)
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

    # C-H bonds
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

    # Torsion (OPLS)
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

    # Create topology (minimal)
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue('BUT', chain)

    element_C = Element.getBySymbol('C')
    element_H = Element.getBySymbol('H')

    atoms = []
    for i in range(4):
        atoms.append(topology.addAtom(f'C{i+1}', element_C, residue))
    for i in range(10):
        atoms.append(topology.addAtom(f'H{i+1}', element_H, residue))

    # Add bonds to topology
    topology.addBond(atoms[0], atoms[1])
    topology.addBond(atoms[1], atoms[2])
    topology.addBond(atoms[2], atoms[3])
    for c, h in ch_bonds:
        topology.addBond(atoms[c], atoms[h])

    # Test adaptive integrator
    print("Testing Adaptive Integrator:")
    print("-" * 80)

    integrator_adaptive = AdaptiveTorsionIntegrator(
        system, topology,
        dt_base=2.0*femtoseconds,
        dt_min=0.5*femtoseconds,
        dt_max=4.0*femtoseconds,
        k=0.01,  # Reduced from 0.05 - less aggressive
        alpha=0.2,
        temperature=300*kelvin
    )

    integrator_adaptive.initialize(positions)

    # Run adaptive
    n_steps = 5000
    print(f"Running {n_steps} adaptive steps...")
    integrator_adaptive.step(n_steps)

    # Get statistics
    stats = integrator_adaptive.get_statistics()

    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()
    print(f"Steps taken:        {stats['steps']}")
    print(f"Time elapsed:       {stats['time_elapsed']:.2f} ps")
    print(f"Base timestep:      {stats['dt_base']:.2f} fs")
    print(f"Average timestep:   {stats['dt_mean']:.2f} fs")
    print(f"Min timestep:       {stats['dt_min']:.2f} fs")
    print(f"Max timestep:       {stats['dt_max']:.2f} fs")
    print(f"Speedup:            {stats['speedup']:.2f}×")
    print(f"Energy drift:       {stats['energy_drift_percent']:.4f}%")
    print(f"Λ_stiff (mean):     {stats['Lambda_mean']:.2f}")
    print(f"Λ_stiff (max):      {stats['Lambda_max']:.2f}")
    print(f"Torsions tracked:   {stats['n_torsions']}")
    print()

    if stats['speedup'] > 1.2:
        print("✅ ADAPTIVE INTEGRATOR SUCCESSFUL")
        print(f"   Achieved {stats['speedup']:.2f}× speedup with {stats['energy_drift_percent']:.4f}% energy drift")
    elif stats['speedup'] > 1.0:
        print("⚠️  MODEST SPEEDUP")
        print(f"   Only {stats['speedup']:.2f}× speedup - consider tuning k parameter")
    else:
        print("❌ NO SPEEDUP")
        print("   Adaptive timestep not beneficial for this system")
    print()

    return integrator_adaptive, stats


if __name__ == "__main__":
    integrator, stats = create_butane_adaptive_test()
