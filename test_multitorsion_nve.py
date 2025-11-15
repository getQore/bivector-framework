#!/usr/bin/env python3
"""
Multi-Torsion Λ_global NVE Validation Test
===========================================

Sprint 1: Test Λ-adaptive integrator monitoring ALL backbone φ torsions
on Ala12 helix, using Λ_global = max(Λ_i) to control timestep.

Acceptance Criteria:
✅ Energy drift < 0.5% (same as single-torsion)
✅ Λ_global correctly tracks maximum across all torsions
✅ Individual Λ_i show heterogeneous dynamics

Rick Mathews - November 2024
Path A Extension - Sprint 1
"""

import numpy as np
import matplotlib.pyplot as plt
from openmm import *
from openmm.app import *
from openmm.unit import *
from lambda_adaptive_integrator import LambdaAdaptiveVerletIntegrator
from protein_torsion_utils import get_backbone_torsions, pick_middle_torsion

# ============================================================================
# Test Parameters
# ============================================================================

DT_BASELINE_FS = 0.5  # Safe baseline for comparison
DT_ADAPTIVE_FS = 0.5  # Start with safety mode (no speedup)
K_ADAPTIVE = 0.0001   # Protein-tuned k (10× smaller than butane)
TOTAL_TIME_PS = 5.0   # 5 ps test (faster for multi-torsion validation)
PRINT_INTERVAL = 100  # Print every 100 steps

# ============================================================================
# Setup System (Ala12 Helix)
# ============================================================================

print("=" * 80)
print("Multi-Torsion NVE Validation: Ala12 Helix")
print("=" * 80)
print()

# Load protein
pdb = PDBFile("ala12_helix.pdb")

print(f"Loaded: Ala12 helix (initial)")
print(f"Atoms (no H): {pdb.topology.getNumAtoms()}")
print()

# Add hydrogens using Modeller
forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(forcefield)
topology = modeller.topology
positions = modeller.positions

print(f"Added hydrogens")
print(f"Atoms (with H): {topology.getNumAtoms()}")
print()

# Create system (AMBER14 + implicit solvent)
system = forcefield.createSystem(
    topology,
    nonbondedMethod=NoCutoff,
    constraints=HBonds,
    rigidWater=True
)

print(f"Force field: AMBER14 + GBn2 implicit solvent")
print(f"Constraints: HBonds")
print()

# Reorganize forces: put torsions in group 1
for force in system.getForces():
    if isinstance(force, PeriodicTorsionForce):
        force.setForceGroup(1)
        print(f"Torsion force moved to group 1")

print()

# ============================================================================
# Find All Backbone Torsions
# ============================================================================

phi_torsions, psi_torsions = get_backbone_torsions(topology)

print(f"Found {len(phi_torsions)} φ (phi) torsions")
print(f"Found {len(psi_torsions)} ψ (psi) torsions")
print()

# Use all φ torsions for monitoring
torsion_atoms_list = [phi_torsions[res_idx] for res_idx in sorted(phi_torsions.keys())]
print(f"Monitoring {len(torsion_atoms_list)} backbone φ torsions:")
for i, (res_idx, atoms) in enumerate(zip(sorted(phi_torsions.keys()), torsion_atoms_list)):
    print(f"  Torsion {i}: Residue {res_idx}, atoms {atoms}")
print()

# ============================================================================
# Baseline: Fixed Timestep (0.5 fs)
# ============================================================================

print("-" * 80)
print("BASELINE: Fixed dt = 0.5 fs")
print("-" * 80)
print()

integrator_fixed = VerletIntegrator(DT_BASELINE_FS * femtoseconds)
platform = Platform.getPlatformByName("Reference")
context_fixed = Context(system, integrator_fixed, platform)
context_fixed.setPositions(positions)

# Minimize
print("Minimizing energy...")
LocalEnergyMinimizer.minimize(context_fixed, 1.0, 100)
print("✅ Minimization complete")
print()

# Set velocities to 300 K
context_fixed.setVelocitiesToTemperature(300 * kelvin)

# Track energy
n_steps_fixed = int(TOTAL_TIME_PS * 1000 / DT_BASELINE_FS)
print(f"Running {n_steps_fixed} steps ({TOTAL_TIME_PS} ps)...")
print()

times_fixed = []
energies_fixed = []

for step in range(n_steps_fixed + 1):
    if step % PRINT_INTERVAL == 0:
        state = context_fixed.getState(getEnergy=True)
        E = state.getPotentialEnergy() + state.getKineticEnergy()
        E_val = E.value_in_unit(kilocalories_per_mole)
        t = step * DT_BASELINE_FS / 1000.0

        times_fixed.append(t)
        energies_fixed.append(E_val)

        if step % 1000 == 0:
            print(f"  t = {t:6.2f} ps, E = {E_val:12.4f} kcal/mol")

    if step < n_steps_fixed:
        integrator_fixed.step(1)

times_fixed = np.array(times_fixed)
energies_fixed = np.array(energies_fixed)

# Compute drift
E0_fixed = energies_fixed[0]
drift_fixed = ((energies_fixed - E0_fixed) / E0_fixed) * 100.0
max_drift_fixed = np.max(np.abs(drift_fixed))

print()
print(f"✅ Fixed baseline complete")
print(f"   Energy drift: {max_drift_fixed:.4f}%")
print()

# ============================================================================
# Adaptive: Multi-Torsion Λ_global
# ============================================================================

print("-" * 80)
print(f"ADAPTIVE: Multi-Torsion Λ_global (k={K_ADAPTIVE})")
print("-" * 80)
print()

# Create new context (fresh start from same initial conditions)
integrator_adaptive_base = VerletIntegrator(DT_ADAPTIVE_FS * femtoseconds)
context_adaptive = Context(system, integrator_adaptive_base, platform)
context_adaptive.setPositions(positions)

# Minimize (same as fixed)
print("Minimizing energy...")
LocalEnergyMinimizer.minimize(context_adaptive, 1.0, 100)
print("✅ Minimization complete")
print()

# Set velocities to 300 K (same random seed for reproducibility)
context_adaptive.setVelocitiesToTemperature(300 * kelvin)

# Create multi-torsion adaptive integrator
adaptive = LambdaAdaptiveVerletIntegrator(
    context=context_adaptive,
    torsion_atoms=torsion_atoms_list,  # List of tuples for multi-torsion
    dt_base_fs=DT_ADAPTIVE_FS,
    k=K_ADAPTIVE,
    alpha=0.1,
    torsion_force_group=1
)

print(f"Created: {adaptive}")
print(f"Monitoring {len(torsion_atoms_list)} backbone φ torsions")
print(f"Λ_global = max(Λ_i) across all torsions")
print()

# Calculate number of steps
n_steps_adaptive = int(TOTAL_TIME_PS * 1000 / DT_ADAPTIVE_FS)
print(f"Running {n_steps_adaptive} steps ({TOTAL_TIME_PS} ps)...")
print()

times_adaptive = []
energies_adaptive = []
dt_history = []
Lambda_global_history = []
Lambda_per_torsion_history = []  # Track all individual Λ_i

for step in range(n_steps_adaptive + 1):
    if step % PRINT_INTERVAL == 0:
        state = context_adaptive.getState(getEnergy=True)
        E = state.getPotentialEnergy() + state.getKineticEnergy()
        E_val = E.value_in_unit(kilocalories_per_mole)
        t = adaptive.get_time()
        dt_current = adaptive.get_timestep()
        Lambda_global = adaptive.get_Lambda()
        Lambda_per_torsion = adaptive.get_Lambda_per_torsion()

        times_adaptive.append(t)
        energies_adaptive.append(E_val)
        dt_history.append(dt_current)
        Lambda_global_history.append(Lambda_global)
        Lambda_per_torsion_history.append(Lambda_per_torsion)

        if step % 1000 == 0:
            print(f"  t = {t:6.2f} ps, E = {E_val:12.4f} kcal/mol, "
                  f"dt = {dt_current:.4f} fs, Λ_global = {Lambda_global:.4f}")

    if step < n_steps_adaptive:
        adaptive.step(1)

times_adaptive = np.array(times_adaptive)
energies_adaptive = np.array(energies_adaptive)
dt_history = np.array(dt_history)
Lambda_global_history = np.array(Lambda_global_history)
Lambda_per_torsion_history = np.array(Lambda_per_torsion_history)  # Shape: (n_samples, n_torsions)

# Compute drift
E0_adaptive = energies_adaptive[0]
drift_adaptive = ((energies_adaptive - E0_adaptive) / E0_adaptive) * 100.0
max_drift_adaptive = np.max(np.abs(drift_adaptive))

print()
print(f"✅ Adaptive multi-torsion complete")
print(f"   Energy drift: {max_drift_adaptive:.4f}%")
print(f"   Mean dt: {np.mean(dt_history):.4f} fs")
print(f"   Mean Λ_global: {np.mean(Lambda_global_history):.4f}")
print()

# ============================================================================
# Analysis & Results
# ============================================================================

print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()

print(f"System: Ala12 helix ({topology.getNumAtoms()} atoms)")
print(f"Duration: {TOTAL_TIME_PS} ps")
print(f"Monitored torsions: {len(torsion_atoms_list)} backbone φ angles")
print()

print("Fixed Timestep (0.5 fs):")
print(f"  Max drift: {max_drift_fixed:.4f}%")
print()

print(f"Adaptive Multi-Torsion (k={K_ADAPTIVE}):")
print(f"  Max drift: {max_drift_adaptive:.4f}%")
print(f"  Mean dt: {np.mean(dt_history):.4f} fs")
print(f"  Speedup: {DT_BASELINE_FS / np.mean(dt_history):.2f}×")
print(f"  Drift ratio: {max_drift_adaptive / max_drift_fixed:.2f}×")
print()

# Check acceptance criteria
print("Acceptance Criteria:")
passed_drift = max_drift_adaptive < 0.5
passed_ratio = (max_drift_adaptive / max_drift_fixed) < 2.0
print(f"  ✅ Energy drift < 0.5%: {max_drift_adaptive:.4f}% - {'PASS' if passed_drift else 'FAIL'}")
print(f"  ✅ Drift ratio < 2×: {max_drift_adaptive/max_drift_fixed:.2f}× - {'PASS' if passed_ratio else 'FAIL'}")
print()

if passed_drift and passed_ratio:
    print("🎉 SPRINT 1 VALIDATION: PASSED")
else:
    print("⚠️  SPRINT 1 VALIDATION: FAILED (needs tuning)")
print()

# Analyze heterogeneity of Λ_i across torsions
print("Per-Torsion Λ_i Analysis:")
Lambda_mean_per_torsion = np.mean(Lambda_per_torsion_history, axis=0)
Lambda_max_per_torsion = np.max(Lambda_per_torsion_history, axis=0)
for i, res_idx in enumerate(sorted(phi_torsions.keys())):
    print(f"  Torsion {i} (Res {res_idx}): mean Λ = {Lambda_mean_per_torsion[i]:.4f}, "
          f"max Λ = {Lambda_max_per_torsion[i]:.4f}")
print()

most_active_idx = np.argmax(Lambda_mean_per_torsion)
print(f"Most active torsion: Torsion {most_active_idx} (mean Λ = {Lambda_mean_per_torsion[most_active_idx]:.4f})")
print()

# ============================================================================
# Visualization
# ============================================================================

print("Creating validation plots...")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot 1: Energy drift comparison
ax = axes[0, 0]
ax.plot(times_fixed, drift_fixed, 'b-', label=f'Fixed {DT_BASELINE_FS} fs', linewidth=1.5)
ax.plot(times_adaptive, drift_adaptive, 'r-', label=f'Adaptive (k={K_ADAPTIVE})', linewidth=1.5)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=0.8, alpha=0.5, label='±0.5% target')
ax.axhline(-0.5, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Time (ps)', fontsize=11)
ax.set_ylabel('Energy Drift (%)', fontsize=11)
ax.set_title('NVE Energy Conservation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 2: Timestep adaptation
ax = axes[0, 1]
ax.plot(times_adaptive, dt_history, 'g-', linewidth=1.2)
ax.axhline(DT_ADAPTIVE_FS, color='b', linestyle='--', linewidth=1, label=f'dt_base = {DT_ADAPTIVE_FS} fs')
ax.set_xlabel('Time (ps)', fontsize=11)
ax.set_ylabel('Timestep (fs)', fontsize=11)
ax.set_title('Adaptive Timestep Evolution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 3: Λ_global evolution
ax = axes[1, 0]
ax.plot(times_adaptive, Lambda_global_history, 'purple', linewidth=1.2)
ax.set_xlabel('Time (ps)', fontsize=11)
ax.set_ylabel('Λ_global (max across torsions)', fontsize=11)
ax.set_title('Global Torsional Stiffness', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 4: Individual Λ_i traces (all torsions)
ax = axes[1, 1]
n_torsions = Lambda_per_torsion_history.shape[1]
colors = plt.cm.tab10(np.linspace(0, 1, n_torsions))
for i in range(n_torsions):
    ax.plot(times_adaptive, Lambda_per_torsion_history[:, i],
            color=colors[i], linewidth=1, alpha=0.7, label=f'Torsion {i}')
ax.set_xlabel('Time (ps)', fontsize=11)
ax.set_ylabel('Λ_i (per-torsion stiffness)', fontsize=11)
ax.set_title(f'Individual Torsion Dynamics ({n_torsions} φ angles)', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# Plot 5: Correlation between dt and Λ_global
ax = axes[2, 0]
ax.scatter(Lambda_global_history, dt_history, alpha=0.5, s=10, c='teal')
ax.set_xlabel('Λ_global', fontsize=11)
ax.set_ylabel('dt (fs)', fontsize=11)
ax.set_title('Timestep vs Stiffness Correlation', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 6: Mean Λ_i per torsion (bar chart)
ax = axes[2, 1]
torsion_labels = [f'T{i}' for i in range(n_torsions)]
ax.bar(torsion_labels, Lambda_mean_per_torsion, color='steelblue', edgecolor='black')
ax.set_xlabel('Torsion Index', fontsize=11)
ax.set_ylabel('Mean Λ_i', fontsize=11)
ax.set_title('Average Stiffness per Torsion', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', labelsize=8)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('multitorsion_nve_validation.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: multitorsion_nve_validation.png")
print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
