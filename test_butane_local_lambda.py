#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Local torsion-aware Lambda for MD
=========================================

System: Butane in vacuum (or light solvent)
Goal : Test whether a *local*, bond-framed diagnostic

    Lambda_i(t) = | phi_dot_i(t) * tau_i(t) |

tracks torsional stiffness and barrier crossings.

This script does:

  1. Builds a butane system from a PDB + force field
  2. Runs a short MD trajectory at 300K
  3. For a chosen dihedral (C1-C2-C3-C4):
       - computes phi(t)
       - computes torsional torque tau(t) about the C2-C3 bond
       - computes phi_dot(t) by finite difference
       - defines Lambda(t) = |phi_dot * tau|
  4. Produces:
       - time series plots (phi, tau, Lambda)
       - scatter plots Lambda vs |tau| and Lambda vs |phi_ddot|
       - R^2 correlations

Static scan hook:
  There is a placeholder function `static_dihedral_scan()` which you can
  complete once you have a rotate_dihedral() utility – the Lambda logic
  doesn't change.

Rick Mathews - MD Patent Stage-1 Validation
November 14, 2024
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from openmm import unit, openmm
    from openmm import app
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    print("[ERROR] OpenMM not installed!")
    print("Install with: conda install -c conda-forge openmm")

# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------

PDB_PATH = "butane.pdb"  # provide a standard butane PDB here
FORCEFIELD_FILES = ["amber14-all.xml", "amber14/tip3p.xml"]  # or GAFF/your choice
N_STEPS = 5000
DT_FS = 2.0
TEMPERATURE = 300.0  # K

# Dihedral atom indices (0-based, matching PDB/topology ordering):
# C1-C2-C3-C4 for butane; adjust as needed once you inspect your topology.
TORSION_NAME = "chi"
TORSION_ATOMS = (0, 4, 6, 10)  # Adjusted for standard butane PDB with hydrogens

# ---------------------------------------------------------------------
# GEOMETRIC UTILITIES
# ---------------------------------------------------------------------

def dihedral_angle(positions, i1, i2, i3, i4):
    """
    Standard robust dihedral angle in radians.

    positions: (N,3) numpy array in *any* length units.
    """
    p1, p2, p3, p4 = positions[[i1, i2, i3, i4]]

    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-12 or n2_norm < 1e-12:
        return 0.0

    n1 /= n1_norm
    n2 /= n2_norm
    b2_hat = b2 / np.linalg.norm(b2)

    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), b2_hat)
    return np.arctan2(y, x)


def torsion_torque_about_bond(positions, forces, i1, i2, i3, i4):
    """
    Scalar torsional torque about the i2-i3 bond axis (in kJ/mol).

    positions: (N,3) [nm]
    forces   : (N,3) [kJ/(mol*nm)]

    We project atom-relative position onto the plane perpendicular to the
    bond axis, then take r_perp x f and project that torque onto the bond
    axis. Sum over the four torsion atoms.
    """
    a, b, c, d = i1, i2, i3, i4
    r = positions
    f = forces

    # bond axis and midpoint
    b_vec = r[c] - r[b]
    b_norm = np.linalg.norm(b_vec)
    if b_norm < 1e-12:
        return 0.0
    b_hat = b_vec / b_norm
    r_mid = 0.5 * (r[b] + r[c])

    torque = 0.0
    for idx in (a, b, c, d):
        r_rel = r[idx] - r_mid
        r_perp = r_rel - np.dot(r_rel, b_hat) * b_hat
        tau_vec = np.cross(r_perp, f[idx])
        torque += np.dot(tau_vec, b_hat)

    # units: (nm × kJ/mol/nm) = kJ/mol
    return torque


# ---------------------------------------------------------------------
# SYSTEM SETUP
# ---------------------------------------------------------------------

def build_butane_system():
    """Build butane system with OpenMM using direct force definitions"""
    if not OPENMM_AVAILABLE:
        raise ImportError("OpenMM required for MD simulation")

    # Create system
    system = openmm.System()

    # Add atoms (masses in amu)
    # 4 carbons + 10 hydrogens
    atom_masses = [
        12.01,  # C1 (0)
        1.008, 1.008, 1.008,  # H11, H12, H13 (1-3)
        12.01,  # C2 (4)
        1.008,  # H21 (5)
        12.01,  # C3 (6)
        1.008, 1.008,  # H31, H32 (7-8)
        1.008,  # H22 (9)
        12.01,  # C4 (10)
        1.008, 1.008, 1.008   # H41, H42, H43 (11-13)
    ]

    for mass in atom_masses:
        system.addParticle(mass * unit.amu)

    # Add harmonic bonds (k in kJ/mol/nm^2, r0 in nm)
    bond_force = openmm.HarmonicBondForce()
    bonds = [
        (0, 1, 340000, 0.109),  # C1-H11
        (0, 2, 340000, 0.109),  # C1-H12
        (0, 3, 340000, 0.109),  # C1-H13
        (0, 4, 260000, 0.154),  # C1-C2
        (4, 5, 340000, 0.109),  # C2-H21
        (4, 6, 260000, 0.154),  # C2-C3
        (4, 9, 340000, 0.109),  # C2-H22
        (6, 7, 340000, 0.109),  # C3-H31
        (6, 8, 340000, 0.109),  # C3-H32
        (6, 10, 260000, 0.154), # C3-C4
        (10, 11, 340000, 0.109),# C4-H41
        (10, 12, 340000, 0.109),# C4-H42
        (10, 13, 340000, 0.109),# C4-H43
    ]
    for i, j, k, r0 in bonds:
        bond_force.addBond(i, j, r0 * unit.nanometer, k * unit.kilojoule_per_mole / (unit.nanometer**2))
    system.addForce(bond_force)

    # Add harmonic angles (k in kJ/mol/rad^2, theta0 in radians)
    angle_force = openmm.HarmonicAngleForce()
    angles = [
        # C-C-C backbone
        (0, 4, 6, 520, np.radians(112.7)),
        (4, 6, 10, 520, np.radians(112.7)),
        # H-C-C
        (1, 0, 4, 330, np.radians(110.7)),
        (2, 0, 4, 330, np.radians(110.7)),
        (3, 0, 4, 330, np.radians(110.7)),
        (5, 4, 6, 330, np.radians(110.7)),
        (9, 4, 6, 330, np.radians(110.7)),
        (7, 6, 10, 330, np.radians(110.7)),
        (8, 6, 10, 330, np.radians(110.7)),
        # H-C-H
        (1, 0, 2, 280, np.radians(107.8)),
        (1, 0, 3, 280, np.radians(107.8)),
        (2, 0, 3, 280, np.radians(107.8)),
        (7, 6, 8, 280, np.radians(107.8)),
        (11, 10, 12, 280, np.radians(107.8)),
        (11, 10, 13, 280, np.radians(107.8)),
        (12, 10, 13, 280, np.radians(107.8)),
    ]
    for i, j, k, kforce, theta0 in angles:
        angle_force.addAngle(i, j, k, theta0 * unit.radian, kforce * unit.kilojoule_per_mole / (unit.radian**2))
    system.addForce(angle_force)

    # Add torsional potential (OPLS-style)
    # V(phi) = sum_n [k_n/2 * (1 + cos(n*phi - phi0))]
    torsion_force = openmm.PeriodicTorsionForce()
    # C-C-C-C dihedral (0-4-6-10)
    # OPLS parameters for alkane C-C-C-C
    torsion_force.addTorsion(0, 4, 6, 10, 3, 0.0, 5.4 * unit.kilojoule_per_mole)  # cos(3phi)
    torsion_force.addTorsion(0, 4, 6, 10, 2, np.pi, 1.3 * unit.kilojoule_per_mole)  # -cos(2phi)
    torsion_force.addTorsion(0, 4, 6, 10, 1, 0.0, 2.5 * unit.kilojoule_per_mole)  # cos(phi)
    system.addForce(torsion_force)

    # Initial positions (all-trans configuration)
    positions = [
        [0.0, 0.0, 0.0],      # C1 (0)
        [-0.06, 0.08, 0.0],   # H11 (1)
        [-0.06, -0.04, -0.08],# H12 (2)
        [-0.06, -0.04, 0.08], # H13 (3)
        [0.154, 0.0, 0.0],    # C2 (4)
        [0.19, 0.103, 0.0],   # H21 (5)
        [0.208, -0.077, 0.12],# C3 (6)
        [0.172, -0.18, 0.12], # H31 (7)
        [0.19, -0.04, 0.22],  # H32 (8)
        [0.19, -0.054, -0.09],# H22 (9)
        [0.362, -0.077, 0.12],# C4 (10)
        [0.40, -0.131, 0.207],# H41 (11)
        [0.40, -0.131, 0.033],# H42 (12)
        [0.40, 0.026, 0.12],  # H43 (13)
    ]

    # Simple Langevin dynamics
    integrator = openmm.LangevinIntegrator(
        TEMPERATURE * unit.kelvin,
        1.0 / unit.picosecond,
        DT_FS * unit.femtosecond,
    )

    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(app.Topology(), system, integrator, platform)
    simulation.context.setPositions([p * unit.nanometer for p in positions])

    # quick minimization to relax bad contacts
    print("Minimizing energy...")
    openmm.LocalEnergyMinimizer.minimize(simulation.context, tolerance=10.0, maxIterations=500)

    # initialize velocities
    simulation.context.setVelocitiesToTemperature(TEMPERATURE * unit.kelvin)

    return simulation, system


# ---------------------------------------------------------------------
# MD TRAJECTORY: LOCAL LAMBDA
# ---------------------------------------------------------------------

def run_md_local_lambda():
    """Run MD and compute local torsion Lambda"""
    simulation, system = build_butane_system()

    n_steps = N_STEPS
    dt_ps = DT_FS * 1e-3  # 2 fs → 0.002 ps

    phi_list = []
    tau_list = []
    time_list = []

    print(f"\nRunning {n_steps} MD steps at {TEMPERATURE}K...")
    print(f"Timestep: {DT_FS} fs ({dt_ps} ps)")
    print(f"Total time: {n_steps * dt_ps:.2f} ps\n")

    # MAIN LOOP
    for step in range(n_steps):
        if step % 500 == 0:
            print(f"Step {step}/{n_steps}")

        state = simulation.context.getState(
            getPositions=True,
            getForces=True,
            enforcePeriodicBox=False,
        )
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        frc = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

        phi = dihedral_angle(pos, *TORSION_ATOMS)
        tau = torsion_torque_about_bond(pos, frc, *TORSION_ATOMS)

        phi_list.append(phi)
        tau_list.append(tau)
        time_list.append(step * dt_ps)

        simulation.step(1)

    phi_arr = np.array(phi_list)
    tau_arr = np.array(tau_list)
    t_arr = np.array(time_list)

    # unwrap angles to avoid ±π jumps
    phi_unwrapped = np.unwrap(phi_arr)

    # central finite difference for phi_dot (rad/ps)
    phi_dot = np.gradient(phi_unwrapped, dt_ps)

    # second derivative (for stiffness / curvature proxy)
    phi_ddot = np.gradient(phi_dot, dt_ps)

    # local torsion-aware Lambda (power-like)
    Lambda = np.abs(phi_dot * tau_arr)  # kJ/mol/ps

    results = {
        "time_ps": t_arr,
        "phi": phi_arr,
        "phi_unwrapped": phi_unwrapped,
        "phi_dot": phi_dot,
        "phi_ddot": phi_ddot,
        "tau": tau_arr,
        "Lambda": Lambda,
    }
    return results


# ---------------------------------------------------------------------
# ANALYSIS / PLOTS
# ---------------------------------------------------------------------

def r2_score(x, y):
    """Calculate R² correlation coefficient"""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2:
        return np.nan
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if den < 1e-20:
        return 0.0
    r = num / den
    return r * r


def analyze_and_plot(results, prefix="butane_local_lambda"):
    """Analyze results and generate plots"""
    t = results["time_ps"]
    phi = results["phi"]
    phi_dot = results["phi_dot"]
    phi_ddot = results["phi_ddot"]
    tau = results["tau"]
    Lambda = results["Lambda"]

    # basic sanity
    print("\n" + "="*80)
    print("STAGE-1 MD RESULTS: Local Torsion Lambda")
    print("="*80)
    print(f"\nphi range (deg): [{np.rad2deg(phi).min():.1f}, {np.rad2deg(phi).max():.1f}]")
    print(f"tau range (kJ/mol): [{tau.min():.3e}, {tau.max():.3e}]")
    print(f"Lambda range (kJ/mol/ps): [{Lambda.min():.3e}, {Lambda.max():.3e}]")

    # correlations
    r2_L_vs_abs_tau = r2_score(Lambda, np.abs(tau))
    r2_L_vs_abs_ddot = r2_score(Lambda, np.abs(phi_ddot))

    print("\n=== Correlations (Stage 1 MD) ===")
    print(f"R²[ Lambda vs |tau| ]      = {r2_L_vs_abs_tau:.4f}")
    print(f"R²[ Lambda vs |phi_ddot| ] = {r2_L_vs_abs_ddot:.4f}")

    # Success criteria
    print("\n=== Stage-1 Success Criteria ===")
    print(f"Target: R²[Lambda, |tau|] ≥ 0.5")
    if r2_L_vs_abs_tau >= 0.5:
        print(f"✓ PASS: R² = {r2_L_vs_abs_tau:.4f}")
    else:
        print(f"✗ FAIL: R² = {r2_L_vs_abs_tau:.4f}")

    # Figure 1: time series
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(t, np.rad2deg(phi))
    axs[0].set_ylabel("phi (deg)")
    axs[0].set_title("Butane torsion angle")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, tau)
    axs[1].set_ylabel("tau (kJ/mol)")
    axs[1].set_title("Torsional torque about bond axis")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t, Lambda)
    axs[2].set_ylabel("Lambda (kJ/mol/ps)")
    axs[2].set_xlabel("time (ps)")
    axs[2].set_title("Local torsion-aware Lambda")
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{prefix}_timeseries.png", dpi=300)
    print(f"\nSaved: {prefix}_timeseries.png")

    # Figure 2: scatter plots for correlations
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))

    axs2[0].scatter(np.abs(tau), Lambda, s=4, alpha=0.4)
    axs2[0].set_xlabel("|tau| (kJ/mol)")
    axs2[0].set_ylabel("Lambda")
    axs2[0].set_title(f"Lambda vs |tau| (R²={r2_L_vs_abs_tau:.3f})")
    axs2[0].grid(True, alpha=0.3)

    axs2[1].scatter(np.abs(phi_ddot), Lambda, s=4, alpha=0.4)
    axs2[1].set_xlabel("|phi_ddot| (rad/ps²)")
    axs2[1].set_ylabel("Lambda")
    axs2[1].set_title(f"Lambda vs |phi_ddot| (R²={r2_L_vs_abs_ddot:.3f})")
    axs2[1].grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(f"{prefix}_correlations.png", dpi=300)
    print(f"Saved: {prefix}_correlations.png")

    # Histogram of Lambda
    plt.figure(figsize=(6,4))
    plt.hist(Lambda, bins=50, density=True, alpha=0.7, edgecolor='black')
    plt.xlabel("Lambda (kJ/mol/ps)")
    plt.ylabel("PDF")
    plt.title("Distribution of local torsion Lambda")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_lambda_hist.png", dpi=300)
    print(f"Saved: {prefix}_lambda_hist.png")

    print("\n" + "="*80)


# ---------------------------------------------------------------------
# OPTIONAL: STATIC DIHEDRAL SCAN HOOK
# ---------------------------------------------------------------------

def static_dihedral_scan():
    """
    Placeholder for the static scan:

      - Loop phi_target in np.linspace(-pi, pi, N)
      - For each:
          * rotate atoms on one side of the bond to set dihedral=phi_target
          * call context.setPositions(...)
          * get potential energy and forces
          * compute tau(phi_target) with torsion_torque_about_bond()
          * define a mock phi_dot (e.g. constant or from small finite diff)
          * compute Lambda(phi_target) = |phi_dot * tau|

      - Plot V(phi), tau(phi), Lambda(phi)
      - Compute R²[Lambda(phi), V(phi)] or vs |d²V/dphi²|.

    Once you add a rotate_dihedral() helper, the Lambda logic here will be
    identical to the MD case.
    """
    print("\n[INFO] Static dihedral scan not yet implemented")
    print("       Implement rotate_dihedral() to enable this test")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    """Main function"""
    print("\n" + "="*80)
    print("STAGE-1: LOCAL TORSION-AWARE LAMBDA FOR MD")
    print("="*80)
    print("\nSystem: Butane")
    print(f"Torsion: atoms {TORSION_ATOMS}")
    print(f"Temperature: {TEMPERATURE} K")
    print(f"Steps: {N_STEPS}")
    print(f"Timestep: {DT_FS} fs")

    if not OPENMM_AVAILABLE:
        print("\n[ERROR] OpenMM not available!")
        print("Install with: conda install -c conda-forge openmm")
        return

    try:
        results = run_md_local_lambda()
        analyze_and_plot(results)
        print("\n[SUCCESS] Stage-1 MD test completed!")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Make sure butane.pdb is in the current directory")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
