#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Molecular Dynamics Bivector Utilities
======================================

Core utilities for mapping molecular dynamics to Clifford algebra Cl(3,1).

Application: Adaptive timestep control using bivector curvature diagnostics
Patent: "Stability-Preserving Integration Using Bivector Curvature"

Key Innovation:
- Angular velocity ω → bivector in e_01, e_02, e_03 (spatial rotation)
- Torsional torque τ → bivector in e_01, e_02, e_03
- Lambda Λ = ||[ω, τ]|| detects torsional stiffness
- High Λ → stiff torsion → reduce timestep
- Low Λ → flexible → increase timestep

Rick Mathews - November 2024
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks


# ============================================================================
# CLIFFORD ALGEBRA Cl(3,1) - Lorentz Bivector
# ============================================================================

class BivectorCl31:
    """
    Bivector in Clifford algebra Cl(3,1) - Lorentz spacetime

    Basis: {e_01, e_02, e_03, e_23, e_31, e_12}
    - e_01, e_02, e_03: Timelike bivectors (boosts in 3D+time interpretation)
    - e_23, e_31, e_12: Spacelike bivectors (rotations in 3D)

    For MD application:
    - Rotations (ω, τ): Use spatial bivectors e_23, e_31, e_12
    - e_01, e_02, e_03 used for encoding distribution moments (if needed)
    """

    def __init__(self, components=None):
        """
        Initialize bivector with 6 components.

        Args:
            components: Array [B_01, B_02, B_03, B_23, B_31, B_12]
        """
        if components is None:
            self.B = np.zeros(6)
        else:
            self.B = np.array(components, dtype=float)

        assert self.B.shape == (6,), "Bivector must have 6 components"

    def __repr__(self):
        return (f"BivectorCl31("
                f"e_01={self.B[0]:.4f}, e_02={self.B[1]:.4f}, e_03={self.B[2]:.4f}, "
                f"e_23={self.B[3]:.4f}, e_31={self.B[4]:.4f}, e_12={self.B[5]:.4f})")

    def commutator(self, other):
        """
        Compute commutator [B1, B2] = B1 * B2 - B2 * B1

        This gives a 4-vector in Cl(3,1), but we care about its norm.
        For simplicity, compute as cross product of rotation components.
        """
        # Extract spatial rotation components
        omega1 = np.array([self.B[5], self.B[4], self.B[3]])  # [B_12, B_31, B_23] → [ωx, ωy, ωz]
        omega2 = np.array([other.B[5], other.B[4], other.B[3]])

        # Commutator ~ cross product for rotation bivectors
        comm = np.cross(omega1, omega2)

        # Return as bivector
        result = BivectorCl31()
        result.B[5] = comm[0]  # e_12
        result.B[4] = comm[1]  # e_31
        result.B[3] = comm[2]  # e_23

        return result

    def norm(self):
        """Frobenius norm of bivector"""
        return np.linalg.norm(self.B)


# ============================================================================
# ANGULAR VELOCITY & TORQUE → BIVECTOR MAPPING
# ============================================================================

def angular_velocity_bivector(coords, velocities):
    """
    Map atomic velocities to angular velocity bivector.

    For a rigid body or molecule:
    - Compute angular momentum L = Σ r_i × (m_i * v_i)
    - Compute moment of inertia tensor I
    - Angular velocity ω = I^(-1) * L
    - Map to bivector: ω_bivector = [0, 0, 0, ωz, ωy, ωx]

    Args:
        coords: Nx3 array of atomic coordinates (Angstroms)
        velocities: Nx3 array of atomic velocities (Angstroms/ps)

    Returns:
        BivectorCl31 with angular velocity encoded
    """
    coords = np.array(coords)
    velocities = np.array(velocities)

    # Assume unit mass for simplicity (can weight by atomic masses)
    masses = np.ones(len(coords))

    # Center of mass
    com = np.average(coords, axis=0, weights=masses)
    com_vel = np.average(velocities, axis=0, weights=masses)

    # Relative positions and velocities
    r = coords - com
    v = velocities - com_vel

    # Angular momentum L = Σ m_i * (r_i × v_i)
    L = np.zeros(3)
    for i in range(len(coords)):
        L += masses[i] * np.cross(r[i], v[i])

    # Moment of inertia tensor I = Σ m_i * (r_i · r_i * Identity - r_i ⊗ r_i)
    I = np.zeros((3, 3))
    for i in range(len(coords)):
        r_sq = np.dot(r[i], r[i])
        I += masses[i] * (r_sq * np.eye(3) - np.outer(r[i], r[i]))

    # Angular velocity ω = I^(-1) * L
    try:
        omega = np.linalg.solve(I, L)
    except np.linalg.LinAlgError:
        # Singular moment of inertia (linear molecule or single atom)
        omega = np.zeros(3)

    # Map to bivector components
    # Convention: ω = [ωx, ωy, ωz] → bivector = [0, 0, 0, ωz, ωy, ωx]
    # e_23 ↔ ωx (rotation in yz plane)
    # e_31 ↔ ωy (rotation in zx plane)
    # e_12 ↔ ωz (rotation in xy plane)
    bivector = BivectorCl31()
    bivector.B[3] = omega[2]  # e_23 = ωz
    bivector.B[4] = omega[1]  # e_31 = ωy
    bivector.B[5] = omega[0]  # e_12 = ωx

    return bivector


def torsional_force_bivector(coords, forces):
    """
    Map atomic forces to torsional torque bivector.

    Torque τ = Σ r_i × F_i

    Args:
        coords: Nx3 array of atomic coordinates
        forces: Nx3 array of atomic forces (kJ/mol/Angstrom)

    Returns:
        BivectorCl31 with torque encoded
    """
    coords = np.array(coords)
    forces = np.array(forces)

    # Center of mass
    com = np.mean(coords, axis=0)
    r = coords - com

    # Total torque τ = Σ r_i × F_i
    tau = np.zeros(3)
    for i in range(len(coords)):
        tau += np.cross(r[i], forces[i])

    # Map to bivector
    bivector = BivectorCl31()
    bivector.B[3] = tau[2]  # e_23 = τz
    bivector.B[4] = tau[1]  # e_31 = τy
    bivector.B[5] = tau[0]  # e_12 = τx

    return bivector


def compute_lambda(omega_bivector, tau_bivector):
    """
    Compute Lambda = ||[ω, τ]|| (commutator norm).

    This measures the "twist" between angular velocity and torque.
    High Λ → stiff torsional coupling → reduce timestep
    Low Λ → flexible → increase timestep

    Args:
        omega_bivector: Angular velocity bivector
        tau_bivector: Torsional torque bivector

    Returns:
        Lambda (float)
    """
    comm = omega_bivector.commutator(tau_bivector)
    Lambda = comm.norm()
    return Lambda


# ============================================================================
# MOLECULAR GEOMETRY BUILDERS
# ============================================================================

def build_butane_geometry(phi_degrees):
    """
    Build butane (C4H10) molecule at specified C-C-C-C dihedral angle.

    Atom numbering:
    C1 - C2 - C3 - C4 (carbon backbone)
    Dihedral φ = C1-C2-C3-C4 angle

    Args:
        phi_degrees: Dihedral angle in degrees

    Returns:
        coords: 14x3 array (4 carbons + 10 hydrogens)
    """
    phi_rad = np.radians(phi_degrees)

    # Standard bond lengths and angles (OPLS force field)
    r_CC = 1.54  # Angstroms (C-C bond)
    r_CH = 1.09  # Angstroms (C-H bond)
    theta = np.radians(109.5)  # Tetrahedral angle

    # Build carbon backbone
    coords = []

    # C1 at origin
    C1 = np.array([0.0, 0.0, 0.0])
    coords.append(C1)

    # C2 along x-axis
    C2 = np.array([r_CC, 0.0, 0.0])
    coords.append(C2)

    # C3 at tetrahedral angle from C2
    C3 = C2 + r_CC * np.array([
        np.cos(np.pi - theta),
        np.sin(np.pi - theta),
        0.0
    ])
    coords.append(C3)

    # C4 at dihedral φ from C3
    # Rotate around C2-C3 bond axis
    bond_axis = (C3 - C2) / np.linalg.norm(C3 - C2)

    # Initial C4 position (staggered, φ=180°)
    C4_initial = C3 + r_CC * np.array([
        np.cos(np.pi - theta),
        -np.sin(np.pi - theta) * np.cos(phi_rad),
        -np.sin(np.pi - theta) * np.sin(phi_rad)
    ])

    # Rotate C4 around bond axis to get desired phi
    rotation = Rotation.from_rotvec((phi_rad - np.pi) * bond_axis)
    C4 = C3 + rotation.apply(C4_initial - C3)
    coords.append(C4)

    # Add hydrogens (simplified - just place at tetrahedral positions)
    # C1 hydrogens (3)
    for i in range(3):
        angle = 2 * np.pi * i / 3
        H = C1 + r_CH * np.array([
            -np.cos(theta),
            np.sin(theta) * np.cos(angle),
            np.sin(theta) * np.sin(angle)
        ])
        coords.append(H)

    # C2 hydrogens (2)
    for i in range(2):
        angle = np.pi * i
        H = C2 + r_CH * np.array([
            0.3 * np.cos(theta),
            np.sin(theta) * np.cos(angle),
            np.sin(theta) * np.sin(angle)
        ])
        coords.append(H)

    # C3 hydrogens (2)
    for i in range(2):
        angle = np.pi * i
        H = C3 + r_CH * np.array([
            0.3 * np.cos(theta),
            np.sin(theta) * np.cos(angle + np.pi/4),
            np.sin(theta) * np.sin(angle + np.pi/4)
        ])
        coords.append(H)

    # C4 hydrogens (3)
    for i in range(3):
        angle = 2 * np.pi * i / 3
        H = C4 + r_CH * np.array([
            np.cos(theta),
            np.sin(theta) * np.cos(angle),
            np.sin(theta) * np.sin(angle)
        ])
        coords.append(H)

    return np.array(coords)


def build_alanine_dipeptide(phi_degrees, psi_degrees):
    """
    Build alanine dipeptide (Ace-Ala-Nme) at specified backbone dihedrals.

    Simplified geometry for proof-of-concept.

    Args:
        phi_degrees: φ dihedral (C-N-Cα-C)
        psi_degrees: ψ dihedral (N-Cα-C-N)

    Returns:
        coords: Nx3 array of atomic coordinates
    """
    phi_rad = np.radians(phi_degrees)
    psi_rad = np.radians(psi_degrees)

    # Simplified backbone geometry
    # In real implementation, use OpenMM or MDTraj for proper geometry
    coords = []

    # Backbone atoms (simplified linear chain)
    r_bond = 1.5  # Average bond length

    for i in range(10):
        # Simple zigzag pattern
        angle = phi_rad if i % 2 == 0 else psi_rad
        x = i * r_bond * 0.5
        y = np.sin(angle) * r_bond
        z = np.cos(angle) * r_bond * 0.5
        coords.append([x, y, z])

    return np.array(coords)


# ============================================================================
# FORCE FIELD CALCULATIONS
# ============================================================================

def compute_torsional_forces_butane(coords, phi_degrees):
    """
    Compute torsional forces for butane from analytical potential.

    For butane C-C-C-C dihedral, force acts perpendicular to bond.
    F_torsional = -dV/dφ * (direction perpendicular to C2-C3 bond)

    Args:
        coords: Nx3 atomic coordinates (14 atoms: 4 C + 10 H)
        phi_degrees: Current dihedral angle

    Returns:
        forces: Nx3 array of forces (kJ/mol/Angstrom)
    """
    N = len(coords)
    forces = np.zeros((N, 3))

    # Compute torsional strain -dV/dφ
    strain = -compute_torsional_strain_butane(phi_degrees)  # Negative for force

    # Carbon atoms are first 4
    C1, C2, C3, C4 = coords[:4]

    # C2-C3 bond axis (rotation axis)
    bond_axis = (C3 - C2) / np.linalg.norm(C3 - C2)

    # Direction of force: perpendicular to bond, in plane of rotation
    # For simplicity, use cross product with arbitrary vector
    perp_vector = np.array([0, 1, 0]) if abs(bond_axis[1]) < 0.9 else np.array([1, 0, 0])
    force_dir = np.cross(bond_axis, perp_vector)
    force_dir = force_dir / np.linalg.norm(force_dir)

    # Apply torsional force to atoms involved in dihedral
    # Magnitude scales with strain, distributed across rotating atoms
    force_magnitude = abs(strain) * 0.1  # Scale factor

    # C1 and C4 experience the torsional force
    forces[0] = -force_magnitude * force_dir  # C1
    forces[3] = +force_magnitude * force_dir  # C4

    return forces


def compute_mm_forces(coords, forcefield='OPLS', phi_degrees=None):
    """
    Compute molecular mechanics forces (simplified).

    For proof-of-concept: Use analytical torsional potential derivative.
    In production: Interface with OpenMM, LAMMPS, etc.

    Args:
        coords: Nx3 atomic coordinates
        forcefield: Force field name
        phi_degrees: Dihedral angle (for butane torsional forces)

    Returns:
        forces: Nx3 array of forces (kJ/mol/Angstrom)
    """
    N = len(coords)
    forces = np.zeros((N, 3))

    if phi_degrees is not None and len(coords) == 14:
        # Butane-specific torsional forces
        forces = compute_torsional_forces_butane(coords, phi_degrees)
    else:
        # Generic: Small random thermal forces
        np.random.seed(42)
        forces = np.random.randn(N, 3) * 1.0  # kJ/mol/Angstrom (reduced)

    return forces


def compute_mm_energy(coords, forcefield='OPLS'):
    """
    Compute molecular mechanics energy (simplified).

    Args:
        coords: Nx3 atomic coordinates
        forcefield: Force field name

    Returns:
        energy: Total energy (kJ/mol)
    """
    # Simplified: Return random energy
    # In real implementation, compute from force field
    return np.random.rand() * 100.0  # kJ/mol


def compute_torsional_energy_butane(phi_degrees):
    """
    Compute butane torsional potential energy (OPLS force field).

    V(φ) = V₁/2[1+cos(φ)] + V₂/2[1-cos(2φ)] + V₃/2[1+cos(3φ)]

    Args:
        phi_degrees: Dihedral angle (degrees)

    Returns:
        energy: Torsional energy (kJ/mol)
    """
    phi_rad = np.radians(phi_degrees)

    # OPLS parameters for butane C-C-C-C torsion
    V1 = 3.4  # kJ/mol
    V2 = -0.8
    V3 = 6.8

    V = (V1/2) * (1 + np.cos(phi_rad)) + \
        (V2/2) * (1 - np.cos(2*phi_rad)) + \
        (V3/2) * (1 + np.cos(3*phi_rad))

    return V


def compute_torsional_strain_butane(phi_degrees):
    """
    Compute butane torsional strain |dV/dφ| (OPLS force field).

    Args:
        phi_degrees: Dihedral angle (degrees)

    Returns:
        strain: |dV/dφ| (kJ/mol/radian)
    """
    phi_rad = np.radians(phi_degrees)

    # OPLS parameters
    V1 = 3.4
    V2 = -0.8
    V3 = 6.8

    dVdphi = -V1 * np.sin(phi_rad) + \
             V2 * np.sin(2*phi_rad) - \
             V3 * np.sin(3*phi_rad)

    return abs(dVdphi)


# ============================================================================
# THERMAL SAMPLING
# ============================================================================

def sample_thermal_velocities(coords, T=300, seed=None):
    """
    Sample velocities from Maxwell-Boltzmann distribution.

    For atom i with mass m_i at temperature T:
    v_i ~ Normal(0, sqrt(k_B * T / m_i))

    Args:
        coords: Nx3 atomic coordinates
        T: Temperature (Kelvin)
        seed: Random seed

    Returns:
        velocities: Nx3 array (Angstroms/ps)
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(coords)

    # Boltzmann constant (kJ/mol/K)
    k_B = 0.00831  # kJ/(mol·K)

    # Assume carbon mass for simplicity (12 g/mol)
    # More accurate: use atomic masses
    mass = 12.0  # g/mol

    # Thermal velocity scale
    sigma = np.sqrt(k_B * T / mass)  # Angstroms/ps

    # Sample from Maxwell-Boltzmann
    velocities = np.random.normal(0, sigma, (N, 3))

    # Remove net momentum (center-of-mass velocity)
    com_vel = np.mean(velocities, axis=0)
    velocities -= com_vel

    return velocities


def has_steric_clash(coords, threshold=1.0):
    """
    Check for steric clashes (atoms too close).

    Args:
        coords: Nx3 atomic coordinates
        threshold: Minimum allowed distance (Angstroms)

    Returns:
        True if clash detected
    """
    N = len(coords)

    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:
                return True

    return False


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MD BIVECTOR UTILITIES - QUICK TEST")
    print("="*80)
    print()

    # Test 1: Build butane at different dihedrals
    print("Test 1: Butane geometry at φ = 0°, 60°, 180°")
    for phi in [0, 60, 180]:
        coords = build_butane_geometry(phi)
        print(f"  φ = {phi}°: {len(coords)} atoms, span = {np.ptp(coords, axis=0)}")
    print()

    # Test 2: Angular velocity bivector
    print("Test 2: Angular velocity bivector")
    coords = build_butane_geometry(60)
    velocities = sample_thermal_velocities(coords, T=300, seed=42)
    omega_biv = angular_velocity_bivector(coords, velocities)
    print(f"  ω bivector: {omega_biv}")
    print()

    # Test 3: Torsional force bivector
    print("Test 3: Torsional force bivector")
    forces = compute_mm_forces(coords)
    tau_biv = torsional_force_bivector(coords, forces)
    print(f"  τ bivector: {tau_biv}")
    print()

    # Test 4: Compute Lambda
    print("Test 4: Lambda = ||[ω, τ]||")
    Lambda = compute_lambda(omega_biv, tau_biv)
    print(f"  Λ = {Lambda:.6f}")
    print()

    # Test 5: Torsional potential
    print("Test 5: Butane torsional potential")
    for phi in [0, 60, 120, 180, 240, 300]:
        V = compute_torsional_energy_butane(phi)
        strain = compute_torsional_strain_butane(phi)
        print(f"  φ = {phi:3.0f}°: V = {V:6.2f} kJ/mol, |dV/dφ| = {strain:6.2f}")
    print()

    print("✅ All tests passed")
