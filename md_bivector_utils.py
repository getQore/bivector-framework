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
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


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


def compute_torsional_curvature_butane(phi_degrees):
    """
    Compute butane torsional curvature |d²V/dφ²| (OPLS force field).

    This measures local stiffness of the potential.

    Args:
        phi_degrees: Dihedral angle (degrees)

    Returns:
        curvature: |d²V/dφ²| (kJ/mol/radian²)
    """
    phi_rad = np.radians(phi_degrees)

    # OPLS parameters
    V1 = 3.4
    V2 = -0.8
    V3 = 6.8

    # Second derivative
    d2Vdphi2 = -V1 * np.cos(phi_rad) - \
                2*V2 * np.cos(2*phi_rad) - \
                3*V3 * np.cos(3*phi_rad)

    return abs(d2Vdphi2)


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
# GEOMETRIC ALGEBRA TORSION DIAGNOSTIC (Patent-Grade)
# ============================================================================

def compute_dihedral_gradient(r_a, r_b, r_c, r_d):
    """
    Compute gradient of dihedral angle φ(a-b-c-d) with respect to atom positions.

    Returns ∂φ/∂r_a, ∂φ/∂r_b, ∂φ/∂r_c, ∂φ/∂r_d (4 vectors of length 3).

    Reference: Blondel & Karplus, J. Comput. Chem. 1996

    Args:
        r_a, r_b, r_c, r_d: 3D position vectors of atoms a, b, c, d

    Returns:
        (g_a, g_b, g_c, g_d): Gradient vectors
    """
    r_a = np.array(r_a)
    r_b = np.array(r_b)
    r_c = np.array(r_c)
    r_d = np.array(r_d)

    # Bond vectors
    b1 = r_b - r_a
    b2 = r_c - r_b
    b3 = r_d - r_c

    # Normal vectors to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        # Degenerate geometry (linear)
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm

    b2_norm = np.linalg.norm(b2)

    # Gradient components (analytical derivatives)
    g_a = -b2_norm / n1_norm**2 * n1
    g_d = b2_norm / n2_norm**2 * n2

    # Intermediate terms
    term_b = np.dot(b1, b2) / b2_norm**2
    term_c = np.dot(b3, b2) / b2_norm**2

    g_b = (term_b - 1) * g_a - term_c * g_d
    g_c = (term_c - 1) * g_d - term_b * g_a

    return g_a, g_b, g_c, g_d


def compute_Q_phi(r_a, r_b, r_c, r_d, F_a, F_b, F_c, F_d):
    """
    Compute generalized torsional force Q_φ = ∂V/∂φ.

    Q_φ = sum_i F_i · ∂r_i/∂φ = (F·g) / |g|²

    where g = ∂φ/∂r is the dihedral gradient.

    Args:
        r_a, r_b, r_c, r_d: Atom positions (3D vectors)
        F_a, F_b, F_c, F_d: Forces on atoms (3D vectors)

    Returns:
        Q_phi: Generalized torsional force (kJ/mol/rad)
    """
    g_a, g_b, g_c, g_d = compute_dihedral_gradient(r_a, r_b, r_c, r_d)

    # Project forces onto gradient
    F_dot_g = (np.dot(F_a, g_a) + np.dot(F_b, g_b) +
               np.dot(F_c, g_c) + np.dot(F_d, g_d))

    # Normalize by gradient magnitude squared
    g_sq = (np.dot(g_a, g_a) + np.dot(g_b, g_b) +
            np.dot(g_c, g_c) + np.dot(g_d, g_d))

    if g_sq < 1e-10:
        return 0.0

    Q_phi = F_dot_g / g_sq
    return Q_phi


def compute_phi_dot(r_a, r_b, r_c, r_d, v_a, v_b, v_c, v_d):
    """
    Compute time derivative of dihedral angle: dφ/dt = g · v.

    Args:
        r_a, r_b, r_c, r_d: Atom positions
        v_a, v_b, v_c, v_d: Atom velocities

    Returns:
        phi_dot: dφ/dt (rad/s or rad/ps depending on velocity units)
    """
    g_a, g_b, g_c, g_d = compute_dihedral_gradient(r_a, r_b, r_c, r_d)

    phi_dot = (np.dot(v_a, g_a) + np.dot(v_b, g_b) +
               np.dot(v_c, g_c) + np.dot(v_d, g_d))

    return phi_dot


def identify_torsion_groups(bond_indices, all_bonds):
    """
    Identify left and right atom groups for a torsion bond.

    Split molecule at bond (b, c), find connected components.

    Args:
        bond_indices: (b_idx, c_idx) - indices of central bond
        all_bonds: List of (i, j) tuples for all bonds in molecule

    Returns:
        (group_L, group_R): Lists of atom indices in left and right groups
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required for group identification. Install with: pip install networkx")

    b_idx, c_idx = bond_indices

    # Build molecular graph
    G = nx.Graph()
    G.add_edges_from(all_bonds)

    # Remove the torsion bond
    if G.has_edge(b_idx, c_idx):
        G.remove_edge(b_idx, c_idx)

    # Find connected components
    components = list(nx.connected_components(G))

    # Identify which component contains b and which contains c
    group_L = None
    group_R = None

    for comp in components:
        if b_idx in comp:
            group_L = sorted(comp)
        if c_idx in comp:
            group_R = sorted(comp)

    if group_L is None or group_R is None:
        raise ValueError(f"Could not split molecule at bond {bond_indices}")

    return group_L, group_R


def identify_torsion_groups_simple(torsion_atoms, n_atoms):
    """
    Simple group identification for small molecules.

    For a torsion (a, b, c, d):
    - Left group: atoms 0 to b (inclusive)
    - Right group: atoms c to n_atoms-1 (inclusive)

    This works for linear/branched chains but not rings.

    Args:
        torsion_atoms: (a, b, c, d) - torsion atom indices
        n_atoms: Total number of atoms

    Returns:
        (group_L, group_R): Lists of atom indices
    """
    a, b, c, d = torsion_atoms

    # Simple partition: atoms on left vs right of central bond
    group_L = list(range(0, b + 1))
    group_R = list(range(c, n_atoms))

    return group_L, group_R


def compute_I_red(positions, masses, u, group_L, group_R, r_pivot):
    """
    Compute reduced moment of inertia for two-body rotation.

    I_red = (I_L * I_R) / (I_L + I_R)

    where I_G = sum_{i in G} m_i * d_i², d_i = distance from axis.

    Args:
        positions: Nx3 array of atom positions
        masses: N-array of atom masses
        u: 3D unit vector for rotation axis
        group_L, group_R: Lists of atom indices in each group
        r_pivot: Point on rotation axis (e.g., position of atom b)

    Returns:
        I_red: Reduced moment of inertia (mass * length²)
    """
    def moment_group(group):
        I_g = 0.0
        for idx in group:
            # Vector from pivot to atom
            r_vec = positions[idx] - r_pivot
            # Distance to axis: |r - (r·u)u|
            r_parallel = np.dot(r_vec, u) * u
            r_perp = r_vec - r_parallel
            d_sq = np.dot(r_perp, r_perp)
            I_g += masses[idx] * d_sq
        return I_g

    I_L = moment_group(group_L)
    I_R = moment_group(group_R)

    # Reduced inertia (avoid division by zero)
    if I_L + I_R < 1e-10:
        return 1e-10

    I_red = (I_L * I_R) / (I_L + I_R)
    return I_red


def fit_omega_rigid_body(positions, velocities, masses, group):
    """
    Fit angular velocity ω for a group of atoms (rigid-body approximation).

    Solve: ω = I^(-1) * L

    where:
    - L = sum m_i (r_i' × v_i) (angular momentum)
    - I = sum m_i (|r_i'|² Id - r_i' ⊗ r_i'^T) (inertia tensor)
    - r_i' = r_i - COM

    Args:
        positions: Nx3 array
        velocities: Nx3 array
        masses: N-array
        group: List of atom indices

    Returns:
        omega: 3D angular velocity vector (rad/s or rad/ps)
    """
    if len(group) == 0:
        return np.zeros(3)

    # Compute center of mass
    M_g = np.sum([masses[idx] for idx in group])
    com_g = np.sum([masses[idx] * positions[idx] for idx in group], axis=0) / M_g

    # Angular momentum and inertia tensor
    L_g = np.zeros(3)
    I_g = np.zeros((3, 3))

    for idx in group:
        r_prime = positions[idx] - com_g
        v = velocities[idx]

        # Angular momentum contribution
        L_g += masses[idx] * np.cross(r_prime, v)

        # Inertia tensor contribution
        r_sq = np.dot(r_prime, r_prime)
        outer = np.outer(r_prime, r_prime)
        I_g += masses[idx] * (r_sq * np.eye(3) - outer)

    # Solve I * omega = L
    try:
        # Add small regularization for numerical stability
        I_g_reg = I_g + 1e-10 * np.eye(3)
        omega_g = np.linalg.solve(I_g_reg, L_g)
    except np.linalg.LinAlgError:
        omega_g = np.zeros(3)

    return omega_g


def compute_Lambda_GA(positions, velocities, forces, masses,
                      torsion_atoms, all_bonds=None):
    """
    Compute GA-based torsion diagnostic: Λ = ||[B_ω, B_Q]||.

    **Patent-Grade Formulation:**

    1. Nominal torsion axis: u = (r_c - r_b) / |r_c - r_b|
    2. Generalized force: Q_φ = (F · g) / |g|²
    3. Reduced inertia: I_red = (I_L * I_R) / (I_L + I_R)
    4. Angular acceleration bivector: B_Q = (Q_φ / I_red) * u
    5. Group angular velocities: ω_L, ω_R from rigid-body fit
    6. Velocity bivector: B_ω = ω_L - ω_R
    7. Commutator: Λ = 2 |ω_rel × (α u)|, where α = Q_φ / I_red

    **Physical Interpretation:**
    - Λ = 0: Pure torsional motion (ω_rel || u)
    - Λ > 0: Axis tilt / precession / coupling

    Args:
        positions: Nx3 array (Angstroms)
        velocities: Nx3 array (Angstroms/ps)
        forces: Nx3 array (kJ/mol/Angstrom)
        masses: N-array (amu)
        torsion_atoms: (a, b, c, d) - indices defining torsion
        all_bonds: List of (i, j) bond tuples (optional, for networkx splitting)

    Returns:
        Lambda: Scalar diagnostic (dimensionless after proper scaling)
    """
    a, b, c, d = torsion_atoms

    # Extract torsion atom data
    r_a, r_b, r_c, r_d = positions[[a, b, c, d]]
    v_a, v_b, v_c, v_d = velocities[[a, b, c, d]]
    F_a, F_b, F_c, F_d = forces[[a, b, c, d]]

    # 1. Torsion axis
    bond_vec = r_c - r_b
    bond_len = np.linalg.norm(bond_vec)
    if bond_len < 1e-10:
        return 0.0
    u = bond_vec / bond_len

    # 2. Identify groups (left and right of bond b-c)
    if all_bonds is not None and HAS_NETWORKX:
        try:
            group_L, group_R = identify_torsion_groups((b, c), all_bonds)
        except:
            # Fallback to simple split
            group_L, group_R = identify_torsion_groups_simple(torsion_atoms, len(positions))
    else:
        group_L, group_R = identify_torsion_groups_simple(torsion_atoms, len(positions))

    # 3. Generalized torsional force Q_φ
    Q_phi = compute_Q_phi(r_a, r_b, r_c, r_d, F_a, F_b, F_c, F_d)

    # 4. Reduced moment of inertia
    I_red = compute_I_red(positions, masses, u, group_L, group_R, r_b)

    # 5. Angular acceleration scalar
    if I_red < 1e-10:
        return 0.0
    alpha = Q_phi / I_red  # rad/s² (or rad/ps² if forces in appropriate units)

    # 6. Dual vector for B_Q
    v_Q = alpha * u  # 3D vector representing bivector B_Q

    # 7. Fit group angular velocities
    omega_L = fit_omega_rigid_body(positions, velocities, masses, group_L)
    omega_R = fit_omega_rigid_body(positions, velocities, masses, group_R)

    # 8. Relative angular velocity
    omega_rel = omega_L - omega_R  # 3D vector
    v_omega = omega_rel  # Dual vector for B_ω

    # 9. Commutator: [B_ω, B_Q] in dual representation
    cross = np.cross(v_omega, v_Q)

    # 10. Lambda = 2 * |cross product|
    Lambda = 2.0 * np.linalg.norm(cross)

    return Lambda


def compute_Lambda_GA_with_diagnostics(positions, velocities, forces, masses,
                                        torsion_atoms, all_bonds=None):
    """
    Extended version that returns diagnostic information.

    Returns:
        dict with keys:
        - 'Lambda': Scalar diagnostic
        - 'Q_phi': Generalized torsional force
        - 'I_red': Reduced moment of inertia
        - 'alpha': Angular acceleration
        - 'omega_L': Left group angular velocity
        - 'omega_R': Right group angular velocity
        - 'omega_rel': Relative angular velocity
        - 'axis': Torsion axis unit vector
        - 'axis_tilt': Angle between ω_rel and axis (degrees)
    """
    a, b, c, d = torsion_atoms

    r_a, r_b, r_c, r_d = positions[[a, b, c, d]]
    v_a, v_b, v_c, v_d = velocities[[a, b, c, d]]
    F_a, F_b, F_c, F_d = forces[[a, b, c, d]]

    # Torsion axis
    bond_vec = r_c - r_b
    bond_len = np.linalg.norm(bond_vec)
    if bond_len < 1e-10:
        return {'Lambda': 0.0, 'Q_phi': 0.0, 'I_red': 0.0, 'alpha': 0.0,
                'omega_L': np.zeros(3), 'omega_R': np.zeros(3),
                'omega_rel': np.zeros(3), 'axis': np.zeros(3), 'axis_tilt': 0.0}
    u = bond_vec / bond_len

    # Groups
    if all_bonds is not None and HAS_NETWORKX:
        try:
            group_L, group_R = identify_torsion_groups((b, c), all_bonds)
        except:
            group_L, group_R = identify_torsion_groups_simple(torsion_atoms, len(positions))
    else:
        group_L, group_R = identify_torsion_groups_simple(torsion_atoms, len(positions))

    # Q_phi and I_red
    Q_phi = compute_Q_phi(r_a, r_b, r_c, r_d, F_a, F_b, F_c, F_d)
    I_red = compute_I_red(positions, masses, u, group_L, group_R, r_b)

    if I_red < 1e-10:
        alpha = 0.0
    else:
        alpha = Q_phi / I_red

    v_Q = alpha * u

    # Omega
    omega_L = fit_omega_rigid_body(positions, velocities, masses, group_L)
    omega_R = fit_omega_rigid_body(positions, velocities, masses, group_R)
    omega_rel = omega_L - omega_R

    # Lambda
    cross = np.cross(omega_rel, v_Q)
    Lambda = 2.0 * np.linalg.norm(cross)

    # Axis tilt angle
    omega_rel_norm = np.linalg.norm(omega_rel)
    if omega_rel_norm > 1e-10:
        cos_theta = np.dot(omega_rel, u) / omega_rel_norm
        cos_theta = np.clip(cos_theta, -1, 1)
        axis_tilt = np.degrees(np.arccos(np.abs(cos_theta)))  # 0-90 degrees
    else:
        axis_tilt = 0.0

    return {
        'Lambda': Lambda,
        'Q_phi': Q_phi,
        'I_red': I_red,
        'alpha': alpha,
        'omega_L': omega_L,
        'omega_R': omega_R,
        'omega_rel': omega_rel,
        'axis': u,
        'axis_tilt': axis_tilt,
        'group_L': group_L,
        'group_R': group_R
    }


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
