#!/usr/bin/env python3
"""
Protein Backbone Torsion Utilities
===================================

Helper functions to identify backbone φ and ψ torsions in proteins.

Rick Mathews - November 2024
"""

import numpy as np
from openmm.app import Topology


def get_backbone_torsions(topology):
    """
    Find all backbone φ and ψ torsions in a protein topology.

    Parameters
    ----------
    topology : openmm.app.Topology
        Protein topology

    Returns
    -------
    phi_torsions : dict
        {residue_index: (C_prev, N, CA, C) atom indices}
    psi_torsions : dict
        {residue_index: (N, CA, C, N_next) atom indices}
    """
    # Map residues to their backbone atoms
    res_atoms = []
    for res in topology.residues():
        atoms = {atom.name: atom.index for atom in res.atoms()}
        res_atoms.append((res, atoms))

    phi_torsions = {}
    psi_torsions = {}

    for i in range(1, len(res_atoms) - 1):
        res_prev, atoms_prev = res_atoms[i - 1]
        res_curr, atoms_curr = res_atoms[i]
        res_next, atoms_next = res_atoms[i + 1]

        # Check for φ: C_{i-1} - N_i - CA_i - C_i
        if "C" in atoms_prev and all(n in atoms_curr for n in ["N", "CA", "C"]):
            phi_torsions[i] = (
                atoms_prev["C"],
                atoms_curr["N"],
                atoms_curr["CA"],
                atoms_curr["C"],
            )

        # Check for ψ: N_i - CA_i - C_i - N_{i+1}
        if all(n in atoms_curr for n in ["N", "CA", "C"]) and "N" in atoms_next:
            psi_torsions[i] = (
                atoms_curr["N"],
                atoms_curr["CA"],
                atoms_curr["C"],
                atoms_next["N"],
            )

    return phi_torsions, psi_torsions


def pick_middle_torsion(phi_torsions, psi_torsions, torsion_type="phi"):
    """
    Pick a middle backbone torsion (away from termini).

    Parameters
    ----------
    phi_torsions : dict
        φ torsions from get_backbone_torsions()
    psi_torsions : dict
        ψ torsions from get_backbone_torsions()
    torsion_type : str
        "phi" or "psi"

    Returns
    -------
    torsion_atoms : tuple
        (a, b, c, d) atom indices for selected torsion
    residue_index : int
        Residue index of the torsion
    """
    if torsion_type == "phi":
        torsions = phi_torsions
    elif torsion_type == "psi":
        torsions = psi_torsions
    else:
        raise ValueError("torsion_type must be 'phi' or 'psi'")

    if not torsions:
        raise ValueError(f"No {torsion_type} torsions found")

    # Pick middle residue
    res_indices = sorted(torsions.keys())
    mid_idx = res_indices[len(res_indices) // 2]

    return torsions[mid_idx], mid_idx


def compute_backbone_rmsd(positions, ref_positions, topology):
    """
    Compute backbone RMSD (N, CA, C atoms only).

    Parameters
    ----------
    positions : list of Vec3
        Current positions
    ref_positions : list of Vec3
        Reference positions
    topology : Topology
        Protein topology

    Returns
    -------
    rmsd : float
        RMSD in nanometers
    """
    from openmm.unit import nanometers

    ref = np.array([p.value_in_unit(nanometers) for p in ref_positions])
    pos = np.array([p.value_in_unit(nanometers) for p in positions])

    backbone_idx = []
    for atom in topology.atoms():
        if atom.name in ("N", "CA", "C"):
            backbone_idx.append(atom.index)

    if not backbone_idx:
        raise ValueError("No backbone atoms found")

    diff = pos[backbone_idx] - ref[backbone_idx]
    rmsd = np.sqrt((diff**2).sum() / len(backbone_idx))

    return rmsd


if __name__ == "__main__":
    # Test with ala12_helix.pdb
    from openmm.app import PDBFile

    print("Testing backbone torsion finder...")
    print()

    pdb = PDBFile("ala12_helix.pdb")

    phi_torsions, psi_torsions = get_backbone_torsions(pdb.topology)

    print(f"Found {len(phi_torsions)} φ torsions")
    print(f"Found {len(psi_torsions)} ψ torsions")
    print()

    print("φ torsions:")
    for res_idx, atoms in sorted(phi_torsions.items()):
        print(f"  Residue {res_idx}: atoms {atoms}")

    print()
    print("ψ torsions:")
    for res_idx, atoms in sorted(psi_torsions.items()):
        print(f"  Residue {res_idx}: atoms {atoms}")

    print()

    # Pick middle φ
    torsion_atoms, res_idx = pick_middle_torsion(phi_torsions, psi_torsions, "phi")
    print(f"Selected middle φ torsion:")
    print(f"  Residue: {res_idx}")
    print(f"  Atoms: {torsion_atoms}")
    print()

    print("✅ Torsion finder working correctly")
