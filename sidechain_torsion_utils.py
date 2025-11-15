#!/usr/bin/env python3
"""
Sidechain Torsion (χ Angle) Utilities for OpenMM
=================================================

Identifies sidechain dihedral angles (χ₁, χ₂, etc.) in protein structures
for Λ-adaptive timestep monitoring of drug binding pocket dynamics.

Key Applications:
- Aromatic ring flips (Phe, Tyr, Trp)
- Charged sidechain motion (Arg, Lys, Asp, Glu)
- Drug binding pocket flexibility

Rick Mathews - November 2024
Path A Extension - Sprint 2
"""

from openmm.app import Topology

# ============================================================================
# Sidechain Chi Angle Definitions
# ============================================================================

# χ₁ angle atom name templates for standard amino acids
# Format: (atom1, atom2, atom3, atom4) defining the dihedral
CHI1_TEMPLATES = {
    # Aromatic residues (important for π-stacking and hydrophobic pockets)
    'PHE': ('N', 'CA', 'CB', 'CG'),   # Phenylalanine (benzyl)
    'TYR': ('N', 'CA', 'CB', 'CG'),   # Tyrosine (phenol)
    'TRP': ('N', 'CA', 'CB', 'CG'),   # Tryptophan (indole)

    # Charged residues (electrostatic interactions)
    'ARG': ('N', 'CA', 'CB', 'CG'),   # Arginine (positive)
    'LYS': ('N', 'CA', 'CB', 'CG'),   # Lysine (positive)
    'ASP': ('N', 'CA', 'CB', 'CG'),   # Aspartate (negative)
    'GLU': ('N', 'CA', 'CB', 'CG'),   # Glutamate (negative)
    'HIS': ('N', 'CA', 'CB', 'CG'),   # Histidine (titratable)

    # Polar residues (H-bonding)
    'SER': ('N', 'CA', 'CB', 'OG'),   # Serine (hydroxyl)
    'THR': ('N', 'CA', 'CB', 'OG1'),  # Threonine (hydroxyl)
    'ASN': ('N', 'CA', 'CB', 'CG'),   # Asparagine (amide)
    'GLN': ('N', 'CA', 'CB', 'CG'),   # Glutamine (amide)
    'CYS': ('N', 'CA', 'CB', 'SG'),   # Cysteine (thiol)

    # Hydrophobic residues
    'VAL': ('N', 'CA', 'CB', 'CG1'),  # Valine (branched)
    'ILE': ('N', 'CA', 'CB', 'CG1'),  # Isoleucine (branched)
    'LEU': ('N', 'CA', 'CB', 'CG'),   # Leucine (branched)
    'MET': ('N', 'CA', 'CB', 'CG'),   # Methionine (sulfur)

    # Note: Glycine (GLY) and Alanine (ALA) have no χ angles
    # Proline (PRO) has constrained χ₁ due to ring structure
}

# χ₂ angle templates (for longer sidechains)
CHI2_TEMPLATES = {
    'PHE': ('CA', 'CB', 'CG', 'CD1'),  # Ring rotation
    'TYR': ('CA', 'CB', 'CG', 'CD1'),  # Ring rotation
    'TRP': ('CA', 'CB', 'CG', 'CD1'),  # Ring rotation
    'ARG': ('CA', 'CB', 'CG', 'CD'),
    'LYS': ('CA', 'CB', 'CG', 'CD'),
    'GLU': ('CA', 'CB', 'CG', 'CD'),
    'GLN': ('CA', 'CB', 'CG', 'CD'),
    'MET': ('CA', 'CB', 'CG', 'SD'),
    'ILE': ('CA', 'CB', 'CG1', 'CD1'),
    'LEU': ('CA', 'CB', 'CG', 'CD1'),
}

# Residues with important drug-binding sidechain dynamics
DRUG_BINDING_RESIDUES = ['PHE', 'TYR', 'TRP', 'ARG', 'LYS', 'ASP', 'GLU', 'HIS']


# ============================================================================
# Sidechain Torsion Finder Functions
# ============================================================================

def get_sidechain_chi1_torsions(topology):
    """
    Find all χ₁ (chi1) sidechain torsions in a protein topology.

    Parameters
    ----------
    topology : openmm.app.Topology
        Protein topology

    Returns
    -------
    chi1_torsions : dict
        Dictionary mapping residue index to (i, j, k, l) atom indices
        Keys: residue indices
        Values: tuple of 4 atom indices defining χ₁ dihedral

    residue_types : dict
        Dictionary mapping residue index to residue name

    Examples
    --------
    >>> chi1_torsions, res_types = get_sidechain_chi1_torsions(topology)
    >>> print(f"Found {len(chi1_torsions)} χ₁ torsions")
    >>> for res_idx, atoms in chi1_torsions.items():
    ...     print(f"Residue {res_idx} ({res_types[res_idx]}): atoms {atoms}")
    """
    chi1_torsions = {}
    residue_types = {}

    for chain in topology.chains():
        for residue in chain.residues():
            res_name = residue.name

            # Skip if no χ₁ template for this residue
            if res_name not in CHI1_TEMPLATES:
                continue

            # Get atom name template
            atom_names = CHI1_TEMPLATES[res_name]

            # Find atoms by name in this residue
            atoms_by_name = {}
            for atom in residue.atoms():
                atoms_by_name[atom.name] = atom.index

            # Check if all required atoms are present
            if all(name in atoms_by_name for name in atom_names):
                atom_indices = tuple(atoms_by_name[name] for name in atom_names)
                chi1_torsions[residue.index] = atom_indices
                residue_types[residue.index] = res_name

    return chi1_torsions, residue_types


def get_sidechain_chi2_torsions(topology):
    """
    Find all χ₂ (chi2) sidechain torsions in a protein topology.

    Parameters
    ----------
    topology : openmm.app.Topology
        Protein topology

    Returns
    -------
    chi2_torsions : dict
        Dictionary mapping residue index to (i, j, k, l) atom indices

    residue_types : dict
        Dictionary mapping residue index to residue name
    """
    chi2_torsions = {}
    residue_types = {}

    for chain in topology.chains():
        for residue in chain.residues():
            res_name = residue.name

            # Skip if no χ₂ template for this residue
            if res_name not in CHI2_TEMPLATES:
                continue

            # Get atom name template
            atom_names = CHI2_TEMPLATES[res_name]

            # Find atoms by name in this residue
            atoms_by_name = {}
            for atom in residue.atoms():
                atoms_by_name[atom.name] = atom.index

            # Check if all required atoms are present
            if all(name in atoms_by_name for name in atom_names):
                atom_indices = tuple(atoms_by_name[name] for name in atom_names)
                chi2_torsions[residue.index] = atom_indices
                residue_types[residue.index] = res_name

    return chi2_torsions, residue_types


def get_drug_binding_chi1_torsions(topology):
    """
    Get χ₁ torsions for residues commonly involved in drug binding.

    Focuses on aromatic (Phe, Tyr, Trp) and charged (Arg, Lys, Asp, Glu, His)
    residues that are critical for binding pocket dynamics.

    Parameters
    ----------
    topology : openmm.app.Topology
        Protein topology

    Returns
    -------
    chi1_torsions : dict
        Dictionary mapping residue index to (i, j, k, l) atom indices

    residue_types : dict
        Dictionary mapping residue index to residue name
    """
    all_chi1, all_types = get_sidechain_chi1_torsions(topology)

    # Filter to only drug-binding residues
    drug_chi1 = {
        res_idx: atoms
        for res_idx, atoms in all_chi1.items()
        if all_types[res_idx] in DRUG_BINDING_RESIDUES
    }

    drug_types = {
        res_idx: res_type
        for res_idx, res_type in all_types.items()
        if res_type in DRUG_BINDING_RESIDUES
    }

    return drug_chi1, drug_types


def get_combined_backbone_sidechain_torsions(topology, include_phi=True, include_psi=False,
                                               include_chi1=True, include_chi2=False,
                                               chi_residue_filter=None):
    """
    Get combined list of backbone and sidechain torsions for comprehensive monitoring.

    Parameters
    ----------
    topology : openmm.app.Topology
        Protein topology
    include_phi : bool, default=True
        Include backbone φ (phi) angles
    include_psi : bool, default=False
        Include backbone ψ (psi) angles
    include_chi1 : bool, default=True
        Include sidechain χ₁ angles
    include_chi2 : bool, default=False
        Include sidechain χ₂ angles
    chi_residue_filter : list of str, optional
        If provided, only include χ angles for these residue types
        (e.g., ['PHE', 'TYR', 'TRP'] for aromatics only)

    Returns
    -------
    torsion_atoms_list : list of tuples
        List of (i, j, k, l) atom index tuples

    torsion_labels : list of str
        Human-readable labels for each torsion
        (e.g., "Res5_PHE_chi1", "Res12_phi")

    Examples
    --------
    >>> # Monitor backbone φ + aromatic χ₁ for drug binding
    >>> torsions, labels = get_combined_backbone_sidechain_torsions(
    ...     topology,
    ...     include_phi=True,
    ...     include_psi=False,
    ...     include_chi1=True,
    ...     chi_residue_filter=['PHE', 'TYR', 'TRP']
    ... )
    """
    from protein_torsion_utils import get_backbone_torsions

    torsion_atoms_list = []
    torsion_labels = []

    # Get backbone torsions
    if include_phi or include_psi:
        phi_torsions, psi_torsions = get_backbone_torsions(topology)

        if include_phi:
            for res_idx in sorted(phi_torsions.keys()):
                torsion_atoms_list.append(phi_torsions[res_idx])
                torsion_labels.append(f"Res{res_idx}_phi")

        if include_psi:
            for res_idx in sorted(psi_torsions.keys()):
                torsion_atoms_list.append(psi_torsions[res_idx])
                torsion_labels.append(f"Res{res_idx}_psi")

    # Get sidechain χ₁ torsions
    if include_chi1:
        chi1_torsions, chi1_types = get_sidechain_chi1_torsions(topology)

        for res_idx in sorted(chi1_torsions.keys()):
            res_type = chi1_types[res_idx]

            # Apply filter if specified
            if chi_residue_filter is not None and res_type not in chi_residue_filter:
                continue

            torsion_atoms_list.append(chi1_torsions[res_idx])
            torsion_labels.append(f"Res{res_idx}_{res_type}_chi1")

    # Get sidechain χ₂ torsions
    if include_chi2:
        chi2_torsions, chi2_types = get_sidechain_chi2_torsions(topology)

        for res_idx in sorted(chi2_torsions.keys()):
            res_type = chi2_types[res_idx]

            # Apply filter if specified
            if chi_residue_filter is not None and res_type not in chi_residue_filter:
                continue

            torsion_atoms_list.append(chi2_torsions[res_idx])
            torsion_labels.append(f"Res{res_idx}_{res_type}_chi2")

    return torsion_atoms_list, torsion_labels


def print_sidechain_summary(topology):
    """
    Print summary of available sidechain torsions in the structure.

    Parameters
    ----------
    topology : openmm.app.Topology
        Protein topology
    """
    chi1_torsions, chi1_types = get_sidechain_chi1_torsions(topology)
    chi2_torsions, chi2_types = get_sidechain_chi2_torsions(topology)
    drug_chi1, drug_types = get_drug_binding_chi1_torsions(topology)

    print("=" * 70)
    print("Sidechain Torsion Summary")
    print("=" * 70)
    print()

    print(f"Total χ₁ torsions: {len(chi1_torsions)}")
    print(f"Total χ₂ torsions: {len(chi2_torsions)}")
    print(f"Drug-binding χ₁ torsions: {len(drug_chi1)}")
    print()

    # Count by residue type
    print("χ₁ Torsions by Residue Type:")
    type_counts = {}
    for res_type in chi1_types.values():
        type_counts[res_type] = type_counts.get(res_type, 0) + 1

    for res_type in sorted(type_counts.keys()):
        count = type_counts[res_type]
        marker = "*" if res_type in DRUG_BINDING_RESIDUES else " "
        print(f"  {marker} {res_type}: {count}")

    print()
    print("  * = Common in drug binding pockets")
    print()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print()
    print("This module provides utilities for finding sidechain χ angles.")
    print()
    print("Example usage:")
    print()
    print("  from openmm.app import PDBFile")
    print("  from sidechain_torsion_utils import get_combined_backbone_sidechain_torsions")
    print()
    print("  pdb = PDBFile('protein.pdb')")
    print("  torsions, labels = get_combined_backbone_sidechain_torsions(")
    print("      pdb.topology,")
    print("      include_phi=True,")
    print("      include_chi1=True,")
    print("      chi_residue_filter=['PHE', 'TYR', 'TRP']  # Aromatics only")
    print("  )")
    print()
    print("  # Use with Λ-adaptive integrator")
    print("  adaptive = LambdaAdaptiveVerletIntegrator(")
    print("      context=context,")
    print("      torsion_atoms=torsions,")
    print("      dt_base_fs=0.5,")
    print("      k=0.0001")
    print("  )")
