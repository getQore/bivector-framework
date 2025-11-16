#!/usr/bin/env python3
"""
Automatic binding site detection from co-crystallized ligands or cavity analysis.

Identifies residues within a distance cutoff from a ligand or cavity center
for focused adaptive timestep monitoring.

Sprint 4: Binding Pocket Adaptive Timestep
Rick Mathews - November 2024
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from openmm.app import Topology

class BindingPocketDetector:
    """Identifies binding pockets from protein structure."""

    # Standard protein residue names
    STANDARD_RESIDUES = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }

    def __init__(self,
                 topology: Topology,
                 distance_cutoff: float = 8.0):  # Angstroms
        """
        Initialize pocket detector.

        Parameters
        ----------
        topology : openmm.app.Topology
            OpenMM topology object
        distance_cutoff : float, default=8.0
            Distance in Angstroms for residue inclusion
        """
        self.topology = topology
        self.distance_cutoff = distance_cutoff / 10.0  # Convert to nm

    def detect_from_ligand(self,
                          positions: np.ndarray,
                          ligand_resname: str) -> List[int]:
        """
        Identify binding pocket residues near a co-crystallized ligand.

        Parameters
        ----------
        positions : numpy.ndarray
            Atomic positions (n_atoms, 3) in nm
        ligand_resname : str
            Residue name of ligand (e.g., 'LIG', 'BEN', 'ATP')

        Returns
        -------
        pocket_residues : list of int
            List of residue indices forming the binding pocket

        Examples
        --------
        >>> detector = BindingPocketDetector(topology, distance_cutoff=8.0)
        >>> pocket = detector.detect_from_ligand(positions, 'LIG')
        >>> print(f"Found {len(pocket)} pocket residues")
        """
        # Convert positions to numpy array (in case it's a list from OpenMM)
        positions = np.array(positions)

        # Get ligand atoms
        ligand_atoms = []
        for atom in self.topology.atoms():
            if atom.residue.name == ligand_resname:
                ligand_atoms.append(atom.index)

        if not ligand_atoms:
            raise ValueError(f"Ligand '{ligand_resname}' not found in topology")

        ligand_pos = positions[ligand_atoms]

        # Find pocket residues
        pocket_residues = set()

        for residue in self.topology.residues():
            # Skip ligand and non-protein residues
            if residue.name == ligand_resname:
                continue
            if residue.name not in self.STANDARD_RESIDUES:
                continue

            # Check if any atom in residue is within cutoff
            for atom in residue.atoms():
                atom_pos = positions[atom.index]

                # Check distance to all ligand atoms
                for lig_pos in ligand_pos:
                    dist = np.linalg.norm(atom_pos - lig_pos)
                    if dist < self.distance_cutoff:
                        pocket_residues.add(residue.index)
                        break

                if residue.index in pocket_residues:
                    break

        return sorted(list(pocket_residues))

    def detect_from_cavity(self,
                          positions: np.ndarray,
                          center: np.ndarray,
                          radius: float = 10.0) -> List[int]:
        """
        Identify pocket residues within a spherical cavity.

        Useful when no co-crystallized ligand is available.

        Parameters
        ----------
        positions : numpy.ndarray
            Atomic positions (n_atoms, 3) in nm
        center : numpy.ndarray
            Center of binding site (x, y, z) in nm
        radius : float, default=10.0
            Cavity radius in Angstroms

        Returns
        -------
        pocket_residues : list of int
            List of residue indices within cavity
        """
        # Convert positions to numpy array (in case it's a list from OpenMM)
        positions = np.array(positions)

        radius_nm = radius / 10.0
        pocket_residues = set()

        for residue in self.topology.residues():
            # Skip non-protein residues
            if residue.name not in self.STANDARD_RESIDUES:
                continue

            # Compute residue center of mass
            res_atoms = [a.index for a in residue.atoms()]
            res_com = positions[res_atoms].mean(axis=0)

            # Check distance from cavity center
            if np.linalg.norm(res_com - center) < radius_nm:
                pocket_residues.add(residue.index)

        return sorted(list(pocket_residues))

    def detect_from_residue_list(self,
                                 residue_indices: List[int],
                                 expand_cutoff: float = 5.0) -> List[int]:
        """
        Identify pocket from a list of known key residues, then expand.

        Useful for known binding sites (e.g., enzyme active sites).

        Parameters
        ----------
        residue_indices : list of int
            Core binding site residues
        expand_cutoff : float, default=5.0
            Expand pocket by this distance in Angstroms

        Returns
        -------
        pocket_residues : list of int
            Expanded list including nearby residues
        """
        # Start with core residues
        pocket_residues = set(residue_indices)

        # For now, just return the input
        # TODO: Implement expansion based on distances
        return sorted(list(pocket_residues))

    def compute_pocket_centroid(self,
                               positions: np.ndarray,
                               pocket_residues: List[int]) -> np.ndarray:
        """
        Compute geometric center of binding pocket.

        Parameters
        ----------
        positions : numpy.ndarray
            Atomic positions (n_atoms, 3) in nm
        pocket_residues : list of int
            Binding pocket residue indices

        Returns
        -------
        centroid : numpy.ndarray
            Pocket center of mass (x, y, z) in nm
        """
        # Convert positions to numpy array (in case it's a list from OpenMM)
        positions = np.array(positions)

        pocket_atoms = []

        for residue in self.topology.residues():
            if residue.index in pocket_residues:
                for atom in residue.atoms():
                    pocket_atoms.append(atom.index)

        if not pocket_atoms:
            raise ValueError("No atoms found in pocket residues")

        pocket_positions = positions[pocket_atoms]
        centroid = pocket_positions.mean(axis=0)

        return centroid

    def print_pocket_summary(self,
                            pocket_residues: List[int],
                            verbose: bool = True):
        """
        Print summary of detected binding pocket.

        Parameters
        ----------
        pocket_residues : list of int
            Binding pocket residue indices
        verbose : bool, default=True
            Print detailed residue list
        """
        print("=" * 70)
        print("Binding Pocket Detection Summary")
        print("=" * 70)
        print()
        print(f"Total pocket residues: {len(pocket_residues)}")
        print(f"Distance cutoff: {self.distance_cutoff * 10:.1f} Å")
        print()

        if verbose and len(pocket_residues) > 0:
            # Count by residue type
            residue_types = {}
            for res_idx in pocket_residues:
                residue = list(self.topology.residues())[res_idx]
                res_name = residue.name
                residue_types[res_name] = residue_types.get(res_name, 0) + 1

            print("Residue composition:")
            for res_name in sorted(residue_types.keys()):
                count = residue_types[res_name]
                print(f"  {res_name}: {count}")
            print()

            # List residues
            print("Pocket residues:")
            for res_idx in pocket_residues[:20]:  # Show first 20
                residue = list(self.topology.residues())[res_idx]
                print(f"  {residue.name}{residue.id} (index {res_idx})")

            if len(pocket_residues) > 20:
                print(f"  ... and {len(pocket_residues) - 20} more")
            print()


# ============================================================================
# Helper Functions
# ============================================================================

def compute_ligand_centroid(topology: Topology,
                           positions: np.ndarray,
                           ligand_resname: str) -> np.ndarray:
    """
    Compute center of mass of ligand.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology object
    positions : numpy.ndarray
        Positions in nm
    ligand_resname : str
        Ligand residue name

    Returns
    -------
    centroid : numpy.ndarray
        Ligand COM in nm
    """
    # Convert positions to numpy array (in case it's a list from OpenMM)
    positions = np.array(positions)

    ligand_atoms = []
    for atom in topology.atoms():
        if atom.residue.name == ligand_resname:
            ligand_atoms.append(atom.index)

    if not ligand_atoms:
        raise ValueError(f"Ligand '{ligand_resname}' not found")

    ligand_positions = positions[ligand_atoms]
    return ligand_positions.mean(axis=0)


if __name__ == "__main__":
    print(__doc__)
    print()
    print("Example usage:")
    print()
    print("  from openmm.app import PDBFile")
    print("  from binding_pocket_detector import BindingPocketDetector")
    print()
    print("  pdb = PDBFile('protein_ligand.pdb')")
    print("  positions = pdb.positions.value_in_unit(unit.nanometer)")
    print()
    print("  detector = BindingPocketDetector(pdb.topology, distance_cutoff=8.0)")
    print("  pocket = detector.detect_from_ligand(positions, 'LIG')")
    print("  detector.print_pocket_summary(pocket)")
