#!/usr/bin/env python3
"""
Generate Ala12 helix PDB using OpenMM
======================================

Simple approach: Use PDBFixer to build from sequence,
then minimize to get a reasonable structure.

Rick Mathews - November 2024
"""

import numpy as np
from openmm import app
from openmm.app import PDBFile, ForceField, Modeller
from openmm import LangevinIntegrator, Platform, Context, LocalEnergyMinimizer, Vec3
from openmm.unit import *

try:
    from pdbfixer import PDBFixer
    has_pdbfixer = True
except ImportError:
    has_pdbfixer = False
    print("PDBFixer not available, will use manual approach")


def build_ala12_simple():
    """
    Build Ala12 using simplified manual construction.
    Creates extended chain, then minimizes.
    """
    from openmm.app import Topology
    import openmm.app.element as elem

    topology = Topology()
    chain = topology.addChain()

    # Simplified: build extended chain with reasonable geometry
    positions = []

    # Spacing between residues (~3.8 Å for extended)
    residue_spacing = 3.8

    for i in range(12):
        res = topology.addResidue("ALA", chain)

        z = i * residue_spacing

        # N
        n = topology.addAtom("N", elem.nitrogen, res)
        positions.append([0.0, 0.0, z])

        # CA
        ca = topology.addAtom("CA", elem.carbon, res)
        positions.append([1.45, 0.0, z])

        # C
        c = topology.addAtom("C", elem.carbon, res)
        positions.append([2.0, 1.5, z])

        # O
        o = topology.addAtom("O", elem.oxygen, res)
        positions.append([3.0, 1.5, z])

        # CB (Ala)
        cb = topology.addAtom("CB", elem.carbon, res)
        positions.append([1.45, 0.0, z + 1.5])

        # Add bonds
        topology.addBond(n, ca)
        topology.addBond(ca, c)
        topology.addBond(c, o)
        topology.addBond(ca, cb)

        # Peptide bond to next residue
        if i > 0:
            prev_c_idx = (i-1) * 5 + 2  # Previous C atom
            prev_c = list(topology.atoms())[prev_c_idx]
            topology.addBond(prev_c, n)

    # Convert to OpenMM positions
    positions_nm = [Vec3(x/10, y/10, z/10) for x, y, z in positions]

    return topology, positions_nm


if __name__ == "__main__":
    print("Generating Ala12 peptide structure...")
    print()

    # Build basic structure
    topology, positions = build_ala12_simple()

    print(f"Initial structure: {topology.getNumAtoms()} atoms, {topology.getNumResidues()} residues")

    # Set up force field
    print("Loading force field...")
    forcefield = ForceField("amber14-all.xml")

    # Add hydrogens
    print("Adding hydrogens...")
    modeller = Modeller(topology, positions)
    modeller.addHydrogens(forcefield, pH=7.0)

    print(f"After adding H: {modeller.topology.getNumAtoms()} atoms")

    # Create system for minimization
    print("Creating system and minimizing...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds
    )

    integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 1.0*femtoseconds)
    platform = Platform.getPlatformByName('CPU')
    context = Context(system, integrator, platform)
    context.setPositions(modeller.positions)

    # Minimize
    print("Minimizing energy...")
    LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=1000)

    # Get minimized positions
    state = context.getState(getPositions=True, getEnergy=True)
    minimized_positions = state.getPositions()
    energy = state.getPotentialEnergy()

    print(f"Final energy: {energy}")

    # Save
    output_file = "ala12_helix.pdb"
    with open(output_file, "w") as f:
        PDBFile.writeFile(modeller.topology, minimized_positions, f)

    print(f"\n✅ Structure saved: {output_file}")
    print(f"   Atoms: {modeller.topology.getNumAtoms()}")
    print(f"   Residues: {modeller.topology.getNumResidues()}")
    print()

    del context, integrator
