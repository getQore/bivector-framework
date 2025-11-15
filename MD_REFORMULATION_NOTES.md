# MD Reformulation: Local Torsional Λ

**Date**: November 14, 2024
**Status**: Guidance for future work

---

## Why Current Approach Failed

**Problem**: Current `md_bivector_utils.py` computes **global molecular rotation**:
```python
def angular_velocity_bivector(positions, velocities):
    # Computes global angular momentum L = Σ rᵢ × vᵢ
    # Small peptide at 300K has tiny global rotation
    # Local torsional motions don't contribute to global ω
```

**Result**: R² = 0.0001 (no correlation with torsional dynamics)

**Root Cause**: Mismatch between what we measure (global rotation) and what matters (local bond torsions)

---

## Correct Formulation: Local Bond-Framed Λ

### Per-Bond Approach

For each torsional degree of freedom φᵢ (dihedral angle):

#### 1. Define Local Frame
```python
def define_local_frame(atom1, atom2, atom3, atom4):
    """
    For dihedral A1-A2-A3-A4:
    - Central bond: A2-A3
    - Bond axis: b̂ᵢ = (r₃ - r₂) / |r₃ - r₂|
    - Orthonormal frame on this bond
    """
    bond_vector = atom3.position - atom2.position
    bond_axis = bond_vector / np.linalg.norm(bond_vector)

    return bond_axis
```

#### 2. Local Angular Velocity
```python
def local_angular_velocity(phi_dot, bond_axis):
    """
    ωᵢ = φ̇ᵢ b̂ᵢ

    where:
    - φ̇ᵢ = time derivative of dihedral angle
    - b̂ᵢ = bond axis unit vector
    """
    omega_i = phi_dot * bond_axis
    return omega_i
```

#### 3. Local Torque
```python
def local_torsional_torque(forces, positions, bond_axis):
    """
    τᵢ = projection of torque onto bond axis

    Torque from forces: τ = Σ rⱼ × Fⱼ (for atoms in torsion)
    Project onto bond: τᵢ = (τ · b̂ᵢ) b̂ᵢ
    """
    # Compute torque from forces
    torque = compute_torque(forces, positions)

    # Project onto bond axis
    tau_parallel = np.dot(torque, bond_axis) * bond_axis

    return tau_parallel
```

#### 4. Per-Bond Λ
```python
def compute_local_lambda(omega_i, tau_i):
    """
    Two options:

    Option A: Bivector commutator (if using full Cl(3,1))
        Λᵢ = ||[ωᵢ, τᵢ]||

    Option B: Simple product (scalar stiffness)
        Λᵢ = |ωᵢ · τᵢ|

    Both measure "coupling strength" between rotation rate and torque
    """
    # Option A: Bivector formalism
    omega_bivector = vector_to_bivector(omega_i)
    tau_bivector = vector_to_bivector(tau_i)
    Lambda_i = commutator_norm(omega_bivector, tau_bivector)

    # Option B: Direct product (simpler, may work just as well)
    # Lambda_i = np.abs(np.dot(omega_i, tau_i))

    return Lambda_i
```

#### 5. Global Λ (Aggregation)
```python
def aggregate_lambda(lambda_per_bond):
    """
    Combine per-bond Λᵢ into global Λ(t)

    Options:
    - Max: Λ(t) = max(Λᵢ) - "bottleneck" bond
    - Mean: Λ(t) = mean(Λᵢ) - average stiffness
    - Weighted: Λ(t) = Σ wᵢ Λᵢ - e.g., weight by barrier height
    - RMS: Λ(t) = sqrt(mean(Λᵢ²)) - emphasize large values
    """
    # For adaptive timestep, probably want max (most conservative)
    Lambda_global = np.max(lambda_per_bond)

    return Lambda_global
```

---

## Complete Reformulated Algorithm

```python
def compute_torsional_lambda_local(coords, velocities, forces, topology):
    """
    Proper torsion-aware Λ for MD

    Args:
        coords: Atomic positions (N, 3)
        velocities: Atomic velocities (N, 3)
        forces: Forces on atoms (N, 3)
        topology: Molecular topology (defines dihedrals)

    Returns:
        Lambda_global: Aggregate stiffness metric
        lambda_per_dihedral: Individual Λᵢ for each torsion
    """
    lambda_per_dihedral = []

    # Loop over all dihedral angles
    for dihedral in topology.dihedrals:
        # Get 4 atoms defining dihedral: A1-A2-A3-A4
        a1, a2, a3, a4 = dihedral.atoms

        # 1. Define local bond frame (A2-A3 bond)
        bond_axis = define_local_frame(
            coords[a1], coords[a2], coords[a3], coords[a4]
        )

        # 2. Calculate dihedral angle φ and its derivative φ̇
        phi = calculate_dihedral_angle(
            coords[a1], coords[a2], coords[a3], coords[a4]
        )
        phi_dot = calculate_dihedral_velocity(
            coords[a1], coords[a2], coords[a3], coords[a4],
            velocities[a1], velocities[a2], velocities[a3], velocities[a4]
        )

        # 3. Local angular velocity
        omega_i = local_angular_velocity(phi_dot, bond_axis)

        # 4. Local torsional torque
        tau_i = local_torsional_torque(
            forces[[a1, a2, a3, a4]],
            coords[[a1, a2, a3, a4]],
            bond_axis
        )

        # 5. Per-bond Λ
        Lambda_i = compute_local_lambda(omega_i, tau_i)
        lambda_per_dihedral.append(Lambda_i)

    # 6. Aggregate to global Λ
    Lambda_global = aggregate_lambda(lambda_per_dihedral)

    return Lambda_global, lambda_per_dihedral
```

---

## Why This Should Work

### 1. **Geometrically Aligned**
- Measures rotation **around the bond** (not global rotation)
- Torque **about the bond** (not global torque)
- These are the actual torsional degrees of freedom

### 2. **Physical Interpretation**
- **High Λᵢ**: Bond rotating fast under large torque = stiff/frustrated
- **Low Λᵢ**: Either slow rotation or weak torque = soft/relaxed
- Directly relates to torsional barrier crossings

### 3. **Adaptive Timestep Connection**
- High Λ → approaching barrier → reduce timestep
- Low Λ → harmonic well → use large timestep
- max(Λᵢ) catches the "bottleneck" bond

---

## Implementation Path Forward

### Phase 1: Proof of Concept (1-2 days)
```python
# File: test_local_torsional_lambda.py

def test_butane_with_local_lambda():
    """
    Butane rotation test with LOCAL Λ

    Should now see:
    - Λ peaks at eclipsed conformations (φ = 0°, 120°, 240°)
    - Λ minima at staggered (φ = 60°, 180°, 300°)
    - R² > 0.8 between Λ and torsional strain
    """

    # Scan dihedral
    for phi in np.linspace(0, 360, 73):
        coords = build_butane_at_dihedral(phi)
        velocities = sample_thermal_velocities(coords, T=300)
        forces = compute_mm_forces(coords)

        # NEW: Local torsional Λ
        Lambda_local, lambdas_per_bond = compute_torsional_lambda_local(
            coords, velocities, forces, butane_topology
        )

        # Compare to torsional potential
        V_torsion = butane_torsional_energy(phi)

        # Should correlate!
        results.append({
            'phi': phi,
            'Lambda': Lambda_local,
            'V': V_torsion
        })

    # Test correlation
    r2 = calculate_r2(results['Lambda'], results['V'])
    print(f"R² (local Λ vs V_torsion) = {r2:.3f}")
    # Expect: R² > 0.8 (vs 0.0001 with global Λ)
```

### Phase 2: Full MD Integration (2-3 days)
1. Implement in OpenMM as custom force/integrator
2. Test on alanine dipeptide (φ, ψ Ramachandran)
3. Validate adaptive timestep on stiff systems
4. Benchmark speedup vs fixed timestep

### Phase 3: Real Applications (1 week)
1. Protein folding (villin headpiece)
2. Drug-protein binding (HIV protease)
3. Publication-quality validation

---

## Key Differences: Global vs Local

| Aspect | Global Λ (Failed) | Local Λ (Proposed) |
|--------|-------------------|-------------------|
| **Measures** | Molecular rotation | Bond torsion |
| **Geometry** | Center-of-mass frame | Per-bond frame |
| **Sensitivity** | Tiny for small molecules | Directly tracks φᵢ(t) |
| **Physical** | L = Σ rᵢ × vᵢ | ωᵢ = φ̇ᵢ b̂ᵢ |
| **Correlation** | R² = 0.0001 ❌ | R² > 0.8 (predicted) ✅ |

---

## Analogy: Why Local Matters

**Global approach**: Like measuring Earth's rotation to detect a door hinge squeaking
- Technically both are rotations
- But completely wrong scale/frame

**Local approach**: Measure the hinge rotation directly
- Right degree of freedom
- Right reference frame
- Detects the squeak!

---

## Next Steps (If Pursuing MD Patent)

### Option 1: Quick Validation (Recommended)
1. Implement `compute_torsional_lambda_local()` (1 day)
2. Test on butane rotation (0.5 day)
3. If R² > 0.8: Continue to full validation
4. If R² < 0.5: Abandon MD patent

### Option 2: Defer MD Patent
1. Focus on RL patent (validated, R² = 0.89)
2. File provisional for RL immediately
3. Revisit MD if local formulation validates

### Option 3: Theoretical Paper First
1. Publish local Λ formulation concept
2. Show it fixes the global rotation problem
3. Demonstrate on simple systems
4. Then pursue patent if adopted by MD community

---

## Bottom Line

**The idea has potential IF reformulated correctly.**

Current implementation used the **wrong geometry** (global vs local).

Fix requires:
1. Per-bond local frames
2. Project ω and τ onto bond axes
3. Aggregate Λᵢ intelligently

This is **doable** but needs 3-5 days of focused MD work.

**Recommendation**:
- Validate RL patent first (it works!)
- Revisit MD with local formulation later
- Or publish as theoretical framework paper

---

**Created**: November 14, 2024
**Status**: Technical guidance for future MD work
**Priority**: MEDIUM (after RL patent filing)
