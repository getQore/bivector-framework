# Stage-1: Local Torsion Lambda Validation

**Date**: November 14, 2024
**Status**: Ready for testing
**System**: Butane (C₄H₁₀)
**Goal**: Validate local bond-framed Λ formulation

---

## Executive Summary

The **global bivector approach failed** (R² = 0.0001) because it measured whole-molecule rotation instead of local bond torsions.

This Stage-1 test implements the **correct local formulation**:

```
Λᵢ(t) = |φ̇ᵢ(t) · τᵢ(t)|
```

where:
- φ̇ᵢ = torsional angular velocity (rad/ps) around bond i
- τᵢ = torque projected onto bond axis i (kJ/mol)
- Λᵢ = coupling strength (kJ/mol/ps) - high when bond is stiff/frustrated

**Success Criterion**: R²[Λ, |τ|] ≥ 0.5 in MD trajectory

---

## What Changed: Global → Local

| Aspect | Global Λ (Failed) | Local Λ (This Test) |
|--------|-------------------|---------------------|
| **Frame** | Center-of-mass | Per-bond axis |
| **ω** | Σ rᵢ × vᵢ (global) | φ̇ᵢ b̂ᵢ (local) |
| **τ** | Global torque | Projected onto b̂ᵢ |
| **Geometry** | 3D bivector | Scalar on bond axis |
| **Sensitivity** | Molecular tumbling | Torsional barriers |

**Key Insight**: For MD torsions, we don't need the full Cl(3,1) bivector machinery - the problem reduces to 1D rotation around each bond axis.

---

## Mathematical Formulation

### Per-Bond Quantities

For dihedral angle φᵢ defined by atoms (a, b, c, d):

1. **Bond axis**:
   ```
   b̂ᵢ = (r_c - r_b) / |r_c - r_b|
   ```

2. **Dihedral angle** (standard robust formula):
   ```python
   n₁ = (r_b - r_a) × (r_c - r_b)
   n₂ = (r_c - r_b) × (r_d - r_c)
   φᵢ = atan2((n₁ × n₂)·b̂ᵢ, n₁·n₂)
   ```

3. **Torsional torque** (projected onto bond):
   ```python
   For each atom k in {a,b,c,d}:
       r_rel = r_k - r_mid
       r_perp = r_rel - (r_rel · b̂ᵢ) b̂ᵢ
       τ_k = (r_perp × F_k) · b̂ᵢ

   τᵢ = Σ_k τ_k
   ```

4. **Angular velocity** (finite difference):
   ```python
   φ̇ᵢ(t) = [φᵢ(t+Δt) - φᵢ(t-Δt)] / (2Δt)
   ```
   Note: Use `np.unwrap()` to handle ±180° discontinuities

5. **Local Lambda**:
   ```python
   Λᵢ(t) = |φ̇ᵢ(t) · τᵢ(t)|
   ```
   Units: (rad/ps) × (kJ/mol) = kJ/mol/ps (power-like)

### Physical Interpretation

- **High Λᵢ**: Bond rotating fast under large torque → stiff/frustrated → need small timestep
- **Low Λᵢ**: Either slow rotation or weak torque → relaxed → can use large timestep
- **For adaptive dt**: `Λ_global = max(Λᵢ)` catches the bottleneck bond

---

## Test System: Butane

### Why Butane?

1. **Simplest torsion**: Single C-C-C-C dihedral
2. **Known barrier**: ~3 kcal/mol (eclipsed vs staggered)
3. **Clean dynamics**: No side chains, no coupling between multiple torsions
4. **Fast to simulate**: 14 atoms, vacuum, 5000 steps

### Conformations

- **Staggered** (φ = 60°, 180°, 300°): Low energy, low τ, low Λ
- **Eclipsed** (φ = 0°, 120°, 240°): High energy, high τ, high Λ (when crossing)

### Expected Behavior

At 300K, butane will:
- Oscillate in staggered wells most of the time (low Λ)
- Occasionally cross barriers (Λ spikes)
- Show correlation between Λ and torsional strain

---

## Files

### 1. `test_butane_local_lambda.py`

Complete MD test script with:

**Functions**:
- `dihedral_angle(positions, i1,i2,i3,i4)` - robust atan2 formula
- `torsion_torque_about_bond(positions, forces, i1,i2,i3,i4)` - projected torque
- `run_md_local_lambda()` - main MD loop
- `analyze_and_plot()` - R² calculations and plots

**Configuration**:
```python
N_STEPS = 5000         # 10 ps at 2 fs/step
DT_FS = 2.0            # 2 fs timestep
TEMPERATURE = 300.0    # K
TORSION_ATOMS = (0, 4, 6, 10)  # C1-C2-C3-C4
```

**Outputs**:
- `butane_local_lambda_timeseries.png` - φ(t), τ(t), Λ(t)
- `butane_local_lambda_correlations.png` - scatter plots with R²
- `butane_local_lambda_lambda_hist.png` - Λ distribution
- Console: R² values and pass/fail

### 2. `butane.pdb`

Standard all-trans butane structure with proper atom ordering:
- Atoms 1, 5, 7, 11 = C1, C2, C3, C4 (PDB 1-indexed)
- Atoms 0, 4, 6, 10 = C1, C2, C3, C4 (Python 0-indexed)

---

## Running the Test

### Prerequisites

```bash
conda install -c conda-forge openmm
conda install matplotlib numpy
```

### Execution

```bash
cd C:\v2_files\md_validation
python test_butane_local_lambda.py
```

### Expected Output

```
================================================================================
STAGE-1: LOCAL TORSION-AWARE LAMBDA FOR MD
================================================================================

System: Butane
Torsion: atoms (0, 4, 6, 10)
Temperature: 300.0 K
Steps: 5000
Timestep: 2.0 fs

Minimizing energy...
Running 5000 MD steps at 300.0K...
Timestep: 2.0 fs (0.002 ps)
Total time: 10.00 ps

Step 0/5000
Step 500/5000
...

================================================================================
STAGE-1 MD RESULTS: Local Torsion Lambda
================================================================================

phi range (deg): [-180.0, 180.0]
tau range (kJ/mol): [...]
Lambda range (kJ/mol/ps): [...]

=== Correlations (Stage 1 MD) ===
R²[ Lambda vs |tau| ]      = 0.XXXX
R²[ Lambda vs |phi_ddot| ] = 0.XXXX

=== Stage-1 Success Criteria ===
Target: R²[Lambda, |tau|] ≥ 0.5
✓ PASS: R² = 0.XXXX
```

---

## Success Criteria

### Stage-1 MD Test

**Minimum viable**:
- R²[Λ, |τ|] ≥ **0.5** → Λ tracks torsional torque

**Strong validation**:
- R²[Λ, |τ|] ≥ **0.7** → Proceed to alanine dipeptide
- Λ distribution shows clear spread (not all near zero)
- Visual correlation in scatter plots

**Failure**:
- R²[Λ, |τ|] < **0.5** → Local formulation also doesn't work; abandon MD patent

### Next Steps After Stage-1

**If R² ≥ 0.7**:
1. Implement static dihedral scan (V(φ) vs Λ(φ))
2. Test on alanine dipeptide (φ/ψ Ramachandran)
3. Proceed to Stage-2 (adaptive timestep)

**If 0.5 ≤ R² < 0.7**:
1. Investigate noise sources (thermal fluctuations?)
2. Try longer trajectory or lower temperature
3. Check torque calculation for sign issues

**If R² < 0.5**:
1. Focus on RL patent (already validated, R² = 0.89)
2. Write up MD as "lessons learned"
3. Publish theoretical framework paper

---

## Understanding the Plots

### `butane_local_lambda_timeseries.png`

Three panels showing time evolution:

**Panel 1: φ(t)**
- Should wander around trans (180°) and gauche (±60°) regions
- Occasional jumps between wells = barrier crossings
- At 300K, mostly stays in one well with thermal oscillations

**Panel 2: τ(t)**
- Torsional torque fluctuates due to:
  - Restoring force from potential V(φ)
  - Thermal collisions with solvent (Langevin)
- Spikes when near eclipsed conformations

**Panel 3: Λ(t)**
- Λ = |φ̇ · τ|, so spikes when BOTH:
  - Bond is rotating (φ̇ large)
  - AND torque is strong (τ large)
- Should see occasional spikes on barrier crossings
- Low baseline in harmonic wells

**Key Visual Check**: Do Λ spikes coincide with φ transitions?

### `butane_local_lambda_correlations.png`

**Left: Λ vs |τ|**
- Should show positive correlation
- R² ≥ 0.5 means Λ tracks torsional stress
- Scatter due to φ̇ modulation

**Right: Λ vs |φ̈|**
- φ̈ = angular acceleration (stiffness proxy)
- May be noisier than |τ| correlation
- Validates that Λ sees "difficult dynamics"

### `butane_local_lambda_lambda_hist.png`

- Distribution of Λ values across trajectory
- Should be right-skewed: mostly low, occasional spikes
- If delta spike at zero → system frozen (check temperature)
- If uniform → no discrimination (check calculation)

---

## Debugging Guide

### Common Issues

**Problem**: R² ≈ 0 (no correlation)
- **Check**: Are torsion atom indices correct? Print φ(t) first step
- **Check**: Is τ calculation returning non-zero values?
- **Check**: Try `np.abs(phi_dot)` instead of raw φ̇

**Problem**: φ jumps wildly between ±180°
- **Fix**: Ensure `phi_unwrapped = np.unwrap(phi_arr)` before computing φ̇
- **Check**: Plot both `phi` and `phi_unwrapped` to verify

**Problem**: Λ is always ~0
- **Check**: Temperature - if too low, no motion → no Λ
- **Check**: Units - τ should be kJ/mol, φ̇ should be rad/ps
- **Check**: Timestep - if too large, finite differences are noisy

**Problem**: OpenMM errors on force field
- **Try**: Different FF: `GAFF.xml`, `openff-2.0.0.offxml`
- **Check**: PDB atom types match force field expectations

### Verification Steps

1. **Print first frame**:
   ```python
   print(f"phi[0] = {np.rad2deg(phi_list[0]):.1f} deg")
   print(f"tau[0] = {tau_list[0]:.3e} kJ/mol")
   ```
   Should give reasonable values (φ ~ 180°, τ small but non-zero)

2. **Check unwrapping**:
   ```python
   plt.plot(phi_arr, label='raw')
   plt.plot(phi_unwrapped, label='unwrapped')
   ```
   Unwrapped should be continuous, raw should have jumps

3. **Sanity check units**:
   ```python
   print(f"phi_dot range: {phi_dot.min():.2e} to {phi_dot.max():.2e} rad/ps")
   ```
   At 300K, expect φ̇ ~ 0.01-0.1 rad/ps during transitions

---

## Theory Notes

### Why This Should Work

**Geometric alignment**:
- Measures rotation AROUND the bond (not global tumbling)
- Torque ABOUT the bond (not global torque)
- Isolates the actual torsional degree of freedom

**Physical meaning**:
- Λ = |φ̇ · τ| is the **rate of work** done by/against the torsional potential
- High when: fast rotation × strong restoring force
- Analogous to mechanical power = ω · τ in rigid body mechanics

**Connection to adaptive timestep**:
- High Λ → system fighting against stiff potential
- Integration error grows with stiffness
- Want small dt when Λ is high

### Comparison to Bivector Formalism

In RL/bandits, we needed full Cl(3,1) bivectors because distributions are inherently multi-dimensional.

In MD torsions:
- Each bond rotation is **1D** (around a fixed axis)
- No need for full bivector commutator
- Scalar product φ̇·τ captures the physics directly

**Possible extension**: If we wanted multi-bond coupling (e.g., φ and ψ interactions in proteins), THEN bivector formalism might help. But for Stage-1, keep it simple.

---

## Static Scan (Future)

The script includes a placeholder `static_dihedral_scan()` for:

1. Loop φ from -180° to 180° (5° steps)
2. For each φ:
   - Rotate atoms to set dihedral = φ
   - Minimize other DOFs
   - Compute V(φ), τ(φ)
   - Define Λ(φ) = |φ̇_mock · τ(φ)|
3. Plot V(φ), τ(φ), Λ(φ) vs φ
4. Check R²[Λ, V] or R²[Λ, |d²V/dφ²|]

**Expected**:
- τ(φ) peaks at eclipsed (high curvature)
- Λ(φ) follows same pattern
- R² ≥ 0.7 for static scan (cleaner than MD)

To implement, you need:
```python
def rotate_dihedral(positions, i1,i2,i3,i4, target_phi):
    """Rotate atoms on one side of bond to set φ = target_phi"""
    # Standard Rodrigues rotation formula
    # Rotate atoms >= i3 around i2-i3 axis
    pass
```

---

## Comparison to RL Validation

### RL Lambda-Bandit (Validated ✓)

| Metric | Value |
|--------|-------|
| R²[Λ, KL] | 0.89 |
| Improvement | 10-72% |
| Status | Ready for patent |

### MD Local Lambda (Testing...)

| Metric | Target | Status |
|--------|--------|--------|
| R²[Λ, τ] | ≥ 0.5 | **TBD** |
| R²[Λ, V] static | ≥ 0.7 | **TBD** |
| Speedup | 5-10x | **TBD** |

**Key Difference**:
- RL: Distribution space (multi-dimensional, needs bivectors)
- MD: Torsional space (1D per bond, scalar sufficient)

---

## Timeline

**Stage-1 MD Test**: 2-3 hours
- Run script: 5 minutes
- Analyze results: 30 minutes
- Debug if needed: 1-2 hours

**Decision point**: Same day
- **If pass** (R² ≥ 0.5): Proceed to static scan + alanine
- **If fail** (R² < 0.5): Abandon MD, focus on RL patent

**Total Stage-1**: 1 day maximum

---

## References

### Torsional Potential

Butane OPLS-AA:
```
V(φ) = Σ [V₁/2 (1 + cos(φ)) + V₂/2 (1 - cos(2φ)) + V₃/2 (1 + cos(3φ))]
```

Barrier heights:
- Trans → gauche: ~3.5 kcal/mol
- Gauche → eclipsed: ~5 kcal/mol

### Dihedral Calculation

Standard in computational chemistry:
- Blondel & Karplus (1996), J. Comp. Chem.
- Used in VMD, MDAnalysis, etc.

### Previous Results

- Global Λ: R² = 0.0001 ❌
- See: `MD_VALIDATION_HONEST_RESULTS.md`
- See: `MD_REFORMULATION_NOTES.md`

---

**Status**: Ready for validation
**Priority**: HIGH (decisive test for MD patent direction)
**Risk**: Low (1 day investment, clear go/no-go)
**Upside**: If works, unlocks pharma/biotech market

**Created**: November 14, 2024
**Author**: Rick Mathews
**For**: Claude Code Web execution
