# Path A Extension Sprints - Implementation Plan

**Date:** November 2024
**Status:** In Progress
**Branch:** `claude/bivector-atomic-physics-day1-01ADXMGPFDQNi9odvadCP2WG`

---

## Overview

Path A (Λ-adaptive timestep for MD) is **production-ready** for single-torsion systems. These three sprints extend the capability to production-scale biomolecular systems with **multiple torsions**, **sidechain dynamics**, and **realistic NVT conditions**.

**Business Value:**
- Strengthen provisional patent with additional claims
- Demonstrate generalization to real drug discovery workflows
- Create publishable validation data for proteins

---

## Sprint 1: Multi-Torsion Λ_global Monitoring

### Objective
Monitor **all backbone φ/ψ torsions simultaneously** and use a global stiffness metric to control timestep.

### Technical Approach

**Current:** Single torsion Λ_stiff = |φ̇ · Q_φ|

**New:** Multi-torsion global stiffness

Two candidate formulas:
1. **Max aggregation:** Λ_global = max(Λ₁, Λ₂, ..., Λₙ)
   - Most conservative (protect against ANY stiff event)
   - Simplest implementation

2. **RMS aggregation:** Λ_global = √(Σᵢ Λᵢ²/n)
   - Smoother behavior
   - Better for averaging over many soft modes

**Decision:** Start with **max aggregation** (safety first)

### Implementation Steps

1. **Modify `LambdaAdaptiveVerletIntegrator`:**
   - Change `torsion_atoms` parameter to `torsion_atoms_list` (list of tuples)
   - Loop over all torsions to compute Λᵢ
   - Apply Λ_global = max(Λᵢ)
   - Use same EMA smoothing and adaptation logic

2. **Test on Ala12:**
   - Use `protein_torsion_utils.py` to find all φ/ψ torsions
   - Monitor all 10-11 backbone torsions
   - Compare vs single-torsion results
   - Validate energy drift still <0.5%

3. **Create validation test:**
   - `test_multitorsion_nve.py`
   - Fixed 0.5 fs baseline
   - Adaptive multi-torsion with k=0.0001
   - Plot: Energy drift, Λ_global(t), individual Λᵢ(t)

### Acceptance Criteria

✅ Code supports arbitrary number of torsions
✅ NVE energy drift <0.5% on Ala12
✅ Λ_global tracking shows correct max-detection behavior
✅ Per-torsion Λᵢ plots show heterogeneous dynamics
✅ Documentation updated with multi-torsion usage example

### Expected Effort
**2-4 hours**

### Datasets
- Ala12 helix (already created)
- Optional: Download small protein from PDB (ubiquitin, villin headpiece)

---

## Sprint 2: Sidechain Torsion (χ) Support

### Objective
Extend beyond backbone to monitor **sidechain χ angles** critical for drug binding pocket dynamics.

### Technical Approach

**Sidechain Torsions (χ₁, χ₂, etc.):**
- χ₁: N - Cα - Cβ - Cγ (first sidechain bond)
- χ₂: Cα - Cβ - Cγ - Cδ (second sidechain bond)
- Critical for: Phe, Tyr, Trp, Arg, Lys (aromatic flips, charged group motion)

**Challenge:** Sidechain topology varies by residue type (need residue-specific templates)

**Solution:** Create `sidechain_torsion_templates.py` with common χ₁ definitions for standard residues

### Implementation Steps

1. **Create sidechain torsion finder:**
   - `sidechain_torsion_utils.py`
   - Templates for χ₁ in all standard amino acids
   - Function: `get_sidechain_chi1_torsions(topology)`

2. **Test on protein with long sidechains:**
   - Create Phe₁₀ poly-phenylalanine (aromatic rings)
   - OR download 1UBQ (ubiquitin) from PDB
   - Monitor backbone + sidechain simultaneously

3. **Validation test:**
   - `test_sidechain_nve.py`
   - Track Λ_global over both backbone AND sidechain
   - Show aromatic ring flips create high Λ events
   - Plot: Λ_backbone vs Λ_sidechain contributions

### Acceptance Criteria

✅ Sidechain χ₁ finder works for standard residues
✅ Multi-torsion integrator accepts mixed backbone/sidechain
✅ NVE validation on protein with sidechains
✅ Visualization shows sidechain-driven Λ spikes
✅ Documentation includes drug discovery use case

### Expected Effort
**2-3 hours**

### Datasets
- Phe₁₀ (create manually, similar to Ala12)
- 1UBQ ubiquitin (PDB download, 76 residues, well-characterized)

---

## Sprint 3: NVT Validation with Langevin Thermostat

### Objective
Validate Λ-adaptive integrator under **realistic production conditions** (constant temperature, not NVE).

### Technical Approach

**Current:** NVE ensemble (microcanonical, energy conservation test)

**New:** NVT ensemble (canonical, constant temperature)

**OpenMM Implementation:**
- Replace VerletIntegrator with LangevinIntegrator
- Temperature: 300 K
- Friction coefficient: 1.0 ps⁻¹
- Adaptive timestep still controlled by Λ_stiff

**Key Validation Metrics:**
1. Temperature distribution (should be Gaussian around 300 K)
2. Kinetic energy fluctuations (validate Maxwell-Boltzmann)
3. Structural stability (backbone RMSD over longer runs)
4. Λ_stiff behavior (should adapt to thermal fluctuations)

### Implementation Steps

1. **Modify integrator class:**
   - Add `thermostat="langevin"` option
   - Use LangevinIntegrator as base instead of VerletIntegrator
   - Apply same dt adaptation logic

2. **Create NVT validation test:**
   - `test_nvt_langevin_protein.py`
   - Run 50 ps (5× longer than NVE tests)
   - Track: T(t), KE(t), RMSD(t), Λ(t)

3. **Temperature validation:**
   - Compute temperature histogram
   - Check mean ≈ 300 K, σ ≈ expected thermal fluctuations
   - Compare fixed vs adaptive thermalization

### Acceptance Criteria

✅ Langevin integration with adaptive timestep
✅ Temperature distribution matches target (300±5 K)
✅ Structural stability over 50 ps
✅ Λ_stiff adapts to thermal fluctuations
✅ Documentation includes production workflow example

### Expected Effort
**1-2 hours**

### Datasets
- Ala12 helix (reuse)
- Optional: Test on folded protein (1UBQ) for longer stability check

---

## Success Metrics (All Sprints)

### Patent Strengthening
- ✅ Add Claim 9: Multi-torsion Λ_global method
- ✅ Add Claim 10: Sidechain χ angle monitoring
- ✅ Add Claim 11: NVT ensemble validation

### Publication Data
- ✅ 3 new validation figures (one per sprint)
- ✅ Performance comparison table (speedup + stability)
- ✅ Drug discovery use case demonstration

### Code Quality
- ✅ All new code in `lambda_adaptive_integrator.py`
- ✅ Comprehensive unit tests
- ✅ Updated documentation and usage examples

---

## Timeline

**Total Estimated Effort:** 5-9 hours

| Sprint | Tasks | Effort | Status |
|--------|-------|--------|--------|
| Sprint 1 | Multi-torsion Λ_global | 2-4 hrs | 🔄 In Progress |
| Sprint 2 | Sidechain χ monitoring | 2-3 hrs | ⏳ Pending |
| Sprint 3 | NVT Langevin validation | 1-2 hrs | ⏳ Pending |

---

## Deliverables

### Code Files
- `lambda_adaptive_integrator.py` (updated with multi-torsion support)
- `sidechain_torsion_utils.py` (new - χ angle finder)
- `test_multitorsion_nve.py` (Sprint 1 validation)
- `test_sidechain_nve.py` (Sprint 2 validation)
- `test_nvt_langevin_protein.py` (Sprint 3 validation)

### Documentation
- `SPRINT_1_RESULTS.md` (multi-torsion validation report)
- `SPRINT_2_RESULTS.md` (sidechain validation report)
- `SPRINT_3_RESULTS.md` (NVT validation report)
- Updated `PATH_A_STATUS.md` with extension results

### Validation Plots
- `multitorsion_nve_validation.png`
- `sidechain_torsion_monitoring.png`
- `nvt_langevin_validation.png`

---

## Risk Assessment

### Low Risk ✅
- Sprint 1: Direct extension of working code
- Sprint 3: Well-established Langevin integration

### Medium Risk ⚠️
- Sprint 2: Sidechain topology is residue-dependent (need templates)

### Mitigation
- Start with simple cases (χ₁ only)
- Use well-characterized proteins (ubiquitin)
- Fallback: Document backbone-only as production mode

---

*Sprint Plan Created: November 2024*
*Rick Mathews - Bivector Framework*
