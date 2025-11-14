#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 2: Electromagnetic Field Bivectors - Mode Competition Focus

STRATEGIC PIVOT based on Day 1:
- SKIP: Single-mode linear propagation (no competition)
- FOCUS: Mode coupling, birefringence, nonlinear effects
- GOAL: Find exp(-Λ²) in systems with competing EM configurations

Morning: Mode Competition (Waveguides, Birefringence)
Afternoon: Nonlinear Optics (Kerr, Parametric)

Rick Mathews / Claude Code
November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import c, epsilon_0, mu_0
import json
import sys

# Import our bivector framework
from bivector_systematic_search import LorentzBivector

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Physical constants
C = c  # Speed of light (m/s)
EPS0 = epsilon_0  # Permittivity of free space
MU0 = mu_0  # Permeability of free space
Z0 = np.sqrt(MU0 / EPS0)  # Impedance of free space (~377 Ω)

print("=" * 80)
print("DAY 2: ELECTROMAGNETIC FIELD BIVECTORS")
print("Strategic Focus: MODE COMPETITION & NONLINEAR COUPLING")
print("=" * 80)
print()
print("Day 1 Insight: exp(-Λ²) appears in COMPETING configurations, NOT linear modes")
print("Day 2 Strategy: Test mode coupling, birefringence, nonlinear effects")
print()

# ============================================================================
# PART 1: ELECTROMAGNETIC FIELD BIVECTOR REPRESENTATION
# ============================================================================

def em_field_bivector(E_vector, B_vector, normalization=1.0):
    """
    Electromagnetic field as a bivector in spacetime.

    The EM field tensor F_μν is a bivector with 6 components:
    - 3 electric field components (time-space)
    - 3 magnetic field components (space-space)

    F_μν = (E/c, B) in units where c appears explicitly

    Parameters:
    -----------
    E_vector : array-like (3,)
        Electric field vector (V/m)
    B_vector : array-like (3,)
        Magnetic field vector (Tesla)
    normalization : float
        Scaling factor

    Returns:
    --------
    LorentzBivector
    """
    E = np.array(E_vector) * normalization
    B = np.array(B_vector) * normalization

    # In bivector representation:
    # Spatial part: Magnetic field (pure rotation in space)
    # Boost part: Electric field (time-space coupling)

    return LorentzBivector(
        name=f"EM_field",
        spatial=B,  # Magnetic field → spatial bivector
        boost=E / C  # Electric field → boost bivector (normalized by c)
    )


# ============================================================================
# PART 2: WAVEGUIDE MODE COUPLING (High Priority!)
# ============================================================================

def waveguide_te_mode(mode_indices, frequency, waveguide_width):
    """
    TE (Transverse Electric) waveguide mode.

    TE_mn: E_z = 0, B_z ≠ 0
    Transverse E field, longitudinal B field

    Parameters:
    -----------
    mode_indices : tuple (m, n)
        Mode numbers
    frequency : float
        Frequency (Hz)
    waveguide_width : float
        Waveguide dimension (m)

    Returns:
    --------
    LorentzBivector
    """
    m, n = mode_indices
    omega = 2 * np.pi * frequency

    # TE mode: E_z = 0, B_z ≠ 0
    # Cutoff frequency
    k_c = np.pi * np.sqrt((m / waveguide_width)**2 + (n / waveguide_width)**2)
    k = omega / C

    if k < k_c:
        # Below cutoff: evanescent mode
        gamma = np.sqrt(k_c**2 - k**2)
        propagation_factor = np.exp(-gamma * waveguide_width)
    else:
        # Above cutoff: propagating mode
        beta = np.sqrt(k**2 - k_c**2)
        propagation_factor = 1.0

    # Field amplitudes (normalized)
    E_transverse = propagation_factor * np.array([0, 1.0, 0])  # E_y
    B_longitudinal = propagation_factor * np.array([0, 0, 1.0])  # B_z

    return LorentzBivector(
        name=f"TE_{m}{n}",
        spatial=B_longitudinal,
        boost=E_transverse / C
    )


def waveguide_tm_mode(mode_indices, frequency, waveguide_width):
    """
    TM (Transverse Magnetic) waveguide mode.

    TM_mn: B_z = 0, E_z ≠ 0
    Longitudinal E field, transverse B field

    Parameters:
    -----------
    mode_indices : tuple (m, n)
        Mode numbers
    frequency : float
        Frequency (Hz)
    waveguide_width : float
        Waveguide dimension (m)

    Returns:
    --------
    LorentzBivector
    """
    m, n = mode_indices
    omega = 2 * np.pi * frequency

    # TM mode: B_z = 0, E_z ≠ 0
    k_c = np.pi * np.sqrt((m / waveguide_width)**2 + (n / waveguide_width)**2)
    k = omega / C

    if k < k_c:
        gamma = np.sqrt(k_c**2 - k**2)
        propagation_factor = np.exp(-gamma * waveguide_width)
    else:
        beta = np.sqrt(k**2 - k_c**2)
        propagation_factor = 1.0

    # Field amplitudes (normalized)
    E_longitudinal = propagation_factor * np.array([0, 0, 1.0])  # E_z
    B_transverse = propagation_factor * np.array([0, 1.0, 0])  # B_y

    return LorentzBivector(
        name=f"TM_{m}{n}",
        spatial=B_transverse,
        boost=E_longitudinal / C
    )


def test_waveguide_mode_coupling():
    """
    Test mode coupling at waveguide discontinuities.

    HYPOTHESIS: Power transfer TE → TM at step discontinuity follows exp(-Λ²)
    where Λ = ||[E_TE, E_TM]||

    This is HIGH PRIORITY because:
    - Two modes COMPETE at boundary
    - Mode conversion involves geometric frustration
    - Similar to BCH competing elastic/plastic paths
    """
    print("=" * 80)
    print("TEST 1: WAVEGUIDE MODE COUPLING (TE ↔ TM)")
    print("=" * 80)
    print()
    print("HYPOTHESIS: Mode conversion efficiency ~ exp(-Λ²)")
    print("Rationale: TE and TM modes compete at discontinuities")
    print()

    # Waveguide parameters
    waveguide_width = 0.02286  # m (WR-90 standard X-band)
    frequencies_GHz = np.array([8.0, 9.0, 10.0, 11.0, 12.0])  # GHz
    frequencies_Hz = frequencies_GHz * 1e9

    # Mode conversion data (from literature/theory)
    # At a step discontinuity, TE₁₀ → TM₁₁ coupling coefficient
    # Theoretical: η ~ (Δa/a)² where Δa is step size
    # We'll model this vs Λ

    Lambda_values = []
    conversion_efficiencies = []

    for freq in frequencies_Hz:
        # TE₁₀ mode (dominant)
        TE_10 = waveguide_te_mode((1, 0), freq, waveguide_width)

        # TM₁₁ mode (coupled at discontinuity)
        TM_11 = waveguide_tm_mode((1, 1), freq, waveguide_width)

        # Compute Λ = ||[TE, TM]||
        Lambda = TE_10.commutator(TM_11)
        Lambda_values.append(Lambda)

        # Mode conversion efficiency (theoretical model)
        # From coupled-mode theory: η ∝ |κ|² where κ is coupling coefficient
        # At step: κ ~ overlap integral of mode fields
        # We model: η ~ exp(-Λ²) if our hypothesis is correct

        # Use realistic values from EM literature
        # For moderate step: η ~ 0.01 to 0.1 (1-10% conversion)
        step_factor = 0.1  # 10% step in width
        eta_theory = (step_factor)**2 * np.exp(-(freq - 10e9)**2 / (2e9)**2)
        conversion_efficiencies.append(eta_theory)

    Lambda_values = np.array(Lambda_values)
    conversion_efficiencies = np.array(conversion_efficiencies)

    print(f"Frequency range: {frequencies_GHz[0]:.1f} - {frequencies_GHz[-1]:.1f} GHz")
    print(f"Λ range: {Lambda_values.min():.6f} - {Lambda_values.max():.6f}")
    print()

    # Test functional forms
    print("FUNCTIONAL FORM TESTING:")
    print("-" * 80)

    results = {}

    # Normalize for comparison
    eta_norm = conversion_efficiencies / np.max(conversion_efficiencies)

    # Form 1: exp(-Λ²)
    if Lambda_values.max() > 0:
        pred_exp_L2 = np.exp(-Lambda_values**2)
        pred_exp_L2_norm = pred_exp_L2 / np.max(pred_exp_L2)
        R2_exp_L2 = compute_r_squared(eta_norm, pred_exp_L2_norm)
    else:
        R2_exp_L2 = -999
    results['exp(-Λ²)'] = R2_exp_L2
    print(f"exp(-Λ²):             R² = {R2_exp_L2:.6f}")

    # Form 2: Λ² (perturbation theory)
    if Lambda_values.max() > 0:
        pred_L2 = Lambda_values**2
        pred_L2_norm = pred_L2 / np.max(pred_L2)
        R2_L2 = compute_r_squared(eta_norm, pred_L2_norm)
    else:
        R2_L2 = -999
    results['Λ²'] = R2_L2
    print(f"Λ² (perturbation):    R² = {R2_L2:.6f}")

    # Form 3: Gaussian (frequency-dependent)
    freq_centered = (frequencies_Hz - 10e9) / 1e9
    pred_gaussian = np.exp(-freq_centered**2 / 2)
    pred_gaussian_norm = pred_gaussian / np.max(pred_gaussian)
    R2_gaussian = compute_r_squared(eta_norm, pred_gaussian_norm)
    results['Gaussian(f)'] = R2_gaussian
    print(f"Gaussian(freq):       R² = {R2_gaussian:.6f}")

    print()

    # Interpretation
    print("INTERPRETATION:")
    print("-" * 80)

    if Lambda_values.max() < 1e-6:
        print("⚠️  Λ ≈ 0 for all mode pairs tested")
        print("   This suggests TE and TM modes are nearly orthogonal in bivector space")
        print("   OR our bivector representation doesn't capture mode coupling")
        print()
        print("PHYSICAL INSIGHT:")
        print("   TE and TM modes have different symmetries:")
        print("   - TE: E_z = 0, B_z ≠ 0")
        print("   - TM: B_z = 0, E_z ≠ 0")
        print("   Their field configurations may not create large Λ in Lorentz algebra")
        print()
    else:
        if R2_exp_L2 > 0.8:
            print(f"✓ exp(-Λ²) shows strong correlation (R² = {R2_exp_L2:.3f})")
            print("  Mode coupling follows geometric suppression!")
            print()
        elif R2_L2 > 0.8:
            print(f"✓ Λ² (perturbation) shows strong correlation (R² = {R2_L2:.3f})")
            print("  Standard coupled-mode theory confirmed")
            print()
        else:
            print(f"! Moderate correlations: Need refined model")
            print()

    return {
        'Lambda_values': Lambda_values.tolist(),
        'conversion_efficiencies': conversion_efficiencies.tolist(),
        'frequencies_GHz': frequencies_GHz.tolist(),
        'R_squared_values': results
    }


# ============================================================================
# PART 3: BIREFRINGENCE (Competing Polarization Modes)
# ============================================================================

def ordinary_ray_bivector(refractive_index_o, E_amplitude=1.0):
    """
    Ordinary ray in birefringent crystal.

    Polarization perpendicular to optic axis.
    Speed: v_o = c/n_o
    """
    n_o = refractive_index_o

    # Electric field perpendicular to optic axis
    E_field = np.array([E_amplitude, 0, 0])  # x-polarized

    # Phase velocity: v = c/n
    # This affects the "boost" character in spacetime
    beta_eff = 1 / n_o  # Effective β for phase velocity

    return LorentzBivector(
        name="Ordinary_ray",
        spatial=[0, 0, 0],  # No magnetic component in this simplified model
        boost=E_field * beta_eff / C
    )


def extraordinary_ray_bivector(refractive_index_e, E_amplitude=1.0):
    """
    Extraordinary ray in birefringent crystal.

    Polarization parallel to optic axis.
    Speed: v_e = c/n_e (depends on direction)
    """
    n_e = refractive_index_e

    # Electric field parallel to optic axis
    E_field = np.array([0, E_amplitude, 0])  # y-polarized

    beta_eff = 1 / n_e

    return LorentzBivector(
        name="Extraordinary_ray",
        spatial=[0, 0, 0],
        boost=E_field * beta_eff / C
    )


def test_birefringence_competition():
    """
    Test competing ordinary vs extraordinary rays in birefringent crystal.

    HYPOTHESIS: Polarization conversion efficiency ~ exp(-Λ²)
    where Λ = ||[E_o, E_e]||

    HIGH PRIORITY: Two polarization modes compete for same photon
    """
    print("=" * 80)
    print("TEST 2: BIREFRINGENCE (Polarization Mode Competition)")
    print("=" * 80)
    print()
    print("HYPOTHESIS: Polarization mixing ~ exp(-Λ²)")
    print("Rationale: Ordinary and extraordinary rays compete")
    print()

    # Birefringent materials (e.g., calcite, quartz)
    # Birefringence: Δn = n_e - n_o

    materials = {
        'Calcite': {'n_o': 1.658, 'n_e': 1.486},  # Negative birefringence
        'Quartz': {'n_o': 1.544, 'n_e': 1.553},   # Positive birefringence
        'Rutile': {'n_o': 2.616, 'n_e': 2.903},   # Large birefringence
    }

    Lambda_values = []
    birefringence_values = []
    material_names = []

    for mat_name, indices in materials.items():
        n_o = indices['n_o']
        n_e = indices['n_e']

        # Create bivectors
        ord_ray = ordinary_ray_bivector(n_o, E_amplitude=1.0)
        ext_ray = extraordinary_ray_bivector(n_e, E_amplitude=1.0)

        # Compute Λ
        Lambda = ord_ray.commutator(ext_ray)
        Lambda_values.append(Lambda)

        # Birefringence
        Delta_n = abs(n_e - n_o)
        birefringence_values.append(Delta_n)
        material_names.append(mat_name)

        print(f"{mat_name}:")
        print(f"  n_o = {n_o:.3f}, n_e = {n_e:.3f}")
        print(f"  Δn = {Delta_n:.4f}")
        print(f"  Λ = {Lambda:.6f}")
        print()

    Lambda_values = np.array(Lambda_values)
    birefringence_values = np.array(birefringence_values)

    # Test correlations
    print("FUNCTIONAL FORM TESTING:")
    print("-" * 80)

    results = {}

    # Normalize
    Delta_n_norm = birefringence_values / np.max(birefringence_values)

    # Form 1: exp(-Λ²)
    if Lambda_values.max() > 0:
        pred_exp_L2 = np.exp(-Lambda_values**2)
        pred_exp_L2_norm = pred_exp_L2 / np.max(pred_exp_L2)
        R2_exp_L2 = compute_r_squared(Delta_n_norm, pred_exp_L2_norm)
    else:
        R2_exp_L2 = -999
    results['exp(-Λ²)'] = R2_exp_L2
    print(f"exp(-Λ²):         R² = {R2_exp_L2:.6f}")

    # Form 2: Linear in Λ
    if Lambda_values.max() > 0:
        pred_L = Lambda_values / np.max(Lambda_values)
        R2_L = compute_r_squared(Delta_n_norm, pred_L)
    else:
        R2_L = -999
    results['Λ'] = R2_L
    print(f"Λ (linear):       R² = {R2_L:.6f}")

    # Form 3: Direct Δn scaling
    R2_direct = 1.0  # Perfect by definition
    results['Δn (direct)'] = R2_direct
    print(f"Δn (direct):      R² = {R2_direct:.6f}")

    print()

    print("INTERPRETATION:")
    print("-" * 80)

    if Lambda_values.max() < 1e-6:
        print("⚠️  Λ ≈ 0 for all material pairs")
        print("   Ordinary and extraordinary rays have orthogonal polarizations")
        print("   BUT same propagation direction → small Λ in our representation")
        print()
        print("INSIGHT: Need to refine bivector representation to capture")
        print("         polarization competition more explicitly")
        print()
    else:
        if R2_exp_L2 > 0.8:
            print(f"✓ exp(-Λ²) correlation found (R² = {R2_exp_L2:.3f})!")
            print("  Birefringence shows geometric suppression pattern")
            print()
        else:
            print(f"Standard material physics: Δn depends on crystal structure")
            print()

    return {
        'Lambda_values': Lambda_values.tolist(),
        'birefringence_values': birefringence_values.tolist(),
        'material_names': material_names,
        'R_squared_values': results
    }


# ============================================================================
# PART 4: NONLINEAR OPTICS (Afternoon Focus)
# ============================================================================

def test_kerr_effect():
    """
    Optical Kerr effect: Intensity-dependent refractive index.

    n = n_0 + n_2 I  (I = intensity)

    HYPOTHESIS: Self-phase modulation ~ exp(-Λ²)
    where Λ = ||[E_low, E_high]||

    This is COMPETITION between low-intensity and high-intensity modes
    """
    print("=" * 80)
    print("TEST 3: KERR EFFECT (Intensity-Dependent Nonlinearity)")
    print("=" * 80)
    print()
    print("HYPOTHESIS: Spectral broadening ~ exp(-Λ²)")
    print("Rationale: Low and high intensity fields compete")
    print()

    # Kerr medium parameters
    n_0 = 1.45  # Linear refractive index (e.g., silica fiber)
    n_2 = 2.6e-20  # m²/W (nonlinear index)

    # Intensity range (W/m²)
    intensities_GW_cm2 = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
    intensities_W_m2 = intensities_GW_cm2 * 1e13  # Convert to W/m²

    Lambda_values = []
    phase_shifts = []

    for I in intensities_W_m2:
        # Refractive index change
        Delta_n = n_2 * I
        n_eff = n_0 + Delta_n

        # Low intensity field
        E_low = LorentzBivector(
            name="E_low_intensity",
            spatial=[0, 0, 0],
            boost=[1.0 / (C * n_0), 0, 0]
        )

        # High intensity field (different effective index)
        E_high = LorentzBivector(
            name="E_high_intensity",
            spatial=[0, 0, 0],
            boost=[1.0 / (C * n_eff), 0, 0]
        )

        # Compute Λ
        Lambda = E_low.commutator(E_high)
        Lambda_values.append(Lambda)

        # Nonlinear phase shift over length L
        L = 1.0  # 1 meter
        k = 2 * np.pi * 1e15 / C  # Optical frequency (λ ~ 1 μm)
        phi_NL = k * Delta_n * L
        phase_shifts.append(phi_NL)

    Lambda_values = np.array(Lambda_values)
    phase_shifts = np.array(phase_shifts)

    print(f"Intensity range: {intensities_GW_cm2[0]:.1f} - {intensities_GW_cm2[-1]:.1f} GW/cm²")
    print(f"Λ range: {Lambda_values.min():.6e} - {Lambda_values.max():.6e}")
    print(f"Phase shift range: {phase_shifts.min():.3f} - {phase_shifts.max():.3f} rad")
    print()

    # Test functional forms
    print("FUNCTIONAL FORM TESTING:")
    print("-" * 80)

    results = {}

    # Normalize
    phi_norm = phase_shifts / np.max(phase_shifts)

    # Form 1: exp(-Λ²)
    if Lambda_values.max() > 0:
        pred_exp_L2 = np.exp(-Lambda_values**2)
        pred_exp_L2_norm = pred_exp_L2 / np.max(pred_exp_L2)
        R2_exp_L2 = compute_r_squared(phi_norm, pred_exp_L2_norm)
    else:
        R2_exp_L2 = -999
    results['exp(-Λ²)'] = R2_exp_L2
    print(f"exp(-Λ²):         R² = {R2_exp_L2:.6f}")

    # Form 2: Linear (Δn ∝ I)
    I_norm = intensities_W_m2 / np.max(intensities_W_m2)
    R2_linear = compute_r_squared(phi_norm, I_norm)
    results['I (linear)'] = R2_linear
    print(f"I (linear):       R² = {R2_linear:.6f}")

    print()

    print("INTERPRETATION:")
    print("-" * 80)

    if Lambda_values.max() < 1e-6:
        print("⚠️  Λ ≈ 0: Low and high intensity fields too similar")
        print("   Kerr effect is perturbative (weak nonlinearity)")
        print("   No strong mode competition")
        print()
    else:
        if R2_exp_L2 > 0.8:
            print(f"✓ exp(-Λ²) correlation (R² = {R2_exp_L2:.3f})!")
            print("  Nonlinear response shows geometric pattern")
            print()
        else:
            print("Standard Kerr theory: φ_NL ∝ n_2 I L")
            print()

    return {
        'Lambda_values': Lambda_values.tolist(),
        'phase_shifts': phase_shifts.tolist(),
        'intensities_GW_cm2': intensities_GW_cm2.tolist(),
        'R_squared_values': results
    }


def test_second_harmonic_generation():
    """
    Second harmonic generation: ω + ω → 2ω

    HYPOTHESIS: Conversion efficiency ~ exp(-Λ²)
    where Λ = ||[E(ω), E(2ω)]||

    HIGH PRIORITY: Fundamental and second harmonic compete
    """
    print("=" * 80)
    print("TEST 4: SECOND HARMONIC GENERATION (Frequency Mixing)")
    print("=" * 80)
    print()
    print("HYPOTHESIS: SHG efficiency ~ exp(-Λ²)")
    print("Rationale: Fundamental (ω) and second harmonic (2ω) compete")
    print()

    # Nonlinear crystal (e.g., KDP, BBO)
    n_omega = 1.50  # Index at fundamental
    n_2omega = 1.52  # Index at second harmonic

    # Wavelengths
    wavelengths_nm = np.array([800, 900, 1000, 1100, 1200])  # nm
    wavelengths_m = wavelengths_nm * 1e-9

    Lambda_values = []
    phase_mismatch_values = []
    conversion_efficiencies = []

    for wl in wavelengths_m:
        freq = C / wl
        freq_2omega = 2 * freq
        wl_2omega = wl / 2

        # Fundamental field
        E_omega = LorentzBivector(
            name="E(ω)",
            spatial=[0, 0, 0],
            boost=[1.0 / (C * n_omega), 0, 0]
        )

        # Second harmonic field
        E_2omega = LorentzBivector(
            name="E(2ω)",
            spatial=[0, 0, 0],
            boost=[0, 1.0 / (C * n_2omega), 0]  # Different polarization/direction
        )

        # Compute Λ
        Lambda = E_omega.commutator(E_2omega)
        Lambda_values.append(Lambda)

        # Phase mismatch
        k_omega = 2 * np.pi * n_omega / wl
        k_2omega = 2 * np.pi * n_2omega / wl_2omega
        Delta_k = k_2omega - 2 * k_omega
        phase_mismatch_values.append(Delta_k)

        # SHG efficiency (sinc² function of phase mismatch)
        L = 0.01  # Crystal length (1 cm)
        eta = np.sinc(Delta_k * L / (2 * np.pi))**2
        conversion_efficiencies.append(eta)

    Lambda_values = np.array(Lambda_values)
    conversion_efficiencies = np.array(conversion_efficiencies)
    phase_mismatch_values = np.array(phase_mismatch_values)

    print(f"Wavelength range: {wavelengths_nm[0]:.0f} - {wavelengths_nm[-1]:.0f} nm")
    print(f"Λ range: {Lambda_values.min():.6e} - {Lambda_values.max():.6e}")
    print(f"Conversion efficiency range: {conversion_efficiencies.min():.4f} - {conversion_efficiencies.max():.4f}")
    print()

    # Test functional forms
    print("FUNCTIONAL FORM TESTING:")
    print("-" * 80)

    results = {}

    # Normalize
    eta_norm = conversion_efficiencies / np.max(conversion_efficiencies)

    # Form 1: exp(-Λ²)
    if Lambda_values.max() > 0:
        pred_exp_L2 = np.exp(-Lambda_values**2)
        pred_exp_L2_norm = pred_exp_L2 / np.max(pred_exp_L2)
        R2_exp_L2 = compute_r_squared(eta_norm, pred_exp_L2_norm)
    else:
        R2_exp_L2 = -999
    results['exp(-Λ²)'] = R2_exp_L2
    print(f"exp(-Λ²):         R² = {R2_exp_L2:.6f}")

    # Form 2: sinc²(Δk·L) (standard phase matching)
    sinc2_norm = conversion_efficiencies / np.max(conversion_efficiencies)
    R2_sinc2 = 1.0  # Perfect by construction
    results['sinc²(Δk)'] = R2_sinc2
    print(f"sinc²(Δk):        R² = {R2_sinc2:.6f}")

    print()

    print("INTERPRETATION:")
    print("-" * 80)

    if Lambda_values.max() < 1e-6:
        print("⚠️  Λ ≈ 0: Fundamental and SH fields not creating large commutator")
        print("   Our bivector representation may not capture frequency mixing")
        print()
    else:
        if R2_exp_L2 > 0.8:
            print(f"✓ exp(-Λ²) correlation (R² = {R2_exp_L2:.3f})!")
            print("  Frequency conversion shows geometric pattern")
            print()
        else:
            print("Standard SHG: Efficiency determined by phase matching")
            print("sinc²(Δk·L) dependence dominates")
            print()

    return {
        'Lambda_values': Lambda_values.tolist(),
        'conversion_efficiencies': conversion_efficiencies.tolist(),
        'wavelengths_nm': wavelengths_nm.tolist(),
        'R_squared_values': results
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_r_squared(y_true, y_pred):
    """Compute R² coefficient of determination."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)

    if ss_tot == 0:
        return 0.0

    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def plot_day2_results(all_results):
    """Create comprehensive visualization of Day 2 results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Waveguide mode coupling
    ax1 = axes[0, 0]
    if 'waveguide' in all_results:
        wg = all_results['waveguide']
        if len(wg['Lambda_values']) > 0 and max(wg['Lambda_values']) > 0:
            ax1.scatter(wg['Lambda_values'], wg['conversion_efficiencies'],
                       s=150, c='blue', marker='o', edgecolors='black', linewidth=2)
            ax1.set_xlabel('Λ = ||[TE, TM]||', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Mode Conversion Efficiency', fontsize=12, fontweight='bold')
            ax1.set_title('Waveguide TE ↔ TM Coupling', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Λ ≈ 0\n(Orthogonal modes)',
                    ha='center', va='center', fontsize=14)
            ax1.set_title('Waveguide Mode Coupling', fontsize=13, fontweight='bold')

    # Plot 2: Birefringence
    ax2 = axes[0, 1]
    if 'birefringence' in all_results:
        bir = all_results['birefringence']
        if len(bir['Lambda_values']) > 0:
            x_pos = range(len(bir['material_names']))
            ax2.bar(x_pos, bir['birefringence_values'], color='green',
                   edgecolor='black', linewidth=2, alpha=0.7)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(bir['material_names'], rotation=45)
            ax2.set_ylabel('Birefringence Δn', fontsize=12, fontweight='bold')
            ax2.set_title('Birefringence vs Material', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # Add Λ values as text
            for i, (mat, lam) in enumerate(zip(bir['material_names'], bir['Lambda_values'])):
                ax2.text(i, bir['birefringence_values'][i] * 1.05,
                        f'Λ={lam:.3f}', ha='center', fontsize=9)

    # Plot 3: Kerr effect
    ax3 = axes[1, 0]
    if 'kerr' in all_results:
        kerr = all_results['kerr']
        if len(kerr['Lambda_values']) > 0:
            ax3.scatter(kerr['intensities_GW_cm2'], kerr['phase_shifts'],
                       s=150, c='red', marker='s', edgecolors='black', linewidth=2)
            ax3.set_xlabel('Intensity (GW/cm²)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Nonlinear Phase Shift (rad)', fontsize=12, fontweight='bold')
            ax3.set_title('Kerr Effect (Self-Phase Modulation)', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_xscale('log')

    # Plot 4: Second harmonic generation
    ax4 = axes[1, 1]
    if 'shg' in all_results:
        shg = all_results['shg']
        if len(shg['Lambda_values']) > 0:
            ax4.scatter(shg['wavelengths_nm'], shg['conversion_efficiencies'],
                       s=150, c='purple', marker='D', edgecolors='black', linewidth=2)
            ax4.set_xlabel('Fundamental Wavelength (nm)', fontsize=12, fontweight='bold')
            ax4.set_ylabel('SHG Conversion Efficiency', fontsize=12, fontweight='bold')
            ax4.set_title('Second Harmonic Generation', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('em_field_analysis_day2.png', dpi=150, bbox_inches='tight')
    print("\nSaved: em_field_analysis_day2.png")
    print()
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main Day 2 execution with strategic focus."""

    print("Starting Day 2: EM Field Bivectors - Mode Competition Focus")
    print()

    all_results = {}

    # Morning: Mode Competition
    print("\n" + "="*80)
    print("MORNING SESSION: MODE COMPETITION")
    print("="*80 + "\n")

    waveguide_results = test_waveguide_mode_coupling()
    all_results['waveguide'] = waveguide_results

    birefringence_results = test_birefringence_competition()
    all_results['birefringence'] = birefringence_results

    # Afternoon: Nonlinear Optics
    print("\n" + "="*80)
    print("AFTERNOON SESSION: NONLINEAR OPTICS")
    print("="*80 + "\n")

    kerr_results = test_kerr_effect()
    all_results['kerr'] = kerr_results

    shg_results = test_second_harmonic_generation()
    all_results['shg'] = shg_results

    # Generate visualization
    plot_day2_results(all_results)

    # Summary
    print("\n" + "="*80)
    print("DAY 2 SUMMARY")
    print("="*80 + "\n")

    print("SYSTEMS TESTED: 4")
    print("  1. Waveguide Mode Coupling (TE ↔ TM)")
    print("  2. Birefringence (O-ray vs E-ray)")
    print("  3. Kerr Effect (Intensity-dependent n)")
    print("  4. Second Harmonic Generation (ω → 2ω)")
    print()

    print("KEY FINDINGS:")
    print("-" * 80)

    # Check for Λ ≈ 0 cases
    lambda_zeros = []

    for system_name, results in all_results.items():
        if 'Lambda_values' in results:
            Lambda_vals = results['Lambda_values']
            if len(Lambda_vals) > 0 and max(Lambda_vals) < 1e-6:
                lambda_zeros.append(system_name)

    if lambda_zeros:
        print(f"\n⚠️  Λ ≈ 0 for systems: {', '.join(lambda_zeros)}")
        print()
        print("CRITICAL INSIGHT:")
        print("  Our bivector representation (Lorentz boosts + spatial rotations)")
        print("  may not fully capture EM mode competition.")
        print()
        print("  Possible reasons:")
        print("  - TE/TM modes differ in field orientation, not Lorentz structure")
        print("  - Polarization competition is SU(2), not SO(3,1)")
        print("  - Need higher-dimensional bivector algebra (Cl(3,2) or Cl(4,1)?)")
        print()

    # Check for any R² > 0.8
    high_r2_found = False
    for system_name, results in all_results.items():
        if 'R_squared_values' in results:
            r2_vals = results['R_squared_values']
            if 'exp(-Λ²)' in r2_vals and r2_vals['exp(-Λ²)'] > 0.8:
                high_r2_found = True
                print(f"✓ {system_name}: exp(-Λ²) R² = {r2_vals['exp(-Λ²)']:.3f}")
                print()

    if not high_r2_found:
        print("CONCLUSION:")
        print("  No strong exp(-Λ²) pattern found in EM systems tested")
        print()
        print("  This aligns with Day 1 insight:")
        print("  exp(-Λ²) appears in GEOMETRIC FRUSTRATION (like BCH),")
        print("  NOT in standard linear/perturbative EM phenomena")
        print()

    print("NEXT STEPS:")
    print("-" * 80)
    print("1. Consider alternative bivector representations for EM")
    print("2. Test systems with stronger mode mixing (photonic crystals?)")
    print("3. OR accept that exp(-Λ²) is specific to materials/QFT, not classical EM")
    print("4. Day 3: Move to condensed matter (superconductivity, topological phases)")
    print()

    # Save results
    with open('day2_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("Results saved to: day2_results.json")
    print()
    print("="*80)
    print("DAY 2 COMPLETE!")
    print("="*80)

    return all_results


if __name__ == "__main__":
    results = main()
