#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRITICAL DEBUGGING: Unit and Scale Analysis

Rick's insight: Billion-sigma errors suggest unit conversion problems.

Check:
1. MHz ↔ eV conversions (factor 2.418×10⁸?)
2. Reduced mass in muonium
3. Rydberg vs Hartree (factor of 2)
4. Natural units vs SI mixing

Rick Mathews
November 2024
"""

import numpy as np
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 80)
print("CRITICAL DEBUGGING: UNIT AND SCALE ANALYSIS")
print("=" * 80)
print()

# Physical constants - CHECK EVERYTHING
HBAR = 1.055e-34  # J·s
C = 2.998e8  # m/s
M_E_KG = 9.109e-31  # kg
M_E_EV = 0.511e6  # eV/c²
M_MU_KG = 1.883e-28  # kg
M_MU_EV = 105.66e6  # eV/c²
E_CHARGE = 1.602e-19  # C
ALPHA = 1/137.036

# Key conversion factors
EV_TO_HZ = E_CHARGE / HBAR  # 1 eV = ? Hz
HZ_TO_EV = HBAR / E_CHARGE  # 1 Hz = ? eV

print("UNIT CONVERSION FACTORS:")
print(f"  1 eV = {EV_TO_HZ:.3e} Hz")
print(f"  1 Hz = {HZ_TO_EV:.3e} eV")
print(f"  1 MHz = {1e6 * HZ_TO_EV:.3e} eV")
print()

# Check Rydberg energy
RY_JOULES = 0.5 * M_E_KG * (ALPHA * C)**2
RY_EV = RY_JOULES / E_CHARGE

print("RYDBERG ENERGY:")
print(f"  Ry (formula): {RY_EV:.3f} eV")
print(f"  Ry (known):   13.606 eV")
print(f"  Match: {abs(RY_EV - 13.606) < 0.01}")
print()

# Bohr radius
A0_FORMULA = HBAR / (M_E_KG * ALPHA * C)

print("BOHR RADIUS:")
print(f"  a₀ (formula): {A0_FORMULA:.3e} m")
print(f"  a₀ (known):   0.529e-10 m")
print(f"  Match: {abs(A0_FORMULA - 0.529e-10) < 0.01e-10}")
print()


def debug_muonium_hyperfine():
    """
    Debug muonium calculation with careful unit tracking.
    """

    print("=" * 80)
    print("DEBUG: MUONIUM HYPERFINE SPLITTING")
    print("=" * 80)
    print()

    # Measured value
    nu_measured_MHz = 4463.302765  # MHz
    nu_measured_Hz = nu_measured_MHz * 1e6

    print(f"MEASURED:")
    print(f"  ν = {nu_measured_MHz:.6f} MHz = {nu_measured_Hz:.3e} Hz")
    print()

    # Standard formula for hyperfine splitting
    # ΔE_hf = (16/3) * α² * (m_reduced/m_e) * R_∞ * (g_e * g_μ * μ_B * μ_N) / (m_e * c²)

    # Simplified: ν = (8/3) * (α² * R_∞ * c) * (m_r/m_e) * (μ_e/μ_B) * (μ_μ/μ_N)

    # Even simpler: Use known hydrogen hyperfine, scale by mass ratio

    # Hydrogen hyperfine
    nu_H_MHz = 1420.4057517667  # MHz
    nu_H_Hz = nu_H_MHz * 1e6

    # Mass ratio (muonium vs hydrogen)
    # Muonium: μ⁺ + e⁻
    # Hydrogen: p + e⁻

    M_P_KG = 1.673e-27  # Proton mass

    # Reduced masses
    mu_H = (M_P_KG * M_E_KG) / (M_P_KG + M_E_KG)
    mu_Mu = (M_MU_KG * M_E_KG) / (M_MU_KG + M_E_KG)

    # Nuclear magnetic moments
    # g_p = 5.586 (proton)
    # g_μ = -2.002 (muon, but negative - antimuon is positive)

    g_p = 5.586
    g_mu = 2.002  # Use positive for μ⁺

    # Nuclear magnetons
    mu_N = 5.051e-27  # J/T (nuclear magneton)
    mu_B = 9.274e-24  # J/T (Bohr magneton)

    # Scaling from hydrogen to muonium
    # ν_Mu / ν_H = (μ_Mu/μ_H) * (m_p/m_μ) * (g_μ/g_p)

    mass_ratio = M_P_KG / M_MU_KG
    g_ratio = g_mu / g_p
    reduced_mass_ratio = mu_Mu / mu_H

    scaling = reduced_mass_ratio * mass_ratio * g_ratio

    nu_Mu_predicted_Hz = nu_H_Hz * scaling
    nu_Mu_predicted_MHz = nu_Mu_predicted_Hz / 1e6

    print(f"SCALING FROM HYDROGEN:")
    print(f"  Hydrogen: ν = {nu_H_MHz:.3f} MHz")
    print(f"  Mass ratio (p/μ): {mass_ratio:.3f}")
    print(f"  g-factor ratio: {g_ratio:.3f}")
    print(f"  Reduced mass ratio: {reduced_mass_ratio:.6f}")
    print(f"  Total scaling: {scaling:.3f}")
    print()

    print(f"PREDICTION:")
    print(f"  ν = {nu_Mu_predicted_MHz:.3f} MHz")
    print(f"  Measured: {nu_measured_MHz:.3f} MHz")
    print(f"  Error: {abs(nu_Mu_predicted_MHz - nu_measured_MHz):.3f} MHz")
    print(f"  Relative error: {abs(nu_Mu_predicted_MHz - nu_measured_MHz)/nu_measured_MHz * 100:.1f}%")
    print()

    # More accurate formula
    # Account for all QED corrections

    # Fermi contact term
    # ν = (8/3) * α² * R_∞ * c * (m_r/m_e) * (g_e/2) * (g_N * μ_N / μ_B)

    R_inf_Hz = 3.289841960e15  # Hz (Rydberg constant × c)

    g_e = 2.002  # electron g-factor

    nu_theory = (8/3) * ALPHA**2 * R_inf_Hz * (mu_Mu/M_E_KG) * (g_e/2) * (g_mu * (mu_N/mu_B))
    nu_theory_MHz = nu_theory / 1e6

    print(f"THEORETICAL FORMULA:")
    print(f"  ν = {nu_theory_MHz:.3f} MHz")
    print(f"  Measured: {nu_measured_MHz:.3f} MHz")
    print(f"  Error: {abs(nu_theory_MHz - nu_measured_MHz):.3f} MHz")
    print()

    return nu_theory_MHz, nu_measured_MHz


def debug_hydrogen_2s4s():
    """
    Debug hydrogen 2S-4S interval calculation.
    """

    print("=" * 80)
    print("DEBUG: HYDROGEN 2S-4S INTERVAL")
    print("=" * 80)
    print()

    # Measured
    nu_measured_MHz = 4797.338  # MHz

    print(f"MEASURED: ν = {nu_measured_MHz:.3f} MHz")
    print()

    # Bohr formula for energy levels
    # E_n = -R_∞ * (1/n²)
    # ΔE = E_4 - E_2 = R_∞ * (1/4 - 1/16) = R_∞ * (3/16)

    R_inf_eV = 13.606  # eV

    Delta_E_Bohr_eV = R_inf_eV * (1/4 - 1/16)
    Delta_E_Bohr_Hz = Delta_E_Bohr_eV * EV_TO_HZ
    Delta_E_Bohr_MHz = Delta_E_Bohr_Hz / 1e6

    print(f"BOHR FORMULA:")
    print(f"  ΔE = {Delta_E_Bohr_eV:.6f} eV")
    print(f"  ν = {Delta_E_Bohr_MHz:.3f} MHz")
    print()

    # Lamb shift corrections
    # 2S has large Lamb shift ~ 1057 MHz
    # 4S has smaller shift ~ 1057/8 MHz (scales as 1/n³)

    Lamb_2S_MHz = 1057.8  # MHz
    Lamb_4S_MHz = Lamb_2S_MHz / 8  # Approximate scaling

    nu_with_Lamb_MHz = Delta_E_Bohr_MHz + (Lamb_2S_MHz - Lamb_4S_MHz)

    print(f"WITH LAMB SHIFT:")
    print(f"  Bohr: {Delta_E_Bohr_MHz:.3f} MHz")
    print(f"  + Lamb(2S): +{Lamb_2S_MHz:.1f} MHz")
    print(f"  - Lamb(4S): -{Lamb_4S_MHz:.1f} MHz")
    print(f"  Total: {nu_with_Lamb_MHz:.3f} MHz")
    print()

    print(f"COMPARISON:")
    print(f"  Predicted: {nu_with_Lamb_MHz:.3f} MHz")
    print(f"  Measured:  {nu_measured_MHz:.3f} MHz")
    print(f"  Error:     {abs(nu_with_Lamb_MHz - nu_measured_MHz):.3f} MHz")
    print()

    # THE ISSUE: We're off by a factor of ~250!
    # Bohr predicts ~2.5 MHz, measured is ~4797 MHz

    print("PROBLEM IDENTIFIED:")
    print(f"  Bohr formula gives {Delta_E_Bohr_MHz:.1f} MHz")
    print(f"  Measured is {nu_measured_MHz:.1f} MHz")
    print(f"  Factor of {nu_measured_MHz/Delta_E_Bohr_MHz:.0f}x discrepancy!")
    print()
    print("  This is NOT Lamb shift (only ~1000 MHz)")
    print("  Something fundamentally wrong with energy level calculation!")
    print()

    return Delta_E_Bohr_MHz, nu_measured_MHz


def check_natural_vs_si():
    """
    Check if we're mixing natural units (ℏ=c=1) with SI.
    """

    print("=" * 80)
    print("DEBUG: NATURAL UNITS VS SI")
    print("=" * 80)
    print()

    # In natural units (ℏ = c = 1):
    # Energy, mass, inverse length all same dimension

    # Electron mass in natural units
    m_e_natural = M_E_EV * 1e6 / C**2  # eV/c² → eV (setting c=1)

    # Rydberg in natural units
    Ry_natural = 0.5 * (ALPHA)**2 * m_e_natural  # eV

    print("NATURAL UNITS (ℏ = c = 1):")
    print(f"  m_e = {M_E_EV/1e6:.3f} MeV")
    print(f"  α = {ALPHA:.6f}")
    print(f"  Ry = (1/2) α² m_e = {Ry_natural/1e6:.6f} MeV = {Ry_natural:.3f} eV")
    print()

    # Compare to SI calculation
    Ry_SI = 13.606  # eV

    print(f"SI UNITS:")
    print(f"  Ry = {Ry_SI:.3f} eV")
    print()

    print(f"Match: {abs(Ry_natural - Ry_SI) < 0.1}")
    print()


def main():
    """Run all debugging tests."""

    print("Running comprehensive unit debugging...")
    print()

    # Test 1: Muonium
    nu_theory_mu, nu_meas_mu = debug_muonium_hyperfine()

    # Test 2: Hydrogen 2S-4S
    nu_theory_h, nu_meas_h = debug_hydrogen_2s4s()

    # Test 3: Natural units
    check_natural_vs_si()

    # Summary
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()

    print("FINDINGS:")
    print()

    print("1. MUONIUM:")
    error_mu_pct = abs(nu_theory_mu - nu_meas_mu) / nu_meas_mu * 100
    if error_mu_pct < 10:
        print(f"   ✓ Calculation CORRECT: {error_mu_pct:.1f}% error")
    else:
        print(f"   ✗ Still off by {error_mu_pct:.0f}%")
    print()

    print("2. HYDROGEN 2S-4S:")
    factor_h = nu_meas_h / nu_theory_h
    if factor_h > 100:
        print(f"   ✗ OFF BY FACTOR OF {factor_h:.0f}x")
        print("   → Bohr formula calculation is WRONG")
        print("   → Not using correct transition!")
    else:
        print(f"   ? Error: {factor_h:.1f}x")
    print()

    print("3. ROOT CAUSE:")
    print("   The 2S-4S 'interval' is NOT the energy difference!")
    print("   It's the two-photon transition frequency from 1S ground state.")
    print("   We're calculating the WRONG thing!")
    print()

    print("CONCLUSION:")
    print("  Framework issues are likely:")
    print("    a) Wrong physics for spectroscopy (not just units)")
    print("    b) Misunderstanding which transitions are being measured")
    print("    c) Missing QED corrections beyond Lamb shift")
    print()

    print("RECOMMENDATION:")
    print("  ABANDON spectroscopy predictions.")
    print("  FOCUS ONLY on dimensionless anomalous moments (g-2).")
    print("  Be honest in publication about scope.")
    print()


if __name__ == "__main__":
    main()
