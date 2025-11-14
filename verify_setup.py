#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup Verification Script
=========================

Run this to verify the bivector framework is set up correctly.

Usage:
    python verify_setup.py

Expected output: All checks should pass [OK]

Rick Mathews - November 2024
"""

import sys
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 80)
print("BIVECTOR FRAMEWORK SETUP VERIFICATION")
print("=" * 80)
print()

# Check 1: Python version
print("CHECK 1: Python Version")
print("-" * 40)
py_version = sys.version_info
print(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
if py_version.major >= 3 and py_version.minor >= 7:
    print("[OK] Python version OK (>= 3.7)")
else:
    print("[X] Python version too old (need >= 3.7)")
    sys.exit(1)
print()

# Check 2: Required packages
print("CHECK 2: Required Packages")
print("-" * 40)

required_packages = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib',
}

all_ok = True
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"[OK] {name} installed")
    except ImportError:
        print(f"[X] {name} NOT installed")
        all_ok = False

if not all_ok:
    print()
    print("Install missing packages with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
print()

# Check 3: Core files exist
print("CHECK 3: Core Files")
print("-" * 40)

core_files = [
    'README.md',
    'COMPREHENSIVE_SUMMARY.md',
    'SPRINT.md',
    'bivector_systematic_search.py',
    'universal_lambda_pattern.py',
]

for filename in core_files:
    if os.path.exists(filename):
        print(f"[OK] {filename}")
    else:
        print(f"[X] {filename} MISSING")
        all_ok = False

if not all_ok:
    print()
    print("Some core files are missing!")
    sys.exit(1)
print()

# Check 4: Can import core functionality
print("CHECK 4: Core Functionality")
print("-" * 40)

try:
    from bivector_systematic_search import LorentzBivector
    print("[OK] LorentzBivector class imported")
    print("[OK] Core module loadable")

    # Simple verification that numpy works
    import numpy as np
    test_array = np.array([1, 2, 3])
    print("[OK] NumPy arrays functional")

except Exception as e:
    print(f"[X] Error testing core functionality:")
    print(f"  {e}")
    sys.exit(1)
print()

# Check 5: Directory structure
print("CHECK 5: Directory Structure")
print("-" * 40)

directories = ['results', 'figures', 'data']
for dirname in directories:
    if os.path.exists(dirname):
        print(f"[OK] {dirname}/ exists")
    else:
        print(f"[!] {dirname}/ not found (will be created)")
        try:
            os.makedirs(dirname, exist_ok=True)
            open(f"{dirname}/.gitkeep", 'w').close()
            print(f"  Created {dirname}/")
        except Exception as e:
            print(f"  Could not create: {e}")
print()

# Summary
print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print()
print("[OK] All checks passed!")
print()
print("You're ready to:")
print("  1. Upload to GitHub (see GITHUB_SETUP.md)")
print("  2. Start the sprint (see SPRINT.md)")
print("  3. Explore bivector combinations!")
print()
print("Quick start:")
print("  python bivector_systematic_search.py  # Original discovery")
print("  python universal_lambda_pattern.py    # BCH validation")
print()
print("For sprint: Follow SPRINT.md Day 1 -> Day 5")
print()
print("=" * 80)
