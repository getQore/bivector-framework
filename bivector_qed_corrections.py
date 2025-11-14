#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bivector Framework for QED Radiative Corrections

Key insight: Don't try to predict absolute g-2 value.
Instead, predict the CORRECTIONS beyond leading-order QED.

QED expansion:
a_e = (alpha/2pi) + C2*(alpha/pi)^2 + C3*(alpha/pi)^3 + ...

where C2, C3, etc. are numerical coefficients from Feynman diagrams.

Hypothesis: C_n = f(Lambda/hbar) from bivector geometry

Rick Mathews
November 2024
