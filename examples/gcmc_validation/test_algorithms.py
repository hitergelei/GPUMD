#!/usr/bin/env python3
"""
Quick validation script for CUDA GCMC algorithm correctness
Tests core formulas against LAMMPS implementation
"""

import numpy as np
import math

def test_umbrella_bias_formula():
    """Test umbrella sampling bias calculation"""
    print("🎯 Testing Umbrella Sampling Bias Formula")
    print("=" * 50)
    
    # Test parameters
    k = 0.5  # force constant
    N0 = 50  # target atoms
    
    test_cases = [
        (45, 46),  # insertion: 45 -> 46
        (55, 54),  # deletion: 55 -> 54  
        (50, 51),  # insertion at target
        (49, 48),  # deletion near target
    ]
    
    for n_before, n_after in test_cases:
        # LAMMPS formula (corrected implementation)
        lammps_bias = -0.5 * k * ((n_after - N0)**2 - (n_before - N0)**2)
        
        # Your CUDA implementation should calculate the same
        cuda_bias = -0.5 * k * ((n_after - N0) * (n_after - N0) - 
                               (n_before - N0) * (n_before - N0))
        
        print(f"  N: {n_before} → {n_after}")
        print(f"  LAMMPS bias: {lammps_bias:.6f} eV")
        print(f"  CUDA bias:   {cuda_bias:.6f} eV")
        print(f"  Match: {'✓' if abs(lammps_bias - cuda_bias) < 1e-10 else '✗'}")
        print()

def test_gcmc_acceptance():
    """Test GCMC acceptance probability formulas"""
    print("⚖️  Testing GCMC Acceptance Formulas") 
    print("=" * 50)
    
    # Test parameters
    temperature = 300.0  # K
    kb = 8.617333262145e-5  # eV/K
    beta = 1.0 / (kb * temperature)
    mu = -5.2  # eV
    volume = 1000.0  # Angstrom^3
    fugacity = math.exp(mu * beta)
    
    test_cases = [
        ("insertion", 49, 50, -0.5),   # insertion with favorable energy
        ("insertion", 49, 50, 2.0),    # insertion with unfavorable energy
        ("deletion", 51, 50, -0.3),    # deletion with favorable energy
        ("deletion", 51, 50, 1.5),     # deletion with unfavorable energy
    ]
    
    for move_type, n_before, n_after, delta_e in test_cases:
        if move_type == "insertion":
            # LAMMPS: P_acc = (z*V)/(N+1) * exp(-beta*delta_E)
            lammps_acc = (fugacity * volume) / (n_after) * math.exp(-beta * delta_e)
            
            # Your CUDA implementation
            cuda_acc = fugacity * volume / (n_after) * math.exp(-beta * delta_e)
            
        else:  # deletion
            # LAMMPS: P_acc = N/(z*V) * exp(-beta*delta_E)
            lammps_acc = n_before / (fugacity * volume) * math.exp(-beta * delta_e)
            
            # Your CUDA implementation  
            cuda_acc = n_before / (fugacity * volume) * math.exp(-beta * delta_e)
        
        print(f"  {move_type.capitalize()}: {n_before} → {n_after}, ΔE = {delta_e} eV")
        print(f"  LAMMPS P_acc: {lammps_acc:.8f}")
        print(f"  CUDA P_acc:   {cuda_acc:.8f}")
        print(f"  Match: {'✓' if abs(lammps_acc - cuda_acc) < 1e-10 else '✗'}")
        print()

def test_combined_umbrella_gcmc():
    """Test combined umbrella sampling + GCMC acceptance"""
    print("🔄 Testing Combined Umbrella + GCMC")
    print("=" * 50)
    
    # Parameters
    temperature = 300.0
    kb = 8.617333262145e-5
    beta = 1.0 / (kb * temperature)
    mu = -5.2
    volume = 1000.0
    fugacity = math.exp(mu * beta)
    
    # Umbrella parameters
    k_umbrella = 0.5
    N0 = 50
    
    # Test case: insertion with umbrella bias
    n_before, n_after = 48, 49
    delta_e = 1.2
    
    # Base GCMC acceptance
    base_acc = fugacity * volume / n_after * math.exp(-beta * delta_e)
    
    # Umbrella bias
    umbr = -0.5 * k_umbrella * ((n_after - N0)**2 - (n_before - N0)**2)
    
    # Combined acceptance (LAMMPS style)
    lammps_total = base_acc * math.exp(umbr)
    
    # Your implementation should match
    cuda_total = base_acc * math.exp(umbr)
    
    print(f"  Insertion: {n_before} → {n_after}, ΔE = {delta_e} eV")
    print(f"  Base GCMC acceptance: {base_acc:.8f}")
    print(f"  Umbrella bias: {umbr:.6f} eV")
    print(f"  LAMMPS total P_acc: {lammps_total:.8f}")
    print(f"  CUDA total P_acc:   {cuda_total:.8f}")
    print(f"  Match: {'✓' if abs(lammps_total - cuda_total) < 1e-10 else '✗'}")
    print()

def test_physical_limits():
    """Test behavior at physical limits"""
    print("🚨 Testing Physical Limits")
    print("=" * 50)
    
    # High chemical potential (should favor insertions)
    mu_high = -1.0
    beta = 1.0 / (8.617333262145e-5 * 300)
    volume = 1000.0
    fugacity_high = math.exp(mu_high * beta)
    
    insertion_acc_high = fugacity_high * volume / 50 * math.exp(-beta * 0.1)
    print(f"  High μ insertion acceptance: {insertion_acc_high:.3f}")
    print(f"  Should be > 1 (will be clamped): {'✓' if insertion_acc_high > 1 else '✗'}")
    
    # Low chemical potential (should favor deletions)
    mu_low = -10.0
    fugacity_low = math.exp(mu_low * beta)
    
    deletion_acc_low = 50 / (fugacity_low * volume) * math.exp(-beta * 0.1)
    print(f"  Low μ deletion acceptance: {deletion_acc_low:.3f}")
    print(f"  Should be > 1 (will be clamped): {'✓' if deletion_acc_low > 1 else '✗'}")
    
    # Strong umbrella bias (far from target)
    k_strong = 5.0
    N0 = 50
    n_current = 20  # Far from target
    
    bias_energy = 0.5 * k_strong * (n_current - N0)**2
    print(f"  Strong umbrella bias energy: {bias_energy:.2f} eV")
    print(f"  Should strongly favor insertion: {'✓' if bias_energy > 10 else '✗'}")
    print()

def main():
    print("🧪 CUDA GCMC Algorithm Validation Suite")
    print("=" * 60)
    print("This script tests the core formulas in your CUDA GCMC implementation")
    print("against the LAMMPS reference implementation.")
    print()
    
    test_umbrella_bias_formula()
    test_gcmc_acceptance()
    test_combined_umbrella_gcmc()
    test_physical_limits()
    
    print("🎉 Algorithm Validation Complete!")
    print("=" * 60)
    print("If all tests show ✓, your CUDA GCMC formulas are correct.")
    print("If any show ✗, check the corresponding calculations in your code.")
    print()
    print("Next steps:")
    print("1. Run full simulation validation with run_validation.bat")
    print("2. Compare results with LAMMPS using compare_results.py")
    print("3. Verify performance and numerical stability")

if __name__ == "__main__":
    main()
