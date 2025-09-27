#!/usr/bin/env python3

"""
Compare virial sign conventions across different GPUMD potential implementations
"""

print("=== GPUMD Virial Sign Convention Analysis ===")
print()

print("Based on code analysis:")
print()

print("1. ADP (adp.cu lines 1139-1147):")
print("   g_virial[n1 + 0 * N] += s_sxx;  // PLUS sign")
print("   g_virial[n1 + 1 * N] += s_syy;")
print("   g_virial[n1 + 2 * N] += s_szz;")
print("   g_virial[n1 + 3 * N] += s_sxy;")
print("   ...")
print()

print("2. EAM_ALLOY (eam_alloy.cu lines 562-570):")
print("   g_virial[n1 + 0 * N] -= sxx;   // MINUS sign")
print("   g_virial[n1 + 1 * N] -= syy;")
print("   g_virial[n1 + 2 * N] -= szz;")
print("   g_virial[n1 + 3 * N] -= sxy;")
print("   ...")
print()

print("3. NEP (nep.cu lines 925-933):")
print("   g_virial[n1 + 0 * N] += s_sxx;  // PLUS sign")
print("   g_virial[n1 + 1 * N] += s_syy;")
print("   g_virial[n1 + 2 * N] += s_szz;")
print("   g_virial[n1 + 3 * N] += s_sxy;")
print("   ...")
print()

print("4. DP (dp.cu lines 200-208):")
print("   v_out[n1] = vxx * v_factor;      // Direct assignment")
print("   v_out[n1 + N] = vyy * v_factor;  // (no += or -=)")
print("   v_out[n1 + N * 2] = vzz * v_factor;")
print("   ...")
print("   Note: DP gets virial directly from DeePMD, no local calculation")
print()

print("=== SUMMARY ===")
print()
print("INCONSISTENCY FOUND!")
print("- ADP:       += (PLUS)")
print("- EAM_ALLOY: -= (MINUS)")  
print("- NEP:       += (PLUS)")
print("- DP:        Direct assignment (no sign issue)")
print()

print("🚨 CRITICAL ISSUE:")
print("ADP and NEP use += while EAM_ALLOY uses -=")
print("This creates inconsistent stress sign conventions!")
print()

print("RECOMMENDATION:")
print("Either:")
print("1. Change ADP to use -= to match EAM_ALLOY")
print("2. Change EAM_ALLOY to use += to match ADP/NEP")
print("3. Document the difference clearly")
print()

print("The current ADP implementation follows NEP convention (+= sign)")
print("but differs from EAM_ALLOY convention (-= sign).")