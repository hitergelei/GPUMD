#!/usr/bin/env python3
"""
Debug ADP element mapping and pair index calculation
"""

def get_pair_index(type1, type2):
    """Mimic the GPU function get_pair_index"""
    a = max(type1, type2)
    b = min(type1, type2)
    return a * (a + 1) // 2 + b

# Test for U-Mo system
print("=== U-Mo ADP File Analysis ===")
elements = ["U", "Mo"]  # Index 0=U, 1=Mo
nelements = len(elements)

print(f"Elements: {elements}")
print(f"Number of elements: {nelements}")

# Calculate pair indices
pairs = []
pair_index = 0
for i in range(nelements):
    for j in range(i + 1):  # j <= i
        pair_idx = get_pair_index(i, j)
        pairs.append((i, j, pair_index, pair_idx))
        print(f"Pair ({elements[i]}-{elements[j]}): storage_index={pair_index}, calculated_index={pair_idx}")
        pair_index += 1

print(f"\nTotal pairs: {pair_index}")

# For Mo-only system using U-Mo file
print("\n=== Mo-only system using U-Mo file ===")
user_element = "Mo"
adp_element_index = elements.index(user_element)  # Should be 1
print(f"User element '{user_element}' maps to ADP index: {adp_element_index}")

# All atoms are type=1 (Mo), so Mo-Mo interaction
mo_mo_pair_index = get_pair_index(1, 1)
print(f"Mo-Mo interaction should use pair index: {mo_mo_pair_index}")

# Check if this matches storage
for storage_idx, (i, j, stored_idx, calc_idx) in enumerate(pairs):
    if calc_idx == mo_mo_pair_index:
        print(f"Mo-Mo interaction found at storage index {stored_idx}: {elements[i]}-{elements[j]}")
        break