#!/usr/bin/env python3
"""
Compare ADP potential values between Mo.adp and U_Mo.alloy.adp_cor.txt
"""

def read_adp_header(filename):
    """Read ADP file header"""
    with open(filename, 'r') as f:
        # Skip first 3 comment lines
        for _ in range(3):
            f.readline()
        
        # Line 4: Nelements Element1 Element2 ...
        line4 = f.readline().strip().split()
        nelements = int(line4[0])
        elements = line4[1:nelements+1]
        
        # Line 5: Nrho, drho, Nr, dr, cutoff
        line5 = f.readline().strip().split()
        nrho = int(line5[0])
        drho = float(line5[1])
        nr = int(line5[2])
        dr = float(line5[3])
        cutoff = float(line5[4])
        
        return {
            'nelements': nelements,
            'elements': elements,
            'nrho': nrho,
            'drho': drho,
            'nr': nr,
            'dr': dr,
            'cutoff': cutoff
        }

def read_adp_functions(filename, info):
    """Read F, rho, phi, u, w functions"""
    data = {}
    
    with open(filename, 'r') as f:
        # Skip header
        for _ in range(5):
            f.readline()
        
        # Read per-element data
        F_data = []
        rho_data = []
        
        for elem_idx in range(info['nelements']):
            # Skip element info line
            f.readline()
            
            # Read F(rho) - embedding function
            F_values = []
            values_read = 0
            while values_read < info['nrho']:
                line = f.readline().strip()
                if line:
                    vals = line.split()
                    for val in vals:
                        if values_read < info['nrho']:
                            F_values.append(float(val))
                            values_read += 1
            F_data.append(F_values)
            
            # Read rho(r) - density function  
            rho_values = []
            values_read = 0
            while values_read < info['nr']:
                line = f.readline().strip()
                if line:
                    vals = line.split()
                    for val in vals:
                        if values_read < info['nr']:
                            rho_values.append(float(val))
                            values_read += 1
            rho_data.append(rho_values)
        
        # Read pair functions phi, u, w
        num_pairs = info['nelements'] * (info['nelements'] + 1) // 2
        phi_data = []
        u_data = []
        w_data = []
        
        # Read phi(r)
        for pair_idx in range(num_pairs):
            phi_values = []
            values_read = 0
            while values_read < info['nr']:
                line = f.readline().strip()
                if line:
                    vals = line.split()
                    for val in vals:
                        if values_read < info['nr']:
                            phi_values.append(float(val))
                            values_read += 1
            phi_data.append(phi_values)
        
        # Read u(r)
        for pair_idx in range(num_pairs):
            u_values = []
            values_read = 0
            while values_read < info['nr']:
                line = f.readline().strip()
                if line:
                    vals = line.split()
                    for val in vals:
                        if values_read < info['nr']:
                            u_values.append(float(val))
                            values_read += 1
            u_data.append(u_values)
        
        # Read w(r)
        for pair_idx in range(num_pairs):
            w_values = []
            values_read = 0
            while values_read < info['nr']:
                line = f.readline().strip()
                if line:
                    vals = line.split()
                    for val in vals:
                        if values_read < info['nr']:
                            w_values.append(float(val))
                            values_read += 1
            w_data.append(w_values)
    
    return {
        'F': F_data,
        'rho': rho_data, 
        'phi': phi_data,
        'u': u_data,
        'w': w_data
    }

# Compare files
mo_file = "/home/hjchen/projects/GPUMD/examples/adp_Mo_Ta/hjchen-Mo.adp"
umo_file = "/home/hjchen/projects/GPUMD/examples/adp_U_Mo/U_Mo.alloy.adp_cor.txt"

print("=== Comparing ADP files ===")

# Read headers
mo_info = read_adp_header(mo_file)
umo_info = read_adp_header(umo_file)

print(f"Mo.adp: {mo_info['elements']}, cutoff={mo_info['cutoff']}")
print(f"U_Mo: {umo_info['elements']}, cutoff={umo_info['cutoff']}")

# Read data
mo_data = read_adp_functions(mo_file, mo_info)
umo_data = read_adp_functions(umo_file, umo_info)

# Compare Mo element data
print("\n=== Mo element comparison ===")

# In Mo.adp: Mo is element 0
# In U_Mo.adp: Mo is element 1 (index 1)
mo_idx_in_mo_file = 0
mo_idx_in_umo_file = 1

# Compare F(rho) - embedding function
print("F(rho) comparison (first 10 values):")
print("Mo.adp (Mo):", mo_data['F'][mo_idx_in_mo_file][:10])
print("U_Mo.adp (Mo):", umo_data['F'][mo_idx_in_umo_file][:10])

# Compare rho(r) - density function
print("\nrho(r) comparison (first 10 values):")
print("Mo.adp (Mo):", mo_data['rho'][mo_idx_in_mo_file][:10])
print("U_Mo.adp (Mo):", umo_data['rho'][mo_idx_in_umo_file][:10])

# Compare Mo-Mo pair interactions
print("\nMo-Mo pair interaction comparison:")
# In Mo.adp: only one pair (Mo-Mo) at index 0  
# In U_Mo.adp: pairs are (U-U)=0, (Mo-U)=1, (Mo-Mo)=2
mo_mo_idx_in_mo_file = 0
mo_mo_idx_in_umo_file = 2

print("phi(r) Mo-Mo (first 10 values):")
print("Mo.adp:", mo_data['phi'][mo_mo_idx_in_mo_file][:10])
print("U_Mo.adp:", umo_data['phi'][mo_mo_idx_in_umo_file][:10])

print("\nu(r) Mo-Mo (first 10 values):")
print("Mo.adp:", mo_data['u'][mo_mo_idx_in_mo_file][:10])
print("U_Mo.adp:", umo_data['u'][mo_mo_idx_in_umo_file][:10])

print("\nw(r) Mo-Mo (first 10 values):")
print("Mo.adp:", mo_data['w'][mo_mo_idx_in_mo_file][:10])
print("U_Mo.adp:", umo_data['w'][mo_mo_idx_in_umo_file][:10])