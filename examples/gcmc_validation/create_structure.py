#!/usr/bin/env python3
"""
Generate initial atomic structure for GCMC validation
Creates a carbon nanotube with empty space for Ar adsorption
"""

import numpy as np
import math

def create_carbon_nanotube(radius=12.0, length=30.0, chirality=(10, 0)):
    """
    Create a carbon nanotube structure
    
    Parameters:
    radius: tube radius in Angstroms
    length: tube length in Angstroms  
    chirality: (n, m) chirality indices
    """
    
    # Carbon-carbon bond length
    a_cc = 1.42  # Angstroms
    
    # Calculate circumference and number of atoms per ring
    circumference = 2 * math.pi * radius
    atoms_per_ring = int(circumference / a_cc)
    
    # Number of rings along tube length
    num_rings = int(length / (a_cc * math.sqrt(3)))
    
    atoms = []
    atom_id = 1
    
    for ring in range(num_rings):
        z = ring * a_cc * math.sqrt(3) / 2
        
        for i in range(atoms_per_ring):
            # Angle around the tube
            theta = 2 * math.pi * i / atoms_per_ring
            
            # Stagger alternate rings
            if ring % 2 == 1:
                theta += math.pi / atoms_per_ring
            
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            
            atoms.append([atom_id, 2, x, y, z])  # type 2 = Carbon
            atom_id += 1
    
    return atoms

def write_xyz_file(atoms, filename, box_size):
    """Write atoms to XYZ format for GPUMD"""
    
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"Lattice=\"{box_size[0]} 0.0 0.0 0.0 {box_size[1]} 0.0 0.0 0.0 {box_size[2]}\" Properties=species:S:1:pos:R:3\n")
        
        for atom in atoms:
            atom_id, atom_type, x, y, z = atom
            element = "C" if atom_type == 2 else "Ar"
            f.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")

def write_lammps_data(atoms, filename, box_size):
    """Write atoms to LAMMPS data format"""
    
    with open(filename, 'w') as f:
        f.write("Carbon nanotube for GCMC validation\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write("2 atom types\n\n")
        
        # Box dimensions
        f.write(f"0.0 {box_size[0]} xlo xhi\n")
        f.write(f"0.0 {box_size[1]} ylo yhi\n") 
        f.write(f"0.0 {box_size[2]} zlo zhi\n\n")
        
        # Masses
        f.write("Masses\n\n")
        f.write("1 39.948  # Ar\n")
        f.write("2 12.011  # C\n\n")
        
        # Atoms
        f.write("Atoms\n\n")
        for atom in atoms:
            atom_id, atom_type, x, y, z = atom
            f.write(f"{atom_id} {atom_type} {x:.6f} {y:.6f} {z:.6f}\n")

def create_nep_potential():
    """Create a simple NEP potential file for testing"""
    nep_content = """nep4 18 2
C Ar
cutoff 6.0 4.0
n_max 8 4
basis_size 8 8
l_max 4 2
ANN 30 30
lambda_1 0.1
lambda_2 0.01
lambda_e 1.0
lambda_f 0.1
lambda_v 0.1
batch 1000
population 50
generation 100000"""
    
    with open('nep.txt', 'w') as f:
        f.write(nep_content)

def main():
    # Parameters
    tube_radius = 12.0  # Angstroms
    tube_length = 30.0  # Angstroms
    box_margin = 10.0   # Extra space around tube
    
    # Calculate box size
    box_x = 2 * (tube_radius + box_margin)
    box_y = 2 * (tube_radius + box_margin) 
    box_z = tube_length + 2 * box_margin
    box_size = [box_x, box_y, box_z]
    
    # Create carbon nanotube
    print("Creating carbon nanotube structure...")
    atoms = create_carbon_nanotube(tube_radius, tube_length)
    
    # Center the tube in the box
    for atom in atoms:
        atom[2] += box_x / 2  # x coordinate
        atom[3] += box_y / 2  # y coordinate  
        atom[4] += box_margin  # z coordinate
    
    print(f"Created nanotube with {len(atoms)} carbon atoms")
    print(f"Box size: {box_x:.1f} x {box_y:.1f} x {box_z:.1f} Angstroms")
    
    # Write files
    write_xyz_file(atoms, 'model.xyz', box_size)
    write_lammps_data(atoms, 'initial_structure.data', box_size)
    create_nep_potential()
    
    print("\nFiles created:")
    print("- model.xyz (GPUMD format)")
    print("- initial_structure.data (LAMMPS format)")
    print("- nep.txt (NEP potential - placeholder)")
    
    print(f"\nNote: Replace nep.txt with a real NEP potential trained for C-Ar interactions")

if __name__ == "__main__":
    main()
