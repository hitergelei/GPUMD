#!/usr/bin/env python3
"""
Convert LAMMPS dump file to GPUMD model.xyz format
"""

def convert_lammps_to_gpumd(lammps_file, output_file):
    """Convert LAMMPS dump file to GPUMD xyz format"""
    
    with open(lammps_file, 'r') as f:
        lines = f.readlines()
    
    # Find number of atoms
    natoms = None
    box_bounds = []
    atoms_data = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if "NUMBER OF ATOMS" in line:
            natoms = int(lines[i+1].strip())
            i += 2
            continue
            
        if "BOX BOUNDS" in line:
            # Read box bounds
            for j in range(3):
                bounds = lines[i+1+j].strip().split()
                box_bounds.append([float(bounds[0]), float(bounds[1])])
            i += 4
            continue
            
        if "ATOMS" in line:
            # Read atom data
            atoms_data = []
            for j in range(natoms):
                atom_line = lines[i+1+j].strip().split()
                atoms_data.append(atom_line)
            break
            
        i += 1
    
    # Calculate box dimensions
    lx = box_bounds[0][1] - box_bounds[0][0]
    ly = box_bounds[1][1] - box_bounds[1][0]
    lz = box_bounds[2][1] - box_bounds[2][0]
    
    # Write GPUMD format
    with open(output_file, 'w') as f:
        f.write(f"{natoms}\n")
        f.write(f"pbc=\"T T T\" Lattice=\"{lx:.6f} 0.0 0.0 0.0 {ly:.6f} 0.0 0.0 0.0 {lz:.6f}\" Properties=species:S:1:pos:R:3\n")
        
        for atom in atoms_data:
            # atom format: id element x y z fx fy fz
            element = atom[1]
            x = float(atom[2])
            y = float(atom[3])
            z = float(atom[4])
            f.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"Converted {natoms} atoms to GPUMD format")
    print(f"Box dimensions: {lx:.3f} x {ly:.3f} x {lz:.3f} Å")

if __name__ == "__main__":
    convert_lammps_to_gpumd("fcc_ni.xyz", "fcc_ni_model.xyz")
