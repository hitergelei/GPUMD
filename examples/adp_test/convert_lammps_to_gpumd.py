#!/usr/bin/env python3
"""
Convert LAMMPS dump file to GPUMD xyz format
"""

def convert_lammps_to_gpumd(input_file, output_file):
    """
    Convert LAMMPS dump file to GPUMD xyz format
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse LAMMPS dump file
    atoms = []
    reading_atoms = False
    natoms = 0
    box_bounds = []
    
    for line in lines:
        line = line.strip()
        if 'NUMBER OF ATOMS' in line:
            continue
        elif line.isdigit() and len(atoms) == 0:
            natoms = int(line)
        elif 'BOX BOUNDS' in line:
            continue
        elif 'ATOMS' in line:
            reading_atoms = True
            continue
        elif reading_atoms and len(line.split()) >= 7:
            parts = line.split()
            atom_id = int(parts[0])
            element = parts[1]
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            fx, fy, fz = float(parts[5]), float(parts[6]), float(parts[7])
            atoms.append((atom_id, element, x, y, z, fx, fy, fz))
        elif len(line.split()) == 2 and not reading_atoms:
            # Box bounds
            box_bounds.append([float(x) for x in line.split()])
    
    # Sort atoms by ID
    atoms.sort(key=lambda x: x[0])
    
    # Calculate box dimensions
    if len(box_bounds) >= 3:
        lx = box_bounds[0][1] - box_bounds[0][0]
        ly = box_bounds[1][1] - box_bounds[1][0]
        lz = box_bounds[2][1] - box_bounds[2][0]
    else:
        # Estimate from atom positions
        xs = [atom[2] for atom in atoms]
        ys = [atom[3] for atom in atoms]
        zs = [atom[4] for atom in atoms]
        lx = max(xs) - min(xs) + 2.0
        ly = max(ys) - min(ys) + 2.0
        lz = max(zs) - min(zs) + 2.0
    
    # Write GPUMD format
    with open(output_file, 'w') as f:
        f.write(f"{natoms}\n")
        f.write(f"Lattice=\"{lx:.6f} 0.0 0.0 0.0 {ly:.6f} 0.0 0.0 0.0 {lz:.6f}\" ")
        f.write("Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n")
        
        for atom in atoms:
            element = atom[1]
            x, y, z = atom[2], atom[3], atom[4]
            f.write(f"{element} {x:.8f} {y:.8f} {z:.8f}\n")

if __name__ == "__main__":
    convert_lammps_to_gpumd("fcc_ni.xyz", "model_gpumd.xyz")
    print("Converted LAMMPS dump to GPUMD format: model_gpumd.xyz")
