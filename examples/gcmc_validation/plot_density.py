#!/usr/bin/env python3
"""
Plot density profiles and other analysis for GCMC validation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import argparse

def read_xyz_file(filename):
    """Read XYZ file and return positions and types"""
    positions = []
    types = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        n_atoms = int(lines[0])
        
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            
            positions.append([x, y, z])
            types.append(element)
    
    return np.array(positions), types

def calculate_radial_density(positions, types, center, max_radius=20.0, n_bins=50):
    """Calculate radial density profile around a center point"""
    
    # Filter for specific atom type (e.g., Ar)
    ar_positions = []
    for i, atom_type in enumerate(types):
        if atom_type == 'Ar':
            ar_positions.append(positions[i])
    
    if len(ar_positions) == 0:
        return np.zeros(n_bins), np.linspace(0, max_radius, n_bins)
    
    ar_positions = np.array(ar_positions)
    
    # Calculate distances from center
    distances = np.linalg.norm(ar_positions - center, axis=1)
    
    # Create radial bins
    r_bins = np.linspace(0, max_radius, n_bins + 1)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    dr = r_bins[1] - r_bins[0]
    
    # Calculate density in each shell
    density = np.zeros(n_bins)
    
    for i in range(n_bins):
        r_inner = r_bins[i]
        r_outer = r_bins[i + 1]
        
        # Count atoms in shell
        in_shell = (distances >= r_inner) & (distances < r_outer)
        n_atoms_shell = np.sum(in_shell)
        
        # Volume of shell
        shell_volume = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        
        # Density (atoms per cubic Angstrom)
        density[i] = n_atoms_shell / shell_volume
    
    return density, r_centers

def plot_density_profiles(gpumd_files, lammps_files=None):
    """Plot radial density profiles for comparison"""
    
    plt.figure(figsize=(12, 8))
    
    # Process GPUMD results
    for i, filename in enumerate(gpumd_files):
        positions, types = read_xyz_file(filename)
        
        # Assume tube center is at box center
        center = np.mean(positions, axis=0)
        
        density, r_centers = calculate_radial_density(positions, types, center)
        
        plt.subplot(2, 2, 1)
        plt.plot(r_centers, density, label=f'GPUMD Step {i}', alpha=0.7)
    
    plt.xlabel('Distance from center (Å)')
    plt.ylabel('Density (atoms/Å³)')
    plt.title('Radial Density Profile - GPUMD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Process LAMMPS results if available
    if lammps_files:
        for i, filename in enumerate(lammps_files):
            # Parse LAMMPS dump file (different format)
            positions, types = read_lammps_dump(filename)
            center = np.mean(positions, axis=0)
            density, r_centers = calculate_radial_density(positions, types, center)
            
            plt.subplot(2, 2, 2)
            plt.plot(r_centers, density, label=f'LAMMPS Step {i}', alpha=0.7)
        
        plt.xlabel('Distance from center (Å)')
        plt.ylabel('Density (atoms/Å³)')
        plt.title('Radial Density Profile - LAMMPS')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot final comparison
    if gpumd_files and lammps_files:
        positions_g, types_g = read_xyz_file(gpumd_files[-1])
        positions_l, types_l = read_lammps_dump(lammps_files[-1])
        
        center_g = np.mean(positions_g, axis=0)
        center_l = np.mean(positions_l, axis=0)
        
        density_g, r_centers = calculate_radial_density(positions_g, types_g, center_g)
        density_l, r_centers = calculate_radial_density(positions_l, types_l, center_l)
        
        plt.subplot(2, 2, 3)
        plt.plot(r_centers, density_g, label='GPUMD', linewidth=2)
        plt.plot(r_centers, density_l, label='LAMMPS', linewidth=2, linestyle='--')
        plt.xlabel('Distance from center (Å)')
        plt.ylabel('Density (atoms/Å³)')
        plt.title('Final Density Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate and plot difference
        plt.subplot(2, 2, 4)
        difference = np.abs(density_g - density_l) / (density_l + 1e-10) * 100
        plt.plot(r_centers, difference, color='red', linewidth=2)
        plt.xlabel('Distance from center (Å)')
        plt.ylabel('Relative Difference (%)')
        plt.title('Density Difference (GPUMD vs LAMMPS)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('density_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()

def read_lammps_dump(filename):
    """Read LAMMPS dump file format"""
    positions = []
    types = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find data section
    i = 0
    while i < len(lines):
        if 'ITEM: ATOMS' in lines[i]:
            i += 1
            break
        i += 1
    
    # Read atom data
    while i < len(lines) and not lines[i].startswith('ITEM:'):
        parts = lines[i].split()
        if len(parts) >= 5:
            atom_type = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            
            positions.append([x, y, z])
            types.append('Ar' if atom_type == 1 else 'C')
        i += 1
    
    return np.array(positions), types

def plot_acceptance_ratios(gcmc_stats_file):
    """Plot GCMC acceptance ratios over time"""
    try:
        data = np.loadtxt(gcmc_stats_file, skiprows=1)
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(data[:, 0], data[:, 2], label='Insertion', alpha=0.7)
        plt.plot(data[:, 0], data[:, 3], label='Deletion', alpha=0.7)
        plt.plot(data[:, 0], data[:, 4], label='Displacement', alpha=0.7)
        plt.xlabel('MC Step')
        plt.ylabel('Acceptance Ratio')
        plt.title('GCMC Acceptance Ratios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(data[:, 0], data[:, 1])
        plt.xlabel('MC Step')
        plt.ylabel('Number of Atoms')
        plt.title('Atom Count Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('acceptance_ratios.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error plotting acceptance ratios: {e}")

def main():
    parser = argparse.ArgumentParser(description='Plot GCMC density profiles and analysis')
    parser.add_argument('--gpumd-xyz', nargs='+', help='GPUMD XYZ files')
    parser.add_argument('--lammps-dump', nargs='+', help='LAMMPS dump files')
    parser.add_argument('--gcmc-stats', help='GCMC statistics file')
    
    args = parser.parse_args()
    
    if args.gpumd_xyz:
        plot_density_profiles(args.gpumd_xyz, args.lammps_dump)
    
    if args.gcmc_stats:
        plot_acceptance_ratios(args.gcmc_stats)

if __name__ == "__main__":
    main()
