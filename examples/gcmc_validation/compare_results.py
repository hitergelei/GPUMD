#!/usr/bin/env python3
"""
GCMC Validation: Compare GPUMD CUDA GCMC results with LAMMPS reference
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import argparse
import os

class GCMCValidator:
    def __init__(self):
        self.gpumd_data = {}
        self.lammps_data = {}
        self.validation_results = {}
    
    def load_gpumd_results(self, directory="."):
        """Load GPUMD GCMC simulation results"""
        try:
            # Load thermodynamic data
            thermo_file = os.path.join(directory, "thermo.out")
            if os.path.exists(thermo_file):
                self.gpumd_data['thermo'] = pd.read_csv(thermo_file, sep='\s+')
            
            # Load GCMC statistics
            gcmc_file = os.path.join(directory, "gcmc_stats.out")
            if os.path.exists(gcmc_file):
                self.gpumd_data['gcmc_stats'] = pd.read_csv(gcmc_file, sep='\s+')
            
            # Load umbrella sampling data
            umbrella_file = os.path.join(directory, "umbrella.out")
            if os.path.exists(umbrella_file):
                self.gpumd_data['umbrella'] = pd.read_csv(umbrella_file, sep='\s+')
            
            # Load energy trajectory
            energy_file = os.path.join(directory, "energy.out")
            if os.path.exists(energy_file):
                self.gpumd_data['energy'] = pd.read_csv(energy_file, sep='\s+')
                
            print("✓ GPUMD results loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading GPUMD results: {e}")
    
    def load_lammps_results(self, directory="."):
        """Load LAMMPS reference results"""
        try:
            # Load LAMMPS log file
            log_file = os.path.join(directory, "log.lammps")
            if os.path.exists(log_file):
                self.lammps_data['log'] = self.parse_lammps_log(log_file)
            
            # Load umbrella statistics
            umbrella_file = os.path.join(directory, "umbrella_statistics.out")
            if os.path.exists(umbrella_file):
                self.lammps_data['umbrella'] = pd.read_csv(umbrella_file, sep='\s+')
            
            # Load atom count data
            natoms_file = os.path.join(directory, "n_atoms.out")
            if os.path.exists(natoms_file):
                self.lammps_data['n_atoms'] = pd.read_csv(natoms_file, sep='\s+')
                
            print("✓ LAMMPS results loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading LAMMPS results: {e}")
    
    def parse_lammps_log(self, log_file):
        """Parse LAMMPS log file to extract thermodynamic data"""
        data = []
        reading_data = False
        
        with open(log_file, 'r') as f:
            for line in f:
                if 'Step' in line and 'Temp' in line:
                    reading_data = True
                    headers = line.split()
                    continue
                
                if reading_data:
                    if line.strip() and not line.startswith('Loop'):
                        values = line.split()
                        if len(values) == len(headers):
                            data.append([float(v) for v in values])
                    else:
                        reading_data = False
        
        return pd.DataFrame(data, columns=headers)
    
    def calculate_validation_metrics(self):
        """Calculate validation metrics comparing GPUMD vs LAMMPS"""
        metrics = {}
        
        # 1. Average particle number comparison
        if 'gcmc_stats' in self.gpumd_data and 'n_atoms' in self.lammps_data:
            gpumd_avg_n = self.gpumd_data['gcmc_stats']['n_atoms'].mean()
            lammps_avg_n = self.lammps_data['n_atoms'].iloc[:, 1].mean()
            
            metrics['avg_particle_difference'] = abs(gpumd_avg_n - lammps_avg_n) / lammps_avg_n * 100
            metrics['gpumd_avg_particles'] = gpumd_avg_n
            metrics['lammps_avg_particles'] = lammps_avg_n
        
        # 2. Acceptance ratio comparison
        if 'gcmc_stats' in self.gpumd_data:
            gpumd_accept = self.gpumd_data['gcmc_stats']['acceptance_ratio'].mean()
            metrics['gpumd_acceptance_ratio'] = gpumd_accept
        
        # 3. Energy distribution comparison
        if 'energy' in self.gpumd_data and 'log' in self.lammps_data:
            gpumd_energy = self.gpumd_data['energy']['total_energy']
            lammps_energy = self.lammps_data['log']['PotEng'] if 'PotEng' in self.lammps_data['log'].columns else None
            
            if lammps_energy is not None:
                # Kolmogorov-Smirnov test for distribution similarity
                ks_stat, p_value = stats.ks_2samp(gpumd_energy, lammps_energy)
                metrics['energy_distribution_similarity'] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'similar': p_value > 0.05  # Accept if p > 0.05
                }
        
        # 4. Umbrella bias comparison
        if 'umbrella' in self.gpumd_data and 'umbrella' in self.lammps_data:
            gpumd_bias = self.gpumd_data['umbrella']['bias_energy'].mean()
            lammps_bias = self.lammps_data['umbrella'].iloc[:, 1].mean()
            
            metrics['umbrella_bias_difference'] = abs(gpumd_bias - lammps_bias) / abs(lammps_bias) * 100
        
        self.validation_results = metrics
        return metrics
    
    def generate_validation_report(self):
        """Generate a comprehensive validation report"""
        print("\n" + "="*60)
        print("GCMC VALIDATION REPORT")
        print("="*60)
        
        if not self.validation_results:
            self.calculate_validation_metrics()
        
        results = self.validation_results
        
        # Particle number validation
        if 'avg_particle_difference' in results:
            print(f"\n🔢 PARTICLE NUMBER VALIDATION:")
            print(f"   GPUMD average particles: {results['gpumd_avg_particles']:.2f}")
            print(f"   LAMMPS average particles: {results['lammps_avg_particles']:.2f}")
            print(f"   Relative difference: {results['avg_particle_difference']:.2f}%")
            
            if results['avg_particle_difference'] < 5.0:
                print("   ✓ PASSED - Particle numbers agree within 5%")
            else:
                print("   ✗ FAILED - Particle numbers differ by >5%")
        
        # Acceptance ratio validation
        if 'gpumd_acceptance_ratio' in results:
            print(f"\n📊 ACCEPTANCE RATIO:")
            print(f"   GPUMD acceptance ratio: {results['gpumd_acceptance_ratio']:.3f}")
            
            if 0.1 <= results['gpumd_acceptance_ratio'] <= 0.6:
                print("   ✓ PASSED - Acceptance ratio in reasonable range (0.1-0.6)")
            else:
                print("   ⚠ WARNING - Acceptance ratio outside typical range")
        
        # Energy distribution validation
        if 'energy_distribution_similarity' in results:
            energy_sim = results['energy_distribution_similarity']
            print(f"\n⚡ ENERGY DISTRIBUTION VALIDATION:")
            print(f"   Kolmogorov-Smirnov statistic: {energy_sim['ks_statistic']:.4f}")
            print(f"   p-value: {energy_sim['p_value']:.4f}")
            
            if energy_sim['similar']:
                print("   ✓ PASSED - Energy distributions are statistically similar")
            else:
                print("   ✗ FAILED - Energy distributions are significantly different")
        
        # Umbrella bias validation
        if 'umbrella_bias_difference' in results:
            print(f"\n🎯 UMBRELLA SAMPLING VALIDATION:")
            print(f"   Umbrella bias difference: {results['umbrella_bias_difference']:.2f}%")
            
            if results['umbrella_bias_difference'] < 10.0:
                print("   ✓ PASSED - Umbrella bias agrees within 10%")
            else:
                print("   ✗ FAILED - Umbrella bias differs by >10%")
        
        print("\n" + "="*60)
    
    def plot_comparison(self, save_plots=True):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('GPUMD vs LAMMPS GCMC Validation', fontsize=16)
        
        # Plot 1: Particle number evolution
        if 'gcmc_stats' in self.gpumd_data and 'n_atoms' in self.lammps_data:
            ax = axes[0, 0]
            ax.plot(self.gpumd_data['gcmc_stats']['step'], 
                   self.gpumd_data['gcmc_stats']['n_atoms'], 
                   label='GPUMD', alpha=0.7)
            ax.plot(self.lammps_data['n_atoms'].iloc[:, 0], 
                   self.lammps_data['n_atoms'].iloc[:, 1], 
                   label='LAMMPS', alpha=0.7)
            ax.set_xlabel('MC Step')
            ax.set_ylabel('Number of Atoms')
            ax.set_title('Particle Number Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Energy comparison
        if 'energy' in self.gpumd_data and 'log' in self.lammps_data:
            ax = axes[0, 1]
            ax.hist(self.gpumd_data['energy']['total_energy'], 
                   alpha=0.6, bins=50, label='GPUMD', density=True)
            if 'PotEng' in self.lammps_data['log'].columns:
                ax.hist(self.lammps_data['log']['PotEng'], 
                       alpha=0.6, bins=50, label='LAMMPS', density=True)
            ax.set_xlabel('Energy (eV)')
            ax.set_ylabel('Probability Density')
            ax.set_title('Energy Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Acceptance ratio
        if 'gcmc_stats' in self.gpumd_data:
            ax = axes[1, 0]
            ax.plot(self.gpumd_data['gcmc_stats']['step'], 
                   self.gpumd_data['gcmc_stats']['acceptance_ratio'])
            ax.set_xlabel('MC Step')
            ax.set_ylabel('Acceptance Ratio')
            ax.set_title('GCMC Acceptance Ratio')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Umbrella bias
        if 'umbrella' in self.gpumd_data:
            ax = axes[1, 1]
            ax.plot(self.gpumd_data['umbrella']['step'], 
                   self.gpumd_data['umbrella']['bias_energy'], 
                   label='GPUMD', alpha=0.7)
            if 'umbrella' in self.lammps_data:
                ax.plot(self.lammps_data['umbrella'].iloc[:, 0], 
                       self.lammps_data['umbrella'].iloc[:, 1], 
                       label='LAMMPS', alpha=0.7)
            ax.set_xlabel('MC Step')
            ax.set_ylabel('Umbrella Bias Energy (eV)')
            ax.set_title('Umbrella Sampling Bias')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('gcmc_validation_plots.png', dpi=300, bbox_inches='tight')
            print("\n📊 Validation plots saved as 'gcmc_validation_plots.png'")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Validate GPUMD CUDA GCMC against LAMMPS')
    parser.add_argument('--gpumd-dir', default='.', help='Directory with GPUMD results')
    parser.add_argument('--lammps-dir', default='.', help='Directory with LAMMPS results') 
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = GCMCValidator()
    
    # Load results
    validator.load_gpumd_results(args.gpumd_dir)
    validator.load_lammps_results(args.lammps_dir)
    
    # Generate validation report
    validator.generate_validation_report()
    
    # Generate plots
    if not args.no_plots:
        validator.plot_comparison()

if __name__ == "__main__":
    main()
