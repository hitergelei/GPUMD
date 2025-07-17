#!/bin/bash

# GCMC Validation Runner
# This script runs both GPUMD and LAMMPS simulations and compares results

echo "🚀 Starting GCMC Validation Suite"
echo "================================="

# Check if required executables exist
GPUMD_EXE="../../src/gpumd"
LAMMPS_EXE="lmp"

if [ ! -f "$GPUMD_EXE" ]; then
    echo "❌ GPUMD executable not found at $GPUMD_EXE"
    echo "   Please compile GPUMD first: cd ../../src && make"
    exit 1
fi

if ! command -v $LAMMPS_EXE &> /dev/null; then
    echo "⚠️  LAMMPS executable not found. Reference comparison will be skipped."
    SKIP_LAMMPS=true
else
    SKIP_LAMMPS=false
fi

# Create initial structure if it doesn't exist
if [ ! -f "model.xyz" ]; then
    echo "📁 Creating initial atomic structure..."
    python3 create_structure.py
fi

# Run GPUMD CUDA GCMC simulation
echo ""
echo "🖥️  Running GPUMD CUDA GCMC simulation..."
echo "========================================"
$GPUMD_EXE 2>&1 | tee gpumd.log

if [ $? -eq 0 ]; then
    echo "✅ GPUMD simulation completed successfully"
else
    echo "❌ GPUMD simulation failed"
    exit 1
fi

# Run LAMMPS reference simulation
if [ "$SKIP_LAMMPS" = false ]; then
    echo ""
    echo "🔬 Running LAMMPS reference simulation..."
    echo "======================================="
    
    # Standard GCMC
    $LAMMPS_EXE -in lammps_gcmc.in 2>&1 | tee lammps_gcmc.log
    
    if [ $? -eq 0 ]; then
        echo "✅ LAMMPS GCMC simulation completed"
    else
        echo "❌ LAMMPS GCMC simulation failed"
    fi
    
    # Umbrella sampling GCMC
    $LAMMPS_EXE -in lammps_umbrella.in 2>&1 | tee lammps_umbrella.log
    
    if [ $? -eq 0 ]; then
        echo "✅ LAMMPS umbrella GCMC simulation completed"
    else
        echo "❌ LAMMPS umbrella GCMC simulation failed"
    fi
fi

# Analyze and compare results
echo ""
echo "📊 Analyzing and comparing results..."
echo "===================================="

# Check if Python packages are available
python3 -c "import numpy, matplotlib, pandas, scipy" 2>/dev/null
if [ $? -eq 0 ]; then
    python3 compare_results.py
    
    # Generate additional plots
    if [ -f "gcmc_stats.out" ]; then
        python3 plot_density.py --gcmc-stats gcmc_stats.out
    fi
    
    echo "✅ Analysis completed - check validation plots"
else
    echo "⚠️  Python packages missing. Install: pip install numpy matplotlib pandas scipy"
    echo "   Manual analysis required."
fi

# Summary
echo ""
echo "📋 Validation Summary"
echo "===================="
echo "Files generated:"
echo "  - gpumd.log (GPUMD simulation log)"
if [ "$SKIP_LAMMPS" = false ]; then
    echo "  - lammps_gcmc.log (LAMMPS GCMC log)"
    echo "  - lammps_umbrella.log (LAMMPS umbrella log)"
fi
echo "  - gcmc_validation_plots.png (comparison plots)"
echo "  - density_profiles.png (density analysis)"
echo "  - acceptance_ratios.png (GCMC statistics)"

echo ""
echo "🎯 Next steps:"
echo "1. Review the validation report printed above"
echo "2. Check the generated plots for visual comparison"
echo "3. If validation fails, check:"
echo "   - Algorithm implementation differences"
echo "   - Parameter settings (temperature, chemical potential)"
echo "   - Potential model consistency"
echo "   - Numerical precision issues"

echo ""
echo "✨ Validation suite completed!"
