@echo off
REM GCMC Validation Runner for Windows
REM This script runs both GPUMD and LAMMPS simulations and compares results

echo 🚀 Starting GCMC Validation Suite
echo =================================

REM Check if required executables exist
set GPUMD_EXE=..\..\src\gpumd.exe
set LAMMPS_EXE=lmp.exe

if not exist "%GPUMD_EXE%" (
    echo ❌ GPUMD executable not found at %GPUMD_EXE%
    echo    Please compile GPUMD first
    exit /b 1
)

where /q %LAMMPS_EXE%
if errorlevel 1 (
    echo ⚠️  LAMMPS executable not found. Reference comparison will be skipped.
    set SKIP_LAMMPS=true
) else (
    set SKIP_LAMMPS=false
)

REM Create initial structure if it doesn't exist
if not exist "model.xyz" (
    echo 📁 Creating initial atomic structure...
    python create_structure.py
)

REM Run GPUMD CUDA GCMC simulation
echo.
echo 🖥️  Running GPUMD CUDA GCMC simulation...
echo ========================================
"%GPUMD_EXE%" > gpumd.log 2>&1

if errorlevel 1 (
    echo ❌ GPUMD simulation failed
    exit /b 1
) else (
    echo ✅ GPUMD simulation completed successfully
)

REM Run LAMMPS reference simulation
if "%SKIP_LAMMPS%"=="false" (
    echo.
    echo 🔬 Running LAMMPS reference simulation...
    echo =======================================
    
    REM Standard GCMC
    %LAMMPS_EXE% -in lammps_gcmc.in > lammps_gcmc.log 2>&1
    
    if errorlevel 1 (
        echo ❌ LAMMPS GCMC simulation failed
    ) else (
        echo ✅ LAMMPS GCMC simulation completed
    )
    
    REM Umbrella sampling GCMC
    %LAMMPS_EXE% -in lammps_umbrella.in > lammps_umbrella.log 2>&1
    
    if errorlevel 1 (
        echo ❌ LAMMPS umbrella GCMC simulation failed
    ) else (
        echo ✅ LAMMPS umbrella GCMC simulation completed
    )
)

REM Analyze and compare results
echo.
echo 📊 Analyzing and comparing results...
echo ====================================

REM Check if Python packages are available
python -c "import numpy, matplotlib, pandas, scipy" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Python packages missing. Install: pip install numpy matplotlib pandas scipy
    echo    Manual analysis required.
) else (
    python compare_results.py
    
    REM Generate additional plots
    if exist "gcmc_stats.out" (
        python plot_density.py --gcmc-stats gcmc_stats.out
    )
    
    echo ✅ Analysis completed - check validation plots
)

REM Summary
echo.
echo 📋 Validation Summary
echo ====================
echo Files generated:
echo   - gpumd.log (GPUMD simulation log)
if "%SKIP_LAMMPS%"=="false" (
    echo   - lammps_gcmc.log (LAMMPS GCMC log)
    echo   - lammps_umbrella.log (LAMMPS umbrella log)
)
echo   - gcmc_validation_plots.png (comparison plots)
echo   - density_profiles.png (density analysis)
echo   - acceptance_ratios.png (GCMC statistics)

echo.
echo 🎯 Next steps:
echo 1. Review the validation report printed above
echo 2. Check the generated plots for visual comparison
echo 3. If validation fails, check:
echo    - Algorithm implementation differences
echo    - Parameter settings (temperature, chemical potential)
echo    - Potential model consistency
echo    - Numerical precision issues

echo.
echo ✨ Validation suite completed!
pause
