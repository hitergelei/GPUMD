# ADP Potential Test Example

This directory contains a simple test example for the ADP (Angular Dependent Potential) implementation in GPUMD.

## Files

- `model.xyz`: A simple copper crystal structure for testing
- `run.in`: GPUMD input file using ADP potential
- `Cu.adp`: Sample ADP potential file for copper

## Usage

To run the test:

```bash
cd /path/to/this/directory
/path/to/gpumd < run.in
```

## Expected Output

The simulation should run without errors and produce output files including:
- `thermo.out`: Thermodynamic properties
- `energy.out`: Energy evolution

## Notes

This is a minimal test to verify the ADP potential implementation. The ADP potential file is a simplified version for testing purposes and may not be suitable for production calculations.
