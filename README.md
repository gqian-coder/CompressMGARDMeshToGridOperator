# CompressMGARDMeshToGridOperator

A plugin operator for ADIOS2 that compresses arbitrary unstructured mesh data using MGARD.
This operator performs compression/decompression inside ADIOS's PUT and GET operations.

## Features

- **Mesh-to-Grid Interpolation**: Separates unstructured data into grid interpolation values and mesh residuals
- **MGARD Compression**: Leverages MGARD's lossy compression with error bounds
- **GPU Acceleration**: Optional GPU support for NVIDIA (CUDA) and AMD (HIP) GPUs
- **Automatic Fallback**: Falls back to CPU when GPU is not available

## How It Works

The operator separates unstructured data into two components:
1. **GridPointVal**: Interpolated values from unstructured mesh to a rectilinear grid
2. **MeshResidVal**: Residuals (differences) between interpolated and original values

Both components are compressed separately using MGARD with specified error tolerances.

## Requirements

- ADIOS2
- MGARD (with GPU support for GPU acceleration)
- For GPU support:
  - HIP (AMD GPUs, e.g., MI250X on Frontier)
  - CUDA (NVIDIA GPUs)

## Building

### CPU-only Build
```bash
mkdir build && cd build
source ../runconf_cpu
make
```

### HIP (AMD GPU) Build (e.g., on Frontier)
```bash
# Load required modules first
module load PrgEnv-amd rocm cmake

mkdir build && cd build
source ../runconf_hip
make
```

### CUDA (NVIDIA GPU) Build
```bash
# Load required modules first
# module load cuda cmake

mkdir build && cd build
source ../runconf_cuda
make
```

## Usage

### XML Configuration
```xml
<variable name="MyVar">
    <operation type="plugin">
        <parameter key="PluginName" value="mgardReMesh"/>
        <parameter key="PluginLibrary" value="CompressMGARDMeshToGridOperator"/>
        <parameter key="meshfile" value="mesh2grid_mapping.bp"/>
        <parameter key="accuracy" value="0.001"/>
        <parameter key="mode" value="ABS"/>
        <parameter key="ebratio" value="0.7"/>
    </operation>
</variable>
```

### In Source Code
```cpp
adios2::Params params;
params["PluginName"] = "mgardReMesh";
params["PluginLibrary"] = "CompressMGARDMeshToGridOperator";
params["meshfile"] = "mesh2grid_mapping.bp";
params["tolerance"] = "0.001";
params["mode"] = "ABS";
params["ebratio"] = "0.7";  // Ratio of tolerance for grid (0.7) vs residual (0.3)
var.AddOperation("plugin", params);
```

### Runtime Setup
```bash
# Set the plugin path to where the library is built
export ADIOS2_PLUGIN_PATH=/path/to/CompressMGARDMeshToGridOperator/build

# Run your application
./your_application
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `meshfile` | Path to mesh-to-grid mapping file (required) | - |
| `tolerance` or `accuracy` | Error tolerance for compression (required) | - |
| `mode` | Error bound type: "ABS" (absolute) | "ABS" |
| `ebratio` | Ratio of tolerance for grid vs residual | 0.1 |
| `threshold` | Minimum data size to trigger compression | 100000 |
| `s` | MGARD smoothness parameter | 0.0 |
| `blockid` | Block ID for multi-block data | 0 |

## GPU Acceleration

When built with GPU support, the operator automatically:
1. Detects if a GPU is available
2. Performs mesh-to-grid interpolation and residual calculation on GPU
3. Falls back to CPU if GPU is not available or fails

The GPU kernels optimize:
- `calc_GridValResi`: Mesh-to-grid interpolation and residual calculation (compression)
- `recompose_remesh`: Reconstructing mesh values from grid and residuals (decompression)

## Files

- `CompressMGARDMeshToGridOperator.cpp` - Main implementation
- `CompressMGARDMeshToGridOperator.h` - Header file
- `CompressMGARDMeshToGridOperator_GPU.hpp` - GPU kernel implementations (CUDA/HIP portable)
- `CMakeLists.txt` - Build configuration
- `runconf_hip` - HIP (AMD GPU) build configuration
- `runconf_cuda` - CUDA (NVIDIA GPU) build configuration
- `runconf_cpu` - CPU-only build configuration

## Examples

For detailed examples, see:
- `Unstructured-ReMesh/mgardPlug_adios_ge.cpp` - Compression example
- `Unstructured-ReMesh/mgardPlug_adios_decompress.cpp` - Decompression example

## Notes

- The operator requires a pre-generated mesh-to-grid mapping file (`meshfile`)
- Use absolute error tolerance (`mode=ABS`) for guaranteed error bounds
- The `ebratio` parameter controls how the tolerance is split between grid and residual compression
