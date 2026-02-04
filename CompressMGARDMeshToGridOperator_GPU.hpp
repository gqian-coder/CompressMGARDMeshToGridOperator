/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * CompressMGARDMeshToGridOperator_GPU.hpp :
 * GPU kernel implementations for mesh-to-grid interpolation and recomposition
 * Portable across NVIDIA (CUDA) and AMD (HIP) GPUs
 *
 *  Created on: Jan 17, 2026
 *      Author: Auto-generated with GPU optimization
 */

#ifndef COMPRESS_MGARD_MESH_TO_GRID_OPERATOR_GPU_HPP
#define COMPRESS_MGARD_MESH_TO_GRID_OPERATOR_GPU_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

// Include MGARD's runtime configuration to detect available backends
#include <mgard/mgard-x/RuntimeX/MGARDXConfig.h>
#include <mgard/mgard-x/RuntimeX/DataTypes.h>

// Portable GPU programming: Use HIP or CUDA based on MGARD's configuration
#if MGARD_ENABLE_HIP
    #include <hip/hip_runtime.h>
    #define GPU_ENABLED 1
    #define GPU_BACKEND_HIP 1
    // HIP compatibility macros
    #define gpuMalloc hipMalloc
    #define gpuFree hipFree
    #define gpuMemcpy hipMemcpy
    #define gpuMemset hipMemset
    #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpuDeviceSynchronize hipDeviceSynchronize
    #define gpuGetLastError hipGetLastError
    #define gpuPeekAtLastError hipPeekAtLastError
    #define gpuGetErrorString hipGetErrorString
    #define gpuSuccess hipSuccess
    #define gpuGetDeviceCount hipGetDeviceCount
    #define gpuSetDevice hipSetDevice
    #define gpuStream_t hipStream_t
    #define gpuStreamCreate hipStreamCreate
    #define gpuStreamDestroy hipStreamDestroy
    #define gpuStreamSynchronize hipStreamSynchronize
    #define gpuMemcpyAsync hipMemcpyAsync
    #define gpuMemsetAsync hipMemsetAsync
    #define gpuError_t hipError_t
    using DeviceType = mgard_x::HIP;
#elif MGARD_ENABLE_CUDA
    #include <cuda_runtime.h>
    #define GPU_ENABLED 1
    #define GPU_BACKEND_CUDA 1
    // CUDA compatibility macros
    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMemcpy cudaMemcpy
    #define gpuMemset cudaMemset
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    #define gpuGetLastError cudaGetLastError
    #define gpuPeekAtLastError cudaPeekAtLastError
    #define gpuGetErrorString cudaGetErrorString
    #define gpuSuccess cudaSuccess
    #define gpuGetDeviceCount cudaGetDeviceCount
    #define gpuSetDevice cudaSetDevice
    #define gpuStream_t cudaStream_t
    #define gpuStreamCreate cudaStreamCreate
    #define gpuStreamDestroy cudaStreamDestroy
    #define gpuStreamSynchronize cudaStreamSynchronize
    #define gpuMemcpyAsync cudaMemcpyAsync
    #define gpuMemsetAsync cudaMemsetAsync
    #define gpuError_t cudaError_t
    using DeviceType = mgard_x::CUDA;
#else
    #define GPU_ENABLED 0
    using DeviceType = mgard_x::SERIAL;
#endif

namespace adios2
{
namespace plugin
{
namespace gpu
{

// GPU error checking macro
#if GPU_ENABLED
#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        gpuError_t err = call;                                                 \
        if (err != gpuSuccess) {                                               \
            std::cerr << "GPU Error: " << gpuGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        }                                                                      \
    } while (0)
#else
#define GPU_CHECK(call) (void)0
#endif

/**
 * @brief Check if GPU device is available
 * @return true if GPU is available, false otherwise
 */
inline bool isGPUAvailable()
{
#if GPU_ENABLED
    int deviceCount = 0;
    gpuError_t err = gpuGetDeviceCount(&deviceCount);
    return (err == gpuSuccess && deviceCount > 0);
#else
    return false;
#endif
}

/**
 * @brief Get the number of available GPU devices
 * @return Number of GPU devices, 0 if none available
 */
inline int getGPUDeviceCount()
{
#if GPU_ENABLED
    int deviceCount = 0;
    gpuError_t err = gpuGetDeviceCount(&deviceCount);
    if (err != gpuSuccess) {
        return 0;
    }
    return deviceCount;
#else
    return 0;
#endif
}

#if GPU_ENABLED

// =============================================================================
// GPU Kernel: Accumulate mesh values to grid points
// Each thread handles one mesh node
// =============================================================================
template <typename T>
__global__ void kernel_accumulate_to_grid(
    const T* __restrict__ meshValues,      // Input: mesh node values
    T* __restrict__ gridValues,            // Output: accumulated grid values (use atomicAdd)
    const size_t* __restrict__ nodeMapGrid, // Mapping: mesh node -> grid point index
    size_t nNodePt)                         // Number of mesh nodes
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nNodePt) {
        size_t gridIdx = nodeMapGrid[idx];
        // Use atomic add to accumulate values from multiple mesh nodes to same grid point
        atomicAdd(&gridValues[gridIdx], meshValues[idx]);
    }
}

// =============================================================================
// GPU Kernel: Average grid values by cluster count
// Each thread handles one grid point
// =============================================================================
template <typename T>
__global__ void kernel_average_grid(
    T* __restrict__ gridValues,           // In/Out: grid values to be averaged
    const size_t* __restrict__ nCluster,  // Cluster count for each grid point
    size_t nGridPt)                        // Number of grid points
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nGridPt) {
        size_t count = nCluster[idx];
        if (count > 0) {
            gridValues[idx] = gridValues[idx] / static_cast<T>(count);
        }
    }
}

// =============================================================================
// GPU Kernel: Calculate residuals on mesh nodes
// Each thread handles one mesh node
// =============================================================================
template <typename T>
__global__ void kernel_calc_residual(
    T* __restrict__ meshResidValues,       // In/Out: mesh values -> residuals
    const T* __restrict__ gridValues,      // Input: averaged grid values
    const size_t* __restrict__ nodeMapGrid, // Mapping: mesh node -> grid point index
    size_t nNodePt)                         // Number of mesh nodes
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nNodePt) {
        size_t gridIdx = nodeMapGrid[idx];
        meshResidValues[idx] = meshResidValues[idx] - gridValues[gridIdx];
    }
}

// =============================================================================
// GPU Kernel: Recompose mesh values from residuals and grid interpolation
// Each thread handles one mesh node
// =============================================================================
template <typename T>
__global__ void kernel_recompose_remesh(
    T* __restrict__ combinedVal,           // In/Out: residuals -> reconstructed values
    const T* __restrict__ gridValues,      // Input: decompressed grid values
    const size_t* __restrict__ nodeMapGrid, // Mapping: mesh node -> grid point index
    size_t nNodePt)                         // Number of mesh nodes
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nNodePt) {
        size_t gridIdx = nodeMapGrid[idx];
        combinedVal[idx] = combinedVal[idx] + gridValues[gridIdx];
    }
}

#endif // GPU_ENABLED

// =============================================================================
// GPU Memory Management Helper Class
// =============================================================================
class GPUMemoryManager
{
public:
    /**
     * @brief Allocate memory on GPU
     * @param size Size in bytes to allocate
     * @return Pointer to allocated GPU memory, or nullptr if failed
     */
    template <typename T>
    static T* allocate(size_t count)
    {
#if GPU_ENABLED
        T* ptr = nullptr;
        gpuError_t err = gpuMalloc(&ptr, count * sizeof(T));
        if (err != gpuSuccess) {
            std::cerr << "GPU Malloc failed: " << gpuGetErrorString(err) << std::endl;
            return nullptr;
        }
        return ptr;
#else
        return nullptr;
#endif
    }

    /**
     * @brief Free GPU memory
     * @param ptr Pointer to GPU memory to free
     */
    template <typename T>
    static void free(T* ptr)
    {
#if GPU_ENABLED
        if (ptr != nullptr) {
            gpuFree(ptr);
        }
#endif
    }

    /**
     * @brief Copy data from host to device
     * @param dst Destination pointer (device)
     * @param src Source pointer (host)
     * @param count Number of elements to copy
     */
    template <typename T>
    static bool copyHostToDevice(T* dst, const T* src, size_t count)
    {
#if GPU_ENABLED
        gpuError_t err = gpuMemcpy(dst, src, count * sizeof(T), gpuMemcpyHostToDevice);
        return (err == gpuSuccess);
#else
        return false;
#endif
    }

    /**
     * @brief Copy data from device to host
     * @param dst Destination pointer (host)
     * @param src Source pointer (device)
     * @param count Number of elements to copy
     */
    template <typename T>
    static bool copyDeviceToHost(T* dst, const T* src, size_t count)
    {
#if GPU_ENABLED
        gpuError_t err = gpuMemcpy(dst, src, count * sizeof(T), gpuMemcpyDeviceToHost);
        return (err == gpuSuccess);
#else
        return false;
#endif
    }

    /**
     * @brief Set GPU memory to zero
     * @param ptr Pointer to GPU memory
     * @param count Number of elements (type T)
     */
    template <typename T>
    static bool memsetZero(T* ptr, size_t count)
    {
#if GPU_ENABLED
        gpuError_t err = gpuMemset(ptr, 0, count * sizeof(T));
        return (err == gpuSuccess);
#else
        return false;
#endif
    }

    /**
     * @brief Synchronize GPU device
     */
    static void synchronize()
    {
#if GPU_ENABLED
        gpuDeviceSynchronize();
#endif
    }
    
    /**
     * @brief Free device memory (void pointer version)
     * @param ptr Pointer to GPU memory to free
     */
    static void freeDevice(void* ptr)
    {
#if GPU_ENABLED
        if (ptr != nullptr) {
            gpuFree(ptr);
        }
#else
        (void)ptr;
#endif
    }
};

// =============================================================================
// GPU Implementation of calc_GridValResi - Keeps data on GPU for MGARD
// This version does NOT copy data back to host, avoiding redundant transfers.
// MGARD can detect device pointers and use them directly.
// Returns: true if GPU was used, false if CPU fallback needed
// =============================================================================
template <typename T>
bool calc_GridValResi_GPU(
    const std::vector<size_t>& nodeMapGrid_h,  // Host: mesh to grid mapping
    const std::vector<size_t>& nCluster_h,     // Host: cluster counts
    const char* dataIn,                         // Host: input mesh values
    size_t nNodePt,
    size_t nGridPt,
    void** d_meshResidVal_out,                 // Output: device pointer to residuals (for MGARD)
    void** d_gridVal_out)                      // Output: device pointer to grid values (for MGARD)
{
#if GPU_ENABLED
    if (!isGPUAvailable()) {
        return false;
    }

    const int blockSize = 256;
    
    // Calculate grid sizes
    int gridSizeNode = (nNodePt + blockSize - 1) / blockSize;
    int gridSizeGrid = (nGridPt + blockSize - 1) / blockSize;
    
    // Allocate GPU memory for mapping data (temporary, freed after kernels)
    size_t* d_nodeMapGrid = GPUMemoryManager::allocate<size_t>(nNodePt);
    size_t* d_nCluster = GPUMemoryManager::allocate<size_t>(nGridPt);
    
    // Allocate GPU memory for data (kept for MGARD to use)
    T* d_meshResidVal = GPUMemoryManager::allocate<T>(nNodePt);
    T* d_gridVal = GPUMemoryManager::allocate<T>(nGridPt);
    
    if (!d_nodeMapGrid || !d_nCluster || !d_meshResidVal || !d_gridVal) {
        // Cleanup on allocation failure
        GPUMemoryManager::free(d_nodeMapGrid);
        GPUMemoryManager::free(d_nCluster);
        GPUMemoryManager::free(d_meshResidVal);
        GPUMemoryManager::free(d_gridVal);
        return false;
    }
    
    // Copy mapping data to GPU
    GPUMemoryManager::copyHostToDevice(d_nodeMapGrid, nodeMapGrid_h.data(), nNodePt);
    GPUMemoryManager::copyHostToDevice(d_nCluster, nCluster_h.data(), nGridPt);
    
    // Copy input mesh values to GPU
    const T* meshVal_h = reinterpret_cast<const T*>(dataIn);
    GPUMemoryManager::copyHostToDevice(d_meshResidVal, meshVal_h, nNodePt);
    
    // Initialize grid values to zero
    GPUMemoryManager::memsetZero(d_gridVal, nGridPt);
    
    // Step 1: Accumulate mesh values to grid points
    kernel_accumulate_to_grid<T><<<gridSizeNode, blockSize>>>(
        d_meshResidVal, d_gridVal, d_nodeMapGrid, nNodePt);
    
    // Step 2: Average grid values
    kernel_average_grid<T><<<gridSizeGrid, blockSize>>>(
        d_gridVal, d_nCluster, nGridPt);
    
    // Step 3: Calculate residuals
    kernel_calc_residual<T><<<gridSizeNode, blockSize>>>(
        d_meshResidVal, d_gridVal, d_nodeMapGrid, nNodePt);
    
    GPUMemoryManager::synchronize();
    
    // Free temporary mapping data (no longer needed after kernels complete)
    GPUMemoryManager::free(d_nodeMapGrid);
    GPUMemoryManager::free(d_nCluster);
    
    // Return device pointers for MGARD to use directly
    // Data stays on GPU - no copy back to host!
    *d_meshResidVal_out = static_cast<void*>(d_meshResidVal);
    *d_gridVal_out = static_cast<void*>(d_gridVal);
    
    return true;
#else
    (void)nodeMapGrid_h;
    (void)nCluster_h;
    (void)dataIn;
    (void)nNodePt;
    (void)nGridPt;
    (void)d_meshResidVal_out;
    (void)d_gridVal_out;
    return false;
#endif
}

// =============================================================================
// Simplified GPU Implementation of calc_GridValResi (copies back to host)
// Use this only when you need data on host (e.g., CPU-based MGARD)
// =============================================================================
template <typename T>
bool calc_GridValResi_GPU_Simple(
    const std::vector<size_t>& nodeMapGrid_h,
    const std::vector<size_t>& nCluster_h,
    char* var_in,
    char* GridPointVal,
    size_t nNodePt,
    size_t nGridPt)
{
#if GPU_ENABLED
    if (!isGPUAvailable()) {
        return false;
    }

    const int blockSize = 256;
    
    int gridSizeNode = (nNodePt + blockSize - 1) / blockSize;
    int gridSizeGrid = (nGridPt + blockSize - 1) / blockSize;
    
    // Allocate GPU memory
    size_t* d_nodeMapGrid = GPUMemoryManager::allocate<size_t>(nNodePt);
    size_t* d_nCluster = GPUMemoryManager::allocate<size_t>(nGridPt);
    T* d_meshResidVal = GPUMemoryManager::allocate<T>(nNodePt);
    T* d_gridVal = GPUMemoryManager::allocate<T>(nGridPt);
    
    if (!d_nodeMapGrid || !d_nCluster || !d_meshResidVal || !d_gridVal) {
        GPUMemoryManager::free(d_nodeMapGrid);
        GPUMemoryManager::free(d_nCluster);
        GPUMemoryManager::free(d_meshResidVal);
        GPUMemoryManager::free(d_gridVal);
        return false;
    }
    
    // Copy data to GPU
    GPUMemoryManager::copyHostToDevice(d_nodeMapGrid, nodeMapGrid_h.data(), nNodePt);
    GPUMemoryManager::copyHostToDevice(d_nCluster, nCluster_h.data(), nGridPt);
    
    T* meshVal_h = reinterpret_cast<T*>(var_in);
    GPUMemoryManager::copyHostToDevice(d_meshResidVal, meshVal_h, nNodePt);
    GPUMemoryManager::memsetZero(d_gridVal, nGridPt);
    
    // Execute kernels
    kernel_accumulate_to_grid<T><<<gridSizeNode, blockSize>>>(
        d_meshResidVal, d_gridVal, d_nodeMapGrid, nNodePt);
    
    kernel_average_grid<T><<<gridSizeGrid, blockSize>>>(
        d_gridVal, d_nCluster, nGridPt);
    
    kernel_calc_residual<T><<<gridSizeNode, blockSize>>>(
        d_meshResidVal, d_gridVal, d_nodeMapGrid, nNodePt);
    
    GPUMemoryManager::synchronize();
    
    // Copy results back to host
    GPUMemoryManager::copyDeviceToHost(meshVal_h, d_meshResidVal, nNodePt);
    T* gridVal_h = reinterpret_cast<T*>(GridPointVal);
    GPUMemoryManager::copyDeviceToHost(gridVal_h, d_gridVal, nGridPt);
    
    // Cleanup
    GPUMemoryManager::free(d_nodeMapGrid);
    GPUMemoryManager::free(d_nCluster);
    GPUMemoryManager::free(d_meshResidVal);
    GPUMemoryManager::free(d_gridVal);
    
    return true;
#else
    (void)nodeMapGrid_h;
    (void)nCluster_h;
    (void)var_in;
    (void)GridPointVal;
    (void)nNodePt;
    (void)nGridPt;
    return false;
#endif
}

// =============================================================================
// GPU Implementation of recompose_remesh
// =============================================================================
template <typename T>
bool recompose_remesh_GPU(
    const std::vector<size_t>& nodeMapGrid_h,
    void* GridPointVal,    // Decompressed grid values (on host)
    void* combinedVal,     // Residuals on host, will become reconstructed values
    size_t nNodePt,
    size_t nGridPt)
{
#if GPU_ENABLED
    if (!isGPUAvailable()) {
        return false;
    }

    const int blockSize = 256;
    int gridSizeNode = (nNodePt + blockSize - 1) / blockSize;
    
    // Allocate GPU memory
    size_t* d_nodeMapGrid = GPUMemoryManager::allocate<size_t>(nNodePt);
    T* d_combinedVal = GPUMemoryManager::allocate<T>(nNodePt);
    T* d_gridVal = GPUMemoryManager::allocate<T>(nGridPt);
    
    if (!d_nodeMapGrid || !d_combinedVal || !d_gridVal) {
        GPUMemoryManager::free(d_nodeMapGrid);
        GPUMemoryManager::free(d_combinedVal);
        GPUMemoryManager::free(d_gridVal);
        return false;
    }
    
    // Copy data to GPU
    GPUMemoryManager::copyHostToDevice(d_nodeMapGrid, nodeMapGrid_h.data(), nNodePt);
    
    T* combinedVal_h = reinterpret_cast<T*>(combinedVal);
    T* gridVal_h = reinterpret_cast<T*>(GridPointVal);
    GPUMemoryManager::copyHostToDevice(d_combinedVal, combinedVal_h, nNodePt);
    GPUMemoryManager::copyHostToDevice(d_gridVal, gridVal_h, nGridPt);
    
    // Execute recomposition kernel
    kernel_recompose_remesh<T><<<gridSizeNode, blockSize>>>(
        d_combinedVal, d_gridVal, d_nodeMapGrid, nNodePt);
    
    GPUMemoryManager::synchronize();
    
    // Copy result back to host
    GPUMemoryManager::copyDeviceToHost(combinedVal_h, d_combinedVal, nNodePt);
    
    // Cleanup
    GPUMemoryManager::free(d_nodeMapGrid);
    GPUMemoryManager::free(d_combinedVal);
    GPUMemoryManager::free(d_gridVal);
    
    return true;
#else
    (void)nodeMapGrid_h;
    (void)GridPointVal;
    (void)combinedVal;
    (void)nNodePt;
    (void)nGridPt;
    return false;
#endif
}

// =============================================================================
// Advanced GPU Implementation with device memory management
// Keeps data on GPU for MGARD compression to avoid redundant transfers
// =============================================================================
class GPUMeshToGridContext
{
public:
#if GPU_ENABLED
    // Device pointers
    size_t* d_nodeMapGrid = nullptr;
    size_t* d_nCluster = nullptr;
    
    // Template pointers for data (void* for type erasure)
    void* d_meshResidVal = nullptr;
    void* d_gridVal = nullptr;
    
    size_t nNodePt = 0;
    size_t nGridPt = 0;
    bool isFloat = true;
    bool initialized = false;
    
    ~GPUMeshToGridContext()
    {
        cleanup();
    }
    
    void cleanup()
    {
        if (d_nodeMapGrid) { GPUMemoryManager::free(d_nodeMapGrid); d_nodeMapGrid = nullptr; }
        if (d_nCluster) { GPUMemoryManager::free(d_nCluster); d_nCluster = nullptr; }
        if (d_meshResidVal) { gpuFree(d_meshResidVal); d_meshResidVal = nullptr; }
        if (d_gridVal) { gpuFree(d_gridVal); d_gridVal = nullptr; }
        initialized = false;
    }
    
    template <typename T>
    bool initialize(
        const std::vector<size_t>& nodeMapGrid_h,
        const std::vector<size_t>& nCluster_h,
        const char* meshData,
        size_t nNode,
        size_t nGrid)
    {
        cleanup();
        
        nNodePt = nNode;
        nGridPt = nGrid;
        isFloat = std::is_same<T, float>::value;
        
        // Allocate mapping arrays
        d_nodeMapGrid = GPUMemoryManager::allocate<size_t>(nNodePt);
        d_nCluster = GPUMemoryManager::allocate<size_t>(nGridPt);
        
        // Allocate data arrays
        T* d_mesh = nullptr;
        T* d_grid = nullptr;
        gpuMalloc(&d_mesh, nNodePt * sizeof(T));
        gpuMalloc(&d_grid, nGridPt * sizeof(T));
        
        if (!d_nodeMapGrid || !d_nCluster || !d_mesh || !d_grid) {
            cleanup();
            if (d_mesh) gpuFree(d_mesh);
            if (d_grid) gpuFree(d_grid);
            return false;
        }
        
        d_meshResidVal = d_mesh;
        d_gridVal = d_grid;
        
        // Copy mapping data
        GPUMemoryManager::copyHostToDevice(d_nodeMapGrid, nodeMapGrid_h.data(), nNodePt);
        GPUMemoryManager::copyHostToDevice(d_nCluster, nCluster_h.data(), nGridPt);
        
        // Copy mesh data
        const T* meshData_h = reinterpret_cast<const T*>(meshData);
        GPUMemoryManager::copyHostToDevice(static_cast<T*>(d_meshResidVal), meshData_h, nNodePt);
        
        // Initialize grid to zero
        gpuMemset(d_gridVal, 0, nGridPt * sizeof(T));
        
        initialized = true;
        return true;
    }
    
    template <typename T>
    bool computeGridValResi()
    {
        if (!initialized) return false;
        
        const int blockSize = 256;
        int gridSizeNode = (nNodePt + blockSize - 1) / blockSize;
        int gridSizeGrid = (nGridPt + blockSize - 1) / blockSize;
        
        T* d_mesh = static_cast<T*>(d_meshResidVal);
        T* d_grid = static_cast<T*>(d_gridVal);
        
        // Step 1: Accumulate
        kernel_accumulate_to_grid<T><<<gridSizeNode, blockSize>>>(
            d_mesh, d_grid, d_nodeMapGrid, nNodePt);
        
        // Step 2: Average
        kernel_average_grid<T><<<gridSizeGrid, blockSize>>>(
            d_grid, d_nCluster, nGridPt);
        
        // Step 3: Calculate residuals
        kernel_calc_residual<T><<<gridSizeNode, blockSize>>>(
            d_mesh, d_grid, d_nodeMapGrid, nNodePt);
        
        GPUMemoryManager::synchronize();
        return true;
    }
    
    // Get device pointers for MGARD (data stays on GPU)
    void* getMeshResidDevicePtr() const { return d_meshResidVal; }
    void* getGridValDevicePtr() const { return d_gridVal; }
    
    // Copy results back to host if needed
    template <typename T>
    bool copyResultsToHost(char* meshResidHost, char* gridValHost)
    {
        if (!initialized) return false;
        
        T* mesh_h = reinterpret_cast<T*>(meshResidHost);
        T* grid_h = reinterpret_cast<T*>(gridValHost);
        T* d_mesh = static_cast<T*>(d_meshResidVal);
        T* d_grid = static_cast<T*>(d_gridVal);
        
        GPUMemoryManager::copyDeviceToHost(mesh_h, d_mesh, nNodePt);
        GPUMemoryManager::copyDeviceToHost(grid_h, d_grid, nGridPt);
        
        return true;
    }
#else
    bool initialized = false;
    void cleanup() {}
    template <typename T>
    bool initialize(const std::vector<size_t>&, const std::vector<size_t>&,
                    const char*, size_t, size_t) { return false; }
    template <typename T>
    bool computeGridValResi() { return false; }
    void* getMeshResidDevicePtr() const { return nullptr; }
    void* getGridValDevicePtr() const { return nullptr; }
    template <typename T>
    bool copyResultsToHost(char*, char*) { return false; }
#endif
};

} // namespace gpu
} // namespace plugin
} // namespace adios2

#endif // COMPRESS_MGARD_MESH_TO_GRID_OPERATOR_GPU_HPP
