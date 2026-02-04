/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * CompressMGARDMeshToGridOperator.cpp :
 *
 *  Created on: Dec 1, 2021
 *      Author: Jason Wang jason.ruonan.wang@gmail.com
 *  Modified on: Jan 17, 2026
 *      Added GPU acceleration for mesh-to-grid interpolation and recomposition
 *      Portable across NVIDIA (CUDA) and AMD (HIP) GPUs
 */

#include "CompressMGARDMeshToGridOperator.h"
#include "CompressMGARDMeshToGridOperator_GPU.hpp"
#include "LosslessCompression.hpp"

#include "adios2/core/Engine.h"
#include "adios2/helper/adiosFunctions.h"
#include <mgard/MGARDConfig.hpp>
#include <mgard/compress_x.hpp>
#include <mgard/compressors.hpp>

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <chrono>  // For timing

namespace adios2
{
namespace plugin
{

/* "STATIC" PART of operator to have read mesh to grid mapping only once */
std::string mappingFileName;
std::mutex readMappingMutex;
// bool meshReadSuccessfully = false;
std::map<size_t, size_t> blockMeshMap;

const bool debugging = false;

// Flag to enable/disable GPU acceleration
// Can be disabled at runtime by setting environment variable: MGARD_MESHGRID_USE_CPU=1
bool useGPU = []() {
    const char* env = std::getenv("MGARD_MESHGRID_USE_CPU");
    if (env && (std::string(env) == "1" || std::string(env) == "true" || std::string(env) == "TRUE"))
        return false;
    return true;
}();

// Flag to print GPU status only once
static bool gpuStatusPrinted = false;

// Residual compression method selection
// Can be set via parameter "residual_method": "mgard" (default), "huffman_zstd", "zstd_only", or "auto"
enum class ResidualMethod {
    MGARD,        // Full MGARD compression (default)
    Huffman_ZSTD, // Quantization + Huffman + ZSTD (better for uncorrelated residuals)
    ZSTD_Only,    // Quantization + ZSTD only
    Auto          // Automatically detect based on data characteristics
};

// Parse residual method from string parameter
ResidualMethod ParseResidualMethod(const std::string& str) {
    if (str == "mgard" || str == "MGARD" || str == "Mgard") {
        return ResidualMethod::MGARD;
    } else if (str == "huffman_zstd" || str == "Huffman_ZSTD" || str == "huffman") {
        return ResidualMethod::Huffman_ZSTD;
    } else if (str == "zstd_only" || str == "ZSTD_Only" || str == "zstd") {
        return ResidualMethod::ZSTD_Only;
    } else if (str == "auto" || str == "Auto" || str == "AUTO") {
        return ResidualMethod::Auto;
    }
    return ResidualMethod::Huffman_ZSTD;  // Default
}

std::vector<std::vector<size_t>> nodeMapGrid;  // stacking multiple blocks of parameters
std::vector<std::vector<size_t>> nCluster;     // stacking multiple blocks of parameters
std::vector<std::vector<size_t>> resampleRate; // stacking multiple blocks of parameters
std::vector<bool> sparsity;                    // size equals to the number of blocks, storing the
                                               // indicator of sparsity

// single process reader for map2grid mapping data
adios2::ADIOS ad;
adios2::IO io_map;
adios2::Engine reader_map;

size_t ReadMapping(std::string mappingfile, size_t blockID)
{
    // meshReadSuccessfully = true;
    mappingFileName = mappingfile;
    if (debugging)
    {
        std::cout << "Reading Mesh File " << mappingfile << " starting from block " << blockID
                  << "\n";
    }

    adios2::Variable<size_t> var_map, var_cluster, var_gridDim;
    adios2::Variable<uint8_t> var_sparse;

    if (!io_map)
    {
        io_map = ad.DeclareIO("InputMap");
    }
    if (!reader_map)
    {
        io_map.SetParameter("SelectSteps", "0");
        reader_map = io_map.Open(mappingFileName, adios2::Mode::ReadRandomAccess);
    }

    size_t offsetGrid = 0, offsetNode = 0;
    
    // Variables might be stored as size_t or uint64_t depending on the version
    // Try with prefix first (newer format), then without prefix (older format)
    var_map = io_map.InquireVariable<size_t>("__mesh_grid_mapping__/MeshGridMap");
    if (!var_map)
    {
        var_map = io_map.InquireVariable<size_t>("MeshGridMap");
    }
    // Try uint64_t if size_t didn't work (ADIOS2 type matching is strict)
    adios2::Variable<uint64_t> var_map_u64;
    if (!var_map)
    {
        var_map_u64 = io_map.InquireVariable<uint64_t>("__mesh_grid_mapping__/MeshGridMap");
        if (!var_map_u64)
        {
            var_map_u64 = io_map.InquireVariable<uint64_t>("MeshGridMap");
        }
    }
    
    var_cluster = io_map.InquireVariable<size_t>("__mesh_grid_mapping__/MeshGridCluster");
    if (!var_cluster)
    {
        var_cluster = io_map.InquireVariable<size_t>("MeshGridCluster");
    }
    adios2::Variable<uint64_t> var_cluster_u64;
    if (!var_cluster)
    {
        var_cluster_u64 = io_map.InquireVariable<uint64_t>("__mesh_grid_mapping__/MeshGridCluster");
        if (!var_cluster_u64)
        {
            var_cluster_u64 = io_map.InquireVariable<uint64_t>("MeshGridCluster");
        }
    }
    
    var_gridDim = io_map.InquireVariable<size_t>("__mesh_grid_mapping__/GridDim");
    if (!var_gridDim)
    {
        var_gridDim = io_map.InquireVariable<size_t>("GridDim");
    }
    adios2::Variable<uint64_t> var_gridDim_u64;
    if (!var_gridDim)
    {
        var_gridDim_u64 = io_map.InquireVariable<uint64_t>("__mesh_grid_mapping__/GridDim");
        if (!var_gridDim_u64)
        {
            var_gridDim_u64 = io_map.InquireVariable<uint64_t>("GridDim");
        }
    }
    
    var_sparse = io_map.InquireVariable<uint8_t>("__mesh_grid_mapping__/GridSparsity");
    if (!var_sparse)
    {
        var_sparse = io_map.InquireVariable<uint8_t>("GridSparsity");
    }
    
    // Determine which type was found
    bool use_uint64 = !var_map && var_map_u64;
    
    // Check if all required variables were found
    if ((!var_map && !var_map_u64) || (!var_cluster && !var_cluster_u64) || (!var_gridDim && !var_gridDim_u64))
    {
        std::cerr << "Error: Could not find required mapping variables in " << mappingfile << std::endl;
        std::cerr << "  MeshGridMap: " << ((var_map || var_map_u64) ? "found" : "NOT FOUND") << std::endl;
        std::cerr << "  MeshGridCluster: " << ((var_cluster || var_cluster_u64) ? "found" : "NOT FOUND") << std::endl;
        std::cerr << "  GridDim: " << ((var_gridDim || var_gridDim_u64) ? "found" : "NOT FOUND") << std::endl;
        std::cerr << "  GridSparsity: " << (var_sparse ? "found" : "NOT FOUND (optional)") << std::endl;
        throw std::runtime_error("Missing required mapping variables");
    }
    
    auto info = use_uint64 ? reader_map.BlocksInfo(var_map_u64, 0) : reader_map.BlocksInfo(var_map, 0);
    if (debugging)
        std::cout << "  number of blocks in total: " << info.size() << std::endl;

    std::vector<size_t> nodeMapGrid_t, nCluster_t, resampleRate_t;
    uint8_t sparsity_t = 0;  // Default to non-sparse
    
    if (use_uint64)
    {
        // Use uint64_t versions
        var_map_u64.SetBlockSelection(blockID);
        var_cluster_u64.SetBlockSelection(blockID);
        var_gridDim_u64.SetBlockSelection(blockID);
        
        std::vector<uint64_t> nodeMapGrid_u64, nCluster_u64, resampleRate_u64;
        reader_map.Get<uint64_t>(var_map_u64, nodeMapGrid_u64, adios2::Mode::Sync);
        reader_map.Get<uint64_t>(var_cluster_u64, nCluster_u64, adios2::Mode::Sync);
        reader_map.Get<uint64_t>(var_gridDim_u64, resampleRate_u64, adios2::Mode::Sync);
        
        // Convert to size_t
        nodeMapGrid_t.assign(nodeMapGrid_u64.begin(), nodeMapGrid_u64.end());
        nCluster_t.assign(nCluster_u64.begin(), nCluster_u64.end());
        resampleRate_t.assign(resampleRate_u64.begin(), resampleRate_u64.end());
    }
    else
    {
        // Use size_t versions
        var_map.SetBlockSelection(blockID);
        var_cluster.SetBlockSelection(blockID);
        var_gridDim.SetBlockSelection(blockID);
        
        reader_map.Get<size_t>(var_map, nodeMapGrid_t, adios2::Mode::Sync);
        reader_map.Get<size_t>(var_cluster, nCluster_t, adios2::Mode::Sync);
        reader_map.Get<size_t>(var_gridDim, resampleRate_t, adios2::Mode::Sync);
    }
    
    // Handle GridSparsity which might be uint8_t or char type, or might not exist
    if (var_sparse)
    {
        var_sparse.SetBlockSelection(blockID);
        reader_map.Get<uint8_t>(var_sparse, &sparsity_t, adios2::Mode::Sync);
    }
    else
    {
        // Try char type (older format)
        auto var_sparse_char = io_map.InquireVariable<char>("__mesh_grid_mapping__/GridSparsity");
        if (!var_sparse_char)
        {
            var_sparse_char = io_map.InquireVariable<char>("GridSparsity");
        }
        if (var_sparse_char)
        {
            var_sparse_char.SetBlockSelection(blockID);
            char sparsity_char;
            reader_map.Get<char>(var_sparse_char, &sparsity_char, adios2::Mode::Sync);
            sparsity_t = static_cast<uint8_t>(sparsity_char);
        }
        // If still not found, use default (non-sparse)
    }
    
    reader_map.PerformGets();
    nodeMapGrid.push_back(nodeMapGrid_t);
    nCluster.push_back(nCluster_t);
    resampleRate.push_back(resampleRate_t);
    sparsity.push_back(sparsity_t);
    return nodeMapGrid.size() - 1;
}

template <typename T>
void calc_GridValResi(std::vector<size_t> nodeMapGrid, std::vector<size_t> nCluster,
                      char *var_in, /* store the residual value back*/
                      char *GridPointVal, size_t nNodePt, size_t nGridPt)
{
    T *gridV_pt = reinterpret_cast<T *>(GridPointVal);
    T *resiV_pt = reinterpret_cast<T *>(var_in);
    for (size_t i = 0; i < nNodePt; i++)
    {
        gridV_pt[nodeMapGrid[i]] += resiV_pt[i];
    }
    for (size_t i = 0; i < nGridPt; i++)
    {
        gridV_pt[i] = gridV_pt[i] / (T)nCluster[i];
    }
    for (size_t i = 0; i < nNodePt; i++)
    {
        resiV_pt[i] -= gridV_pt[nodeMapGrid[i]];
    }
}

template <typename T>
void recompose_remesh(std::vector<size_t> nodeMapGrid, void *GridPointVal, void *combinedVal)
{
    T *gridV_pt = reinterpret_cast<T *>(GridPointVal);
    T *resiV_pt = reinterpret_cast<T *>(combinedVal);
    size_t nNodePt = nodeMapGrid.size();
    for (size_t i = 0; i < nNodePt; i++)
    {
        resiV_pt[i] += gridV_pt[nodeMapGrid[i]];
    }
}

/* END OF "STATIC" PART */

CompressMGARDMeshToGridOperator::CompressMGARDMeshToGridOperator(const Params &parameters)
: PluginOperatorInterface(parameters)
{
    if (debugging)
    {
        std::cout << "=== CompressMGARDMeshToGridOperator constructor ===" << std::endl;
        for (auto &it : m_Parameters)
        {
            std::cout << "-- parameter " << it.first << " = " << it.second << std::endl;
        }
    }
}

CompressMGARDMeshToGridOperator::~CompressMGARDMeshToGridOperator()
{
    if (debugging)
        std::cout << "**** CompressMGARDMeshToGridOperator destructor called " << std::endl;

    if (reader_map)
    {
        reader_map.Close();
    }
}

void CompressMGARDMeshToGridOperator::AddExtraParameters(const Params &params)
{
    if (debugging)
        std::cout << "===== CompressMGARDMeshToGridOperator::AddExtraParameters() got params "
                  << std::endl;
    for (auto &it : params)
    {
        if (it.first == "EngineName")
            m_EngineName = it.second;
        else if (it.first == "VariableName")
            m_VariableName = it.second;
        else
            std::cout << "Unrecognized extra parameter '" << it.first
                      << "' passed to CompressMGARDMeshToGridOperator" << std::endl;
    }
    if (debugging)
    {
        std::cout << "    engine " << m_EngineName << std::endl;
        std::cout << "    variable " << m_VariableName << std::endl;
    }
};

size_t CompressMGARDMeshToGridOperator::Operate(const char *dataIn, const Dims &blockStart,
                                                const Dims &blockCount, const DataType type,
                                                char *bufferOut)
{
    // Print GPU status once at first use
    if (!gpuStatusPrinted)
    {
        gpuStatusPrinted = true;
        std::cout << "[CompressMGARDMeshToGridOperator] GPU Support: ";
#if defined(ENABLE_HIP)
        if (!useGPU)
            std::cout << "HIP/ROCm - DISABLED via MGARD_MESHGRID_USE_CPU=1\n";
        else if (gpu::isGPUAvailable())
            std::cout << "HIP/ROCm (AMD GPU) - ENABLED and AVAILABLE\n";
        else
            std::cout << "HIP/ROCm (AMD GPU) - compiled but NO GPU AVAILABLE\n";
#elif defined(ENABLE_CUDA)
        if (!useGPU)
            std::cout << "CUDA - DISABLED via MGARD_MESHGRID_USE_CPU=1\n";
        else if (gpu::isGPUAvailable())
            std::cout << "CUDA (NVIDIA GPU) - ENABLED and AVAILABLE\n";
        else
            std::cout << "CUDA (NVIDIA GPU) - compiled but NO GPU AVAILABLE\n";
#else
        std::cout << "DISABLED (CPU-only build)\n";
#endif
    }

    if (debugging)
        std::cout << "=== CompressMGARDMeshToGridOperator::Operate() ===" << std::endl;

    auto itMappingFileName = m_Parameters.find("meshfile");
    if (itMappingFileName == m_Parameters.end())
    {
        helper::Throw<std::invalid_argument>(
            "Operator", "CompressMGARDMeshToGridOperator", "Operate",
            "This operator needs an unstructured mesh input file with "
            "parameter name 'meshfile' for compression");
    }
    m_MappingFile = itMappingFileName->second;

    // Each var.AddOperator() will call this constructor, and we set blockId back
    // to 0 to read meshMap from the beginning
    m_BlockId = 0;
    auto itBlockId = m_Parameters.find("blockid");
    if (itBlockId == m_Parameters.end())
    {
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator",
                                             "Operate",
                                             "This operator needs the blockId with "
                                             "parameter name 'blockid' for compression");
    }
    m_BlockId = adios2::helper::StringToSizeT(itBlockId->second, "blockID in compression operator");

    // Read mapping for this block if not read before (by another operator instance on this process)
    if (blockMeshMap.find(m_BlockId) == blockMeshMap.end())
    {
        // if the requested block has not been loaded
        if (!m_MappingFile.empty())
        {
            std::lock_guard<std::mutex> lck(readMappingMutex);
            if (!mappingFileName.empty() && m_MappingFile != mappingFileName)
            {
                helper::Throw<std::invalid_argument>(
                    "Operator", "CompressMGARDMeshToGridOperator", "Operate",
                    "Cannot process more than one mesh files. Already read " + mappingFileName);
            }
            m_MapId = ReadMapping(m_MappingFile, m_BlockId);
            // insert a new blockId
            blockMeshMap[m_BlockId] = m_MapId;
        }
    }
    m_MapId = blockMeshMap[m_BlockId];
    if (debugging)
        std::cout << "  working with blockId " << m_BlockId << " and mapping id " << m_MapId
                  << std::endl;

    // Buffer version 2: includes residual compression method marker
    // Version 1 was MGARD-only, kept for backward compatibility
    const uint8_t bufferVersion = 2;
    size_t bufferOutOffset = 0;

    MakeCommonHeader(bufferOut, bufferOutOffset, bufferVersion);

    Dims convertedDims = ConvertDims(blockCount, type, 3);

    const size_t ndims = convertedDims.size();
    if (ndims > 5)
    {
        helper::Throw<std::invalid_argument>(
            "Operator", "CompressMGARDMeshToGridOperator", "Operate",
            "MGARD does not support data in " + std::to_string(ndims) + " dimensions");
    }
    // mgard V1 metadata
    // DEBUG: store block Id
    PutParameter(bufferOut, bufferOutOffset, m_BlockId);
    if (debugging)
        std::cout << "store blockId " << m_BlockId << " into metadata\n";

    PutParameter(bufferOut, bufferOutOffset, ndims);
    for (const auto &d : convertedDims)
    {
        PutParameter(bufferOut, bufferOutOffset, d);
    }
    PutParameter(bufferOut, bufferOutOffset, type);
    PutParameter(bufferOut, bufferOutOffset, static_cast<uint8_t>(MGARD_VERSION_MAJOR));
    PutParameter(bufferOut, bufferOutOffset, static_cast<uint8_t>(MGARD_VERSION_MINOR));
    PutParameter(bufferOut, bufferOutOffset, static_cast<uint8_t>(MGARD_VERSION_PATCH));
    // mgard V1 metadata end

    // set type
    mgard_x::data_type mgardType;
    if (type == helper::GetDataType<float>())
    {
        mgardType = mgard_x::data_type::Float;
    }
    else if (type == helper::GetDataType<double>())
    {
        mgardType = mgard_x::data_type::Double;
    }
    else if (type == helper::GetDataType<std::complex<float>>())
    {
        mgardType = mgard_x::data_type::Float;
    }
    else if (type == helper::GetDataType<std::complex<double>>())
    {
        mgardType = mgard_x::data_type::Double;
    }
    else
    {
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator",
                                             "Operate",
                                             "MGARD only supports float and double types");
    }
    // set type end

    // set mgard style dim info
    mgard_x::DIM mgardDim = ndims;
    std::vector<mgard_x::SIZE> mgardCount;
    for (const auto &c : convertedDims)
    {
        mgardCount.push_back(c);
    }
    // set mgard style dim info end

    // Parameters
    bool hasTolerance = false;
    double tolerance = 0.0;
    double s = 0.0;
    double ratio_t = 0.1; // ratio of tolerance used for compressing grid and residuals
    double tol_data = 0.0, tol_resi = 0.0;
    auto errorBoundType = mgard_x::error_bound_type::ABS;

    // input size under this bound will not compress
    size_t thresholdSize = 100000;

    auto itThreshold = m_Parameters.find("threshold");
    if (itThreshold != m_Parameters.end())
    {
        thresholdSize = std::stod(itThreshold->second);
    }
    auto itAccuracy = m_Parameters.find("accuracy");
    if (itAccuracy != m_Parameters.end())
    {
        tolerance = std::stod(itAccuracy->second);
        hasTolerance = true;
    }
    auto itTolerance = m_Parameters.find("tolerance");
    if (itTolerance != m_Parameters.end())
    {
        tolerance = std::stod(itTolerance->second);
        hasTolerance = true;
    }
    if (!hasTolerance)
    {
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator",
                                             "Operate",
                                             "missing mandatory parameter tolerance / accuracy");
    }
    auto itTolRatio = m_Parameters.find("ebratio");
    if (itTolRatio != m_Parameters.end())
    {
        ratio_t = std::stof(itTolRatio->second);
    }
    auto itSParameter = m_Parameters.find("s");
    if (itSParameter != m_Parameters.end())
    {
        s = std::stod(itSParameter->second);
    }

    auto itMode = m_Parameters.find("mode");
    if (itMode != m_Parameters.end())
    {
        if (itMode->second == "ABS")
        {
            tol_data = tolerance * ratio_t;
            tol_resi = tolerance * (1 - ratio_t);
        }
        else if (itMode->second == "REL")
        {
            /* Cannot use REL to compress both grid interpolation and mesh residual
             * as we must ensure the total error in recomposed data to stay below
             * prescribed error tolerance */
            helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator",
                                                 "Operate",
                                                 "must convert the relative tolerance to abs");
        }
    }
    
    // Parse residual compression method
    // Options: "mgard", "huffman_zstd" (default), "zstd_only", "auto"
    ResidualMethod residualMethod = ResidualMethod::Huffman_ZSTD;
    auto itResidualMethod = m_Parameters.find("residual_method");
    if (itResidualMethod != m_Parameters.end())
    {
        residualMethod = ParseResidualMethod(itResidualMethod->second);
        if (debugging)
        {
            std::cout << "Residual compression method: " << itResidualMethod->second << "\n";
        }
    }

    // let mgard know the output buffer size
    size_t sizeOut = helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type));
    if (sizeOut < thresholdSize)
    {
        /* disable compression and add marker in the header*/
        PutParameter(bufferOut, bufferOutOffset, false);
        headerSize = bufferOutOffset;
        return 0;
    }
    mgard_x::Config config;
    config.lossless = mgard_x::lossless_type::Huffman_Zstd;
    
    // Set device type for MGARD GPU acceleration
#if defined(ENABLE_HIP)
    if (useGPU && gpu::isGPUAvailable())
    {
        config.dev_type = mgard_x::device_type::HIP;
        if (debugging)
            std::cout << "MGARD config: using HIP device\n";
    }
#elif defined(ENABLE_CUDA)
    if (useGPU && gpu::isGPUAvailable())
    {
        config.dev_type = mgard_x::device_type::CUDA;
        if (debugging)
            std::cout << "MGARD config: using CUDA device\n";
    }
#endif

    PutParameter(bufferOut, bufferOutOffset, true);

    // remesh start here
    mgard_x::DIM mgardDim_gd = ndims;
    std::vector<mgard_x::SIZE> mgardCount_gd;

    size_t nNodePt = nodeMapGrid[m_MapId].size();
    size_t nGridPt = nCluster[m_MapId].size();
    size_t nbytes = 1;
    if (type == helper::GetDataType<float>())
    {
        nbytes = sizeof(float);
    }
    else if (type == helper::GetDataType<double>())
    {
        nbytes = sizeof(double);
    }
    
    // Pointers for mesh residual and grid values
    // These can be either host or device pointers depending on GPU availability
    void *MeshResidVal = nullptr;
    void *GridPointVal = nullptr;
    void *d_MeshResidVal = nullptr;  // Device pointer (if GPU used)
    void *d_GridPointVal = nullptr;  // Device pointer (if GPU used)
    bool gpuUsed = false;
    
    // Timing for performance analysis
    static double total_meshgrid_time = 0.0;
    static double total_mgard_resi_time = 0.0;
    static double total_mgard_grid_time = 0.0;
    static int block_count = 0;
    auto t_start = std::chrono::steady_clock::now();
    
    // Try GPU acceleration for mesh-to-grid interpolation and residual calculation
    // This keeps data on GPU to avoid redundant CPU<->GPU transfers with MGARD
    if (useGPU && gpu::isGPUAvailable())
    {
        if (debugging)
        {
#if defined(ENABLE_HIP)
            std::cout << "Block " << m_BlockId << ": Attempting GPU acceleration (HIP/ROCm backend)\n";
#elif defined(ENABLE_CUDA)
            std::cout << "Block " << m_BlockId << ": Attempting GPU acceleration (CUDA backend)\n";
#else
            std::cout << "Block " << m_BlockId << ": Attempting GPU acceleration (unknown backend)\n";
#endif
        }
        
        if (type == helper::GetDataType<float>())
        {
            gpuUsed = gpu::calc_GridValResi_GPU<float>(
                nodeMapGrid[m_MapId], nCluster[m_MapId], 
                dataIn, nNodePt, nGridPt,
                &d_MeshResidVal, &d_GridPointVal);
        }
        else if (type == helper::GetDataType<double>())
        {
            gpuUsed = gpu::calc_GridValResi_GPU<double>(
                nodeMapGrid[m_MapId], nCluster[m_MapId], 
                dataIn, nNodePt, nGridPt,
                &d_MeshResidVal, &d_GridPointVal);
        }
        if (gpuUsed)
        {
            // Use device pointers - MGARD will detect and use them directly
            MeshResidVal = d_MeshResidVal;
            GridPointVal = d_GridPointVal;
            if (debugging)
            {
#if defined(ENABLE_HIP)
                std::cout << "Block " << m_BlockId << ": GPU acceleration active (HIP/ROCm), data stays on GPU\n";
#elif defined(ENABLE_CUDA)
                std::cout << "Block " << m_BlockId << ": GPU acceleration active (CUDA), data stays on GPU\n";
#endif
            }
        }
    }
    
    // Fall back to CPU if GPU is not available or failed
    if (!gpuUsed)
    {
        if (debugging)
        {
            if (useGPU)
                std::cout << "Block " << m_BlockId << ": GPU not available or failed, using CPU fallback\n";
            else
                std::cout << "Block " << m_BlockId << ": Using CPU (GPU disabled)\n";
        }
        
        // Allocate host memory
        MeshResidVal = malloc(nbytes * nNodePt);
        memcpy(MeshResidVal, dataIn, nbytes * nNodePt);
        GridPointVal = malloc(nGridPt * nbytes);
        memset(GridPointVal, 0, nbytes * nGridPt);
        
        if (type == helper::GetDataType<float>())
        {
            calc_GridValResi<float>(nodeMapGrid[m_MapId], nCluster[m_MapId], 
                                    (char*)MeshResidVal, (char*)GridPointVal,
                                    nNodePt, nGridPt);
        }
        else if (type == helper::GetDataType<double>())
        {
            calc_GridValResi<double>(nodeMapGrid[m_MapId], nCluster[m_MapId], 
                                     (char*)MeshResidVal, (char*)GridPointVal,
                                     nNodePt, nGridPt);
        }
        if (debugging)
            std::cout << "Block " << m_BlockId << ": CPU used for calc_GridValResi\n";
    }
    
    auto t_meshgrid_done = std::chrono::steady_clock::now();
    double meshgrid_time = std::chrono::duration<double>(t_meshgrid_done - t_start).count();
    total_meshgrid_time += meshgrid_time;
    
    // compress mesh residuals...saving 8 bytes for storing the compressed
    // meshResi size
    // Also save 1 byte for residual compression method marker
    // Note: MGARD will detect if MeshResidVal is a device pointer and use it directly
    void *compressedData = bufferOut + bufferOutOffset + sizeof(size_t) + 1;
    
    // Determine actual residual method (handle Auto mode)
    ResidualMethod actualMethod = residualMethod;
    if (residualMethod == ResidualMethod::Auto)
    {
        // Analyze data characteristics to choose method
        if (type == helper::GetDataType<float>())
        {
            if (lossless::ShouldUseLosslessForResidual(reinterpret_cast<float*>(MeshResidVal), nNodePt, tol_data))
                actualMethod = ResidualMethod::Huffman_ZSTD;
            else
                actualMethod = ResidualMethod::MGARD;
        }
        else if (type == helper::GetDataType<double>())
        {
            if (lossless::ShouldUseLosslessForResidual(reinterpret_cast<double*>(MeshResidVal), nNodePt, tol_data))
                actualMethod = ResidualMethod::Huffman_ZSTD;
            else
                actualMethod = ResidualMethod::MGARD;
        }
        if (debugging)
        {
            std::cout << "Block " << m_BlockId << ": Auto-selected residual method = " 
                      << (actualMethod == ResidualMethod::Huffman_ZSTD ? "Huffman_ZSTD" : "MGARD") << "\n";
        }
    }
    
    // Store the residual compression method marker (1 byte)
    // 0 = MGARD, 1 = Huffman_ZSTD, 2 = ZSTD_Only
    uint8_t methodMarker = static_cast<uint8_t>(actualMethod);
    PutParameter(bufferOut, bufferOutOffset, methodMarker);
    
    auto t_resi_start = std::chrono::steady_clock::now();
    
    // Compress residuals using selected method
    if (actualMethod == ResidualMethod::Huffman_ZSTD || actualMethod == ResidualMethod::ZSTD_Only)
    {
        // Use Quantization + Huffman + ZSTD (or just ZSTD)
        bool success = false;
        if (type == helper::GetDataType<float>())
        {
            if (actualMethod == ResidualMethod::Huffman_ZSTD)
            {
                success = lossless::CompressHuffmanZstd<float>(
                    reinterpret_cast<float*>(MeshResidVal), nNodePt, tol_data,
                    compressedData, sizeOut);
            }
            else
            {
                success = lossless::CompressZstdOnly<float>(
                    reinterpret_cast<float*>(MeshResidVal), nNodePt, tol_data,
                    compressedData, sizeOut);
            }
        }
        else if (type == helper::GetDataType<double>())
        {
            if (actualMethod == ResidualMethod::Huffman_ZSTD)
            {
                success = lossless::CompressHuffmanZstd<double>(
                    reinterpret_cast<double*>(MeshResidVal), nNodePt, tol_data,
                    compressedData, sizeOut);
            }
            else
            {
                success = lossless::CompressZstdOnly<double>(
                    reinterpret_cast<double*>(MeshResidVal), nNodePt, tol_data,
                    compressedData, sizeOut);
            }
        }
        
        if (!success)
        {
            // Fall back to MGARD if Huffman_ZSTD compression fails
            if (debugging)
                std::cout << "Block " << m_BlockId << ": Huffman_ZSTD compression failed, falling back to MGARD\n";
            methodMarker = 0;
            // Rewrite marker
            bufferOutOffset -= 1;
            PutParameter(bufferOut, bufferOutOffset, methodMarker);
            mgard_x::compress(mgardDim, mgardType, mgardCount, tol_data, s, errorBoundType, MeshResidVal,
                              compressedData, sizeOut, config, true);
        }
        else if (debugging)
        {
            std::cout << "Block " << m_BlockId << ": Quantization + Huffman + ZSTD compression successful, size = " << sizeOut << "\n";
        }
    }
    else
    {
        // Use MGARD compression (default)
        mgard_x::compress(mgardDim, mgardType, mgardCount, tol_data, s, errorBoundType, MeshResidVal,
                          compressedData, sizeOut, config, true);
    }
    
    auto t_resi_done = std::chrono::steady_clock::now();
    double resi_compress_time = std::chrono::duration<double>(t_resi_done - t_resi_start).count();
    total_mgard_resi_time += resi_compress_time;
    
    free(MeshResidVal);

    if (debugging)
        std::cout << "Block " << m_BlockId << ": bufferOutOffset before = " << bufferOutOffset;

    // Important!!! store the compressed size of the mesh residual data before the
    // compressed bytes
    PutParameter(bufferOut, bufferOutOffset, sizeOut);
    bufferOutOffset += sizeOut;

    // size_t compressedDataSize = sizeOut;
    if (debugging)
        std::cout << ", (compressed meshResi size = " << sizeOut << "), ";

    // compress grid interpolation
    compressedData = bufferOut + bufferOutOffset;
    sizeOut = helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type));
    // have to use a different shape config representing grid architecture
    if (sparsity[m_MapId] == 0)
    {
        for (int d = 0; d < ndims; d++)
        {
            mgardCount_gd.push_back(resampleRate[m_MapId].data()[d]);
        }
    }
    else
    {
        mgardDim_gd = 1;
        mgardCount_gd.push_back(nGridPt);
    }
    // Note: MGARD will detect if GridPointVal is a device pointer and use it directly
    auto t_grid_start = std::chrono::steady_clock::now();
    mgard_x::compress(mgardDim_gd, mgardType, mgardCount_gd, tol_resi, s, errorBoundType,
                      GridPointVal, compressedData, sizeOut, config, true);
    auto t_grid_done = std::chrono::steady_clock::now();
    double grid_compress_time = std::chrono::duration<double>(t_grid_done - t_grid_start).count();
    total_mgard_grid_time += grid_compress_time;
    
    bufferOutOffset += sizeOut;
    
    block_count++;
    
    // Print timing summary periodically (every 10 blocks or at specific blocks)
    // Disabled for now - uncomment to enable timing output
    // if (m_BlockId == 0 || (block_count % 10 == 0))
    // {
    //     std::cout << "[TIMING] Block " << m_BlockId << ": mesh-to-grid=" << meshgrid_time*1000 << "ms, "
    //               << "MGARD MeshResid=" << resi_compress_time*1000 << "ms, "
    //               << "MGARD GridPt=" << grid_compress_time*1000 << "ms, "
    //               << "GPU=" << (gpuUsed ? "YES" : "NO") << "\n";
    // }
    // // Print cumulative at end (block 95 for 96 blocks)
    // if (m_BlockId == 95)
    // {
    //     std::cout << "[TIMING SUMMARY] Total mesh-to-grid: " << total_meshgrid_time << "s, "
    //               << "Total MGARD MeshResid: " << total_mgard_resi_time << "s, "
    //               << "Total MGARD GridPt: " << total_mgard_grid_time << "s, "
    //               << "Blocks: " << block_count << "\n";
    // }
    
    if (debugging)
        std::cout << "final bufferOutOffset = " << bufferOutOffset
                  << " (compressed GridPt size = " << sizeOut << ")\n";
    
    // Cleanup: free memory (GPU or CPU depending on what was used)
    if (gpuUsed)
    {
        gpu::GPUMemoryManager::freeDevice(d_MeshResidVal);
        gpu::GPUMemoryManager::freeDevice(d_GridPointVal);
    }
    else
    {
        free(MeshResidVal);
        free(GridPointVal);
    }
    
    return bufferOutOffset;
}

size_t CompressMGARDMeshToGridOperator::GetHeaderSize() const { return headerSize; }

size_t CompressMGARDMeshToGridOperator::DecompressV1(const char *bufferIn, const size_t sizeIn,
                                                     char *dataOut)
{
    // V1 format: MGARD-only compression, no residual method marker
    // Kept for backward compatibility with data compressed before lossless support

    size_t bufferInOffset = 0;

    size_t blockID = GetParameter<size_t>(bufferIn, bufferInOffset);
    if (debugging)
        std::cout << "Decompressing blockId = " << blockID << " (V1 format, MGARD-only)\n";
    if (blockMeshMap.find(blockID) == blockMeshMap.end())
    {
        // the requested block has not been loaded
        std::lock_guard<std::mutex> lck(readMappingMutex);
        blockMeshMap[blockID] = ReadMapping(m_EngineName, blockID);
    }
    size_t mapID = blockMeshMap[blockID];
    if (debugging)
        std::cout << "Mapping id = " << mapID << "\n";

    const size_t ndims = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    Dims blockCount(ndims);
    for (size_t i = 0; i < ndims; ++i)
    {
        blockCount[i] = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    }
    const DataType type = GetParameter<DataType>(bufferIn, bufferInOffset);
    m_VersionInfo = " Data is compressed using MGARD Version " +
                    std::to_string(GetParameter<uint8_t>(bufferIn, bufferInOffset)) + "." +
                    std::to_string(GetParameter<uint8_t>(bufferIn, bufferInOffset)) + "." +
                    std::to_string(GetParameter<uint8_t>(bufferIn, bufferInOffset)) +
                    ". Please make sure a compatible version is used for decompression.";

    const bool isCompressed = GetParameter<bool>(bufferIn, bufferInOffset);
    
    // V1 format: no method marker, directly read compressed size
    size_t compressedSize_resi = GetParameter<size_t>(bufferIn, bufferInOffset);

    size_t sizeOut = helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type));

    if (type == DataType::FloatComplex || type == DataType::DoubleComplex)
    {
        sizeOut /= 2;
    }

    if (isCompressed)
    {
        try
        {
            void *dataOutVoid = dataOut;
            void *GridPointVal = NULL;
            
            // V1: Always use MGARD decompression
            mgard_x::decompress(bufferIn + bufferInOffset, compressedSize_resi, dataOutVoid, true);
            
            mgard_x::decompress(bufferIn + bufferInOffset + compressedSize_resi,
                                sizeIn - bufferInOffset - compressedSize_resi, GridPointVal, false);
            // std::cout << "nodeMapGrid: " << nodeMapGrid.size() << ", " <<
            // nodeMapGrid[mapID].size() << "\n";
            
            size_t nNodePt = nodeMapGrid[mapID].size();
            size_t nGridPt = nCluster[mapID].size();
            
            // Try GPU acceleration for recomposition
            bool gpuUsed = false;
            if (useGPU && gpu::isGPUAvailable())
            {
                if (debugging)
                {
#if defined(ENABLE_HIP)
                    std::cout << "Block " << blockID << ": Attempting GPU recomposition (HIP/ROCm backend)\n";
#elif defined(ENABLE_CUDA)
                    std::cout << "Block " << blockID << ": Attempting GPU recomposition (CUDA backend)\n";
#else
                    std::cout << "Block " << blockID << ": Attempting GPU recomposition (unknown backend)\n";
#endif
                }
                
                if (type == helper::GetDataType<float>())
                {
                    gpuUsed = gpu::recompose_remesh_GPU<float>(
                        nodeMapGrid[mapID], GridPointVal, dataOutVoid, nNodePt, nGridPt);
                }
                else if (type == helper::GetDataType<double>())
                {
                    gpuUsed = gpu::recompose_remesh_GPU<double>(
                        nodeMapGrid[mapID], GridPointVal, dataOutVoid, nNodePt, nGridPt);
                }
                if (debugging && gpuUsed)
                {
#if defined(ENABLE_HIP)
                    std::cout << "Block " << blockID << ": GPU recomposition complete (HIP/ROCm)\n";
#elif defined(ENABLE_CUDA)
                    std::cout << "Block " << blockID << ": GPU recomposition complete (CUDA)\n";
#endif
                }
            }
            
            // Fall back to CPU if GPU is not available or failed
            if (!gpuUsed)
            {
                if (debugging)
                {
                    if (useGPU)
                        std::cout << "Block " << blockID << ": GPU not available or failed for recomposition, using CPU\n";
                    else
                        std::cout << "Block " << blockID << ": Using CPU for recomposition (GPU disabled)\n";
                }
                if (type == helper::GetDataType<float>())
                {
                    recompose_remesh<float>(nodeMapGrid[mapID], GridPointVal, dataOutVoid);
                }
                else if (type == helper::GetDataType<double>())
                {
                    recompose_remesh<double>(nodeMapGrid[mapID], GridPointVal, dataOutVoid);
                }
            }
        }
        catch (...)
        {
            helper::Throw<std::runtime_error>("Operator", "CompressMGARDMeshToGridOperator",
                                              "DecompressV1", m_VersionInfo);
        }
        return sizeOut;
    }

    headerSize += bufferInOffset;
    return 0;
}

size_t CompressMGARDMeshToGridOperator::DecompressV2(const char *bufferIn, const size_t sizeIn,
                                                     char *dataOut)
{
    // V2 format: includes residual compression method marker (1 byte)
    // Supports MGARD, Huffman_ZSTD, and ZSTD_Only for residual compression

    size_t bufferInOffset = 0;

    size_t blockID = GetParameter<size_t>(bufferIn, bufferInOffset);
    if (debugging)
        std::cout << "Decompressing blockId = " << blockID << " (V2 format)\n";
    if (blockMeshMap.find(blockID) == blockMeshMap.end())
    {
        std::lock_guard<std::mutex> lck(readMappingMutex);
        blockMeshMap[blockID] = ReadMapping(m_EngineName, blockID);
    }
    size_t mapID = blockMeshMap[blockID];
    if (debugging)
        std::cout << "Mapping id = " << mapID << "\n";

    const size_t ndims = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    Dims blockCount(ndims);
    for (size_t i = 0; i < ndims; ++i)
    {
        blockCount[i] = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    }
    const DataType type = GetParameter<DataType>(bufferIn, bufferInOffset);
    m_VersionInfo = " Data is compressed using MGARD Version " +
                    std::to_string(GetParameter<uint8_t>(bufferIn, bufferInOffset)) + "." +
                    std::to_string(GetParameter<uint8_t>(bufferIn, bufferInOffset)) + "." +
                    std::to_string(GetParameter<uint8_t>(bufferIn, bufferInOffset)) +
                    ". Please make sure a compatible version is used for decompression.";

    const bool isCompressed = GetParameter<bool>(bufferIn, bufferInOffset);
    
    // V2: Read residual compression method marker (1 byte)
    uint8_t residualMethodMarker = GetParameter<uint8_t>(bufferIn, bufferInOffset);
    
    if (debugging)
    {
        const char* methodName = "Unknown";
        switch (residualMethodMarker)
        {
            case 0: methodName = "MGARD"; break;
            case 1: methodName = "Huffman_ZSTD"; break;
            case 2: methodName = "ZSTD_Only"; break;
            case 3: methodName = "Auto"; break;
        }
        std::cout << "Block " << blockID << ": Residual method marker = " 
                  << (int)residualMethodMarker << " (" << methodName << ")\n";
    }
    
    size_t compressedSize_resi = GetParameter<size_t>(bufferIn, bufferInOffset);

    size_t sizeOut = helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type));
    size_t nNodePt = nodeMapGrid[mapID].size();

    if (type == DataType::FloatComplex || type == DataType::DoubleComplex)
    {
        sizeOut /= 2;
    }

    if (isCompressed)
    {
        try
        {
            void *dataOutVoid = dataOut;
            void *GridPointVal = NULL;
            
            // Decompress residuals based on method marker
            ResidualMethod residualMethod = static_cast<ResidualMethod>(residualMethodMarker);
            
            if (residualMethod == ResidualMethod::Huffman_ZSTD || residualMethod == ResidualMethod::ZSTD_Only)
            {
                // Use Quantization + Huffman + ZSTD decompression (dequantize after decoding)
                if (type == helper::GetDataType<float>())
                {
                    bool success = lossless::DecompressHuffmanZstd<float>(
                        bufferIn + bufferInOffset, compressedSize_resi,
                        reinterpret_cast<float*>(dataOutVoid), nNodePt);
                    if (!success)
                    {
                        helper::Throw<std::runtime_error>("Operator", "CompressMGARDMeshToGridOperator",
                                                          "DecompressV2", "Huffman_ZSTD decompression failed");
                    }
                }
                else if (type == helper::GetDataType<double>())
                {
                    bool success = lossless::DecompressHuffmanZstd<double>(
                        bufferIn + bufferInOffset, compressedSize_resi,
                        reinterpret_cast<double*>(dataOutVoid), nNodePt);
                    if (!success)
                    {
                        helper::Throw<std::runtime_error>("Operator", "CompressMGARDMeshToGridOperator",
                                                          "DecompressV2", "Huffman_ZSTD decompression failed");
                    }
                }
                if (debugging)
                    std::cout << "Block " << blockID << ": Decompressed using Quantization + Huffman + ZSTD\n";
            }
            else
            {
                // Use MGARD decompression
                mgard_x::decompress(bufferIn + bufferInOffset, compressedSize_resi, dataOutVoid, true);
            }
            
            // Decompress grid interpolation (always uses MGARD)
            mgard_x::decompress(bufferIn + bufferInOffset + compressedSize_resi,
                                sizeIn - bufferInOffset - compressedSize_resi, GridPointVal, false);
            
            // Recompose mesh values
            if (type == helper::GetDataType<float>())
            {
                recompose_remesh<float>(nodeMapGrid[mapID], GridPointVal, dataOutVoid);
            }
            else if (type == helper::GetDataType<double>())
            {
                recompose_remesh<double>(nodeMapGrid[mapID], GridPointVal, dataOutVoid);
            }
        }
        catch (...)
        {
            helper::Throw<std::runtime_error>("Operator", "CompressMGARDMeshToGridOperator",
                                              "DecompressV2", m_VersionInfo);
        }
        return sizeOut;
    }

    headerSize += bufferInOffset;
    return 0;
}

size_t CompressMGARDMeshToGridOperator::InverseOperate(const char *bufferIn, const size_t sizeIn,
                                                       char *dataOut)
{
    if (debugging)
    {
        std::cout << "=== CompressMGARDMeshToGridOperator::InverseOperate() ===" << std::endl;
    }
    size_t bufferInOffset = 1; // skip operator type
    const uint8_t bufferVersion = GetParameter<uint8_t>(bufferIn, bufferInOffset);
    bufferInOffset += 2; // skip two reserved bytes
    headerSize = bufferInOffset;

    if (bufferVersion == 1)
    {
        // V1: MGARD-only format (backward compatibility)
        return DecompressV1(bufferIn + bufferInOffset, sizeIn - bufferInOffset, dataOut);
    }
    else if (bufferVersion == 2)
    {
        // V2: Supports multiple residual compression methods
        return DecompressV2(bufferIn + bufferInOffset, sizeIn - bufferInOffset, dataOut);
    }
    else
    {
        helper::Throw<std::runtime_error>("Operator", "CompressMGARDMeshToGridOperator",
                                          "InverseOperate", "invalid mgard buffer version");
    }
    return 0;
}

bool CompressMGARDMeshToGridOperator::IsDataTypeValid(const DataType type) const
{
    if (type == DataType::Double || type == DataType::Float || type == DataType::DoubleComplex ||
        type == DataType::FloatComplex)
    {
        return true;
    }
    return false;
}

} // end namespace plugin
} // end namespace adios2

extern "C" {

adios2::plugin::CompressMGARDMeshToGridOperator *OperatorCreate(const adios2::Params &parameters)
{
    return new adios2::plugin::CompressMGARDMeshToGridOperator(parameters);
}

void OperatorDestroy(adios2::plugin::CompressMGARDMeshToGridOperator *obj) { delete obj; }
}
