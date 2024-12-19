/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * CompressMGARDMeshToGridOperator.cpp :
 *
 *  Created on: Dec 1, 2021
 *      Author: Jason Wang jason.ruonan.wang@gmail.com
 */

#include "CompressMGARDMeshToGridOperator.h"

#include "adios2.h"
#include "adios2/helper/adiosFunctions.h"
#include <mgard/MGARDConfig.hpp>
#include <mgard/compress_x.hpp>

#include <cstring>
#include <mutex>

namespace adios2
{
namespace plugin
{

/* "STATIC" PART of operator to have read mesh only once */
std::string meshFileName;
std::mutex readMeshMutex;
bool meshReadSuccessfully = false;

size_t blockId = 0;
size_t nBlocks;

std::vector<std::vector<size_t>> nodeMapGrid;  // stacking multiple blocks of parameters 
std::vector<std::vector<size_t>> nCluster;     // stacking multiple blocks of parameters 
std::vector<std::vector<size_t>> resampleRate; // stacking multiple blocks of parameters 
std::vector<bool>   sparsity;                  // size equals to the number of blocks, storing the indicator of sparsity

// define mesh mapping variable files

void ReadMesh(std::string meshfile, std::string Blc, std::string numBlc)
{
    std::cout << "=== Reading Mesh File " << meshfile << " ===" << std::endl;
    meshReadSuccessfully = true;
    meshFileName = meshfile;
    size_t bId = (size_t)std::stoi(Blc);
    nBlocks = (size_t)std::stoi(numBlc);
    std::cout << "readin starting from block " << bId << " for " << nBlocks << " blocks\n"; 

    adios2::Variable<size_t> var_map, var_cluster, var_gridDim;
    adios2::Variable<char> var_sparse;

    adios2::ADIOS ad;
    adios2::IO reader_io_m = ad.DeclareIO("InputMap");
    adios2::Engine reader_mesh = reader_io_m.Open(meshFileName, adios2::Mode::Read);

    size_t offsetGrid = 0, offsetNode = 0;
    while (true) {
        adios2::StepStatus read_status = reader_mesh.BeginStep(adios2::StepMode::Read, 10.0f);
        if (read_status == adios2::StepStatus::NotReady) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        else if (read_status != adios2::StepStatus::OK) {
            break;
        }
        var_map     = reader_io_m.InquireVariable<size_t>("MeshGridMap");
        var_cluster = reader_io_m.InquireVariable<size_t>("MeshGridCluster");
        var_gridDim = reader_io_m.InquireVariable<size_t>("GridDim");
        var_sparse  = reader_io_m.InquireVariable<char>("GridSparsity");
        auto info = reader_mesh.BlocksInfo(var_map, 0);
        //std::cout << "number of blocks in total: " << info.size() << "\n"; 
        for (size_t bi=bId; bi<bId+nBlocks; bi++) {
            var_map.SetBlockSelection(bi);
            var_cluster.SetBlockSelection(bi);
            var_gridDim.SetBlockSelection(bi);
            var_sparse.SetBlockSelection(bi);
            std::vector<size_t> nodeMapGrid_t, nCluster_t, resampleRate_t;
            char sparsity_t;
            reader_mesh.Get<size_t>(var_map    , nodeMapGrid_t , adios2::Mode::Sync);
            reader_mesh.Get<size_t>(var_cluster, nCluster_t    , adios2::Mode::Sync);
            reader_mesh.Get<size_t>(var_gridDim, resampleRate_t, adios2::Mode::Sync);
            reader_mesh.Get<char>(var_sparse ,   &sparsity_t   , adios2::Mode::Sync);
            reader_mesh.PerformGets();
            nodeMapGrid.push_back(nodeMapGrid_t);
            nCluster.push_back(nCluster_t);
            resampleRate.push_back(resampleRate_t);
            sparsity.push_back(sparsity_t);
        }
        reader_mesh.EndStep();
    }
    reader_mesh.Close();
    return;
}

template <typename T>
void calc_GridValResi(std::vector<size_t> nodeMapGrid,
                      std::vector<size_t> nCluster,
                      char * var_in, /* store the residual value back*/
                      char * GridPointVal,
                      size_t nNodePt,
                      size_t nGridPt)
{
    T * gridV_pt = reinterpret_cast<T*> (GridPointVal);
    T * resiV_pt = reinterpret_cast<T*> (var_in);
    for (size_t i=0; i<nNodePt; i++) {
        gridV_pt[nodeMapGrid[i]] += resiV_pt[i];
    }
    for (size_t i=0; i<nGridPt; i++) {
        gridV_pt[i] = gridV_pt[i] / (T)nCluster[i];
    }
    for (size_t i=0; i<nNodePt; i++) {
        resiV_pt[i] -= gridV_pt[nodeMapGrid[i]];
    }
}

template <typename T>
void recompose_remesh(std::vector<size_t> nodeMapGrid,
                      void * GridPointVal,
                      void * combinedVal)
{
    T * gridV_pt = reinterpret_cast<T*> (GridPointVal);
    T * resiV_pt = reinterpret_cast<T*> (combinedVal);
    size_t nNodePt = nodeMapGrid.size();
    for (size_t i=0; i<nNodePt; i++) {
        resiV_pt[i] += gridV_pt[nodeMapGrid[i]];
    }
}

/* END OF "STATIC" PART */

CompressMGARDMeshToGridOperator::CompressMGARDMeshToGridOperator(const Params &parameters) : PluginOperatorInterface(parameters)
{
    std::cout << "=== CompressMGARDMeshToGridOperator constructor ===" << std::endl;
    blockId = 0;
    auto itMeshFileName = m_Parameters.find("meshfile");
    if (itMeshFileName == m_Parameters.end())
    {
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator", "constructor",
                                             "This operator needs an unstructured mesh input file with parameter name 'meshfile'");
    }
    std::string &meshfile = itMeshFileName->second;

    std::cout << "meshfile = " << meshfile << "\n";
    auto itBlockId = m_Parameters.find("block");
    if (itBlockId == m_Parameters.end()) {
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator", "constructor",
                                             "This operator needs a blockId input with parameter name 'block'");
    }
    auto itBlock_N = m_Parameters.find("nblocks");
    if (itBlock_N == m_Parameters.end()) {
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator", "constructor",
                                             "This operator needs a number of blocks input with parameter name 'nblocks'");
    }
    std::cout << "meshfile = " << meshfile << ", nblocks = " << itBlock_N->second << "\n";
    if (!meshReadSuccessfully)
    {
        std::lock_guard<std::mutex> lck(readMeshMutex);
        if (!meshFileName.empty() && meshfile != meshFileName)
        {
            helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator", "constructor",
                                                 "Cannot process more than one mesh files. Already read " + meshFileName);
        }
        ReadMesh(itMeshFileName->second, itBlockId->second, itBlock_N->second);
    }
}

size_t CompressMGARDMeshToGridOperator::Operate(const char *dataIn, const Dims &blockStart, const Dims &blockCount,
                                                const DataType type, char *bufferOut)
{
    //std::cout << "=== CompressMGARDMeshToGridOperator::Operate() ===" << std::endl;
    const uint8_t bufferVersion = 1;
    size_t bufferOutOffset = 0;

    MakeCommonHeader(bufferOut, bufferOutOffset, bufferVersion);

    Dims convertedDims = ConvertDims(blockCount, type, 3);

    const size_t ndims = convertedDims.size();
    if (ndims > 5)
    {
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator", "Operate",
                                             "MGARD does not support data in " + std::to_string(ndims) + " dimensions");
    }

    // mgard V1 metadata
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
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator", "Operate",
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
    double ratio_t = 0.1;    // ratio of tolerance used for compressing grid and residuals
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
        helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator", "Operate",
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
            tol_resi = tolerance * (1-ratio_t); 
        }
        else if (itMode->second == "REL")
        {
            /* Cannot use REL to compress both grid interpolation and mesh residual
             * as we must ensure the total error in recomposed data to stay below
             * prescribed error tolerance */
            helper::Throw<std::invalid_argument>("Operator", "CompressMGARDMeshToGridOperator", "Operate",
                                             "must convert the relative tolerance to abs");
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

    PutParameter(bufferOut, bufferOutOffset, true);
    // remesh filename
    //PutParameter(bufferOut, bufferOutOffset, meshFileName);

    // remesh start here
    mgard_x::DIM mgardDim_gd = ndims;
    std::vector<mgard_x::SIZE> mgardCount_gd;
    
    size_t nNodePt = nodeMapGrid[blockId].size();
    size_t nGridPt = nCluster[blockId].size();
    size_t nbytes = 1;
    if (type == helper::GetDataType<float>()) {
        nbytes = sizeof(float);
    } else if (type == helper::GetDataType<double>()) {
         nbytes = sizeof(double);
    }
    char * MeshResidVal = (char *)malloc(nbytes * nNodePt);
    memcpy(MeshResidVal, dataIn, nbytes * nNodePt);
    char * GridPointVal =  (char *) malloc(nGridPt * nbytes);;
    memset(GridPointVal, 0, nbytes * nGridPt);
    if (type == helper::GetDataType<float>()) { 
        calc_GridValResi<float>(nodeMapGrid[blockId], nCluster[blockId], MeshResidVal, GridPointVal, nNodePt, nGridPt);
    } else if (type == helper::GetDataType<double>()) {
        calc_GridValResi<double>(nodeMapGrid[blockId], nCluster[blockId], MeshResidVal, GridPointVal, nNodePt, nGridPt);
    }
    // compress mesh residuals
    void *compressedData = bufferOut + bufferOutOffset + sizeof(size_t);
    mgard_x::compress(mgardDim, mgardType, mgardCount, tol_data, s, errorBoundType, MeshResidVal, compressedData, sizeOut, config, true);
    free(MeshResidVal);
    std::cout << "ndims = " << ndims << ", number of Grid points: " << nGridPt << ", mesh points: " << nNodePt << "\n";
    //std::cout << "tolerance = " << tolerance << ", data tol = " << tol_data << ", resi tol = " << tol_resi << "\n";
    std::cout << "Block " << blockId << ": bufferOutOffset before = " << bufferOutOffset;
    // store the compressed size of the mesh residual data
    PutParameter(bufferOut, bufferOutOffset, sizeOut);
    bufferOutOffset += sizeOut;
    std::cout << ", (compressed meshResi size = " << sizeOut << "), ";
    // compress grid interpolation
    compressedData = bufferOut + bufferOutOffset;
    sizeOut = helper::GetTotalSize(blockCount, helper::GetDataTypeSize(type));
    // have to use a different shape config representing grid architecture
    if (sparsity[blockId]==0) { 
         for (int d=0; d<ndims; d++) {
            mgardCount_gd.push_back(resampleRate[blockId].data()[d]);
        }
    } else {
        mgardDim_gd   = 1;
        mgardCount_gd.push_back(nGridPt); 
    }
    mgard_x::compress(mgardDim_gd, mgardType, mgardCount_gd, tol_resi, s, errorBoundType, GridPointVal, compressedData, sizeOut, config, true);
    bufferOutOffset += sizeOut;
    std::cout << "final bufferOutOffset = " << bufferOutOffset << " (compressed GridPt size = " << sizeOut << ")\n"; 
    free(GridPointVal); 
    blockId ++;
    
    return bufferOutOffset;
}

size_t CompressMGARDMeshToGridOperator::GetHeaderSize() const { return headerSize; }

size_t CompressMGARDMeshToGridOperator::DecompressV1(const char *bufferIn, const size_t sizeIn, char *dataOut)
{
    // Do NOT remove even if the buffer version is updated. Data might be still
    // in lagacy formats. This function must be kept for backward compatibility.
    // If a newer buffer format is implemented, create another function, e.g.
    // DecompressV2 and keep this function for decompressing lagacy data.

    size_t bufferInOffset = 0;

    const size_t ndims = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    Dims blockCount(ndims);
    for (size_t i = 0; i < ndims; ++i)
    {
        blockCount[i] = GetParameter<size_t, size_t>(bufferIn, bufferInOffset);
    }
    const DataType type = GetParameter<DataType>(bufferIn, bufferInOffset);
    m_VersionInfo = " Data is compressed using MGARD Version " + std::to_string(GetParameter<uint8_t>(bufferIn, bufferInOffset)) +
                    "." + std::to_string(GetParameter<uint8_t>(bufferIn, bufferInOffset)) + "." +
                    std::to_string(GetParameter<uint8_t>(bufferIn, bufferInOffset)) +
                    ". Please make sure a compatible version is used for decompression.";

    const bool isCompressed = GetParameter<bool>(bufferIn, bufferInOffset);
    size_t compressedSize_resi = GetParameter<size_t>(bufferIn, bufferInOffset);
    std::cout << "compressed residual data bytes = " << compressedSize_resi << ", sizeIn = " << sizeIn - bufferInOffset <<"\n";

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
            mgard_x::decompress(bufferIn + bufferInOffset, compressedSize_resi, dataOutVoid, true);
            mgard_x::decompress(bufferIn + bufferInOffset + compressedSize_resi, sizeIn - bufferInOffset - compressedSize_resi, GridPointVal, false);
            if (type == helper::GetDataType<float>()) {
                recompose_remesh<float>(nodeMapGrid[0], GridPointVal, dataOutVoid);
            } else if (type == helper::GetDataType<double>()) {
                recompose_remesh<double>(nodeMapGrid[0], GridPointVal, dataOutVoid);
            }
            free(GridPointVal);
        }
        catch (...)
        {
            helper::Throw<std::runtime_error>("Operator", "CompressMGARDMeshToGridOperator", "DecompressV1", m_VersionInfo);
        }
        return sizeOut;
    }

    headerSize += bufferInOffset;
    return 0;
}

size_t CompressMGARDMeshToGridOperator::InverseOperate(const char *bufferIn, const size_t sizeIn, char *dataOut)
{
    std::cout << "=== CompressMGARDMeshToGridOperator::InverseOperate() ===" << std::endl; 
    size_t bufferInOffset = 1; // skip operator type
    const uint8_t bufferVersion = GetParameter<uint8_t>(bufferIn, bufferInOffset);
    bufferInOffset += 2; // skip two reserved bytes
    headerSize = bufferInOffset;

    if (bufferVersion == 1)
    {
        // Need to think how to load the meshfile
        //ReadMesh(meshFileName, std::to_string(blockId), std::to_string(1));
        return DecompressV1(bufferIn + bufferInOffset, sizeIn - bufferInOffset, dataOut);
    }
    else if (bufferVersion == 2)
    {
        // TODO: if a Version 2 mgard buffer is being implemented, put it here
        // and keep the DecompressV1 routine for backward compatibility
    }
    else
    {
        helper::Throw<std::runtime_error>("Operator", "CompressMGARDMeshToGridOperator", "InverseOperate",
                                          "invalid mgard buffer version");
    }
    blockId ++;
    return 0;
}

bool CompressMGARDMeshToGridOperator::IsDataTypeValid(const DataType type) const
{
    if (type == DataType::Double || type == DataType::Float || type == DataType::DoubleComplex || type == DataType::FloatComplex)
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
