/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * CompressMGARDMeshToGrid.h :
 *
 *  Created on: Dec 9, 2024
 *      Author: Qian Gong
 */

#ifndef COMPRESSMGARDMESHTOGRIDOPERATOR_H_
#define COMPRESSMGARDMESHTOGRIDOPERATOR_H_

#include "adios2/common/ADIOSTypes.h"
#include "adios2/operator/plugin/PluginOperatorInterface.h"

namespace adios2
{
namespace plugin
{

class CompressMGARDMeshToGridOperator : public PluginOperatorInterface
{

public:
    CompressMGARDMeshToGridOperator(const Params &parameters);

    ~CompressMGARDMeshToGridOperator() = default;

    /**
     * @param dataIn
     * @param blockStart
     * @param blockCount
     * @param type
     * @param bufferOut
     * @return size of compressed buffer
     */
    size_t Operate(const char *dataIn, const Dims &blockStart, const Dims &blockCount, const DataType type,
                   char *bufferOut) override;

    /**
     * @param bufferIn
     * @param sizeIn
     * @param dataOut
     * @return size of decompressed buffer
     */
    size_t InverseOperate(const char *bufferIn, const size_t sizeIn, char *dataOut) override;

    bool IsDataTypeValid(const DataType type) const override;

    size_t GetHeaderSize() const;

private:
    size_t headerSize = 0;
    /**
     * Decompress function for V1 buffer. Do NOT remove even if the buffer
     * version is updated. Data might be still in lagacy formats. This function
     * must be kept for backward compatibility
     * @param bufferIn : compressed data buffer (V1 only)
     * @param sizeIn : number of bytes in bufferIn
     * @param dataOut : decompressed data buffer
     * @return : number of bytes in dataOut
     */
    size_t DecompressV1(const char *bufferIn, const size_t sizeIn, char *dataOut);

    std::string m_VersionInfo;
};

} // end namespace plugin
} // end namespace adios2

extern "C" {

adios2::plugin::CompressMGARDMeshToGridOperator *OperatorCreate(const adios2::Params &parameters);
void OperatorDestroy(adios2::plugin::CompressMGARDMeshToGridOperator *obj);
}

#endif /* COMPRESSMGARDMESHTOGRIDOPERATOR_H_ */
