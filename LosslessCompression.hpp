/**
 * @file LosslessCompression.hpp
 * @brief Lossless compression for residual data using Quantization + Huffman + ZSTD.
 * 
 * This module provides an alternative to MGARD for residual data compression.
 * For residuals without clear spatial correlations, using direct lossless 
 * compression (quantization + Huffman + ZSTD) can be more efficient than
 * MGARD's wavelet-like decomposition.
 * 
 * Created: February 4, 2026
 * Based on: /ccs/proj/cfd164/gongq/refactorMesh/p_multilevel/src/lossless_compression.cpp
 */

#ifndef LOSSLESS_COMPRESSION_HPP
#define LOSSLESS_COMPRESSION_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

// MGARD headers for Huffman compression
#include <mgard/compressors.hpp>
#include <mgard/utilities.hpp>

// Check for ZSTD support
#ifdef MGARD_ZSTD
#include <zstd.h>
#define HAVE_ZSTD 1
#else
// Try to include ZSTD directly
#if __has_include(<zstd.h>)
#include <zstd.h>
#define HAVE_ZSTD 1
#endif
#endif

namespace lossless {

//=============================================================================
// Compression Method Selection
//=============================================================================

enum class CompressionMethod {
    MGARD,           // Full MGARD with multilevel decomposition
    Huffman_ZSTD,    // Quantization + Huffman + ZSTD (no multilevel decomposition)
    ZSTD_Only        // Quantization + ZSTD (no Huffman)
};

//=============================================================================
// Linear Quantizer
//=============================================================================

/**
 * @brief Uniform linear quantizer.
 * Maps floating-point values to integers: q = round(x / quantum)
 */
class LinearQuantizer {
public:
    explicit LinearQuantizer(double quantum) 
        : quantum_(quantum), inv_quantum_(1.0 / quantum) {}
    
    static LinearQuantizer FromTolerance(double tolerance) {
        // quantum = 2 * tolerance ensures max error <= tolerance after dequantization
        return LinearQuantizer(2.0 * tolerance);
    }
    
    int64_t Quantize(double x) const {
        return static_cast<int64_t>(std::round(x * inv_quantum_));
    }
    
    double Dequantize(int64_t q) const {
        return static_cast<double>(q) * quantum_;
    }
    
    template<typename T>
    int64_t QuantizeArray(const T* input, int64_t* output, std::size_t n) const {
        int64_t max_abs = 0;
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = Quantize(static_cast<double>(input[i]));
            int64_t abs_val = std::abs(output[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        return max_abs;
    }
    
    template<typename T>
    void DequantizeArray(const int64_t* input, T* output, std::size_t n) const {
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = static_cast<T>(Dequantize(input[i]));
        }
    }
    
    double GetQuantum() const { return quantum_; }

private:
    double quantum_;
    double inv_quantum_;
};

//=============================================================================
// Compressed Data Header
//=============================================================================

/**
 * @brief Header for lossless compressed data.
 * Stored at the beginning of compressed buffer for proper decompression.
 */
struct CompressedHeader {
    uint32_t magic = 0x51485A53;  // "QHZS" (Quant-Huffman-Zstd)
    uint8_t version = 1;
    uint8_t method;               // CompressionMethod enum
    uint8_t element_size;         // sizeof(float) or sizeof(double)
    uint8_t reserved = 0;
    uint64_t num_elements;        // Number of original elements
    double quantum;               // Quantization step for dequantization
    uint64_t compressed_size;     // Size of compressed payload (excluding header)
    
    static constexpr std::size_t Size() { return 40; }
    
    void Serialize(char* buffer) const {
        char* ptr = buffer;
        std::memcpy(ptr, &magic, sizeof(magic)); ptr += sizeof(magic);
        *ptr++ = version;
        *ptr++ = method;
        *ptr++ = element_size;
        *ptr++ = reserved;
        std::memcpy(ptr, &num_elements, sizeof(num_elements)); ptr += sizeof(num_elements);
        std::memcpy(ptr, &quantum, sizeof(quantum)); ptr += sizeof(quantum);
        std::memcpy(ptr, &compressed_size, sizeof(compressed_size));
    }
    
    static bool Deserialize(const char* buffer, std::size_t size, CompressedHeader& header) {
        if (size < Size()) return false;
        
        const char* ptr = buffer;
        std::memcpy(&header.magic, ptr, sizeof(header.magic)); ptr += sizeof(header.magic);
        
        if (header.magic != 0x51485A53) return false;
        
        header.version = static_cast<uint8_t>(*ptr++);
        header.method = static_cast<uint8_t>(*ptr++);
        header.element_size = static_cast<uint8_t>(*ptr++);
        header.reserved = static_cast<uint8_t>(*ptr++);
        std::memcpy(&header.num_elements, ptr, sizeof(header.num_elements)); ptr += sizeof(header.num_elements);
        std::memcpy(&header.quantum, ptr, sizeof(header.quantum)); ptr += sizeof(header.quantum);
        std::memcpy(&header.compressed_size, ptr, sizeof(header.compressed_size));
        
        return true;
    }
};

//=============================================================================
// Lossless Compression Functions
//=============================================================================

/**
 * @brief Compress residual data using Quantization + MGARD Huffman + ZSTD.
 * 
 * @tparam T float or double
 * @param data Input residual data
 * @param n Number of elements
 * @param tolerance Absolute error tolerance
 * @param compressed Output compressed buffer
 * @param compressedSize Output: size of compressed data
 * @return true if successful
 */
template<typename T>
bool CompressHuffmanZstd(const T* data, std::size_t n, double tolerance,
                         void* compressed, std::size_t& compressedSize) {
    if (n == 0) {
        compressedSize = 0;
        return false;
    }
    
    // Step 1: Quantize data to int64_t
    LinearQuantizer quantizer = LinearQuantizer::FromTolerance(tolerance);
    
    std::vector<int64_t> quantized(n);
    for (std::size_t i = 0; i < n; ++i) {
        quantized[i] = quantizer.Quantize(static_cast<double>(data[i]));
    }
    
    // Step 2: Compress using MGARD's Huffman + ZSTD
    // MGARD's compress_memory_huffman expects long int*
    // Optimization: avoid copy if sizeof(int64_t) == sizeof(long int) (true on 64-bit systems)
    long int* huffman_input;
    std::vector<long int> quantized_long;
    
    if constexpr (sizeof(int64_t) == sizeof(long int)) {
        // Zero-copy: reinterpret int64_t* as long int*
        huffman_input = reinterpret_cast<long int*>(quantized.data());
    } else {
        // Fallback: element-by-element conversion (rare: only on systems where long != 64-bit)
        quantized_long.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            quantized_long[i] = static_cast<long int>(quantized[i]);
        }
        huffman_input = quantized_long.data();
    }
    
    mgard::MemoryBuffer<unsigned char> huffman_result = 
        mgard::compress_memory_huffman(huffman_input, n);
    
    if (huffman_result.size == 0) {
        std::cerr << "[LosslessCompression] Huffman compression failed\n";
        return false;
    }
    
    // Step 3: Create header and copy compressed data
    CompressedHeader header;
    header.method = static_cast<uint8_t>(CompressionMethod::Huffman_ZSTD);
    header.element_size = sizeof(T);
    header.num_elements = n;
    header.quantum = quantizer.GetQuantum();
    header.compressed_size = huffman_result.size;
    
    // Write header + compressed data
    char* out = static_cast<char*>(compressed);
    header.Serialize(out);
    std::memcpy(out + CompressedHeader::Size(), huffman_result.data.get(), huffman_result.size);
    
    compressedSize = CompressedHeader::Size() + huffman_result.size;
    
    return true;
}

/**
 * @brief Compress residual data using Quantization + ZSTD only (no Huffman).
 */
template<typename T>
bool CompressZstdOnly(const T* data, std::size_t n, double tolerance,
                      void* compressed, std::size_t& compressedSize) {
#ifndef HAVE_ZSTD
    std::cerr << "[LosslessCompression] ZSTD not available\n";
    return false;
#else
    if (n == 0) {
        compressedSize = 0;
        return false;
    }
    
    // Step 1: Quantize data to int64_t
    LinearQuantizer quantizer = LinearQuantizer::FromTolerance(tolerance);
    
    std::vector<int64_t> quantized(n);
    for (std::size_t i = 0; i < n; ++i) {
        quantized[i] = quantizer.Quantize(static_cast<double>(data[i]));
    }
    
    // Step 2: ZSTD compress the quantized data
    std::size_t src_size = n * sizeof(int64_t);
    std::size_t max_dst_size = ZSTD_compressBound(src_size);
    
    char* out = static_cast<char*>(compressed);
    
    std::size_t zstd_size = ZSTD_compress(
        out + CompressedHeader::Size(), max_dst_size,
        quantized.data(), src_size,
        3);  // compression level 3
    
    if (ZSTD_isError(zstd_size)) {
        std::cerr << "[LosslessCompression] ZSTD compression failed: " 
                  << ZSTD_getErrorName(zstd_size) << "\n";
        return false;
    }
    
    // Step 3: Write header
    CompressedHeader header;
    header.method = static_cast<uint8_t>(CompressionMethod::ZSTD_Only);
    header.element_size = sizeof(T);
    header.num_elements = n;
    header.quantum = quantizer.GetQuantum();
    header.compressed_size = zstd_size;
    
    header.Serialize(out);
    
    compressedSize = CompressedHeader::Size() + zstd_size;
    
    return true;
#endif
}

/**
 * @brief Decompress data that was compressed with CompressHuffmanZstd.
 */
template<typename T>
bool DecompressHuffmanZstd(const void* compressed, std::size_t compressedSize,
                           T* data, std::size_t n) {
    const char* in = static_cast<const char*>(compressed);
    
    // Parse header
    CompressedHeader header;
    if (!CompressedHeader::Deserialize(in, compressedSize, header)) {
        std::cerr << "[LosslessCompression] Invalid header\n";
        return false;
    }
    
    if (header.num_elements != n) {
        std::cerr << "[LosslessCompression] Element count mismatch: expected " 
                  << n << ", got " << header.num_elements << "\n";
        return false;
    }
    
    CompressionMethod method = static_cast<CompressionMethod>(header.method);
    
    if (method == CompressionMethod::Huffman_ZSTD) {
        // Decompress using MGARD's Huffman
        // MGARD outputs to long int*, we need int64_t for dequantization
        // Optimization: avoid copy if sizeof(int64_t) == sizeof(long int) (true on 64-bit systems)
        
        std::vector<int64_t> quantized(n);
        long int* huffman_output;
        std::vector<long int> quantized_long;
        
        if constexpr (sizeof(int64_t) == sizeof(long int)) {
            // Zero-copy: decompress directly into int64_t buffer
            huffman_output = reinterpret_cast<long int*>(quantized.data());
        } else {
            // Fallback: use separate buffer
            quantized_long.resize(n);
            huffman_output = quantized_long.data();
        }
        
        std::size_t dst_len = n * sizeof(long int);
        
        mgard::decompress_memory_huffman(
            const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(in + CompressedHeader::Size())),
            header.compressed_size,
            huffman_output,
            dst_len);
        
        // Copy from long int to int64_t if needed (only on non-64-bit systems)
        if constexpr (sizeof(int64_t) != sizeof(long int)) {
            for (std::size_t i = 0; i < n; ++i) {
                quantized[i] = static_cast<int64_t>(quantized_long[i]);
            }
        }
        
        // Dequantize
        LinearQuantizer quantizer(header.quantum);
        for (std::size_t i = 0; i < n; ++i) {
            data[i] = static_cast<T>(quantizer.Dequantize(quantized[i]));
        }
        
        return true;
    }
    else if (method == CompressionMethod::ZSTD_Only) {
#ifndef HAVE_ZSTD
        std::cerr << "[LosslessCompression] ZSTD not available for decompression\n";
        return false;
#else
        // ZSTD decompress
        std::vector<int64_t> quantized(n);
        std::size_t dst_size = n * sizeof(int64_t);
        
        std::size_t result = ZSTD_decompress(
            quantized.data(), dst_size,
            in + CompressedHeader::Size(), header.compressed_size);
        
        if (ZSTD_isError(result) || result != dst_size) {
            std::cerr << "[LosslessCompression] ZSTD decompression failed\n";
            return false;
        }
        
        // Dequantize
        LinearQuantizer quantizer(header.quantum);
        for (std::size_t i = 0; i < n; ++i) {
            data[i] = static_cast<T>(quantizer.Dequantize(quantized[i]));
        }
        
        return true;
#endif
    }
    
    std::cerr << "[LosslessCompression] Unknown compression method: " << (int)header.method << "\n";
    return false;
}

/**
 * @brief Get maximum compressed size for buffer allocation.
 */
inline std::size_t GetMaxCompressedSize(std::size_t n, std::size_t element_size) {
    // Conservative estimate: header + (quantized data with some overhead)
    // Huffman/ZSTD typically reduces size, but worst case is expansion
    return CompressedHeader::Size() + (n * sizeof(int64_t)) + 1024;
}

/**
 * @brief Helper to determine if lossless compression should be used for residuals.
 * 
 * Analyzes data characteristics to suggest optimal compression method.
 * Returns true if Huffman+ZSTD is likely better than MGARD for this data.
 */
template<typename T>
bool ShouldUseLosslessForResidual(const T* data, std::size_t n, double tolerance) {
    if (n < 1000) return false;  // Too small to analyze
    
    // Sample the data to estimate correlation
    const std::size_t sample_size = std::min(n, std::size_t(10000));
    const std::size_t step = n / sample_size;
    
    double sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0;
    std::size_t count = 0;
    
    for (std::size_t i = 0; i + step < n; i += step) {
        double x = static_cast<double>(data[i]);
        double y = static_cast<double>(data[i + step]);
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        count++;
    }
    
    if (count < 2) return false;
    
    // Compute Pearson correlation coefficient
    double n_d = static_cast<double>(count);
    double denom = std::sqrt((n_d * sum_x2 - sum_x * sum_x) * (n_d * sum_y2 - sum_y * sum_y));
    
    if (denom < 1e-15) return true;  // No variation = no correlation
    
    double correlation = (n_d * sum_xy - sum_x * sum_y) / denom;
    
    // Low correlation (|r| < 0.3) suggests lossless compression is better
    return std::abs(correlation) < 0.3;
}

} // namespace lossless

#endif // LOSSLESS_COMPRESSION_HPP
