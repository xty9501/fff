#pragma once

#include <cstddef>
#include <cstdint>

#include "sz_def.hpp"

namespace cpsz {

template<typename T_data, typename LayerFetcher, typename EbFetcher>
unsigned char* omp_sz_compress_cp_preserve_sos_2p5d_streaming_blocked(
    LayerFetcher&& fetch_layer,
    EbFetcher&& fetch_required_eb,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode,
    int num_threads);

template<typename T_data>
unsigned char* omp_sz_compress_cp_preserve_sos_2p5d_streaming_blocked(
    const T_data* U,
    const T_data* V,
    const int64_t* required_eb,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode,
    int num_threads);

template<typename T_data>
bool omp_sz_decompress_cp_preserve_sos_2p5d_streaming_blocked(
    const unsigned char* compressed,
    size_t r1, size_t r2, size_t r3,
    T_data*& U,
    T_data*& V);

} // namespace cpsz

#include "sz_compress_cp_preserve_sos_2p5d_blocked.tpp"
