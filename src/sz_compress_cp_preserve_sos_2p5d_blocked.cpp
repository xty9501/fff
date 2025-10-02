#include "sz_compress_cp_preserve_sos_2p5d_blocked.hpp"

#include <cstdint>
#include <functional>

namespace cpsz {

// Explicit instantiations so that the linker will emit the symbols even when the
// translation units including the templated definitions are not directly referenced.

template unsigned char* omp_sz_compress_cp_preserve_sos_2p5d_streaming_blocked<float,
    std::function<bool(size_t, float*, float*)>,
    std::function<bool(size_t, int64_t*)>>(
    std::function<bool(size_t, float*, float*)>&&,
    std::function<bool(size_t, int64_t*)>&&,
    size_t, size_t, size_t, size_t&, double, EbMode, int);

template unsigned char* omp_sz_compress_cp_preserve_sos_2p5d_streaming_blocked<double,
    std::function<bool(size_t, double*, double*)>,
    std::function<bool(size_t, int64_t*)>>(
    std::function<bool(size_t, double*, double*)>&&,
    std::function<bool(size_t, int64_t*)>&&,
    size_t, size_t, size_t, size_t&, double, EbMode, int);

template unsigned char* omp_sz_compress_cp_preserve_sos_2p5d_streaming_blocked<float>(
    const float*, const float*, const int64_t*, size_t, size_t, size_t,
    size_t&, double, EbMode, int);

template unsigned char* omp_sz_compress_cp_preserve_sos_2p5d_streaming_blocked<double>(
    const double*, const double*, const int64_t*, size_t, size_t, size_t,
    size_t&, double, EbMode, int);

template bool omp_sz_decompress_cp_preserve_sos_2p5d_streaming_blocked<float>(
    const unsigned char*, size_t, size_t, size_t, float*&, float*&);

template bool omp_sz_decompress_cp_preserve_sos_2p5d_streaming_blocked<double>(
    const unsigned char*, size_t, size_t, size_t, double*&, double*&);

} // namespace cpsz
