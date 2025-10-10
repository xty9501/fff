#include "sz_compress_cp_preserve_sos_2p5d.hpp"

template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);

template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);

template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v2<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);

template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v3<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);

template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_warped_lorenzo<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);

template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_AR2_2D_LORENZO<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);

template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_3DL_AR2<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);

template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_streaming<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);

template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_mop<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);

template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_mixed_order<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);
template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_chanrot<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);

template unsigned char* sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_simple<float>(
    const float*, const float*, size_t, size_t, size_t, size_t&, double, EbMode);
