#include "sz_decompress_cp_preserve_sos_2p5d.hpp"

template bool sz_decompress_cp_preserve_sos_2p5d_fp<float>(
    const unsigned char*, size_t, size_t, size_t, float*&, float*&);

template bool sz_decompress_cp_preserve_sos_2p5d_fp_warped_Lorenzo<float>(
    const unsigned char*, size_t, size_t, size_t, float*&, float*&);

template bool sz_decompress_cp_preserve_sos_2p5d_fp_AR2_2D_Lorenzo<float>(
    const unsigned char*, size_t, size_t, size_t, float*&, float*&);

template bool sz_decompress_cp_preserve_sos_2p5d_fp_3DL_AR2<float>(
    const unsigned char*, size_t, size_t, size_t, float*&, float*&);

template bool sz_decompress_cp_preserve_sos_2p5d_fp_chanrot<float>(
    const unsigned char*, size_t, size_t, size_t, float*&, float*&);

template bool sz_decompress_cp_preserve_sos_2p5d_fp_mop<float>(
    const unsigned char*, size_t, size_t, size_t, float*&, float*&);