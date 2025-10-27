#define CP_DEBUG_VISIT 0
#define DEBUG_USE 0
#define VERBOSE 0
#define VERIFY 0 //验证 0 when test
#define DETAIL 0 //详细 0 when test
#define CPSZ_BASELINE 0 //baseline results for cpsz
#define STREAMING 0 //streaming mode
#define VISUALIZE 0 //visualization mode 0 when test
#define WRITEOUT 0 //write output 0 when test
#define STOP_AFTER_VERIFY 1 //stop after verification 1 when test

#ifdef _OPENMP
#include <omp.h>
#endif

#include <atomic>
#ifndef BITMAP_ATOMIC
#define BITMAP_ATOMIC 0
#endif

#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <utility>
#include <initializer_list>
#include <ftk/numeric/critical_point_type.hh>
#include <ftk/numeric/critical_point_test.hh>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <ftk/numeric/clamp.hh>
#include <ftk/numeric/linear_interpolation.hh>


#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include "sz_compress_cp_preserve_2d.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_cp_preserve_utils.hpp"
#include "sz_def.hpp"
#include "sz_compression_utils.hpp"
#include "sz_lossless.hpp"
#include "utils.hpp"
#include "sz_decompression_utils.hpp"
#include "sz_prediction.hpp"
#include "sz_cp_preserve_sos_2p5d_common.hpp"
#include "sz_compress_cp_preserve_sos_2p5d.hpp"
#include "sz_decompress_cp_preserve_sos_2p5d.hpp"



// main.cpp
// #include "mesh_index_stream.hpp"
#include <iostream>
#include <vector>
#include <array>
#include <unordered_set>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <iomanip>
#include <algorithm>

int main(int argc, char** argv) {
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif
    // args: U_file V_file H W T [mode] [eb] [threads]
    // example : ./test ~/data/CBA_full/u.bin ~/data/CBA_full/v.bin 450 150 2001 abs 0.005
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " U_file V_file H W T [mode=abs/rel] [eb=0.01] [threads]\n";
        return 1;
    }
    std::string u_path = argv[1];
    std::string v_path = argv[2];
    size_t dim_W = std::stoul(argv[3]); //fastest
    size_t dim_H = std::stoul(argv[4]); //medium
    size_t dim_T = std::stoul(argv[5]);  //slowest
    EbMode mode = EbMode::Absolute;
    if (argc >= 7) {
        std::string mode_str = argv[6];
        if (mode_str == "abs") mode = EbMode::Absolute;
        else if (mode_str == "rel") mode = EbMode::Relative;
        else {
            std::cerr << "Invalid mode: " << mode_str << ". Use 'abs' or 'rel'.\n";
            return 1;
        }
    }
    double error_bound = 0.01; // default
    if (argc >= 8) {
        error_bound = std::stod(argv[7]);
        if (error_bound <= 0) {
            std::cerr << "Error bound must be positive.\n";
            return 1;
        }
    }

#ifdef _OPENMP
    if (argc >= 9) {
        int requested_threads = std::stoi(argv[8]);
        if (requested_threads <= 0) {
            std::cerr << "Thread count must be positive.\n";
            return 1;
        }
        num_threads = requested_threads;
    }
    omp_set_num_threads(num_threads);
    printf("Using OpenMP with %d threads\n", num_threads);
#else
    if (argc >= 9) {
        std::cerr << "Warning: thread count argument ignored because OpenMP is not enabled.\n";
    }
    printf("OpenMP is not enabled.\n");
#endif
    std::string method;
    if (argc >= 10) {
        method = std::string(argv[9]);
    }
    else{
        method = "default"; //default
    }
    std::cout << "U_file: " << u_path << ", V_file: " << v_path << "\n";
    std::cout << "Dimensions: H=" << dim_H << ", W=" << dim_W << ", T=" << dim_T << "\n";
    std::cout << "Error mode: " << (mode == EbMode::Absolute ? "Absolute" : "Relative") << ", Error bound: " << error_bound << "\n";
    const size_t total_elements = dim_H * dim_W * dim_T;
    size_t num_elements = total_elements;
    float* U_ptr = nullptr;
    float* V_ptr = nullptr;

#if STREAMING
    const size_t layer_elems = dim_H * dim_W;
    const std::streamsize layer_bytes =
        static_cast<std::streamsize>(layer_elems) * static_cast<std::streamsize>(sizeof(float));
    std::ifstream fin_u(u_path, std::ios::binary);
    std::ifstream fin_v(v_path, std::ios::binary);
    if (!fin_u.is_open() || !fin_v.is_open()) {
        std::cerr << "[ERR] failed to open input files for streaming mode." << std::endl;
        return 1;
    }
    auto loader = [&](size_t t, float* slabU, float* slabV) -> bool {
        const std::streamoff offset =
            static_cast<std::streamoff>(t) * static_cast<std::streamoff>(layer_bytes);
        fin_u.clear();
        fin_v.clear();
        fin_u.seekg(offset);
        fin_v.seekg(offset);
        if (!fin_u.good() || !fin_v.good()) return false;
        fin_u.read(reinterpret_cast<char*>(slabU), layer_bytes);
        fin_v.read(reinterpret_cast<char*>(slabV), layer_bytes);
        return fin_u.gcount() == layer_bytes && fin_v.gcount() == layer_bytes;
    };
#else
    num_elements = 0;
    U_ptr = readfile<float>(u_path.c_str(), num_elements);
    V_ptr = readfile<float>(v_path.c_str(), num_elements);
#endif

    auto start_t = std::chrono::high_resolution_clock::now();
    size_t compressed_size = 0;

    //把dimension当作filename转换成string
    std::string dim_str = std::to_string(dim_W) + "x" + std::to_string(dim_H) + "x" + std::to_string(dim_T);

#if STREAMING
    auto* compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_streaming<float>(
        loader,
        dim_H, dim_W, dim_T,
        compressed_size, error_bound, mode);
#else
    unsigned char* compressed = nullptr;
    if (method == "default"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode);
    }
    else if (method == "parallel"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode);
    }
    else if (method == "parallel_v2"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v2...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v2(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode);
    }
    else if (method == "parallel_v3"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v3...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v3(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode);
    }
    else if (method == "warped_lorenzo"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_warped_lorenzo...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_warped_lorenzo(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode);
    }
    else if (method == "AR2_2D_LORENZO"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_AR2_2D_LORENZO...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_AR2_2D_LORENZO(
        U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
        compressed_size, error_bound, mode);
    }
    else if (method == "3DL_AR2"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_3DL_AR2...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_3DL_AR2(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode);
    }
    else if (method == "mop"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_mop...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_mop(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode);
    }
    else if (method == "mop_p"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_mop_parallel...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_mop_parallel(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode);
    }
    else if (method == "mixed_order"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_mixed_order...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_mixed_order(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode);
    }
    else if (method == "chanrot"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_chanrot...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_chanrot(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode);
    }
    else if( method == "simple"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_simple...\n");
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_simple(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode);
    }
    else if (method == "sl"){
        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_sl...\n");
        //临时写法
        double dt_dx, dt_dy;
        if (dim_W ==640 && dim_H == 80){
            dt_dx = 0.8;
            dt_dy = 0.8;
        }
        else if (dim_W == 150 && dim_H == 450){
            dt_dx = 0.01 / 0.66666666;
            dt_dy = 0.01 / 0.66666666;
        }
        else if (dim_W == 450 && dim_H == 200){
            dt_dx = (4.428 / 499) / (10.0 / 450);
            dt_dy = (4.428 / 499) / (4.0 / 200);
        }
        else if (dim_W == 512 && dim_H == 512){
            dt_dx = (10.0 / 1000.0) / ( 1.0 / 512.0);
            dt_dy = (10.0 / 1000.0) / ( 1.0 / 512.0);
        }
        else{
            std::cout << "Please provide dt/dx and dt/dy for this dataset!" << std::endl;
            exit(0);
        }
        compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_sl(
            U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
            compressed_size, error_bound, mode, dt_dx, dt_dy);
    }
    else{
        std::cerr << "Unknown method: " << method << "\n";
        return 1;
    }

#endif

#if STREAMING
    fin_u.close();
    fin_v.close();
#endif

    if (!compressed){
        std::cerr << "[ERR] compression returned null.\n";
        return 2;
    }
    // exit(0);
    unsigned char * result_after_lossless = NULL;
    size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, compressed, compressed_size, &result_after_lossless);
    //释放返回的缓冲区（由函数 malloc 分配）
    std::free(compressed);
    auto end_t = std::chrono::high_resolution_clock::now();
    cout << "compression time in sec:" << std::chrono::duration<double>(end_t - start_t).count() << endl;
    cout << "compressed size (after lossless): " << lossless_outsize << " bytes. "
         << "ratio = "
         << (2.0 * static_cast<double>(total_elements) * sizeof(float)) / double(lossless_outsize)
         << endl;
    
    //exit(0);
    //调用解压
    auto dec_start_t = std::chrono::high_resolution_clock::now();
    unsigned char * decompressed = NULL;
    size_t decompressed_size = sz_lossless_decompress(ZSTD_COMPRESSOR,result_after_lossless, lossless_outsize, &decompressed,compressed_size);
    //释放返回的缓冲区（由函数 malloc 分配）
    std::free(result_after_lossless);
    if (!decompressed){
        std::cerr << "[ERR] lossless decompression returned null.\n";
        return 3;
    }
    //调用sz_decompress_cp_preserve_sos_2p5d_online_fp
    float * U_dec = NULL;
    float * V_dec = NULL;
    int H = dim_H; //450;
    int W = dim_W; //150;
    int T = dim_T; //2001;
    printf("start decompression...\n");

    if(method == "default" || method == "parallel" || method == "parallel_v2" || method == "parallel_v3"){
        printf("sz_decompress_cp_preserve_sos_2p5d_fp...\n");
        sz_decompress_cp_preserve_sos_2p5d_fp<float>(decompressed, H, W, T, U_dec, V_dec);
    }
    else if (method == "warped_lorenzo"){
        printf("sz_decompress_cp_preserve_sos_2p5d_fp_warped_Lorenzo...\n");
        sz_decompress_cp_preserve_sos_2p5d_fp_warped_Lorenzo<float>(decompressed, H, W, T, U_dec, V_dec);
    }
    else if (method == "AR2_2D_LORENZO"){
        printf("sz_decompress_cp_preserve_sos_2p5d_fp_AR2_2D_Lorenzo...\n");
        sz_decompress_cp_preserve_sos_2p5d_fp_AR2_2D_Lorenzo<float>(decompressed, H, W, T, U_dec, V_dec);
    }
    else if (method == "3DL_AR2"){
        printf("sz_decompress_cp_preserve_sos_2p5d_fp_3DL_AR2...\n");
        sz_decompress_cp_preserve_sos_2p5d_fp_3DL_AR2<float>(decompressed, H, W, T, U_dec, V_dec);
    }
    else if (method == "chanrot"){
        printf("sz_decompress_cp_preserve_sos_2p5d_fp_chanrot...\n");
        sz_decompress_cp_preserve_sos_2p5d_fp_chanrot<float>(decompressed, H, W, T, U_dec, V_dec);
    }
    else if (method == "mop"){
        printf("sz_decompress_cp_preserve_sos_2p5d_fp_mop...\n");
        sz_decompress_cp_preserve_sos_2p5d_fp_mop<float>(decompressed, H, W, T, U_dec, V_dec);
    }
    // else if (method == "mixed_order"){
    //     printf("sz_decompress_cp_preserve_sos_2p5d_fp_mixed_order...\n");
    //     sz_decompress_cp_preserve_sos_2p5d_fp_mixed_order<float>(decompressed, H, W, T, U_dec, V_dec);
    // }
    // else if( method == "simple"){
    //     printf("sz_decompress_cp_preserve_sos_2p5d_fp_simple...\n");
    //     sz_decompress_cp_preserve_sos_2p5d_fp_simple<float>(decompressed, H, W, T, U_dec, V_dec);
    // }
    else if (method =="sl"){
        printf("sz_decompress_cp_preserve_sos_2p5d_fp_sl...\n");
        sz_decompress_cp_preserve_sos_2p5d_fp_sl<float>(decompressed, H, W, T, U_dec, V_dec);
    }
    else{
        std::cerr << "Unknown method: " << method << "\n";
        return 1;
    }


    // printf("sz_decompress_cp_preserve_sos_2p5d_fp_warped_Lorenzo...\n");
    // sz_decompress_cp_preserve_sos_2p5d_fp_warped_Lorenzo<float>(decompressed, H, W, T, U_dec, V_dec);

    // printf("sz_decompress_cp_preserve_sos_2p5d_fp_AR2_2D_Lorenzo...\n");
    // sz_decompress_cp_preserve_sos_2p5d_fp_AR2_2D_Lorenzo<float>(decompressed, H, W, T, U_dec, V_dec);

    auto dec_end_t = std::chrono::high_resolution_clock::now();
    cout << "decompression time in sec:" << std::chrono::duration<double>(dec_end_t - dec_start_t).count() << endl;


#if VERIFY
    printf("start verify....\n");
    size_t verify_elements = total_elements;
    float* U_ori = nullptr;
    float* V_ori = nullptr;
    bool free_original_buffers = false;
    #if STREAMING
    size_t file_elements_u = 0, file_elements_v = 0;
    U_ori = readfile<float>(u_path.c_str(), file_elements_u);
    V_ori = readfile<float>(v_path.c_str(), file_elements_v);
    if (!U_ori || !V_ori || file_elements_u != total_elements || file_elements_v != total_elements) {
        std::cerr << "[ERR] streaming verification failed to load original data." << std::endl;
        std::free(decompressed);
        std::free(U_dec);
        std::free(V_dec);
        if (U_ori) std::free(U_ori);
        if (V_ori) std::free(V_ori);
        return 4;
    }
    verify_elements = file_elements_u;
    free_original_buffers = true;
    #else
    verify_elements = 0;
    U_ori = readfile<float>(u_path.c_str(), verify_elements);
    V_ori = readfile<float>(v_path.c_str(), verify_elements);
    #endif

    verify(U_ori, V_ori, U_dec, V_dec, H, W, T);
    #if STOP_AFTER_VERIFY
    exit(0);
    #endif

    int64_t* U_ori_fp = (int64_t*)std::malloc(verify_elements * sizeof(int64_t));
    int64_t* V_ori_fp = (int64_t*)std::malloc(verify_elements * sizeof(int64_t));
    int64_t range_ori = 0;
    int64_t scale_ori = convert_to_fixed_point(U_ori, V_ori, verify_elements, U_ori_fp, V_ori_fp, range_ori);
    printf("original scale = %ld\n", scale_ori);

    int64_t* U_dec_fp = (int64_t*)std::malloc(verify_elements * sizeof(int64_t));
    int64_t* V_dec_fp = (int64_t*)std::malloc(verify_elements * sizeof(int64_t));
    convert_to_fixed_point_given_factor(U_dec, V_dec, verify_elements, U_dec_fp, V_dec_fp, scale_ori);

    auto cp_faces_dec = compute_cp_2p5d_faces_new(U_dec_fp, V_dec_fp, H, W, T);
    std::cout << "Total faces with CP dec: " << cp_faces_dec.size() << std::endl;

    auto cp_faces_ori = compute_cp_2p5d_faces_new(U_ori_fp, V_ori_fp, H, W, T);
    std::cout << "Total faces with CP ori: " << cp_faces_ori.size() << std::endl;

    // std::cout << (are_unordered_sets_equal(cp_faces_dec, cp_faces_ori) ? "CP face sets are equal." : "CP face sets differ!") << std::endl;
    bool cp_equal = are_unordered_sets_equal(cp_faces_dec, cp_faces_ori);
    std::cout << (cp_equal ? "CP face sets are equal." : "CP face sets differ!") << std::endl;
    if (!cp_equal) {
        print_cp_face_diffs(cp_faces_dec, cp_faces_ori, U_dec, V_dec, U_ori, V_ori,U_dec_fp, V_dec_fp, U_ori_fp, V_ori_fp, H, W, T);
    }

    #if VISUALIZE
        {
            // Export traced critical point trajectories to VTK PolyData (.vtp)
            const std::string out_dir = "/project/xli281_uksr/mxia/tmp_output/tracing_vtk";
            const std::string base_name_ori = std::to_string(dim_W) + "_" + std::to_string(dim_H) + "_" + std::to_string(dim_T) + ".vtp";
            const std::string base_name_dec = std::to_string(dim_W) + "_" + std::to_string(dim_H) + "_" + std::to_string(dim_T) + "_" + method + "_" + std::to_string(error_bound) + ".vtp";
            const std::string f_ori = out_dir + "/ori_" + base_name_ori;
            const std::string f_dec = out_dir + "/dec_" + base_name_dec;
            // best-effort ensure directory exists
            std::string mkdir_cmd = std::string("mkdir -p ") + out_dir;
            (void)std::system(mkdir_cmd.c_str());

            const Size3 sz_dims{H, W, T};
            bool ok1 = ftk_write_traced_critical_point_vtp(f_ori, U_ori, V_ori, sz_dims, cp_faces_ori);
            bool ok2 = ftk_write_traced_critical_point_vtp(f_dec, U_dec, V_dec, sz_dims, cp_faces_dec);
            std::cout << "[VISUALIZE] Wrote VTP ori: " << f_ori << " (" << (ok1?"ok":"fail") << ")\n";
            std::cout << "[VISUALIZE] Wrote VTP dec: " << f_dec << " (" << (ok2?"ok":"fail") << ")\n";
        }
    #endif


    #if DETAIL
    //   const std::string csv_name = std::to_string(H) + "_" + std::to_string(W) + "_" + std::to_string(T) + ".csv";
    //   summarize_cp_faces_per_layer_and_slab(cp_faces_dec, cp_faces_ori, H, W, T, csv_name);
    summarize_cp_faces_per_layer_and_slab(cp_faces_dec, cp_faces_ori, H, W, T, "");
    #endif



    if (free_original_buffers) {
        std::free(U_ori);
        std::free(V_ori);
    }
    std::free(U_ori_fp);
    std::free(V_ori_fp);
    std::free(U_dec_fp);
    std::free(V_dec_fp);
#endif

    #if CPSZ_BASELINE && !STREAMING
    float *U_dec_cpsz = nullptr;
    float *V_dec_cpsz = nullptr;
    int64_t *U_dec_cpsz_fp = nullptr;
    int64_t *V_dec_cpsz_fp = nullptr;
    int64_t *U_ori_cpsz_fp = nullptr;
    int64_t *V_ori_cpsz_fp = nullptr;

    const size_t frame_elems = dim_H * dim_W;
    const size_t total_elems = frame_elems * dim_T;
    int64_t global_scalar_cpsz = 0;
    int64_t global_range_cpsz = 0;

    if (total_elems != num_elements) {
        std::cerr << "[ERR] CPSZ baseline skipped: element count mismatch (dims="
                  << total_elems << ", file=" << num_elements << ")." << std::endl;
    } else {
        double cpsz_max_pwr_eb = error_bound;
        if (mode == EbMode::Relative) {
            // relative mode expects a ratio; keep the user-specified error_bound as-is
        } else if (mode != EbMode::Absolute) {
            std::cerr << "[ERR] Unsupported error mode for CPSZ baseline." << std::endl;
        }

        size_t cpsz_compressed_size = 0;
        const auto cpsz_2d_comp_time_start = std::chrono::high_resolution_clock::now();
        unsigned char *cpsz_compressed = sz_compress_cp_preserve_sos_2d_time_online_fp(
            U_ptr,
            V_ptr,
            dim_T,
            dim_H,
            dim_W,
            cpsz_compressed_size,
            /*transpose=*/false,
            cpsz_max_pwr_eb,
            mode);
        const auto cpsz_2d_comp_time_end = std::chrono::high_resolution_clock::now();
        std::cout << "CPSZ 2D-time compression time in sec: "
                  << std::chrono::duration<double>(cpsz_2d_comp_time_end - cpsz_2d_comp_time_start).count() << std::endl;
        std::cout << "CPSZ compressed size: " << cpsz_compressed_size
                  << " bytes. ratio = " << (2 * total_elems * sizeof(float)) / double(cpsz_compressed_size) << std::endl;

        bool parsed_global_params = false;
        if (cpsz_compressed && cpsz_compressed_size >= sizeof(size_t) * 4 + sizeof(int64_t) * 2) {
            size_t cursor = 0;
            auto read_size_t = [&](size_t &value) -> bool {
                if (cursor + sizeof(size_t) > cpsz_compressed_size) return false;
                std::memcpy(&value, cpsz_compressed + cursor, sizeof(size_t));
                cursor += sizeof(size_t);
                return true;
            };
            auto read_int64 = [&](int64_t &value) -> bool {
                if (cursor + sizeof(int64_t) > cpsz_compressed_size) return false;
                std::memcpy(&value, cpsz_compressed + cursor, sizeof(int64_t));
                cursor += sizeof(int64_t);
                return true;
            };

            size_t header_time = 0;
            size_t header_r2 = 0;
            size_t header_r3 = 0;
            parsed_global_params = read_size_t(header_time) && read_size_t(header_r2) && read_size_t(header_r3);
            if (parsed_global_params && header_time > 0) {
                size_t first_frame_size = 0;
                parsed_global_params = read_size_t(first_frame_size) && read_int64(global_scalar_cpsz) && read_int64(global_range_cpsz);
            } else {
                parsed_global_params = false;
            }
        }

        if (!cpsz_compressed) {
            std::cerr << "[ERR] CPSZ compression failed." << std::endl;
        } else {
            size_t dec_time_dim = 0;
            size_t dec_H = 0;
            size_t dec_W = 0;
            const auto cpsz_2d_decomp_time_start = std::chrono::high_resolution_clock::now();
            sz_decompress_cp_preserve_2d_time_online_fp<float>(
                cpsz_compressed,
                dec_time_dim,
                dec_H,
                dec_W,
                U_dec_cpsz,
                V_dec_cpsz);
            const auto cpsz_2d_decomp_time_end = std::chrono::high_resolution_clock::now();
            std::cout << "CPSZ 2D-time decompression time in sec: "
                      << std::chrono::duration<double>(cpsz_2d_decomp_time_end - cpsz_2d_decomp_time_start).count() << std::endl;
            std::free(cpsz_compressed);

            if (!U_dec_cpsz || !V_dec_cpsz) {
                std::cerr << "[ERR] CPSZ decompression returned null buffers." << std::endl;
            } else if (dec_time_dim != dim_T || dec_H != dim_H || dec_W != dim_W) {
                std::cerr << "[ERR] CPSZ decompression dimension mismatch: "
                          << "time=" << dec_time_dim << " (expected " << dim_T << ") "
                          << "H=" << dec_H << " (expected " << dim_H << ") "
                          << "W=" << dec_W << " (expected " << dim_W << ")." << std::endl;
            } else {
                if ((!parsed_global_params || global_scalar_cpsz <= 0) && total_elems > 0) {
                    std::vector<int64_t> tmp_u(total_elems);
                    std::vector<int64_t> tmp_v(total_elems);
                    int64_t tmp_range = 0;
                    global_scalar_cpsz = convert_to_fixed_point<float, int64_t>(
                        U_ptr,
                        V_ptr,
                        total_elems,
                        tmp_u.data(),
                        tmp_v.data(),
                        tmp_range);
                    global_range_cpsz = tmp_range;
                    parsed_global_params = (global_scalar_cpsz > 0);
                }

                if (global_scalar_cpsz <= 0) {
                    global_scalar_cpsz = 1;
                }

                U_dec_cpsz_fp = static_cast<int64_t *>(std::malloc(total_elems * sizeof(int64_t)));
                V_dec_cpsz_fp = static_cast<int64_t *>(std::malloc(total_elems * sizeof(int64_t)));
                U_ori_cpsz_fp = static_cast<int64_t *>(std::malloc(total_elems * sizeof(int64_t)));
                V_ori_cpsz_fp = static_cast<int64_t *>(std::malloc(total_elems * sizeof(int64_t)));
                if (!U_dec_cpsz_fp || !V_dec_cpsz_fp || !U_ori_cpsz_fp || !V_ori_cpsz_fp) {
                    std::cerr << "[ERR] CPSZ fixed-point buffer allocation failed." << std::endl;
                    std::free(U_dec_cpsz_fp); U_dec_cpsz_fp = nullptr;
                    std::free(V_dec_cpsz_fp); V_dec_cpsz_fp = nullptr;
                    std::free(U_ori_cpsz_fp); U_ori_cpsz_fp = nullptr;
                    std::free(V_ori_cpsz_fp); V_ori_cpsz_fp = nullptr;
                } else {
                    convert_to_fixed_point_given_factor(
                        U_dec_cpsz,
                        V_dec_cpsz,
                        total_elems,
                        U_dec_cpsz_fp,
                        V_dec_cpsz_fp,
                        global_scalar_cpsz);
                    convert_to_fixed_point_given_factor(
                        U_ptr,
                        V_ptr,
                        total_elems,
                        U_ori_cpsz_fp,
                        V_ori_cpsz_fp,
                        global_scalar_cpsz);
                }
            }
        }
    }

    if (U_dec_cpsz_fp && V_dec_cpsz_fp && U_ori_cpsz_fp && V_ori_cpsz_fp) {
        auto cp_faces_cpsz = compute_cp_2p5d_faces_new(U_dec_cpsz_fp, V_dec_cpsz_fp, H, W, T);
        std::cout << "Total faces with CP dec CPSZ baseline: " << cp_faces_cpsz.size() << std::endl;

        auto cp_faces_ori_cpsz = compute_cp_2p5d_faces_new(U_ori_cpsz_fp, V_ori_cpsz_fp, H, W, T);
        // const std::string csv_name_cpsz =
        //     std::to_string(H) + "_" + std::to_string(W) + "_" + std::to_string(T) + "_" + std::to_string(error_bound) + "_cpsz.csv";
        // summarize_cp_faces_per_layer_and_slab(cp_faces_cpsz, cp_faces_ori_cpsz, H, W, T, csv_name_cpsz);
        // std::cout << "CPSZ baseline summary written to " << csv_name_cpsz << std::endl;
        summarize_cp_faces_per_layer_and_slab(cp_faces_cpsz, cp_faces_ori_cpsz, H, W, T, "");
    }

    if (U_dec_cpsz && V_dec_cpsz) {
        printf("start verify CPSZ baseline....\n");
        verify(U_ptr, V_ptr, U_dec_cpsz, V_dec_cpsz, H, W, T);
    }
        #if VISUALIZE
        {
            // Export traced critical point trajectories to VTK PolyData (.vtp)
            const std::string out_dir = "/project/xli281_uksr/mxia/tmp_output/tracing_vtk";
            const std::string base_name_cpsz = std::to_string(dim_W) + "_" + std::to_string(dim_H) + "_" + std::to_string(dim_T) + "_" + std::to_string(error_bound) + "_cpsz.vtp";
            const std::string f_cpsz = out_dir + "/dec_cpsz_" + base_name_cpsz;
            // best-effort ensure directory exists
            std::string mkdir_cmd = std::string("mkdir -p ") + out_dir;
            (void)std::system(mkdir_cmd.c_str());

            const Size3 sz_dims{H, W, T};
            bool ok = ftk_write_traced_critical_point_vtp(f_cpsz, U_dec_cpsz, V_dec_cpsz, sz_dims, cp_faces_ori);
            std::cout << "[VISUALIZE] Wrote VTP dec CPSZ: " << f_cpsz << " (" << (ok?"ok":"fail") << ")\n";
        }
        #endif
    #endif

    #if WRITEOUT
    const std::string out_dir = "/project/xli281_uksr/mxia/tmp_output/tracing_vtk";
    std::string out_U = out_dir + "/U_dec_" + dim_str + "_" + method + "_" + std::to_string(error_bound) + ".bin";
    std::string out_V = out_dir + "/V_dec_" + dim_str + "_" + method + "_" + std::to_string(error_bound) + ".bin";
    writefile<float>(out_U.c_str(), U_dec, total_elements);
    writefile<float>(out_V.c_str(), V_dec, total_elements);
    std::cout << "Decompressed data written to " << out_U << " and " << out_V << std::endl;
        #if CPSZ_BASELINE
        if (U_dec_cpsz && V_dec_cpsz) {
            std::string out_U_cpsz = "U_dec_cpsz_" + dim_str + "_" + std::to_string(error_bound) + ".bin";
            std::string out_V_cpsz = "V_dec_cpsz_" + dim_str + "_" + std::to_string(error_bound) + ".bin";
            writefile<float>(U_dec_cpsz, total_elements, out_U_cpsz.c_str());
            writefile<float>(V_dec_cpsz, total_elements, out_V_cpsz.c_str());
            std::cout << "CPSZ baseline decompressed data written to " << out_U_cpsz << " and " << out_V_cpsz << std::endl;
        }
        #endif
    #endif
    
    //释放返回的缓冲区（由函数 malloc 分配）
    #if CPSZ_BASELINE
    std::free(U_dec_cpsz);
    std::free(V_dec_cpsz);
    std::free(U_dec_cpsz_fp);
    std::free(V_dec_cpsz_fp);
    std::free(U_ori_cpsz_fp);
    std::free(V_ori_cpsz_fp);
    #endif
    std::free(decompressed);
    std::free(U_ptr);
    std::free(V_ptr);
    std::free(U_dec);
    std::free(V_dec);

    return 0;
}
