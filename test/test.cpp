// mesh_index_stream.hpp
#include <atomic>
#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <ftk/numeric/critical_point_type.hh>
#include <ftk/numeric/critical_point_test.hh>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <ftk/numeric/clamp.hh>
#include <ftk/numeric/linear_interpolation.hh>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef BITMAP_ATOMIC
#define BITMAP_ATOMIC 0
#endif

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include "sz_compress_cp_preserve_2d.hpp"
#include "sz_cp_preserve_utils.hpp"
#include "sz_def.hpp"
#include "sz_compression_utils.hpp"
#include "sz_lossless.hpp"
#include "utils.hpp"
#include "sz_decompression_utils.hpp"




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


template<typename T, typename T_fp>
static int64_t 
convert_to_fixed_point(const T * U, const T * V, size_t num_elements, T_fp * U_fp, T_fp * V_fp, T_fp& range, int type_bits=63){
	// find the max value in all U and V
	double vector_field_resolution = 0;
	int64_t vector_field_scaling_factor = 1;
	for (int i=0; i<num_elements; i++){
		double min_val = std::max(fabs(U[i]), fabs(V[i]));
		vector_field_resolution = std::max(vector_field_resolution, min_val);
	}
	// take log2 and ceil
	int vbits = std::ceil(std::log2(vector_field_resolution));
	int nbits = (type_bits - 3) / 2;
	// int nbits = 30;
	//vector_field_scaling_factor = 2^(30-vbits) 
	vector_field_scaling_factor = 1 << (nbits - vbits);
	std::cerr << "resolution=" << vector_field_resolution 
	<< ", factor=" << vector_field_scaling_factor 
	<< ", nbits=" << nbits << ", vbits=" << vbits << ", shift_bits=" << nbits - vbits << std::endl;
	int64_t max = std::numeric_limits<int64_t>::min();
	int64_t min = std::numeric_limits<int64_t>::max();
	printf("max = %lld, min = %lld\n", max, min);
	for(int i=0; i<num_elements; i++){
		U_fp[i] = U[i] * vector_field_scaling_factor;
		V_fp[i] = V[i] * vector_field_scaling_factor;
		max = std::max(max, U_fp[i]);
		max = std::max(max, V_fp[i]);
		min = std::min(min, U_fp[i]);
		min = std::min(min, V_fp[i]);
	}
	printf("max = %lld, min = %lld\n", max, min);
	range = max - min;
	return vector_field_scaling_factor;
}

template<typename T, typename T_fp>
static void 
convert_to_floating_point(const T_fp * U_fp, const T_fp * V_fp, size_t num_elements, T * U, T * V, int64_t vector_field_scaling_factor){
	for(int i=0; i<num_elements; i++){
		U[i] = U_fp[i] * (T)1.0 / vector_field_scaling_factor;
		V[i] = V_fp[i] * (T)1.0 / vector_field_scaling_factor;
	}
}

// ================== 网格/索引基础 ==================
struct Size3 { int H, W, T; }; // H=450, W=150, T=2001
inline int vid(int t,int i,int j,const Size3& sz){
    return t*(sz.H*sz.W) + i*sz.W + j; // 时间最慢
}
enum class TriInCell : unsigned char { Upper=0, Lower=1 };

// cell(i,j) 内，两种2D三角（底/顶层都用相同拓扑）
inline std::array<int,3> tri_vertices_2d(int i,int j,TriInCell which,int t,const Size3& sz){
    const int v00=vid(t,i,  j,  sz), v01=vid(t,i,  j+1,sz),
              v10=vid(t,i+1,j,  sz), v11=vid(t,i+1,j+1,sz);
    return (which==TriInCell::Upper) ? std::array<int,3>{v00,v01,v11}
                                     : std::array<int,3>{v00,v10,v11};
}

// 指定的 3-tet 切分（用于“内部剖分面”）
struct Tet { int v[4]; };
inline std::array<Tet,3> prism_split_3tets(const std::array<int,3>& a,
                                           const std::array<int,3>& b){
    // T0={a0,a1,a2,b2}, T1={a0,a1,b1,b2}, T2={a0,b0,b1,b2}
    return {{
        Tet{ a[0], a[1], a[2], b[2] },
        Tet{ a[0], a[1], b[1], b[2] },
        Tet{ a[0], b[0], b[1], b[2] }
    }};
}

// ============ 辅助：对一个三角做一次“CP/eb”并回写到 eb_min ============
template<typename Tfp>
inline void consider_triangle_and_update_ebmin(
    int a,int b,int c,
    const Tfp* U_fp,const Tfp* V_fp,
    std::vector<Tfp>& eb_min, size_t &cp_count)
{
    // CP 检测（定点）
    int64_t vf[3][2] = {
        { (int64_t)U_fp[a], (int64_t)V_fp[a] },
        { (int64_t)U_fp[b], (int64_t)V_fp[b] },
        { (int64_t)U_fp[c], (int64_t)V_fp[c] }
    };
    int idxs[3] = {a,b,c};
    if (ftk::robust_critical_point_in_simplex2(vf, idxs)) {
        eb_min[a]=0; eb_min[b]=0; eb_min[c]=0;
        cp_count++;
        return;
    }
    // eb 推导（一次）
    Tfp eb = derive_cp_abs_eb_sos_online<Tfp>(
                U_fp[a],U_fp[b],U_fp[c],
                V_fp[a],V_fp[b],V_fp[c]);
    if (eb < eb_min[a]) eb_min[a]=eb;
    if (eb < eb_min[b]) eb_min[b]=eb;
    if (eb < eb_min[c]) eb_min[c]=eb;
}

// ================== 全局唯一三角遍历 + 内部面 ==================
// 思路：对每个 slab [t,t+1]
//   (1) 底面：cell(i,j) 的 Upper/Lower（只在层 t 发一次）
//   (2) 侧面：以“边×时间”为基本单元（横/竖/对角 v00-v11），统一拆成两三角——全局唯一
//   (3) 内部剖分面：按您的 3-tet 切分，每个棱柱发两张内部面（只归属本棱柱）
//   (4) 顶面：仅在最后一个 slab（t==T-2）把层 t+1 的 Upper/Lower 发一次
template<typename Tfp>
inline void accumulate_eb_min_global_unique_with_internal(
    const Size3& sz, const Tfp* U_fp, const Tfp* V_fp,
    std::vector<Tfp>& eb_min)
{
    const int H=sz.H, W=sz.W, T=sz.T;
    const int dv = H*W; // = sk
    size_t cp_count =0;

    for (int t=0; t<T-1; ++t){
        if (t % 100 == 0){
            printf("processing slab %d / %d\n", t, T-1);
        }
        // -------- (1) 底面：每 cell 两三角（仅层 t） --------
        for (int i=0; i<H-1; ++i){
            int base_i   = vid(t,i,0,sz);
            int base_ip1 = vid(t,i+1,0,sz);
            for (int j=0; j<W-1; ++j){
                int v00 = base_i   + j;
                int v01 = base_i   + (j+1);
                int v10 = base_ip1 + j;
                int v11 = base_ip1 + (j+1);
                // Upper: (v00,v01,v11)
                consider_triangle_and_update_ebmin(v00,v01,v11,U_fp,V_fp,eb_min,cp_count);
                // Lower: (v00,v10,v11)
                consider_triangle_and_update_ebmin(v00,v10,v11,U_fp,V_fp,eb_min,cp_count);
            }
        }

        // -------- (2) 侧面：边×时间 → 两三角（全局唯一） --------
        // 2.1 横边 (i,j)-(i,j+1)
        for (int i=0; i<H; ++i){
            int row = vid(t,i,0,sz);
            for (int j=0; j<W-1; ++j){
                int a=row+j, b=a+1, ap=a+dv, bp=b+dv;
                consider_triangle_and_update_ebmin(a,b,bp,U_fp,V_fp,eb_min,cp_count);
                consider_triangle_and_update_ebmin(a,bp,ap,U_fp,V_fp,eb_min,cp_count);
            }
        }
        // 2.2 竖边 (i,j)-(i+1,j)
        for (int i=0; i<H-1; ++i){
            int row_i   = vid(t,i,0,sz);
            int row_ip1 = vid(t,i+1,0,sz);
            for (int j=0; j<W; ++j){
                int a=row_i+j, b=row_ip1+j, ap=a+dv, bp=b+dv;
                consider_triangle_and_update_ebmin(a,b,bp,U_fp,V_fp,eb_min,cp_count);
                consider_triangle_and_update_ebmin(a,bp,ap,U_fp,V_fp,eb_min,cp_count);
            }
        }
        // 2.3 对角边 (i,j)-(i+1,j+1)（与三角网格一致，仅此一条对角）
        for (int i=0; i<H-1; ++i){
            int row_i   = vid(t,i,0,sz);
            int row_ip1 = vid(t,i+1,0,sz);
            for (int j=0; j<W-1; ++j){
                int a=row_i+j, b=row_ip1+(j+1), ap=a+dv, bp=b+dv;
                consider_triangle_and_update_ebmin(a,b,bp,U_fp,V_fp,eb_min,cp_count);
                consider_triangle_and_update_ebmin(a,bp,ap,U_fp,V_fp,eb_min,cp_count);
            }
        }

        // -------- (3) 内部剖分面：每棱柱两面（仅归属本棱柱） --------
        for (int i=0; i<H-1; ++i){
            for (int j=0; j<W-1; ++j){
                // Upper prism
                {
                    auto a = tri_vertices_2d(i,j,TriInCell::Upper,t,sz);
                    std::array<int,3> b{ a[0]+dv, a[1]+dv, a[2]+dv };
                    // 两张内部面： (a0,a1,b2), (a0,b1,b2)
                    consider_triangle_and_update_ebmin(a[0],a[1],b[2],U_fp,V_fp,eb_min,cp_count);
                    consider_triangle_and_update_ebmin(a[0],b[1],b[2],U_fp,V_fp,eb_min,cp_count);
                }
                // Lower prism
                {
                    auto a = tri_vertices_2d(i,j,TriInCell::Lower,t,sz);
                    std::array<int,3> b{ a[0]+dv, a[1]+dv, a[2]+dv };
                    consider_triangle_and_update_ebmin(a[0],a[1],b[2],U_fp,V_fp,eb_min,cp_count);
                    consider_triangle_and_update_ebmin(a[0],b[1],b[2],U_fp,V_fp,eb_min,cp_count);
                }
            }
        }

        // -------- (4) 顶面：仅最后一片把层 t+1 发一次 --------
        if (t==T-2){
            int tp=t+1;
            for (int i=0; i<H-1; ++i){
                int base_i   = vid(tp,i,0,sz);
                int base_ip1 = vid(tp,i+1,0,sz);
                for (int j=0; j<W-1; ++j){
                    int v00=base_i+j, v01=base_i+(j+1);
                    int v10=base_ip1+j, v11=base_ip1+(j+1);
                    consider_triangle_and_update_ebmin(v00,v01,v11,U_fp,V_fp,eb_min,cp_count);
                    consider_triangle_and_update_ebmin(v00,v10,v11,U_fp,V_fp,eb_min,cp_count);
                }
            }
        }
    }
    printf("total cp count = %ld\n",cp_count);
}

struct Metrics {
    double min_orig = +std::numeric_limits<double>::infinity();
    double max_orig = -std::numeric_limits<double>::infinity();
    double min_recon= +std::numeric_limits<double>::infinity();
    double max_recon= -std::numeric_limits<double>::infinity();
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double rmse = 0.0;
    double nrmse = 0.0;
    double psnr = 0.0;
};

template<typename T>
static inline Metrics compute_metrics(const T* orig, const T* recon, size_t N, double eps=1e-30)
{
    Metrics m;
    long double sse = 0.0L; // 用长双精度累加，降低数值误差

    for (size_t i=0;i<N;++i){
        const double o = static_cast<double>(orig[i]);
        const double r = static_cast<double>(recon[i]);
        const double e = std::abs(o - r);

        m.min_orig  = std::min(m.min_orig,  o);
        m.max_orig  = std::max(m.max_orig,  o);
        m.min_recon = std::min(m.min_recon, r);
        m.max_recon = std::max(m.max_recon, r);

        m.max_abs_err = std::max(m.max_abs_err, e);
        if (std::abs(o) > 1e-5){
            m.max_rel_err = std::max(m.max_rel_err, e / std::abs(o));
        }
        const double de = (o - r);
        sse += static_cast<long double>(de) * static_cast<long double>(de);
    }

    const double rmse = std::sqrt(static_cast<double>(sse / (N>0?N:1)));
    m.rmse = rmse;

    const double range = m.max_orig - m.min_orig; // 原始数据动态范围
    m.nrmse = (range > 0.0) ? rmse / range : rmse;

    if (rmse == 0.0){
        m.psnr = std::numeric_limits<double>::infinity();
    } else if (range > 0.0){
        m.psnr = 20.0 * std::log10(range / rmse);
    } else {
        // 原始数据为常数场：以 |max_orig| 作为峰值尝试给出 PSNR，否则报告 -inf
        const double peak = std::abs(m.max_orig);
        m.psnr = (peak > 0.0) ? 20.0 * std::log10(peak / rmse)
                              : -std::numeric_limits<double>::infinity();
    }
    return m;
}

static inline void print_metrics(const char* name, const Metrics& m)
{
    std::cout.setf(std::ios::fixed); std::cout<<std::setprecision(6);
    std::cout << "["<<name<<"]\n"
              << "  orig min/max : " << m.min_orig  << " / " << m.max_orig  << "\n"
              << "  recon min/max: " << m.min_recon << " / " << m.max_recon << "\n"
              << "  max |err|    : " << m.max_abs_err << "\n"
              << "  max rel err  : " << m.max_rel_err << "\n"
              << "  RMSE         : " << m.rmse  << "\n"
              << "  NRMSE        : " << m.nrmse << "  (normalized by orig range)\n"
              << "  PSNR [dB]    : " << m.psnr  << "\n";
}

// 入口：同时验证 U/V 以及合并后的整体（两分量拼接）
template<typename T>
static inline void verify(const T* U_orig, const T* V_orig,
                          const T* U_recon, const T* V_recon,
                          size_t r1, size_t r2, size_t r3,
                          double eps=1e-30)
{
    const size_t N = r1*r2*r3;

    Metrics mu = compute_metrics(U_orig, U_recon, N, eps);
    Metrics mv = compute_metrics(V_orig, V_recon, N, eps);

    // 合并统计（把 U/V 当作 2N 长度的单一序列）
    // 这里避免额外分配，直接分别累计再合并
    Metrics mall;
    {
        // 先 U
        mall.min_orig  = std::min(mu.min_orig, mv.min_orig);
        mall.max_orig  = std::max(mu.max_orig, mv.max_orig);
        mall.min_recon = std::min(mu.min_recon, mv.min_recon);
        mall.max_recon = std::max(mu.max_recon, mv.max_recon);
        mall.max_abs_err = std::max(mu.max_abs_err, mv.max_abs_err);
        mall.max_rel_err = std::max(mu.max_rel_err, mv.max_rel_err);

        // RMSE 合并（基于 SSE 累加）
        const long double sse_u = static_cast<long double>(mu.rmse) * mu.rmse * N;
        const long double sse_v = static_cast<long double>(mv.rmse) * mv.rmse * N;
        const size_t      Nall  = 2*N;
        const double rmse_all = std::sqrt(static_cast<double>((sse_u + sse_v) / (Nall>0?Nall:1)));
        mall.rmse = rmse_all;

        const double range_all = (mall.max_orig - mall.min_orig);
        mall.nrmse = (range_all>0.0) ? (rmse_all / range_all) : rmse_all;
        if (rmse_all == 0.0){
            mall.psnr = std::numeric_limits<double>::infinity();
        } else if (range_all > 0.0){
            mall.psnr = 20.0 * std::log10(range_all / rmse_all);
        } else {
            const double peak = std::max(std::abs(mall.max_orig), std::abs(mall.min_orig));
            mall.psnr = (peak>0.0) ? 20.0*std::log10(peak / rmse_all)
                                   : -std::numeric_limits<double>::infinity();
        }
    }

    // 打印
    print_metrics("U", mu);
    print_metrics("V", mv);
    print_metrics("U+V (combined)", mall);
}

// ========== 误差模式 ==========
enum class EbMode : uint8_t { Absolute=0, Relative=1 };


// ================== 压缩主函数（时间最慢 + 全局唯一三角 + 内部面） ==================
template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_fp(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,      // r1=H, r2=W, r3=T
    size_t& compressed_size,
    EbMode mode,                          // 误差模式
    double eb_param                       // Absolute: 相对range的因子；Relative: 相对比例 r
){
    using T = int64_t;
    const Size3 sz{ (int)r1,(int)r2,(int)r3 };
    const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

    // 0) 定点化
    T* U_fp=(T*)std::malloc(N*sizeof(T));
    T* V_fp=(T*)std::malloc(N*sizeof(T));
    if(!U_fp || !V_fp){ if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp); compressed_size=0; return nullptr; }
    T range=0, scale=convert_to_fixed_point<T_data,T>(U,V,N,U_fp,V_fp,range);

    // 参数
    const int   base=2; const double log_of_base=std::log2(base);
    const int   capacity=65536; const int intv_radius=(capacity>>1);
    const T     threshold=1;
    const T     eb_floor = 1; // 防止过小 eb
    const T     max_abs_eb = (mode==EbMode::Absolute)
                            ? (T)std::llround((long double)range * (long double)eb_param)
                            : std::numeric_limits<T>::max();

    // 1) eb_min 预计算
    std::vector<T> eb_min(N, max_abs_eb);
    accumulate_eb_min_global_unique_with_internal(sz, U_fp, V_fp, eb_min);

    // 2) Lorenzo + 量化 + Huffman
    int* eb_q  = (int*)std::malloc(N*sizeof(int));
    int* dq    = (int*)std::malloc(2*N*sizeof(int)); // U/V 交错
    if(!eb_q || !dq){
        if(eb_q) std::free(eb_q); if(dq) std::free(dq);
        std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
    }
    int* eb_q_pos=eb_q; int* dq_pos=dq;
    std::vector<T_data> unpred; unpred.reserve(N/10*2);

    const ptrdiff_t si=W, sj=1, sk=(ptrdiff_t)(H*W);

    for (int t=0; t<(int)Tt; ++t){
        if (t % 100 == 0){
            //printf("processing slab %d / %d\n", t, Tt);
        }
        for (int i=0; i<(int)H; ++i){
            for (int j=0; j<(int)W; ++j){
                const ptrdiff_t v = (ptrdiff_t)vid(t,i,j,sz);

                // 误差设定：模式 → eb_setting
                T eb_setting = max_abs_eb;
                if (mode==EbMode::Relative){
                    // 相对误差（基于定点幅度的 L_inf ）
                    T amp = std::max<T>( std::llabs(U_fp[v]), std::llabs(V_fp[v]) );
                    long double cand = (long double)amp * (long double)eb_param;
                    if (cand > (long double)std::numeric_limits<T>::max()) cand = (long double)std::numeric_limits<T>::max();
                    eb_setting = (T)std::llround(cand);
                }
                // CP 约束收紧 + floor
                T abs_eb = std::min<T>( eb_setting, eb_min[v] );
                // if (abs_eb < eb_floor) abs_eb = eb_floor;

                // 量化 eb（按引用，abs_eb 被替换为离散代表值）
                int eb_idx = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
                *eb_q_pos = eb_idx;

                if (abs_eb > 0){
                    bool unp=false;
                    T dec[2];
                    const T* pos[2] = { U_fp+v, V_fp+v };
                    for (int p=0; p<2; ++p){
                        const T* cur = pos[p];
                        T curv = *cur;
                        // 3D 一阶 Lorenzo（时间最慢）
                        T d0 = (t&&i&&j)? cur[-sk - si - sj]:0;
                        T d1 = (t&&i)   ? cur[-sk - si]     :0;
                        T d2 = (t&&j)   ? cur[-sk - sj]     :0;
                        T d3 = (t)      ? cur[-sk]          :0;
                        T d4 = (i&&j)   ? cur[-si - sj]     :0;
                        T d5 = (i)      ? cur[-si]          :0;
                        T d6 = (j)      ? cur[-sj]          :0;
                        T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
                        T diff = curv - pred;

                        T qd = (std::llabs(diff)/abs_eb) + 1;
                        if (qd < capacity){
                            qd = (diff>0) ? qd : -qd;
                            int qidx = int(qd/2) + intv_radius;
                            dq_pos[p] = qidx;
                            dec[p] = pred + 2*(qidx-intv_radius)*abs_eb;
                            if (std::llabs(dec[p]-curv) >= abs_eb){ unp=true; break; }
                        } else { unp=true; break; }
                    }
                    if (unp){
                        *(eb_q_pos++) = 0;
                        unpred.push_back(U[v]);
                        unpred.push_back(V[v]);
                    } else {
                        ++eb_q_pos;
                        dq_pos += 2;
                        U_fp[v] = dec[0];
                        V_fp[v] = dec[1];
                    }
                }else{
                    *(eb_q_pos++) = 0;
                    unpred.push_back(U[v]);
                    unpred.push_back(V[v]);
                }
            }
        }
    }

    // 3) 打包：加上误差模式与参数（便于排查 / 统计）
    unsigned char* out = (unsigned char*)std::malloc( (size_t) (2*N*sizeof(T)) );
    unsigned char* pos = out;
    write_variable_to_dst(pos, scale);
    write_variable_to_dst(pos, base);
    write_variable_to_dst(pos, threshold);
    write_variable_to_dst(pos, intv_radius);

    // 新增：误差模式与参数
    uint8_t mode_byte = static_cast<uint8_t>(mode);
    write_variable_to_dst(pos, mode_byte);
    double eb_param_d = eb_param;
    write_variable_to_dst(pos, eb_param_d);

    size_t unpred_cnt = unpred.size();
    write_variable_to_dst(pos, unpred_cnt);
    if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);

    size_t eb_num = (size_t)(eb_q_pos - eb_q);
    write_variable_to_dst(pos, eb_num);
    Huffman_encode_tree_and_data(/*state_num=*/1024, eb_q, eb_num, pos);
    std::free(eb_q);

    size_t dq_num = (size_t)(dq_pos - dq);
    write_variable_to_dst(pos, dq_num);
    Huffman_encode_tree_and_data(/*state_num=*/65536, dq, dq_num, pos);
    std::free(dq);

    compressed_size = (size_t)(pos - out);
    std::free(U_fp); std::free(V_fp);
    return out;
}
// ---------------- 解压主函数 ----------------
template<typename T_data>
bool sz_decompress_cp_preserve_sos_2p5d_fp(
    const unsigned char* compressed,
    size_t r1, size_t r2, size_t r3,
    T_data*& U, T_data*& V)
{
    using T = int64_t;
    const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

    if (U) std::free(U);
    if (V) std::free(V);

    // 头部
    const unsigned char* p = compressed;
    T scale=0; read_variable_from_src(p, scale);
    int base=0; read_variable_from_src(p, base);
    T threshold=0; read_variable_from_src(p, threshold);
    int intv_radius=0; read_variable_from_src(p, intv_radius);
    const int capacity = (intv_radius<<1);

    // 读取误差模式与参数（当前不参与重建，仅用于一致性/调试）
    uint8_t mode_byte=0; read_variable_from_src(p, mode_byte);
    double eb_param=0.0; read_variable_from_src(p, eb_param);
    (void)mode_byte; (void)eb_param;

    size_t unpred_cnt=0; read_variable_from_src(p, unpred_cnt);
    if (unpred_cnt % 2 != 0) return false;

    const T_data* unpred_data = reinterpret_cast<const T_data*>(p);
    const T_data* unpred_pos  = unpred_data;
    p += unpred_cnt * sizeof(T_data);

    size_t eb_num=0; read_variable_from_src(p, eb_num);
    int* eb_idx = Huffman_decode_tree_and_data(/*state_num=*/1024, eb_num, p);
    if (!eb_idx) return false;

    size_t dq_num=0; read_variable_from_src(p, dq_num);
    int* dq = Huffman_decode_tree_and_data(/*state_num=*/65536, dq_num, p);
    if (!dq){ std::free(eb_idx); return false; }

    if (eb_num != N) { std::free(eb_idx); std::free(dq); return false; }
    const size_t n_unpred_points = unpred_cnt/2;
    if (dq_num != 2*(N - n_unpred_points)) { std::free(eb_idx); std::free(dq); return false; }

    // 定点缓冲
    T* U_fp=(T*)std::malloc(N*sizeof(T));
    T* V_fp=(T*)std::malloc(N*sizeof(T));
    if(!U_fp || !V_fp){ if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp);
        std::free(eb_idx); std::free(dq); return false; }

    T* U_pos=U_fp; T* V_pos=V_fp;
    int* eb_pos=eb_idx; int* dq_pos=dq;
    std::vector<size_t> unpred_indices; unpred_indices.reserve(n_unpred_points);

    const ptrdiff_t si=W, sj=1, sk=(ptrdiff_t)(H*W);

    for (int t=0; t<(int)Tt; ++t){
        for (int i=0; i<(int)H; ++i){
            for (int j=0; j<(int)W; ++j){
                int ebid = *eb_pos++;

                if (ebid == 0){
                    size_t off = (size_t)(U_pos - U_fp);
                    unpred_indices.push_back(off);
                    *U_pos = (T)std::llround((long double)(*unpred_pos++) * (long double)scale);
                    *V_pos = (T)std::llround((long double)(*unpred_pos++) * (long double)scale);
                } else {
                    long double eb_ld = std::pow((long double)base, (long double)ebid)
                                      * (long double)threshold;
                    T abs_eb = (T)std::llround(eb_ld);

                    for (int pcomp=0; pcomp<2; ++pcomp){
                        T* cur = (pcomp==0) ? U_pos : V_pos;
                        T d0 = (t&&i&&j)? cur[-sk - si - sj]:0;
                        T d1 = (t&&i)   ? cur[-sk - si]     : 0;
                        T d2 = (t&&j)   ? cur[-sk - sj]     : 0;
                        T d3 = (t)      ? cur[-sk]          : 0;
                        T d4 = (i&&j)   ? cur[-si - sj]     : 0;
                        T d5 = (i)      ? cur[-si]          : 0;
                        T d6 = (j)      ? cur[-sj]          : 0;
                        T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

                        int qidx = *dq_pos++;
                        *cur = pred + (T) (2LL * ( (long long)qidx - (long long)intv_radius ) ) * abs_eb;
                    }
                }
                ++U_pos; ++V_pos;
            }
        }
    }

    // 回浮点并覆盖不可预测点
    U = (T_data*)std::malloc(N*sizeof(T_data));
    V = (T_data*)std::malloc(N*sizeof(T_data));
    if(!U || !V){
        std::free(U); std::free(V);
        std::free(U_fp); std::free(V_fp);
        std::free(eb_idx); std::free(dq);
        return false;
    }
    convert_to_floating_point(U_fp, V_fp, N, U, V, scale);

    unpred_pos = unpred_data;
    for (size_t off : unpred_indices){
        U[off] = *unpred_pos++;
        V[off] = *unpred_pos++;
    }

    std::free(U_fp); std::free(V_fp);
    std::free(eb_idx); std::free(dq);
    return true;
}

int main(int argc, char** argv) {
    // 文件路径：可从命令行传入；否则用默认名
    std::string u_path = (argc > 1 ? argv[1] : "u.bin");
    std::string v_path = (argc > 2 ? argv[2] : "v.bin");
    EbMode mode;
    std::string mode_str = (argc > 3 ? argv[3] : "abs");
    if(mode_str == "abs") {
        mode = EbMode::Absolute;
        printf("selected mode: absolute\n");
    }
    
    else if(mode_str == "rel") {
        mode = EbMode::Relative;
        printf("selected mode: relative\n");
    }
    else{
        std::cerr << "[ERR] unknown mode: " << mode_str << "\n";
        return 1;
    }
    //read data
    size_t num_elements =0;
    float * U_ptr = readfile<float>(u_path.c_str(),num_elements);
    float * V_ptr = readfile<float>(v_path.c_str(),num_elements);
    //调用压缩
    auto start_t = std::chrono::high_resolution_clock::now();
    size_t compressed_size = 0;
    unsigned char* compressed =
        sz_compress_cp_preserve_sos_2p5d_fp<float>(
    U_ptr, V_ptr, 450, 150, 2001, compressed_size, mode,0.01);

    if (!compressed){
        std::cerr << "[ERR] compression returned null.\n";
        return 2;
    }
    unsigned char * result_after_lossless = NULL;
    size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, compressed, compressed_size, &result_after_lossless);
    //释放返回的缓冲区（由函数 malloc 分配）
    std::free(compressed);
    auto end_t = std::chrono::high_resolution_clock::now();
    cout << "compression time in sec:" << std::chrono::duration<double>(end_t - start_t).count() << endl;
    cout << "compressed size (after lossless): " << lossless_outsize << " bytes." << "ratio = " << (2*num_elements*sizeof(float))/double(lossless_outsize) << endl;
    
    //调用解压
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
    int H = 450;
    int W = 150;
    int T = 2001;
    printf("start decompression...\n");
    sz_decompress_cp_preserve_sos_2p5d_fp<float>(decompressed, H, W, T, U_dec, V_dec);

    //verify
    printf("start verify...\n");
    float * U_ori = readfile<float>(u_path.c_str(),num_elements);
    float * V_ori = readfile<float>(v_path.c_str(),num_elements);
    verify(U_ori, V_ori, U_dec, V_dec, H, W, T);
    //check cp_count for U_dec and V_dec
    printf("calculating cp_count for decompressed data...\n");
    const Size3 sz{ (int)H,(int)W,(int)T };
    int64_t* U_fp=(int64_t*)std::malloc(num_elements*sizeof(int64_t));
    int64_t* V_fp=(int64_t*)std::malloc(num_elements*sizeof(int64_t));
    int64_t range=0, scale=convert_to_fixed_point<float,int64_t>(U_dec,V_dec,num_elements,U_fp,V_fp,range);
    std::vector<int64_t> eb_min(num_elements, 0.01);
    accumulate_eb_min_global_unique_with_internal(sz, U_fp, V_fp, eb_min);


    return 0;
}
