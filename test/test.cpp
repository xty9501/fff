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
	for (size_t i=0; i<num_elements; i++){
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
	for(size_t i=0; i<num_elements; i++){
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
void
convert_to_fixed_point_given_factor(const T * U, const T * V, size_t num_elements, T_fp * U_fp, T_fp * V_fp,int64_t vector_field_scaling_factor){
  for(size_t i=0; i<num_elements; i++){
    U_fp[i] = U[i] * vector_field_scaling_factor;
    V_fp[i] = V[i] * vector_field_scaling_factor;
  }
}

template<typename T, typename T_fp>
static void 
convert_to_floating_point(const T_fp * U_fp, const T_fp * V_fp, size_t num_elements, T * U, T * V, int64_t vector_field_scaling_factor){
	for(int i=0; i<num_elements; i++){
		U[i] = U_fp[i] * (T)1.0 / vector_field_scaling_factor;
		V[i] = V_fp[i] * (T)1.0 / vector_field_scaling_factor;
	}
}


// template<typename T, typename T_fp>
// static int64_t
// convert_to_fixed_point(const T* U, const T* V,
//                        size_t N, T_fp* U_fp, T_fp* V_fp,
//                        T_fp& range, int type_bits = 63)
// {
//     // 1) 最大幅值（用 long double 降误差；fabs 支持 long double）
//     long double max_abs = 0.0L;
//     for (size_t i = 0; i < N; ++i) {
//         long double a = std::fabs((long double)U[i]);
//         long double b = std::fabs((long double)V[i]);
//         long double m = (a > b ? a : b);
//         if (m > max_abs) max_abs = m;
//     }

//     // 2) 计算移位位数（base=2），并做安全夹取
//     const int nbits = (type_bits - 3) / 2;
//     int vbits = (max_abs > 0.0L) ? (int)std::ceil(std::log2((double)max_abs)) : 0;
//     int shift = nbits - vbits;
//     if (shift < 0)  shift = 0;
//     if (shift > 62) shift = 62;

//     // 3) 缩放因子（2^shift）
//     const int64_t scale = (int64_t)1 << shift;

//     // 4) 先乘后“最近整数”取整，避免近零符号丢失
//     int64_t fp_max = std::numeric_limits<int64_t>::min();
//     int64_t fp_min = std::numeric_limits<int64_t>::max();
//     const long double s = (long double)scale;

//     for (size_t i = 0; i < N; ++i) {
//         long double su = (long double)U[i] * s;
//         long double sv = (long double)V[i] * s;
//         int64_t iu = (int64_t)std::llround(su);
//         int64_t iv = (int64_t)std::llround(sv);

//         U_fp[i] = (T_fp)iu;
//         V_fp[i] = (T_fp)iv;

//         if (iu > fp_max) fp_max = iu;
//         if (iv > fp_max) fp_max = iv;
//         if (iu < fp_min) fp_min = iu;
//         if (iv < fp_min) fp_min = iv;
//     }

//     range = (T_fp)(fp_max - fp_min);
//     return scale;
// }

// template<typename T, typename T_fp>
// static void
// convert_to_floating_point(const T_fp* U_fp, const T_fp* V_fp,
//                           size_t N, T* U, T* V, int64_t scale)
// {
//     const long double inv = 1.0L / (long double)scale;
//     for (size_t i = 0; i < N; ++i) {
//         U[i] = (T)((long double)U_fp[i] * inv);
//         V[i] = (T)((long double)V_fp[i] * inv);
//     }
// }



// ================== 网格/索引基础 ==================
struct Size3 { int H, W, T; }; // H=450, W=150, T=2001
inline size_t vid(int t,int i,int j,const Size3& sz){
    return t*(sz.H*sz.W) + i*sz.W + j; // 时间t最慢, 行i次之, 列j最快
}
enum class TriInCell : unsigned char { Upper=0, Lower=1 };

// cell(i,j) 内，两种2D三角（底/顶层都用相同拓扑）
inline std::array<size_t,3> tri_vertices_2d(int i,int j,TriInCell which,int t,const Size3& sz){
    const size_t v00=vid(t,i,  j,  sz), v01=vid(t,i,  j+1,sz),
              v10=vid(t,i+1,j,  sz), v11=vid(t,i+1,j+1,sz);
    return (which==TriInCell::Upper) ? std::array<size_t,3>{v00,v01,v11}
                                     : std::array<size_t,3>{v00,v10,v11};
}

inline std::vector<size_t> all_connected_vertices(size_t vertice_idx, const Size3& sz){
    std::vector<size_t> neighbors;
    int t = vertice_idx / (sz.H * sz.W);
    int rem = vertice_idx % (sz.H * sz.W);
    int i = rem / sz.W;
    int j = rem % sz.W;

    // 3D grid neighbors (6-connectivity)
    for (int dt = -1; dt <= 1; ++dt) {
        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                if (std::abs(dt) + std::abs(di) + std::abs(dj) == 1) { // only direct neighbors
                    int nt = t + dt;
                    int ni = i + di;
                    int nj = j + dj;
                    if (nt >= 0 && nt < sz.T && ni >= 0 && ni < sz.H && nj >= 0 && nj < sz.W) {
                        neighbors.push_back(vid(nt, ni, nj, sz));
                    }
                }
            }
        }
    }
    return neighbors;
}

// 指定的 3-tet 切分（用于“内部剖分面”）
struct Tet { size_t v[4]; };
inline std::array<Tet,3> prism_split_3tets(const std::array<size_t,3>& a,
                                           const std::array<size_t,3>& b){
    // T0={a0,a1,a2,b2}, T1={a0,a1,b1,b2}, T2={a0,b0,b1,b2}
    return {{
        Tet{ a[0], a[1], a[2], b[2] },
        Tet{ a[0], a[1], b[1], b[2] },
        Tet{ a[0], b[0], b[1], b[2] }
    }};
}

// -------- 面键（无向三角，升序） ----------
struct FaceKeySZ {
  std::array<size_t,3> v;
  FaceKeySZ() = default;
  FaceKeySZ(size_t a,size_t b,size_t c){ v = {a,b,c}; std::sort(v.begin(), v.end()); }
  bool operator==(const FaceKeySZ& o) const { return v == o.v; }
};
struct FaceKeySZHash {
  size_t operator()(const FaceKeySZ& k) const noexcept {
    uint64_t h = 1469598103934665603ull;
    for (auto x : k.v){
      uint64_t y = (uint64_t)x;
      h ^= y + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
    }
    return (size_t)h;
  }
};

inline std::ostream& operator<<(std::ostream& os, const FaceKeySZ& f){
  return os << "(" << f.v[0] << ", " << f.v[1] << ", " << f.v[2] << ")";
}

// 线性索引 vid ⇢ (t,i,j)，假设：vid = t*(H*W) + i*W + j
inline void inv_vid(size_t vid, int H, int W, int &t, int &i, int &j) {
  const size_t dv = (size_t)H * (size_t)W;
  t = (int)(vid / dv);
  size_t rem = vid % dv;
  i = (int)(rem / (size_t)W);
  j = (int)(rem % (size_t)W);
}

// 针对单个三角，分别在 ori / dec 上做 robust CP 判定并打印
template<typename T_fp>
static inline void print_face_cp_robust_compare(
    const FaceKeySZ& face,
    const T_fp* U_fp_ori, const T_fp* V_fp_ori,
    const T_fp* U_fp_dec, const T_fp* V_fp_dec)
{
    auto a = face.v[0], b = face.v[1], c = face.v[2];
    bool cp_ori = face_has_cp_robust<T_fp>(a,b,c, U_fp_ori, V_fp_ori);
    bool cp_dec = face_has_cp_robust<T_fp>(a,b,c, U_fp_dec, V_fp_dec);
    std::cout << "  robust_cp(ori)=" << (cp_ori?1:0)
              << "  |  robust_cp(dec)=" << (cp_dec?1:0) << "\n";
}

// 打印一个三角面三个顶点在 ori/dec 两个场中的 U,V 值
template<typename T>
void print_face_uv(
    const FaceKeySZ& face,
    const T* U_ori, const T* V_ori,
    const T* U_dec, const T* V_dec,
    const int64_t* U_fp_ori, const int64_t* V_fp_ori,
    const int64_t* U_fp_dec, const int64_t* V_fp_dec,
    int H, int W, int Tt)
{
  (void)Tt; // 未直接使用，但保留以表意
  const size_t dv = (size_t)H * (size_t)W;
  auto prec = std::numeric_limits<T>::max_digits10;

  std::cout << "  vertices (t,i,j) and U/V (ori | dec):\n";
  for (int k = 0; k < 3; ++k) {
    const size_t vid = face.v[k];
    int t,i,j; inv_vid(vid, H, W, t, i, j);
    const size_t idx = (size_t)t * dv + (size_t)i * (size_t)W + (size_t)j;
    std::cout << "    v" << k << " @ (t=" << t << ", i=" << i << ", j=" << j << ")  "
              << "U: " << std::setprecision(prec) << U_ori[idx] << " | " << U_dec[idx]
              << "   V: " << std::setprecision(prec) << V_ori[idx] << " | " << V_dec[idx]
              << "   U_fp: " << U_fp_ori[idx] << " | " << U_fp_dec[idx]
              << "   V_fp: " << V_fp_ori[idx] << " | " << V_fp_dec[idx]
              << "\n";
  }
}

// 打印差异集合：分别列出「仅在 dec 有」与「仅在 ori 有」的面，并逐面打印 U,V
template<typename T>
void print_cp_face_diffs(
    const std::unordered_set<FaceKeySZ, FaceKeySZHash>& cp_faces_dec,
    const std::unordered_set<FaceKeySZ, FaceKeySZHash>& cp_faces_ori,
    const T* U_dec, const T* V_dec,
    const T* U_ori, const T* V_ori,
    const int64_t* U_fp_dec, const int64_t* V_fp_dec,
    const int64_t* U_fp_ori, const int64_t* V_fp_ori,
    int H, int W, int Tt)
{
  int max_print = 20;
  std::cout << "printing up to " << max_print << " differing CP faces:\n";
  int count = 0;
  // 仅在 decompressed 中出现
  for (const auto& f : cp_faces_dec) {
    if (!cp_faces_ori.count(f) && count < max_print) {
      std::cout << "[ONLY in DEC] face " << f << "\n";
      print_face_uv(f, U_ori, V_ori, U_dec, V_dec, U_fp_ori, V_fp_ori, U_fp_dec, V_fp_dec, H, W, Tt);
      print_face_cp_robust_compare<int64_t>(f, U_fp_ori, V_fp_ori, U_fp_dec, V_fp_dec);
      count++;
    }
  }
  count = 0;
  // 仅在 original 中出现
  for (const auto& f : cp_faces_ori) {
    if (!cp_faces_dec.count(f) && count < max_print) {
      std::cout << "[ONLY in ORI] face " << f << "\n";
      print_face_uv(f, U_ori, V_ori, U_dec, V_dec, U_fp_ori, V_fp_ori, U_fp_dec, V_fp_dec, H, W, Tt);
      print_face_cp_robust_compare<int64_t>(f, U_fp_ori, V_fp_ori, U_fp_dec, V_fp_dec);
      count++;
    }
  }
}

template<typename T, typename Hash, typename Equal>
bool are_unordered_sets_equal(
    const std::unordered_set<T, Hash, Equal>& a,
    const std::unordered_set<T, Hash, Equal>& b) {
    // 1. 检查大小
    if (a.size() != b.size()) {
        return false;
    }
    // 2. 遍历其中一个集合，检查所有元素是否存在于另一个集合
    for (const auto& elem : a) {
        if (b.find(elem) == b.end()) {
            return false;
        }
    }
    return true;
}

template<typename T_fp>
static inline bool face_has_cp_robust(size_t a,size_t b,size_t c,
                                      const T_fp* U_fp, const T_fp* V_fp)
{
  auto to_int = [](size_t x)->int { return (int)x; }; // 如需超大网格可加范围检查
  int idxs[3] = { to_int(a), to_int(b), to_int(c) };
  long long vf[3][2] = {
    { (long long)U_fp[a], (long long)V_fp[a] },
    { (long long)U_fp[b], (long long)V_fp[b] },
    { (long long)U_fp[c], (long long)V_fp[c] }
  };
  return ftk::robust_critical_point_in_simplex2(vf, idxs);
}

// =====================================================================
// ===============  2.5D 全局 compute_cp （返回哈希集合）  ==============
// =====================================================================
template<typename T_fp>
static std::unordered_set<FaceKeySZ, FaceKeySZHash>
compute_cp_2p5d_faces(const T_fp* U_fp, const T_fp* V_fp,
                      int H, int W, int T, std::string filenam="")
{
  bool write_to_file = !filenam.empty();
  std::unordered_set<FaceKeySZ, FaceKeySZHash> faces_with_cp;
  faces_with_cp.reserve((size_t)(H*(size_t)W*(size_t)T / 8)); // 经验值，预留一些空间

  const Size3 sz{H,W,T};
  const size_t dv = (size_t)H * (size_t)W;
  std::vector<int> cp_per_layer(T, 0);
  std::vector<int> cp_per_slab(T-1, 0);

#ifdef _OPENMP
  // 使用 OpenMP 并行：每个线程使用本地缓冲，最后统一去重合并
  printf("Using OpenMP with %d threads\n", omp_get_max_threads());
  const int nthreads = omp_get_max_threads();
  std::vector<std::vector<FaceKeySZ>> thread_faces(nthreads);
  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    auto &local = thread_faces[tid];

    // ---------------- (1) 层内面：每一层 t ∈ [0..T-1] ----------------
    #pragma omp for collapse(3) schedule(static)
    for (int t=0; t<T; ++t){
      for (int i=0; i<H-1; ++i){
        for (int j=0; j<W-1; ++j){
          if (i==0 && j==0 && (t % 1000 == 0)){
            printf("pre-compute cp lower layer %d / %d\n", t, T);
          }
          size_t v00 = vid(t,i,  j,  sz); 
          size_t v10 = vid(t,i,  j+1,sz); // v10=> x=1,y=0
          size_t v01 = vid(t,i+1,j,  sz);
          size_t v11 = vid(t,i+1,j+1,sz);

          if (face_has_cp_robust(v00,v01,v11, U_fp,V_fp)){
            local.emplace_back(v00,v01,v11);
            #pragma omp atomic update
            cp_per_layer[t]++;
          }
          if (face_has_cp_robust(v00,v10,v11, U_fp,V_fp)){
            local.emplace_back(v00,v10,v11);
            #pragma omp atomic update
            cp_per_layer[t]++;
          }
        }
      }
    }

    // ---------------- (2) 侧面：片 [t, t+1]，t ∈ [0..T-2] ----------------
    // 2.1 横边：i∈[0..H-1], j∈[0..W-2]
    #pragma omp for collapse(3) schedule(static)
    for (int t=0; t<T-1; ++t){
      for (int i=0; i<H; ++i){
        for (int j=0; j<W-1; ++j){
          if (i==0 && j==0 && (t % 1000 == 0)){
            printf("pre-compute cp side_hor layer %d / %d\n", t, T);
          }
          size_t a  = vid(t, i, j,   sz);
          size_t b  = vid(t, i, j+1, sz);
          size_t ap = a + dv, bp = b + dv;
          if (face_has_cp_robust(a,b,bp, U_fp,V_fp))  local.emplace_back(a,b,bp);
          if (face_has_cp_robust(a,bp,ap, U_fp,V_fp)) local.emplace_back(a,bp,ap);
        }
      }
    }

    // 2.2 竖边：i∈[0..H-2], j∈[0..W-1]
    #pragma omp for collapse(3) schedule(static)
    for (int t=0; t<T-1; ++t){
      for (int i=0; i<H-1; ++i){
        for (int j=0; j<W; ++j){
          if (i==0 && j==0 && (t % 1000 == 0)){
            printf("pre-compute cp side_ver layer %d / %d\n", t, T);
          }
          // size_t a  = vid(t, i,   j, sz);
          // size_t b  = vid(t, i+1, j, sz);
          // size_t ap = a + dv, bp = b + dv;
          // if (face_has_cp_robust(a,b,bp, U_fp,V_fp))  local.emplace_back(a,b,bp);
          // if (face_has_cp_robust(a,bp,ap, U_fp,V_fp)) local.emplace_back(a,bp,ap);
          size_t ax0y0 = vid(t,i,  j,  sz);
          size_t ax0y1 = vid(t,i+1,j,  sz);
          size_t bx0y0 = ax0y0 + dv, bx0y1 = ax0y1 + dv;
          if (face_has_cp_robust(ax0y0,ax0y1,bx0y0, U_fp,V_fp))  local.emplace_back(ax0y0,ax0y1,bx0y0);
          if (face_has_cp_robust(ax0y1,bx0y0,bx0y1, U_fp,V_fp)) local.emplace_back(ax0y1,bx0y0,bx0y1);
        }
      }
    }

    // 2.3 对角边：i∈[0..H-2], j∈[0..W-2]
    #pragma omp for collapse(3) schedule(static)
    for (int t=0; t<T-1; ++t){
      for (int i=0; i<H-1; ++i){
        for (int j=0; j<W-1; ++j){
          if (i==0 && j==0 && (t % 1000 == 0)){
            printf("pre-compute cp diag layer %d / %d\n", t, T);
          }
          // size_t a  = vid(t, i,   j,   sz);
          // size_t b  = vid(t, i+1, j+1, sz);
          // size_t ap = a + dv, bp = b + dv;
          // if (face_has_cp_robust(a,b,bp, U_fp,V_fp))  local.emplace_back(a,b,bp);
          // if (face_has_cp_robust(a,bp,ap, U_fp,V_fp)) local.emplace_back(a,bp,ap);
          size_t ax0y0 = vid(t,i,  j,  sz);
          size_t ax1y1 = vid(t,i+1,j+1,sz);
          size_t bx0y0 = ax0y0 + dv, bx1y1 = ax1y1 + dv;
          if (face_has_cp_robust(ax0y0,ax1y1,bx0y0, U_fp,V_fp))  local.emplace_back(ax0y0,ax1y1,bx0y0);
          if (face_has_cp_robust(ax1y1,bx0y0,bx1y1, U_fp,V_fp)) local.emplace_back(ax1y1,bx0y0,bx1y1);
        }
      }
    }

    // ---------------- (3) 内部剖分面：按 3-tet 切分 ----------------
    #pragma omp for collapse(3) schedule(static)
    for (int t=0; t<T-1; ++t){
      for (int i=0; i<H-1; ++i){
        for (int j=0; j<W-1; ++j){
          if (i==0 && j==0 && (t % 1000 == 0)){
            printf("pre-compute cp inside layer %d / %d\n", t, T);
          }
          // Upper prism
          {
            // size_t a0 = vid(t,i,  j,  sz);
            // size_t a1 = vid(t,i,  j+1,sz);
            // size_t a2 = vid(t,i+1,j+1,sz);
            // size_t b0 = a0 + dv, b1 = a1 + dv, b2 = a2 + dv;
            // if (face_has_cp_robust(a0,a1,b2, U_fp,V_fp)) local.emplace_back(a0,a1,b2);
            // if (face_has_cp_robust(a0,b1,b2, U_fp,V_fp)) local.emplace_back(a0,b1,b2);
            size_t ax0y0 = vid(t,i,  j,  sz);
            size_t ax0y1 = vid(t,i+1,  j,sz);
            size_t ax1y1 = vid(t,i+1,j+1,sz);
            size_t bx0y0 = ax0y0 + dv, bx0y1 = ax0y1 + dv, bx1y1 = ax1y1 + dv;
            if (face_has_cp_robust(ax0y1,bx0y0,ax1y1, U_fp,V_fp)) local.emplace_back(ax0y1,bx0y0,ax1y1);
            if (face_has_cp_robust(ax0y1,bx0y0,bx1y1, U_fp,V_fp)) local.emplace_back(ax0y1,bx0y0,bx1y1);
          }
          // Lower prism
          {
            // size_t a0 = vid(t,i,  j,  sz);
            // size_t a1 = vid(t,i+1,j,  sz);
            // size_t a2 = vid(t,i+1,j+1,sz);
            // size_t b0 = a0 + dv, b1 = a1 + dv, b2 = a2 + dv;
            // if (face_has_cp_robust(a0,a1,b2, U_fp,V_fp)) local.emplace_back(a0,a1,b2);
            // if (face_has_cp_robust(a0,b1,b2, U_fp,V_fp)) local.emplace_back(a0,b1,b2);
            size_t ax0y0 = vid(t,i,  j,  sz);
            size_t ax1y0 = vid(t,i,j+1,  sz);
            size_t ax1y1 = vid(t,i+1,j+1,sz);
            size_t bx0y0 = ax0y0 + dv, bx1y0 = ax1y0 + dv, bx1y1 = ax1y1 + dv;
            if (face_has_cp_robust(ax0y0,ax1y1,bx1y0, U_fp,V_fp)) local.emplace_back(ax0y0,ax1y1,bx1y0);
            if (face_has_cp_robust(ax1y1,bx0y0,bx1y0, U_fp,V_fp)) local.emplace_back(ax1y1,bx0y0,bx1y0);
          }
        }
      }
    }
  } // end parallel

  // 合并去重
  for (auto &vec : thread_faces){
    for (const auto &fk : vec){
      faces_with_cp.emplace(fk);
    }
  }
#else
  // // ---------------- (1) 层内面：每一层 t ∈ [0..T-1] ----------------
  // for (int t=0; t<T; ++t){
  //   if(t % 1000 == 0){
  //     printf("pre-compute cp lower layer %d / %d\n", t, T);
  //   }
  //   for (int i=0; i<H-1; ++i){
  //     for (int j=0; j<W-1; ++j){
  //       size_t v00 = vid(t,i,  j,  sz);
  //       size_t v01 = vid(t,i,  j+1,sz);
  //       size_t v10 = vid(t,i+1,j,  sz);
  //       size_t v11 = vid(t,i+1,j+1,sz);

  //       // Upper: (v00,v01,v11)
  //       if (face_has_cp_robust(v00,v01,v11, U_fp,V_fp)){
  //           faces_with_cp.emplace(v00,v01,v11);
  //           cp_per_layer[t]++;
  //       }

  //       // Lower: (v00,v10,v11)
  //       if (face_has_cp_robust(v00,v10,v11, U_fp,V_fp)){
  //           faces_with_cp.emplace(v00,v10,v11);
  //           cp_per_layer[t]++;
  //       }
  //     }
  //   }
  // }

  // // ---------------- (2) 侧面：片 [t, t+1]，t ∈ [0..T-2] ----------------
  // // 2.1 横边：i∈[0..H-1], j∈[0..W-2]
  // for (int t=0; t<T-1; ++t){
  //   if(t % 1000 == 0){
  //     printf("pre-compute cp side_hor layer %d / %d\n", t, T);
  //   }
  //   for (int i=0; i<H; ++i){
  //     for (int j=0; j<W-1; ++j){
  //       size_t a  = vid(t, i, j,   sz);
  //       size_t b  = vid(t, i, j+1, sz);
  //       size_t ap = a + dv, bp = b + dv;
  //       // 两个侧三角
  //       if (face_has_cp_robust(a,b,bp, U_fp,V_fp))  faces_with_cp.emplace(a,b,bp);
  //       if (face_has_cp_robust(a,bp,ap, U_fp,V_fp)) faces_with_cp.emplace(a,bp,ap);
  //     }
  //   }
  // }
  // // 2.2 竖边：i∈[0..H-2], j∈[0..W-1]
  // for (int t=0; t<T-1; ++t){
  //   if(t % 1000 == 0){
  //     printf("pre-compute cp side_ver layer %d / %d\n", t, T);
  //   }
  //   for (int i=0; i<H-1; ++i){
  //     for (int j=0; j<W; ++j){
  //       size_t a  = vid(t, i,   j, sz);
  //       size_t b  = vid(t, i+1, j, sz);
  //       size_t ap = a + dv, bp = b + dv;
  //       if (face_has_cp_robust(a,b,bp, U_fp,V_fp))  faces_with_cp.emplace(a,b,bp);
  //       if (face_has_cp_robust(a,bp,ap, U_fp,V_fp)) faces_with_cp.emplace(a,bp,ap);
  //     }
  //   }
  // }
  // // 2.3 对角边：i∈[0..H-2], j∈[0..W-2]
  // for (int t=0; t<T-1; ++t){
  //   if(t % 1000 == 0){
  //     printf("pre-compute cp diag layer %d / %d\n", t, T);
  //   }
  //   for (int i=0; i<H-1; ++i){
  //     for (int j=0; j<W-1; ++j){
  //       size_t a  = vid(t, i,   j,   sz);
  //       size_t b  = vid(t, i+1, j+1, sz);
  //       size_t ap = a + dv, bp = b + dv;
  //       if (face_has_cp_robust(a,b,bp, U_fp,V_fp))  faces_with_cp.emplace(a,b,bp);
  //       if (face_has_cp_robust(a,bp,ap, U_fp,V_fp)) faces_with_cp.emplace(a,bp,ap);
  //     }
  //   }
  // }

  // // ---------------- (3) 内部剖分面：按你的 3-tet 切分 ----------------
  // // 每个棱柱（由 cell (i,j) 的 Upper/Lower 三角沿时间 extrude）有 2 个内部三角面
  // // Upper 三角 a=(v00,v01,v11), b=a+dv -> 内部面: (a0,a1,b2), (a0,b1,b2)
  // // Lower 三角 a=(v00,v10,v11), b=a+dv -> 内部面: (a0,a1,b2), (a0,b1,b2)
  // for (int t=0; t<T-1; ++t){
  //   if(t % 1000 == 0){
  //     printf("pre-compute cp inside layer %d / %d\n", t, T);
  //   }
  //   for (int i=0; i<H-1; ++i){
  //     for (int j=0; j<W-1; ++j){
  //       // Upper
  //       {
  //         size_t a0 = vid(t,i,  j,  sz);
  //         size_t a1 = vid(t,i,  j+1,sz);
  //         size_t a2 = vid(t,i+1,j+1,sz);
  //         size_t b0 = a0 + dv, b1 = a1 + dv, b2 = a2 + dv;

  //         if (face_has_cp_robust(a0,a1,b2, U_fp,V_fp)) faces_with_cp.emplace(a0,a1,b2);
  //         if (face_has_cp_robust(a0,b1,b2, U_fp,V_fp)) faces_with_cp.emplace(a0,b1,b2);
  //       }
  //       // Lower
  //       {
  //         size_t a0 = vid(t,i,  j,  sz);
  //         size_t a1 = vid(t,i+1,j,  sz);
  //         size_t a2 = vid(t,i+1,j+1,sz);
  //         size_t b0 = a0 + dv, b1 = a1 + dv, b2 = a2 + dv;

  //         if (face_has_cp_robust(a0,a1,b2, U_fp,V_fp)) faces_with_cp.emplace(a0,a1,b2);
  //         if (face_has_cp_robust(a0,b1,b2, U_fp,V_fp)) faces_with_cp.emplace(a0,b1,b2);
  //       }
  //     }
  //   }
  // }
#endif

  // 打印每层 cp 数量（保持原行为）
  // for (int t=0; t<T; t+=10){
  //     printf("  cp in layer %d = %d\n", t, cp_per_layer[t]);
  // }

  //cp_layer 写入文件
  if (write_to_file){
  // write cp_per_layer to file
  std::ofstream ofs(filenam);
  if (ofs){
    for (int t=0; t<T; ++t){
      ofs << cp_per_layer[t] << "\n";
    }
    ofs.close();
    printf("Wrote cp per layer to %s\n", filenam.c_str());
  } else {
    printf("Failed to open file %s for writing cp per layer\n", filenam.c_str());
  }
}

  return faces_with_cp;
}


// -------- 便捷查询（可在逐顶点流程里用）----------
static inline bool has_cp(const std::unordered_set<FaceKeySZ,FaceKeySZHash>& cp_faces,
                          size_t a,size_t b,size_t c)
{
  return cp_faces.find(FaceKeySZ(a,b,c)) != cp_faces.end();
}


// ============ 辅助：对一个三角做一次“CP/eb”并回写到 eb_min ============
template<typename Tfp>
inline void consider_triangle_and_update_ebmin(
    size_t a,size_t b,size_t c,
    const Tfp* U_fp,const Tfp* V_fp,
    std::vector<Tfp>& eb_min, size_t &cp_count, std::vector<int> &cp_vector,int time_dim)
{
    // CP 检测（定点）
    int64_t vf[3][2] = {
        { (int64_t)U_fp[a], (int64_t)V_fp[a] },
        { (int64_t)U_fp[b], (int64_t)V_fp[b] },
        { (int64_t)U_fp[c], (int64_t)V_fp[c] }
    };
    int idxs[3] = {static_cast<int>(a), static_cast<int>(b), static_cast<int>(c)}; //这里的idx只能用int..ftk的接口就是这样

    if (ftk::robust_critical_point_in_simplex2(vf, idxs)) {
        eb_min[a]=0; eb_min[b]=0; eb_min[c]=0;
        cp_count++;
        cp_vector[time_dim]++;
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
    std::vector<int> cp_per_layer(T,0);
    std::vector<int> cp_per_slab(T-1,0);

    for (int t=0; t<T-1; ++t){

        if (t % 100 == 0){
            printf("processing slab %d / %d\n", t, T-1);
        }
        // -------- (1) 底面：每 cell 两三角（仅层 t） --------
        for (int i=0; i<H-1; ++i){
            size_t base_i   = vid(t,i,0,sz);
            size_t base_ip1 = vid(t,i+1,0,sz);
            for (int j=0; j<W-1; ++j){
                size_t v00 = base_i   + j;
                size_t v01 = base_i   + (j+1);
                size_t v10 = base_ip1 + j;
                size_t v11 = base_ip1 + (j+1);
                // Upper: (v00,v01,v11)
                consider_triangle_and_update_ebmin(v00,v01,v11,U_fp,V_fp,eb_min,cp_count,cp_per_layer,t);
                // Lower: (v00,v10,v11)
                consider_triangle_and_update_ebmin(v00,v10,v11,U_fp,V_fp,eb_min,cp_count,cp_per_layer,t);
            }
        }

        // -------- (2) 侧面：边×时间 → 两三角（全局唯一） --------
        // 2.1 横边 (i,j)-(i,j+1)
        for (int i=0; i<H; ++i){
            size_t row = vid(t,i,0,sz);
            for (int j=0; j<W-1; ++j){
                size_t a=row+j, b=a+1, ap=a+dv, bp=b+dv;
                consider_triangle_and_update_ebmin(a,b,bp,U_fp,V_fp,eb_min,cp_count,cp_per_slab,t);
                consider_triangle_and_update_ebmin(a,bp,ap,U_fp,V_fp,eb_min,cp_count,cp_per_slab,t);
            }
        }
        // 2.2 竖边 (i,j)-(i+1,j)
        for (int i=0; i<H-1; ++i){
            size_t row_i   = vid(t,i,0,sz);
            size_t row_ip1 = vid(t,i+1,0,sz);
            for (int j=0; j<W; ++j){
                size_t a=row_i+j, b=row_ip1+j, ap=a+dv, bp=b+dv;
                consider_triangle_and_update_ebmin(a,b,bp,U_fp,V_fp,eb_min,cp_count,cp_per_slab,t);
                consider_triangle_and_update_ebmin(a,bp,ap,U_fp,V_fp,eb_min,cp_count,cp_per_slab,t);
            }
        }
        // 2.3 对角边 (i,j)-(i+1,j+1)（与三角网格一致，仅此一条对角）
        for (int i=0; i<H-1; ++i){
            size_t row_i   = vid(t,i,0,sz);
            size_t row_ip1 = vid(t,i+1,0,sz);
            for (int j=0; j<W-1; ++j){
                size_t a=row_i+j, b=row_ip1+(j+1), ap=a+dv, bp=b+dv;
                consider_triangle_and_update_ebmin(a,b,bp,U_fp,V_fp,eb_min,cp_count,cp_per_slab,t);
                consider_triangle_and_update_ebmin(a,bp,ap,U_fp,V_fp,eb_min,cp_count,cp_per_slab,t);
            }
        }

        // -------- (3) 内部剖分面：每棱柱两面（仅归属本棱柱） --------
        for (int i=0; i<H-1; ++i){
            for (int j=0; j<W-1; ++j){
                // Upper prism
                {
                    auto a = tri_vertices_2d(i,j,TriInCell::Upper,t,sz);
                    std::array<size_t,3> b{ a[0]+dv, a[1]+dv, a[2]+dv };
                    // 两张内部面： (a0,a1,b2), (a0,b1,b2)
                    consider_triangle_and_update_ebmin(a[0],a[1],b[2],U_fp,V_fp,eb_min,cp_count,cp_per_slab,t);
                    consider_triangle_and_update_ebmin(a[0],b[1],b[2],U_fp,V_fp,eb_min,cp_count,cp_per_slab,t);
                }
                // Lower prism
                {
                    auto a = tri_vertices_2d(i,j,TriInCell::Lower,t,sz);
                    std::array<size_t,3> b{ a[0]+dv, a[1]+dv, a[2]+dv };
                    consider_triangle_and_update_ebmin(a[0],a[1],b[2],U_fp,V_fp,eb_min,cp_count,cp_per_slab,t);
                    consider_triangle_and_update_ebmin(a[0],b[1],b[2],U_fp,V_fp,eb_min,cp_count,cp_per_slab,t);
                }
            }
        }

        // -------- (4) 顶面：仅最后一片把层 t+1 发一次 --------
        if (t==T-2){
            int tp=t+1;
            for (int i=0; i<H-1; ++i){
                size_t base_i   = vid(tp,i,0,sz);
                size_t base_ip1 = vid(tp,i+1,0,sz);
                for (int j=0; j<W-1; ++j){
                    size_t v00=base_i+j, v01=base_i+(j+1);
                    size_t v10=base_ip1+j, v11=base_ip1+(j+1);
                    consider_triangle_and_update_ebmin(v00,v01,v11,U_fp,V_fp,eb_min,cp_count);
                    consider_triangle_and_update_ebmin(v00,v10,v11,U_fp,V_fp,eb_min,cp_count);
                }
            }
        }
    }
    printf("total cp count = %ld\n",cp_count);
    for (int t=0; t<T; t+=10){
        printf("  cp in layer %d = %d\n", t, cp_per_layer[t]);
        printf("  cp in slab  %d = %d\n", t, (t<T-1)?cp_per_slab[t]:0);
    }
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
static inline Metrics compute_metrics(const T* orig, const T* recon, size_t N)
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
                          size_t r1, size_t r2, size_t r3)
{
    const size_t N = r1*r2*r3;

    Metrics mu = compute_metrics(U_orig, U_recon, N);
    Metrics mv = compute_metrics(V_orig, V_recon, N);

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

// ===== 主函数：逐顶点 + CP 面集合 =====
template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap1(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb                  // 全局绝对误差上限：max_eb = range * max_pwr_eb
){
  using T = int64_t;
  const Size3 sz{ (int)r1,(int)r2,(int)r3 };
  const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

  // 1) 定点化
  T *U_fp=(T*)std::malloc(N*sizeof(T));
  T *V_fp=(T*)std::malloc(N*sizeof(T));
  if(!U_fp || !V_fp){ if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp); compressed_size=0; return nullptr; }
  T range=0;
  T scale = convert_to_fixed_point<T_data,T>(U, V, N, U_fp, V_fp, range);

  // 2) 预计算：全局“含 CP 的三角面”集合（一次性，后续 O(1) 查询）
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  cout << "pre-compute cp faces time second: " << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count() << endl;
  //print the size of cp_faces
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // 3) 量化/编码缓冲
  int* eb_quant_index = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index = (int*)std::malloc(2*N*sizeof(int));
  double enc_max_abs_eb_fp   = 0.0; //check 用
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred; unpred.reserve((N/10)*2);

  const int base = 2;
  const double log_of_base = std::log2(base);
  const int capacity = 65536;
  const int intv_radius = (capacity >> 1);
//   const T max_eb = range * max_pwr_eb; // 全局绝对误差上限max_eb
  const T max_eb = (T) std::llround( (long double)max_pwr_eb * (long double)scale );
  const T threshold = 1;
  const T eb_floor = 1; // 防止为 0 的步长（量化后代表值）

  // 4) 逐顶点：枚举与该顶点相关的三角面 → min eb
  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W;

  // 6 个平面邻接方向（左右、上下、主对角）
  const int di[6] = { 0,  0,  1, -1,  1, -1 };
  const int dj[6] = { 1, -1,  0,  0,  1, -1 };

  for (int t=0; t<(int)Tt; ++t){
    if(t % 100 == 0){
      printf("processing slice %d / %d\n", t, (int)Tt);
    }
    for (int i=0; i<(int)H; ++i){
      for (int j=0; j<(int)W; ++j){
        const size_t v = vid(t,i,j,sz);
        // 进入单点处理前，先缓存原始定点值，便于自检统计：
        T *curU = U_fp + v;
        T *curV = V_fp + v;
        const T curU_val = *curU;
        const T curV_val = *curV;

        // —— 收集最小 eb——
        T required_eb = max_eb;

        // (A) 层内 t：影响 (i-1..i, j-1..j) 的 4 个 cell，每 cell 两个三角
        for (int ci=i-1; ci<=i && required_eb>0; ++ci){
          if (!in_range(ci, (int)H-1)) continue;
          for (int cj=j-1; cj<=j && required_eb>0; ++cj){
            if (!in_range(cj, (int)W-1)) continue;

            size_t v00 = vid(t,ci,  cj,  sz);
            size_t v01 = vid(t,ci,  cj+1,sz);
            size_t v10 = vid(t,ci+1,cj,  sz);
            size_t v11 = vid(t,ci+1,cj+1,sz);

            // Upper: (v00,v01,v11)
            if (v==v00 || v==v01 || v==v11){
              if (has_cp(cp_faces, v00,v01,v11)) { required_eb = 0; goto after_min_eb; }
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v00],U_fp[v01],U_fp[v11],
                                                     V_fp[v00],V_fp[v01],V_fp[v11]);
              if (eb < required_eb) required_eb = eb;
            }
            // Lower: (v00,v10,v11)
            if (v==v00 || v==v10 || v==v11){
              if (has_cp(cp_faces, v00,v10,v11)) { required_eb = 0; goto after_min_eb; }
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v00],U_fp[v10],U_fp[v11],
                                                     V_fp[v00],V_fp[v10],V_fp[v11]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        // (B1) 侧面 [t, t+1]
        if (t < (int)Tt-1 && required_eb>0){
          for (int k=0; k<6 && required_eb>0; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a  = vid(t, i, j, sz);
            size_t b  = vid(t, ni,nj, sz);
            size_t ap = a + dv, bp = b + dv;
            // (a,b,bp)
            if (has_cp(cp_faces, a,b,bp)) { required_eb = 0; goto after_min_eb; }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[b],U_fp[bp],
                                                     V_fp[a],V_fp[b],V_fp[bp]);
              if (eb < required_eb) required_eb = eb;
            }
            // (a,bp,ap)
            if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0; goto after_min_eb; }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[bp],U_fp[ap],
                                                     V_fp[a],V_fp[bp],V_fp[ap]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        // (B2) 侧面 [t-1, t]
        if (t > 0 && required_eb>0){
          for (int k=0; k<6 && required_eb>0; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a  = vid(t-1, i, j, sz);
            size_t b  = vid(t-1, ni,nj, sz);
            size_t ap = a + dv; // = v
            size_t bp = b + dv;
            // (a,b,bp)
            if (has_cp(cp_faces, a,b,bp)) { required_eb = 0; goto after_min_eb; }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[b],U_fp[bp],
                                                     V_fp[a],V_fp[b],V_fp[bp]);
              if (eb < required_eb) required_eb = eb;
            }
            // (a,bp,ap)
            if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0; goto after_min_eb; }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[bp],U_fp[ap],
                                                     V_fp[a],V_fp[bp],V_fp[ap]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        // (C) 内部剖分面：两片 ts ∈ {t, t-1}；每相邻 cell 的 Upper/Lower 各 2 面
        if (required_eb>0){
          auto eval_internal = [&](int ts)->bool{
            if (!in_range(ts,(int)Tt-1)) return false;
            for (int ci=i-1; ci<=i; ++ci){
              if (!in_range(ci,(int)H-1)) continue;
              for (int cj=j-1; cj<=j; ++cj){
                if (!in_range(cj,(int)W-1)) continue;
                // Upper: a=(v00,v01,v11), b=a+dv → (a0,a1,b2), (a0,b1,b2)
                {
                  size_t a0 = vid(ts,ci,  cj,  sz);
                  size_t a1 = vid(ts,ci,  cj+1,sz);
                  size_t a2 = vid(ts,ci+1,cj+1,sz);
                  size_t b0 = a0+dv, b1=a1+dv, b2=a2+dv;
                  if (v==a0 || v==a1 || v==b2){
                    if (has_cp(cp_faces,a0,a1,b2)) { required_eb=0; return true; }
                    T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a0],U_fp[a1],U_fp[b2],
                                                           V_fp[a0],V_fp[a1],V_fp[b2]);
                    if (eb < required_eb) required_eb = eb;
                    if (required_eb==0) return true;
                  }
                  if (v==a0 || v==b1 || v==b2){
                    if (has_cp(cp_faces,a0,b1,b2)) { required_eb=0; return true; }
                    T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a0],U_fp[b1],U_fp[b2],
                                                           V_fp[a0],V_fp[b1],V_fp[b2]);
                    if (eb < required_eb) required_eb = eb;
                    if (required_eb==0) return true;
                  }
                }
                // Lower: a=(v00,v10,v11), b=a+dv → (a0,a1,b2), (a0,b1,b2)
                {
                  size_t a0 = vid(ts,ci,  cj,  sz);
                  size_t a1 = vid(ts,ci+1,cj,  sz);
                  size_t a2 = vid(ts,ci+1,cj+1,sz);
                  size_t b0 = a0+dv, b1=a1+dv, b2=a2+dv;
                  if (v==a0 || v==a1 || v==b2){
                    if (has_cp(cp_faces,a0,a1,b2)) { required_eb=0; return true; }
                    T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a0],U_fp[a1],U_fp[b2],
                                                           V_fp[a0],V_fp[a1],V_fp[b2]);
                    if (eb < required_eb) required_eb = eb;
                    if (required_eb==0) return true;
                  }
                  if (v==a0 || v==b1 || v==b2){
                    if (has_cp(cp_faces,a0,b1,b2)) { required_eb=0; return true; }
                    T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a0],U_fp[b1],U_fp[b2],
                                                           V_fp[a0],V_fp[b1],V_fp[b2]);
                    if (eb < required_eb) required_eb = eb;
                    if (required_eb==0) return true;
                  }
                }
              }
            }
            return false;
          };
          if (eval_internal(t)) goto after_min_eb;
          if (eval_internal(t-1)) goto after_min_eb;
        }

      after_min_eb:
        // —— 量化 eb 并压缩该顶点的 U/V —— 
        {
          T abs_eb = required_eb;
          // *eb_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
          *eb_pos = eb_exponential_quantize_new(abs_eb,threshold); //新量化

          if (abs_eb > 0){
            bool unpred_flag=false;
            T dec[2];
            // T *curU = U_fp + v;
            // T *curV = V_fp + v;
            T abs_err_fp_q[2] = {0,0};


            for (int p=0; p<2; ++p){
              T *cur = (p==0)? curU : curV;
              T  curv  = (p==0) ? curU_val : curV_val;

              // 3D Lorenzo（时间最慢）
              T d0 = (t&&i&&j)? cur[-sk - si - sj] : 0;
              T d1 = (t&&i)   ? cur[-sk - si]      : 0;
              T d2 = (t&&j)   ? cur[-sk - sj]      : 0;
              T d3 = (t)      ? cur[-sk]           : 0;
              T d4 = (i&&j)   ? cur[-si - sj]      : 0;
              T d5 = (i)      ? cur[-si]           : 0;
              T d6 = (j)      ? cur[-sj]           : 0;
              T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

              T diff = curv - pred;
              T qd = (std::llabs(diff)/abs_eb) + 1;
              if (qd < capacity){
                qd = (diff > 0) ? qd : -qd;
                int qindex = (int)(qd/2) + intv_radius;
                dq_pos[p] = qindex;
                dec[p] = pred + 2*(qindex - intv_radius)*abs_eb;
                //if (std::llabs(dec[p]-curv) > required_eb){ unpred_flag=true; break; }
                // **守门：必须用 abs_eb（实际传输的步长）校验**
                if (std::llabs(dec[p] - curv) > abs_eb){
                    unpred_flag = true;
                    break;
                }
                // 暂存本分量的真实误差(定点)
                abs_err_fp_q[p] = std::llabs(dec[p] - curv);
              }else{
                unpred_flag=true; break;
              }
            }

            if (unpred_flag){
              *(eb_pos++) = 0;
              unpred.push_back(U[v]);
              unpred.push_back(V[v]);
            }else{
                ++eb_pos;
                dq_pos += 2;
                *curU = dec[0];
                *curV = dec[1];
                // **编码端自检统计（浮点域）**
                double abs_eb_fp = (double)abs_eb / (double)scale;
                double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
                double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
                enc_max_abs_eb_fp   = std::max(enc_max_abs_eb_fp, abs_eb_fp);
                enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
            }
          }else{
            *(eb_pos++) = 0;
            unpred.push_back(U[v]);
            unpred.push_back(V[v]);
          }
        }
      }
    }
  }

    std::cerr << "[ENC] max abs_eb(fp) = " << enc_max_abs_eb_fp
    << ", max actual |err|(fp) = " << enc_max_real_err_fp << "\n";
  // 5) 打包码流（与 2D 版一致）
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  printf("write scale = %ld\n", scale);
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  write_array_to_dst(pos, unpred.data(), unpred_cnt);

  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::free(eb_quant_index);

  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  std::free(U_fp); std::free(V_fp);
  return compressed;
}

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap2(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb                  // 浮点域绝对误差上限（如 0.005）
){
  using T = int64_t;
  const Size3 sz{ (int)r1,(int)r2,(int)r3 };
  const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

  // ========== 1) 浮点→定点 ==========
  T *U_fp=(T*)std::malloc(N*sizeof(T));
  T *V_fp=(T*)std::malloc(N*sizeof(T));
  if(!U_fp || !V_fp){
    if(U_fp) std::free(U_fp);
    if(V_fp) std::free(V_fp);
    compressed_size=0; return nullptr;
  }
  T range=0;
  T scale = convert_to_fixed_point<T_data,T>(U, V, N, U_fp, V_fp, range);

  // ========== 2) 预计算：全局“含 CP 的三角面”集合 ==========
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // === PATCH A: 构建 CP 顶点掩码（凡在 CP 面上的顶点，一律 unpred）===
  std::vector<uint8_t> cp_vertex(N, 0);
  for (const auto& f : cp_faces){
    cp_vertex[f.v[0]] = 1;
    cp_vertex[f.v[1]] = 1;
    cp_vertex[f.v[2]] = 1;
  }

  // ========== 3) 量化/编码缓冲 ==========
  int* eb_quant_index = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index = (int*)std::malloc(2*N*sizeof(int));
  double enc_max_abs_eb_fp   = 0.0; //check 用
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred; unpred.reserve((N/10)*2);

  const int base = 2;
  const int capacity = 65536;
  const int intv_radius = (capacity >> 1);
  const T threshold = 1;                  // 幂指数量化的最小步长
  const T max_eb = (T) std::llround( (long double)max_pwr_eb * (long double)scale ); // 浮点→定点
  const size_t dv = (size_t)H*(size_t)W;

  // Lorenzo 预测的位移
  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);

  // 2D 6 邻（含 2 条对角，和你现有的三角剖分一致）
  const int di[6] = { 0,  0,  1, -1,  1, -1 };
  const int dj[6] = { 1, -1,  0,  0,  1, -1 };

  // ===== 小工具：对一张三角既查 CP 又取 eb 最小 =====
  auto consider_face = [&](size_t x,size_t y,size_t z, T &required_eb)->bool{
    if (has_cp(cp_faces, x,y,z)) { required_eb = 0; return true; }
    T eb = derive_cp_abs_eb_sos_online<T>(
              U_fp[x],U_fp[y],U_fp[z],
              V_fp[x],V_fp[y],V_fp[z]);
    if (eb < required_eb) required_eb = eb;
    return required_eb==0;
  };

  // ========== 4) 逐顶点：收紧 eb，然后量化并编码 ==========
  for (int t=0; t<(int)Tt; ++t){
    if(t % 100 == 0){
      printf("processing slice %d / %d\n", t, (int)Tt);
    }
    for (int i=0; i<(int)H; ++i){
      for (int j=0; j<(int)W; ++j){
        const size_t v = vid(t,i,j,sz);

        // === PATCH A-1: 凡是 CP 顶点，直接不预测 ===
        if (cp_vertex[v]) {
          *(eb_pos++) = 0;
          unpred.push_back(U[v]);
          unpred.push_back(V[v]);
          continue; // 这个点不再做预测/量化
        }

        // 缓存原始定点值（用于预测和误差检查）
        T *curU = U_fp + v;
        T *curV = V_fp + v;
        const T curU_val = *curU;
        const T curV_val = *curV;

        // —— 收集最小 eb ——（先设为全局上限）
        T required_eb = max_eb;

        // (A) 层内四邻 cell 的两张三角（Upper/Lower）
        for (int ci=i-1; ci<=i && required_eb>0; ++ci){
          if (ci < 0 || ci >= (int)H-1) continue;
          for (int cj=j-1; cj<=j && required_eb>0; ++cj){
            if (cj < 0 || cj >= (int)W-1) continue;

            size_t v00 = vid(t,ci,  cj,  sz);
            size_t v01 = vid(t,ci,  cj+1,sz);
            size_t v10 = vid(t,ci+1,cj,  sz);
            size_t v11 = vid(t,ci+1,cj+1,sz);

            if (v==v00 || v==v01 || v==v11){
              if (consider_face(v00,v01,v11, required_eb)) goto after_min_eb;
            }
            if (v==v00 || v==v10 || v==v11){
              if (consider_face(v00,v10,v11, required_eb)) goto after_min_eb;
            }
          }
        }

        // (B1) 侧面 [t, t+1] —— “四三角全覆盖”
        if (t < (int)Tt-1 && required_eb>0){
          for (int k=0; k<6 && required_eb>0; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (ni<0 || ni>=(int)H || nj<0 || nj>=(int)W) continue;

            size_t a  = vid(t, i, j,   sz);
            size_t b  = vid(t, ni,nj,  sz);
            size_t ap = a + dv,        bp = b + dv;

            // 以当前点 a 为锚的两张
            if (consider_face(a,b,bp, required_eb))  goto after_min_eb;
            if (consider_face(a,bp,ap, required_eb)) goto after_min_eb;

            // === PATCH B: 以邻点 b 为锚的两张（补齐覆盖）
            if (consider_face(b,a,ap, required_eb))  goto after_min_eb;
            if (consider_face(b,ap,bp, required_eb)) goto after_min_eb;
          }
        }

        // (B2) 侧面 [t-1, t] —— “四三角全覆盖”
        if (t > 0 && required_eb>0){
          for (int k=0; k<6 && required_eb>0; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (ni<0 || ni>=(int)H || nj<0 || nj>=(int)W) continue;

            size_t ap = vid(t,   i, j,  sz);  // 当前层
            size_t bp = vid(t,  ni,nj,  sz);
            size_t a  = ap - dv;              // 上一层
            size_t b  = bp - dv;

            if (consider_face(a,b,bp, required_eb))  goto after_min_eb;
            if (consider_face(a,bp,ap, required_eb)) goto after_min_eb;

            // === PATCH B: 以邻点 b 为锚的两张（补齐覆盖）
            if (consider_face(b,a,ap, required_eb))  goto after_min_eb;
            if (consider_face(b,ap,bp, required_eb)) goto after_min_eb;
          }
        }

        // (C) 内部剖分面：与原逻辑一致（Upper/Lower 两个内部三角），ts ∈ {t, t-1}
        if (required_eb>0){
          auto eval_internal = [&](int ts)->bool{
            if (ts < 0 || ts >= (int)Tt-1) return false;
            for (int ci=i-1; ci<=i; ++ci){
              if (ci<0 || ci>=(int)H-1) continue;
              for (int cj=j-1; cj<=j; ++cj){
                if (cj<0 || cj>=(int)W-1) continue;

                // Upper: a=(v00,v01,v11), b=a+dv → (a0,a1,b2), (a0,b1,b2)
                {
                  size_t a0 = vid(ts,ci,  cj,  sz);
                  size_t a1 = vid(ts,ci,  cj+1,sz);
                  size_t a2 = vid(ts,ci+1,cj+1,sz);
                  size_t b0 = a0+dv, b1=a1+dv, b2=a2+dv;

                  if (v==a0 || v==a1 || v==b2){
                    if (consider_face(a0,a1,b2, required_eb)) return true;
                  }
                  if (v==a0 || v==b1 || v==b2){
                    if (consider_face(a0,b1,b2, required_eb)) return true;
                  }
                }
                // Lower: a=(v00,v10,v11), b=a+dv → (a0,a1,b2), (a0,b1,b2)
                {
                  size_t a0 = vid(ts,ci,  cj,  sz);
                  size_t a1 = vid(ts,ci+1,cj,  sz);
                  size_t a2 = vid(ts,ci+1,cj+1,sz);
                  size_t b0 = a0+dv, b1=a1+dv, b2=a2+dv;

                  if (v==a0 || v==a1 || v==b2){
                    if (consider_face(a0,a1,b2, required_eb)) return true;
                  }
                  if (v==a0 || v==b1 || v==b2){
                    if (consider_face(a0,b1,b2, required_eb)) return true;
                  }
                }
              }
            }
            return false;
          };
          if (eval_internal(t))    goto after_min_eb;
          if (eval_internal(t-1))  goto after_min_eb;
        }

after_min_eb:
        // —— 量化 eb 并压缩该顶点的 U/V —— 
        {
          T abs_eb = required_eb;
          *eb_pos = eb_exponential_quantize_new(abs_eb, threshold); // 与解码端一致

          if (abs_eb > 0){
            bool unpred_flag=false;
            T dec[2];
            T abs_err_fp_q[2] = {0,0};

            for (int p=0; p<2; ++p){
              T *cur = (p==0)? curU : curV;
              T  curv  = (p==0) ? curU_val : curV_val;

              // 3D 一阶 Lorenzo（时间最慢）
              T d0 = (t&&i&&j)? cur[-sk - si - sj] : 0;
              T d1 = (t&&i)   ? cur[-sk - si]      : 0;
              T d2 = (t&&j)   ? cur[-sk - sj]      : 0;
              T d3 = (t)      ? cur[-sk]           : 0;
              T d4 = (i&&j)   ? cur[-si - sj]      : 0;
              T d5 = (i)      ? cur[-si]           : 0;
              T d6 = (j)      ? cur[-sj]           : 0;
              T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

              T diff = curv - pred;
              T qd = (std::llabs(diff)/abs_eb) + 1;
              if (qd < capacity){
                qd = (diff > 0) ? qd : -qd;
                int qindex = (int)(qd/2) + intv_radius;
                dq_pos[p] = qindex;
                dec[p] = pred + 2*(qindex - intv_radius)*abs_eb;

                // 守门：必须用 abs_eb（实际传输的步长）校验
                if (std::llabs(dec[p] - curv) > abs_eb){
                  unpred_flag = true; break;
                }
                abs_err_fp_q[p] = std::llabs(dec[p] - curv);
              }else{
                unpred_flag = true; break;
              }
            }

            if (unpred_flag){
              *(eb_pos++) = 0;
              unpred.push_back(U[v]);
              unpred.push_back(V[v]);
            }else{
              ++eb_pos;
              dq_pos += 2;
              *curU = dec[0];
              *curV = dec[1];

              // 仅用于编码端自检统计（浮点域）
              double abs_eb_fp = (double)abs_eb / (double)scale;
              double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
              double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
              enc_max_abs_eb_fp   = std::max(enc_max_abs_eb_fp, abs_eb_fp);
              enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
            }
          }else{
            *(eb_pos++) = 0;
            unpred.push_back(U[v]);
            unpred.push_back(V[v]);
          }
        } // encode one vertex
      } // j
    }   // i
  }     // t

  std::cerr << "[ENC] max abs_eb(fp) = " << enc_max_abs_eb_fp
            << ", max actual |err|(fp) = " << enc_max_real_err_fp << "\n";

  // ========== 5) 打包码流 ==========
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::printf("write scale = %ld\n", (long)scale);
  write_variable_to_dst(pos, base);       // 仍写入 base=2（便于兼容旧格式）
  write_variable_to_dst(pos, threshold);  // 幂指数量化阈值
  write_variable_to_dst(pos, intv_radius);

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);

  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::free(eb_quant_index);

  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  std::free(U_fp); std::free(V_fp);
  return compressed;
}

//添加了把接近0的vector直接lossless
template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb,                  // 全局绝对误差上限（浮点域）：max_eb(fp)=max_pwr_eb
    EbMode mode 
){
  using T = int64_t;
  const Size3 sz{ (int)r1,(int)r2,(int)r3 };
  const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

  // 1) 定点化（先乘后 llround）
  T *U_fp=(T*)std::malloc(N*sizeof(T));
  T *V_fp=(T*)std::malloc(N*sizeof(T));
  if(!U_fp || !V_fp){
    if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp);
    compressed_size=0; return nullptr;
  }
  T range=0;
  T scale = convert_to_fixed_point<T_data,T>(U, V, N, U_fp, V_fp, range);

  // 2) 预计算：全局 CP 面集合（一次性）
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();

  // === 覆盖调试：记录编码端“触达”的三角面 ===
  #ifndef CP_DEBUG_VISIT
  #define CP_DEBUG_VISIT 0   // 调试时开，性能敏感时关
  #endif
  #ifndef DEBUG_USE
  #define DEBUG_USE 0   
  #endif


  #if CP_DEBUG_VISIT
  // 1.1 记录：编码端枚举/检查过的三角面
  std::unordered_set<FaceKeySZ, FaceKeySZHash> visited_faces;

  // 1.2 小工具：规范化存一下 (a,b,c)
  auto MARK_FACE = [&](size_t a, size_t b, size_t c){
      visited_faces.emplace(a,b,c); // FaceKeySZ 构造里已做排序/规范化
  };

  // 1.3 帮助打印：把 vid -> (t,i,j)
  auto decode_tij = [&](size_t v){
      int t = (int)(v / (H*W));
      size_t rem = v - (size_t)t * H * W;
      int i = (int)(rem / W);
      int j = (int)(rem % W);
      return std::tuple<int,int,int>(t,i,j);
  };

  // 1.4（可选）分类一下是哪类面
  auto classify_face = [&](size_t a, size_t b, size_t c){
      auto [ta,ia,ja] = decode_tij(a);
      auto [tb,ib,jb] = decode_tij(b);
      auto [tc,ic,jc] = decode_tij(c);
      int st = (ta==tb) + (ta==tc) + (tb==tc);
      if (ta==tb && tb==tc) {
          // 同一层
          bool has_diag = ((ia!=ib)||(ja!=jb)) && ((ia!=ic)||(ja!=jc)) && ((ib!=ic)||(jb!=jc));
          return has_diag ? "layer(tri)" : "layer(?)";
      } else {
          // 跨层
          int cnt_t0 = (ta==tb) + (ta==tc) + (ta==ta); // 粗略判断：有2点在同一层
          (void)cnt_t0;
          return "slab";
      }
  };
  #endif

  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // 3) 量化/编码缓冲
  int* eb_quant_index  = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index= (int*)std::malloc(2*N*sizeof(int)); // U/V 交错
  double enc_max_abs_eb_fp   = 0.0; //编码端自检（浮点）
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred; unpred.reserve((N/10)*2);

  // 参数与量化槽
  const int base = 2;                             // 仍写头，方便向后兼容
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);

  // 把浮点域绝对误差上限转到定点域（LSB）
  // const T max_eb = (T) std::llround( (long double)max_pwr_eb * (long double)scale );

  T max_eb = 0;
  if(mode == EbMode::Relative){

    max_eb = max_pwr_eb * range; // 相对误差转绝对误差
    printf("Compression Using Relative Eb Mode!\n");
  }
  else if (mode == EbMode::Absolute){
    printf("Compression Using Absolute Eb Mode!\n");
    max_eb = max_pwr_eb * scale; // 浮点→定点
  }
  else{
    std::cerr << "Error: Unsupported EbMode!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  // T max_eb = range * max_pwr_eb;


  // 定点 LSB 的阈值（退化门控）
  const T threshold = 1;                          // 幂指数量化阈值（LSB）


  // 4) 逐顶点：枚举与该顶点相关的三角面 → 最小 eb
  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W; // 层间位移

  // 6 个平面邻接方向（左右、上下、主对角）
  const int di[6] = { 0,  0,  1, -1,  1, -1 }; //y-axis
  const int dj[6] = { 1, -1,  0,  0,  1, -1 }; //x-axis

  for (int t=0; t<(int)Tt; ++t){
    if(t % 100 == 0){
      printf("processing slice %d / %d\n", t, (int)Tt);
    }
    for (int i=0; i<(int)H; ++i){
      for (int j=0; j<(int)W; ++j){
        const size_t v = vid(t,i,j,sz);
        // 缓存原始定点值（预测/回写前）
        T *curU = U_fp + v;
        T *curV = V_fp + v;
        const T curU_val = *curU;
        const T curV_val = *curV;

        // —— 收集最小 eb——
        T required_eb = max_eb;

        // (A) 层内 t：影响 (i-1..i, j-1..j) 的 4 个 cell，每 cell 两三角
        for (int ci=i-1; ci<=i; ++ci){
          if (!in_range(ci, (int)H-1)) continue;
          for (int cj=j-1; cj<=j; ++cj){
            if (!in_range(cj, (int)W-1)) continue;

            size_t v00 = vid(t,ci,  cj,  sz);
            size_t v10 = vid(t,ci,  cj+1,sz); //v10=> x=1,y=0
            size_t v01 = vid(t,ci+1,cj,  sz);
            size_t v11 = vid(t,ci+1,cj+1,sz);

            // Upper: (v00,v01,v11)
            if (v==v00 || v==v01 || v==v11){
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v01,v11);
              #endif
              if (has_cp(cp_faces, v00,v01,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v01],U_fp[v00],
                                                    V_fp[v11],V_fp[v01],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
              # if DEBUG_USE
              if (v00 ==  44324190  || v01 ==  44324190 || v11 ==  44324190 ) {
                  std::cout << "consider face v00,v01,v11 (" << v00 << "," << v01 << "," << v11 << ") "
                            << " with eb = " << eb
                            << "current required_eb = " << required_eb
                            << std::endl;
              }
              # endif
            }
            // Lower: (v00,v10,v11)
            if (v==v00 || v==v10 || v==v11){
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v10,v11);
              #endif
              if (has_cp(cp_faces, v00,v10,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v10],U_fp[v00],
                                                    V_fp[v11],V_fp[v10],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
              # if DEBUG_USE
              if (v00 ==  44324190  || v10 ==  44324190 || v11 ==  44324190 ) {
                  std::cout << "consider face v00,v10,v11 (" << v00 << "," << v10 << "," << v11 << ") "
                            << " with eb = " << eb
                            << "current required_eb = " << required_eb
                            << std::endl;
              }
              # endif
            }
          }
        }
        
        // (B1) 侧面 [t, t+1]
        if (t < (int)Tt-1){
          for (int k=0; k<6; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a  = vid(t, i, j, sz);
            size_t b  = vid(t, ni,nj, sz);
            size_t ap = a + dv, bp = b + dv;
            if (k == 0 || k==3 || k==5){
              // (a,b,bp) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (a,bp,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,bp,ap);
              #endif
              if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[bp],U_fp[a],
                                                        V_fp[ap],V_fp[bp],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,ap) for k = 1,2,4
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (b,ap,bp) for k = 1,2,4 //好像与点无关
              // if (has_cp(cp_faces, b,ap,bp)) { required_eb = 0; goto after_min_eb; }
              //   {
              //     T eb = derive_cp_abs_eb_sos_online<T>(U_fp[b],U_fp[ap],U_fp[bp],
              //                                           V_fp[b],V_fp[ap],V_fp[bp]);
              //     if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
              //     if (eb < required_eb) required_eb = eb;
              //   }
            }

          }
        }

        // (B2) 侧面 [t-1, t]
        if (t > 0){
          for (int k=0; k<6 ; ++k){
            int ni = i + di[k], nj = j + dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a = vid(t,i,j,sz); //v
            size_t b = vid(t,ni,nj,sz);
            size_t ap = a - dv, bp = b - dv; //ap,bp为上一层
            if (k == 0 || k==3 || k==5){
              // (a,b,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,bp) for k = 1,2,4 //侧面上三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
                
              // (a,ap,bp) for k = 1,2,4 //侧面下三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,ap,bp);
              #endif
              if (has_cp(cp_faces, a,ap,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[ap],U_fp[a],
                                                        V_fp[bp],V_fp[ap],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            // int ni=i+di[k], nj=j+dj[k];
            // if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            // size_t a  = vid(t-1, i, j, sz);
            // size_t b  = vid(t-1, ni,nj, sz);
            // size_t ap = a + dv; // = v
            // size_t bp = b + dv;
            // if (k==0 || k==3 || k==5){
            //   //(ap,bp,a) for k = 0,3,5
            //   if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0; goto after_min_eb; }
            //   {
            //     T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[bp],U_fp[ap],
            //                                           V_fp[a],V_fp[bp],V_fp[ap]);
            //     if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
            //     if (eb < required_eb) required_eb = eb;
            //   }
            //   //(a,b,bp) for k = 0,3,5
            //   if (has_cp(cp_faces, a,b,bp)) { required_eb = 0; goto after_min_eb; }
            //   {
            //     T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[b],U_fp[bp],
            //                                           V_fp[a],V_fp[b],V_fp[bp]);
            //     if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
            //     if (eb < required_eb) required_eb = eb;
            //   }
            // }
            // else{
            //   // (ap,bp,b) for k = 1,2,4
            //   if (has_cp(cp_faces, ap,bp,b)) { required_eb = 0; goto after_min_eb; }
            //   {
            //     T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[bp],U_fp[b],
            //                                           V_fp[ap],V_fp[bp],V_fp[b]);
            //     if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
            //     if (eb < required_eb) required_eb = eb;
            //   }
            //   // (ap,a,b) for k = 1,2,4
            //   if (has_cp(cp_faces, ap,a,b)) { required_eb = 0; goto after_min_eb; }
            //   {
            //     T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[a],U_fp[b],
            //                                           V_fp[ap],V_fp[a],V_fp[b]);
            //     if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
            //     if (eb < required_eb) required_eb = eb;
            //   }
            // }
          }
        }

        #if 0

        // (B1) 侧面 [t, t+1]
        if (t < (int)Tt-1 && required_eb>0 && !degenerate_face){
          for (int k=0; k<6 && required_eb>0 && !degenerate_face; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;

            size_t a  = vid(t, i,  j,  sz);
            size_t b  = vid(t, ni, nj, sz);
            size_t ap = a + dv,       bp = b + dv;

            bool is_hor  = (ni==i && nj!=j);          // 横边：j±1
            bool is_ver  = (ni!=i && nj==j);          // 竖边：i±1
            bool is_diag = (ni!=i && nj!=j);          // 对角：i±1,j±1

            if (is_hor) {
              // —— 完全对齐 compute 2.1 —— (a,b,bp) & (a,bp,ap)
              if (has_cp(cp_faces, a,b,bp)) { required_eb=0; goto after_min_eb; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[b],U_fp[bp],
                                                      V_fp[a],V_fp[b],V_fp[bp]);
                if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, a,bp,ap)) { required_eb=0; goto after_min_eb; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[bp],U_fp[ap],
                                                      V_fp[a],V_fp[bp],V_fp[ap]);
                if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
                if (eb < required_eb) required_eb = eb;
              }
            } else if (is_ver || is_diag) {
              // —— 完全对齐 compute 2.2/2.3 —— (a,b,ap) & (b,ap,bp)
              if (has_cp(cp_faces, a,b,ap)) { required_eb=0; goto after_min_eb; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[b],U_fp[ap],
                                                      V_fp[a],V_fp[b],V_fp[ap]);
                if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, b,ap,bp)) { required_eb=0; goto after_min_eb; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[b],U_fp[ap],U_fp[bp],
                                                      V_fp[b],V_fp[ap],V_fp[bp]);
                if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }
        // (B2) 侧面 [t-1, t]
        if (t > 0 && required_eb>0 && !degenerate_face){
          for (int k=0; k<6 && required_eb>0 && !degenerate_face; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;

            size_t a  = vid(t-1, i,  j,  sz);   // t-1
            size_t b  = vid(t-1, ni, nj, sz);   // t-1 邻居
            size_t ap = a + dv,       bp = b + dv; // t

            bool is_hor  = (ni==i && nj!=j);
            bool is_ver  = (ni!=i && nj==j);
            bool is_diag = (ni!=i && nj!=j);

            if (is_hor) {
              // 与 B1 横边保持一致（同一两张面）
              if (has_cp(cp_faces, a,b,bp)) { required_eb=0; goto after_min_eb; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[b],U_fp[bp],
                                                      V_fp[a],V_fp[b],V_fp[bp]);
                if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, a,bp,ap)) { required_eb=0; goto after_min_eb; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[bp],U_fp[ap],
                                                      V_fp[a],V_fp[bp],V_fp[ap]);
                if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
                if (eb < required_eb) required_eb = eb;
              }
            } else if (is_ver || is_diag) {
              if (has_cp(cp_faces, a,b,ap)) { required_eb=0; goto after_min_eb; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[b],U_fp[ap],
                                                      V_fp[a],V_fp[b],V_fp[ap]);
                if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, b,ap,bp)) { required_eb=0; goto after_min_eb; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[b],U_fp[ap],U_fp[bp],
                                                      V_fp[b],V_fp[ap],V_fp[bp]);
                if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }
        
        // (B3) 对角侧面 [t, t+1] —— 与 precompute 2.3 对齐
        if (t < (int)Tt-1 && required_eb>0 && !degenerate_face){
          for (int ci=i-1; ci<=i && required_eb>0 && !degenerate_face; ++ci){
            if (!in_range(ci,(int)H-1)) continue;
            for (int cj=j-1; cj<=j && required_eb>0 && !degenerate_face; ++cj){
              if (!in_range(cj,(int)W-1)) continue;

              size_t a  = vid(t,   ci,   cj,   sz);
              size_t b  = vid(t,   ci+1, cj+1, sz);
              size_t ap = a + dv;
              size_t bp = b + dv;

              // (a, b, ap)
              if (v==a || v==b || v==ap){
                if (has_cp(cp_faces, a,b,ap)) { required_eb = 0; goto after_min_eb; }
                T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[b],U_fp[ap],
                                                      V_fp[a],V_fp[b],V_fp[ap]);
                if (eb < required_eb) required_eb = eb;
              }
              // (b, ap, bp)
              if (v==b || v==ap || v==bp){
                if (has_cp(cp_faces, b,ap,bp)) { required_eb = 0; goto after_min_eb; }
                T eb = derive_cp_abs_eb_sos_online<T>(U_fp[b],U_fp[ap],U_fp[bp],
                                                      V_fp[b],V_fp[ap],V_fp[bp]);
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }

        // (B3') 对角侧面 [t-1, t] —— 与 (B3) 对称
        if (t > 0 && required_eb>0 && !degenerate_face){
          for (int ci=i-1; ci<=i && required_eb>0 && !degenerate_face; ++ci){
            if (!in_range(ci,(int)H-1)) continue;
            for (int cj=j-1; cj<=j && required_eb>0 && !degenerate_face; ++cj){
              if (!in_range(cj,(int)W-1)) continue;

              size_t a  = vid(t-1, ci,   cj,   sz);
              size_t b  = vid(t-1, ci+1, cj+1, sz);
              size_t ap = a + dv;   // 当前层
              size_t bp = b + dv;

              // (a, b, ap)
              if (v==a || v==b || v==ap){
                if (has_cp(cp_faces, a,b,ap)) { required_eb = 0; goto after_min_eb; }
                T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a],U_fp[b],U_fp[ap],
                                                      V_fp[a],V_fp[b],V_fp[ap]);
                if (eb < required_eb) required_eb = eb;
              }
              // (b, ap, bp)
              if (v==b || v==ap || v==bp){
                if (has_cp(cp_faces, b,ap,bp)) { required_eb = 0; goto after_min_eb; }
                T eb = derive_cp_abs_eb_sos_online<T>(U_fp[b],U_fp[ap],U_fp[bp],
                                                      V_fp[b],V_fp[ap],V_fp[bp]);
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }
        #endif

        // (C) 内部剖分面：两片 ts ∈ {t, t-1}；每相邻 cell 的 Upper/Lower 各 2 面
        
        #if 0
        if (required_eb>0 && !degenerate_face){
          auto eval_internal = [&](int ts)->bool{
            if (!in_range(ts,(int)Tt-1)) return false;
            for (int ci=i-1; ci<=i; ++ci){
              if (!in_range(ci,(int)H-1)) continue;
              for (int cj=j-1; cj<=j; ++cj){
                if (!in_range(cj,(int)W-1)) continue;

                // ===== Upper prism: a=(v00,v01,v11), b=a+dv =====
                {
                  size_t a0 = vid(ts,ci,  cj,  sz);     // v00
                  size_t a1 = vid(ts,ci+1,cj,  sz);     // v01  (y+1,x)
                  size_t a2 = vid(ts,ci+1,cj+1,sz);     // v11
                  size_t b0 = a0 + dv, b1 = a1 + dv, b2 = a2 + dv;

                  // (a1, b0, a2)
                  if (v==a1 || v==b0 || v==a2){
                    if (has_cp(cp_faces, a1,b0,a2)) { required_eb=0; return true; }
                    T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a1],U_fp[b0],U_fp[a2],
                                                          V_fp[a1],V_fp[b0],V_fp[a2]);
                    if (eb <= degenerate_lsb){ degenerate_face = true; return true; }
                    if (eb < required_eb) required_eb = eb;
                    if (required_eb==0) return true;
                  }
                  // (a1, b0, b2)
                  if (v==a1 || v==b0 || v==b2){
                    if (has_cp(cp_faces, a1,b0,b2)) { required_eb=0; return true; }
                    T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a1],U_fp[b0],U_fp[b2],
                                                          V_fp[a1],V_fp[b0],V_fp[b2]);
                    if (eb <= degenerate_lsb){ degenerate_face = true; return true; }
                    if (eb < required_eb) required_eb = eb;
                    if (required_eb==0) return true;
                  }
                }

                // ===== Lower prism: a=(v00,v10,v11), b=a+dv =====
                {
                  size_t a0 = vid(ts,ci,  cj,  sz);     // v00
                  size_t a1 = vid(ts,ci,  cj+1,sz);     // v10  (y,x+1)
                  size_t a2 = vid(ts,ci+1,cj+1,sz);     // v11
                  size_t b0 = a0 + dv, b1 = a1 + dv, b2 = a2 + dv;

                  // (a0, a2, b1)
                  if (v==a0 || v==a2 || v==b1){
                    if (has_cp(cp_faces, a0,a2,b1)) { required_eb=0; return true; }
                    T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a0],U_fp[a2],U_fp[b1],
                                                          V_fp[a0],V_fp[a2],V_fp[b1]);
                  if (eb <= degenerate_lsb){ degenerate_face = true; return true; }
                    if (eb < required_eb) required_eb = eb;
                    if (required_eb==0) return true;
                  }
                  // (a2, b0, b1)
                  if (v==a2 || v==b0 || v==b1){
                    if (has_cp(cp_faces, a2,b0,b1)) { required_eb=0; return true; }
                    T eb = derive_cp_abs_eb_sos_online<T>(U_fp[a2],U_fp[b0],U_fp[b1],
                                                          V_fp[a2],V_fp[b0],V_fp[b1]);
                    if (eb <= degenerate_lsb){ degenerate_face = true; return true; }
                    if (eb < required_eb) required_eb = eb;
                    if (required_eb==0) return true;
                  }
                }
              }
            }
            return false;
          };
          if (eval_internal(t))   goto after_min_eb;
          if (eval_internal(t-1)) goto after_min_eb;
        }
        #endif
        // (C1) [t, t+1] 内部剖分面
        if (t < (int)Tt-1){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|
   
      
          //triange 1 [x,y,t] = [(0,0,0),(1,0,0),(0,-1,0)]: has 2 faces
          size_t f1a = vid(t,  i,  j,  sz);
          size_t f1b = vid(t,  i,j+1,  sz);
          size_t f1c = vid(t,  i-1,j,  sz);
          size_t f1ap = f1a + dv, f1bp = f1b + dv, f1cp = f1c + dv;
          if (in_range(i-1,(int)H) && in_range(j+1,(int)W)){
            // (f1a,f1cp,f1b)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1b);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1b)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1b],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1b],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f1a,f1cp,f1bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1bp);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1bp],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1bp],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          //triange 2 [x,y,t] = [(0,0,0),(0,-1,0),(-1,-1,0)]: has 2 faces
          size_t f2a = vid(t,  i,  j,  sz);
          size_t f2b = vid(t,  i-1,j,  sz);
          size_t f2c = vid(t,  i-1,j-1,sz);
          size_t f2ap = f2a + dv, f2bp = f2b + dv, f2cp = f2c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f2a,f2c,f2bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2c,f2bp);
            #endif
            if (has_cp(cp_faces, f2a,f2c,f2bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2bp],U_fp[f2c],U_fp[f2a],
                                                    V_fp[f2bp],V_fp[f2c],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f2a,f2bp,f2cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2bp,f2cp);
            #endif
            if (has_cp(cp_faces, f2a,f2bp,f2cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2cp],U_fp[f2bp],U_fp[f2a],
                                                    V_fp[f2cp],V_fp[f2bp],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          // triange 3 [x,y,t] = [(0,0,0),(-1,-1,0),(-1,0,0)]: has 1 faces
          size_t f3a = vid(t,  i,  j,  sz);
          size_t f3b = vid(t,  i-1,j-1,sz);
          size_t f3c = vid(t,  i,j-1,  sz);
          size_t f3ap = f3a + dv, f3bp = f3b + dv, f3cp = f3c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3bp,f3c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3bp,f3c);
            #endif
            if (has_cp(cp_faces, f3a,f3bp,f3c)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3c],U_fp[f3bp],U_fp[f3a],
                                                    V_fp[f3c],V_fp[f3bp],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // triange 4 0 faces
          // triange 5 0 faces
          // triange 6 [x,y,t] = [(0,0,0),(1,0,0),(1,1,0)]: has 1 faces
          size_t f6a = vid(t,  i,  j,  sz);
          size_t f6b = vid(t,  i,j+1,  sz);
          size_t f6c = vid(t,  i+1,j+1,sz);
          size_t f6ap = f6a + dv, f6bp = f6b + dv, f6cp = f6c + dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6bp,f6c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6bp,f6c);
            #endif
            if (has_cp(cp_faces, f6a,f6bp,f6c)) {
              required_eb=0; 
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6c],U_fp[f6bp],U_fp[f6a],
                                                    V_fp[f6c],V_fp[f6bp],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }


        // (C2) [t-1, t] 内部剖分面 并不与C1对称
        if (t > 0){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|

          //triange 1 0 face
          //triange 2 0 face
          //triange 3  has 1 faces
          size_t f3a = vid(t, i,  j,  sz);   
          size_t f3b = vid(t, i-1,j-1,sz);
          size_t f3c = vid(t, i,j-1,  sz);
          size_t f3ap = f3a - dv, f3bp = f3b - dv, f3cp = f3c - dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3b,f3cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3b,f3cp);
            #endif
            if (has_cp(cp_faces, f3a,f3b,f3cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3cp],U_fp[f3b],U_fp[f3a],
                                                    V_fp[f3cp],V_fp[f3b],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 4 has 2 faces
          size_t f4a = vid(t, i,  j,  sz);
          size_t f4b = vid(t, i,j-1,  sz);
          size_t f4c = vid(t,i+1,j,  sz);
          size_t f4ap = f4a - dv, f4bp = f4b - dv, f4cp = f4c - dv;
          if (in_range(i+1,(int)H) && in_range(j-1,(int)W)){
            // (f4a,f4bp,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4bp,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4bp,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4bp],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4bp],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f4a,f4b,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4b,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4b,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4b],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4b],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 5 has 2 faces
          size_t f5a = vid(t, i,  j,  sz);
          size_t f5b = vid(t, i+1,j,  sz);
          size_t f5c = vid(t, i+1,j+1,sz);
          size_t f5ap = f5a - dv, f5bp = f5b - dv, f5cp = f5c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f5a,f5bp,f5cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5bp,f5cp);
            #endif
            if (has_cp(cp_faces, f5a,f5bp,f5cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5cp],U_fp[f5bp],U_fp[f5a],
                                                    V_fp[f5cp],V_fp[f5bp],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f5a,f5c,f5bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5c,f5bp);
            #endif
            if (has_cp(cp_faces, f5a,f5c,f5bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5bp],U_fp[f5c],U_fp[f5a],
                                                    V_fp[f5bp],V_fp[f5c],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 6 has 1 faces
          size_t f6a = vid(t, i,  j,  sz);
          size_t f6b = vid(t, i,j+1,  sz);
          size_t f6c = vid(t, i+1,j+1,sz);
          size_t f6ap = f6a - dv, f6bp = f6b - dv, f6cp = f6c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6b,f6cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6b,f6cp);
            #endif
            if (has_cp(cp_faces, f6a,f6b,f6cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6cp],U_fp[f6b],U_fp[f6a],
                                                    V_fp[f6cp],V_fp[f6b],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }
        // {
        //   // 保证 |dec-orig| ≤ abs_eb < |orig| ⇒ 不会跨 0 翻符号
        //   T sign_guard = std::min(std::llabs(curU_val), std::llabs(curV_val));
        //   if (sign_guard > 0 && required_eb >= sign_guard){
        //       T max_eb_to_preserve_sign = sign_guard - 1; // 留 1 LSB 余量
        //       if (max_eb_to_preserve_sign < required_eb)
        //           required_eb = max_eb_to_preserve_sign;
        //   }
        // }
        
        #if DEBUG_USE
        if (v == 44324190){
          fprintf(stderr, "[DEBUG] v=%zu  required_eb=%lld\n",
                  v, (long long)required_eb);
        }
        #endif
        T abs_eb = required_eb;
        int id = eb_exponential_quantize(abs_eb,base,log_of_base,threshold);
        
        if (abs_eb == 0 )
        {
          *(eb_pos++) = 0;            // ebid=0 → 无损点
          unpred.push_back(U[v]);     // 原始浮点（解码端直接回填）
          unpred.push_back(V[v]);
          continue;
        }

        // ===== 量化 eb（配套的新幂指数量化）并编码该顶点 =====
        {
          // T abs_eb = required_eb;
          // // int id = eb_exponential_quantize_new(abs_eb, threshold); // id>=1，abs_eb 替换为代表值
          
          // // *eb_pos = id;
          // int id = eb_exponential_quantize(abs_eb,base,log_of_base,threshold);
          *eb_pos = id;

          bool unpred_flag=false;
          T dec[2];
          T abs_err_fp_q[2] = {0,0};

          for (int p=0; p<2; ++p){
            T *cur  = (p==0)? curU : curV;
            T  curv = (p==0) ? curU_val : curV_val;

            // 3D 一阶 Lorenzo（时间最慢）
            T d0 = (t&&i&&j)? cur[-sk - si - sj] : 0;
            T d1 = (t&&i)   ? cur[-sk - si]      : 0;
            T d2 = (t&&j)   ? cur[-sk - sj]      : 0;
            T d3 = (t)      ? cur[-sk]           : 0;
            T d4 = (i&&j)   ? cur[-si - sj]      : 0;
            T d5 = (i)      ? cur[-si]           : 0;
            T d6 = (j)      ? cur[-sj]           : 0;
            T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

            T diff = curv - pred;
            T qd = (std::llabs(diff)/abs_eb) + 1;
            if (qd < capacity){
              qd = (diff > 0) ? qd : -qd;
              int qindex = (int)(qd/2) + intv_radius;
              dq_pos[p] = qindex;
              dec[p] = pred + 2*(qindex - intv_radius)*abs_eb;

              // 守门：必须用 *代表值* abs_eb 校验
              if (std::llabs(dec[p] - curv) > abs_eb){
              // if (std::llabs(dec[p] - curv) > required_eb){
                unpred_flag = true; break;
              }
              abs_err_fp_q[p] = std::llabs(dec[p] - curv);
            }else{
              unpred_flag = true; break;
            }
          }

          if (unpred_flag){
            *(eb_pos++) = 0;                  // 改回无损
            unpred.push_back(U[v]);
            unpred.push_back(V[v]);
          }else{
            ++eb_pos;
            dq_pos += 2;
            *curU = dec[0];
            *curV = dec[1];

            // 编码端自检（浮点域）
            double abs_eb_fp = (double)abs_eb / (double)scale;
            double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
            double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
            enc_max_abs_eb_fp   = std::max(enc_max_abs_eb_fp, abs_eb_fp);
            enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
          }
        }
      }
    }
  }

  // std::cerr << "[ENC] max abs_eb(fp) = " << enc_max_abs_eb_fp
  //           << ", max actual |err|(fp) = " << enc_max_real_err_fp << "\n";

  
  #if CP_DEBUG_VISIT
  {
      size_t miss = 0, shown = 0;
      for (const auto &fk : cp_faces){
          if (!visited_faces.count(fk)){
              ++miss;
              if (shown < 50){ // 只演示前 50 个，避免刷屏
                  size_t a = fk.v[0], b = fk.v[1], c = fk.v[2];
                  auto [ta,ia,ja] = decode_tij(a);
                  auto [tb,ib,jb] = decode_tij(b);
                  auto [tc,ic,jc] = decode_tij(c);
                  fprintf(stderr,
                    "[MISSING] face (%zu,%zu,%zu)  "
                    "A(t=%d,i=%d,j=%d)  B(t=%d,i=%d,j=%d)  C(t=%d,i=%d,j=%d)  type=%s\n",
                    a,b,c, ta,ia,ja, tb,ib,jb, tc,ic,jc,
                    classify_face(a,b,c));
                  ++shown;
              }
          }
      }
      fprintf(stderr, "[COVERAGE] encoder touched faces = %zu, cp_faces = %zu, missing = %zu\n",
              visited_faces.size(), cp_faces.size(), miss);
  }
  #endif
  // 5) 打包码流
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);

  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::free(eb_quant_index);

  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  std::free(U_fp); std::free(V_fp);
  return compressed;
}


// template<typename T_data>
// unsigned char*
// sz_compress_cp_preserve_sos_2p5d_fp(
//     const T_data* U, const T_data* V,
//     size_t r1, size_t r2, size_t r3,      // r1=H, r2=W, r3=T
//     size_t& compressed_size,
//     EbMode mode,                          // 误差模式
//     double eb_param                       // Absolute: 相对range的因子；Relative: 相对比例 r
// ){
//     using T = int64_t;
//     const Size3 sz{ (int)r1,(int)r2,(int)r3 };
//     const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;
//     // 0) 定点化
//     T* U_fp=(T*)std::malloc(N*sizeof(T));
//     T* V_fp=(T*)std::malloc(N*sizeof(T));
//     if(!U_fp || !V_fp){ if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp); compressed_size=0; return nullptr; }
//     T range=0, scale=convert_to_fixed_point<T_data,T>(U,V,N,U_fp,V_fp,range);
//     // 参数
//     const int   base=2; const double log_of_base=std::log2(base);
//     const int   capacity=65536; const int intv_radius=(capacity>>1);
//     const T     threshold=1;
//     // const T     eb_floor = 1; // 防止过小 eb
//     const T     max_abs_eb = (mode==EbMode::Absolute)
//                             ? (T)std::llround((long double)range * (long double)eb_param)
//                             : std::numeric_limits<T>::max();
//     // 1) eb_min 预计算
//     std::vector<T> eb_min(N, max_abs_eb);
//     accumulate_eb_min_global_unique_with_internal(sz, U_fp, V_fp, eb_min);
//     // 2) Lorenzo + 量化 + Huffman
//     int* eb_q  = (int*)std::malloc(N*sizeof(int));
//     int* dq    = (int*)std::malloc(2*N*sizeof(int)); // U/V 交错
//     if(!eb_q || !dq){
//         if(eb_q) std::free(eb_q); if(dq) std::free(dq);
//         std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
//     }
//     int* eb_q_pos=eb_q; int* dq_pos=dq;
//     std::vector<T_data> unpred; unpred.reserve(N/10*2);
//     const ptrdiff_t si=W, sj=1, sk=(ptrdiff_t)(H*W);
//     for (int t=0; t<(int)Tt; ++t){
//         if (t % 100 == 0){
//             //printf("processing slab %d / %d\n", t, Tt);
//         }
//         for (int i=0; i<(int)H; ++i){
//             for (int j=0; j<(int)W; ++j){
//                 const ptrdiff_t v = (ptrdiff_t)vid(t,i,j,sz);
//                 // 误差设定：模式 → eb_setting
//                 T eb_setting = max_abs_eb;
//                 if (mode==EbMode::Relative){
//                     // 相对误差（基于定点幅度的 L_inf ）
//                     T amp = std::max<T>( std::llabs(U_fp[v]), std::llabs(V_fp[v]) );
//                     long double cand = (long double)amp * (long double)eb_param;
//                     if (cand > (long double)std::numeric_limits<T>::max()) cand = (long double)std::numeric_limits<T>::max();
//                     eb_setting = (T)std::llround(cand);
//                 }
//                 // CP 约束收紧 + floor
//                 T abs_eb = std::min<T>( eb_setting, eb_min[v] );
//                 // if (abs_eb < eb_floor) abs_eb = eb_floor;
//                 // 量化 eb（按引用，abs_eb 被替换为离散代表值）
//                 int eb_idx = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
//                 *eb_q_pos = eb_idx;
//                 if (abs_eb > 0){
//                     bool unp=false;
//                     T dec[2];
//                     const T* pos[2] = { U_fp+v, V_fp+v };
//                     for (int p=0; p<2; ++p){
//                         const T* cur = pos[p];
//                         T curv = *cur;
//                         // 3D 一阶 Lorenzo（时间最慢）
//                         T d0 = (t&&i&&j)? cur[-sk - si - sj]:0;
//                         T d1 = (t&&i)   ? cur[-sk - si]     :0;
//                         T d2 = (t&&j)   ? cur[-sk - sj]     :0;
//                         T d3 = (t)      ? cur[-sk]          :0;
//                         T d4 = (i&&j)   ? cur[-si - sj]     :0;
//                         T d5 = (i)      ? cur[-si]          :0;
//                         T d6 = (j)      ? cur[-sj]          :0;
//                         T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
//                         T diff = curv - pred;

//                         T qd = (std::llabs(diff)/abs_eb) + 1;
//                         if (qd < capacity){
//                             qd = (diff>0) ? qd : -qd;
//                             int qidx = int(qd/2) + intv_radius;
//                             dq_pos[p] = qidx;
//                             dec[p] = pred + 2*(qidx-intv_radius)*abs_eb;
//                             if (std::llabs(dec[p]-curv) >= abs_eb){ unp=true; break; }
//                         } else { unp=true; break; }
//                     }
//                     if (unp){
//                         *(eb_q_pos++) = 0;
//                         unpred.push_back(U[v]);
//                         unpred.push_back(V[v]);
//                     } else {
//                         ++eb_q_pos;
//                         dq_pos += 2;
//                         U_fp[v] = dec[0];
//                         V_fp[v] = dec[1];
//                     }
//                 }else{
//                     *(eb_q_pos++) = 0;
//                     unpred.push_back(U[v]);
//                     unpred.push_back(V[v]);
//                 }
//             }
//         }
//     }
//     // 3) 打包：加上误差模式与参数（便于排查 / 统计）
//     unsigned char* out = (unsigned char*)std::malloc( (size_t) (2*N*sizeof(T)) );
//     unsigned char* pos = out;
//     write_variable_to_dst(pos, scale);
//     write_variable_to_dst(pos, base);
//     write_variable_to_dst(pos, threshold);
//     write_variable_to_dst(pos, intv_radius);
//     // // 新增：误差模式与参数
//     // uint8_t mode_byte = static_cast<uint8_t>(mode);
//     // write_variable_to_dst(pos, mode_byte);
//     // double eb_param_d = eb_param;
//     // write_variable_to_dst(pos, eb_param_d);
//     size_t unpred_cnt = unpred.size();
//     write_variable_to_dst(pos, unpred_cnt);
//     if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);
//     size_t eb_num = (size_t)(eb_q_pos - eb_q);
//     write_variable_to_dst(pos, eb_num);
//     Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_q, eb_num, pos);
//     std::free(eb_q);
//     size_t dq_num = (size_t)(dq_pos - dq);
//     write_variable_to_dst(pos, dq_num);
//     Huffman_encode_tree_and_data(/*state_num=*/2*65536, dq, dq_num, pos);
//     std::free(dq);
//     compressed_size = (size_t)(pos - out);
//     std::free(U_fp); std::free(V_fp);
//     return out;
// }

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
    printf("read scale = %ld\n", scale);
    int base=0; read_variable_from_src(p, base);
    printf("read base = %d\n", base);
    T threshold=0; read_variable_from_src(p, threshold);
    printf("read threshold = %ld\n", threshold);
    int intv_radius=0; read_variable_from_src(p, intv_radius);
    printf("read intv_radius = %d\n", intv_radius);
    const int capacity = (intv_radius<<1);

    // // 读取误差模式与参数（当前不参与重建，仅用于一致性/调试）
    // uint8_t mode_byte=0; read_variable_from_src(p, mode_byte);
    // double eb_param=0.0; read_variable_from_src(p, eb_param);
    // (void)mode_byte; (void)eb_param;

    size_t unpred_cnt=0; read_variable_from_src(p, unpred_cnt);
    if (unpred_cnt % 2 != 0) return false;

    const T_data* unpred_data = reinterpret_cast<const T_data*>(p);
    const T_data* unpred_pos  = unpred_data;
    p += unpred_cnt * sizeof(T_data);

    size_t eb_num=0; read_variable_from_src(p, eb_num);
    int* eb_idx = Huffman_decode_tree_and_data(/*state_num=*/2*1024, eb_num, p);
    if (!eb_idx) return false;

    size_t dq_num=0; read_variable_from_src(p, dq_num);
    int* dq = Huffman_decode_tree_and_data(/*state_num=*/2*65536, dq_num, p);
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
                    *U_pos = *(unpred_pos++) * scale;
                    *V_pos = *(unpred_pos++) * scale;
                }

                else {
                    T abs_eb = pow(base,ebid) * threshold;
                    //long double eb_ld = std::pow((long double)base, (long double)ebid) * (long double)threshold;
                    //T abs_eb = (T)std::llround(eb_ld);

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
                        // *cur = pred + (T) (2LL * ( (long long)qidx - (long long)intv_radius ) ) * abs_eb;
                        *cur = pred + 2* (qidx - intv_radius) * abs_eb;
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

template<typename T_data>
bool sz_decompress_cp_preserve_sos_2p5d_fp_latest(
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
    int base=0; read_variable_from_src(p, base);           (void)base;       // 保持头一致，但不再用于反量化
    T threshold=0; read_variable_from_src(p, threshold);
    int intv_radius=0; read_variable_from_src(p, intv_radius);
    const int capacity = (intv_radius<<1);                  (void)capacity;

    size_t unpred_cnt=0; read_variable_from_src(p, unpred_cnt);
    if (unpred_cnt % 2 != 0) return false;

    const T_data* unpred_data = reinterpret_cast<const T_data*>(p);
    const T_data* unpred_pos  = unpred_data;
    p += unpred_cnt * sizeof(T_data);

    size_t eb_num=0; read_variable_from_src(p, eb_num);
    int* eb_idx = Huffman_decode_tree_and_data(/*state_num=*/2*1024, eb_num, p);
    if (!eb_idx) return false;

    size_t dq_num=0; read_variable_from_src(p, dq_num);
    int* dq = Huffman_decode_tree_and_data(/*state_num=*/2*65536, dq_num, p);
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

                int64_t abs_eb = 0;
                if (!eb_dequant_from_id(ebid, threshold, abs_eb)) {
                    // ebid==0 → 无损点：直接用原始浮点（按写入顺序 U,V）
                    size_t off = (size_t)(U_pos - U_fp);
                    unpred_indices.push_back(off);
                    // // 回到定点（与编码端 convert_to_fixed_point 的 llround 对齐）
                    *U_pos = (T)std::llround((long double)(*unpred_pos++) * (long double)scale);
                    *V_pos = (T)std::llround((long double)(*unpred_pos++) * (long double)scale);
                } else {
                    // 预测重建（3D Lorenzo）
                    for (int pcomp=0; pcomp<2; ++pcomp){
                        T* cur = (pcomp==0) ? U_pos : V_pos;
                        T d0 = (t&&i&&j)? cur[-sk - si - sj]:0;
                        T d1 = (t&&i)   ? cur[-sk - si]     :0;
                        T d2 = (t&&j)   ? cur[-sk - sj]     :0;
                        T d3 = (t)      ? cur[-sk]          :0;
                        T d4 = (i&&j)   ? cur[-si - sj]     :0;
                        T d5 = (i)      ? cur[-si]          :0;
                        T d6 = (j)      ? cur[-sj]          :0;
                        T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

                        int qidx = *dq_pos++;
                        *cur = pred + 2 * ( (T)qidx - (T)intv_radius ) * abs_eb;
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
    #ifdef _OPENMP
    printf("Using OpenMP with %d threads\n", omp_get_max_threads());
    #else
    printf("OpenMP is not enabled.\n");
    #endif
    // args: U_file V_file H W T [mode] [eb]
    // example : ./test ~/data/CBA_full/u.bin ~/data/CBA_full/v.bin 450 150 2001 abs 0.005
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " U_file V_file H W T [mode=abs/rel] [eb=0.01]\n";
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
    std::cout << "U_file: " << u_path << ", V_file: " << v_path << "\n";
    std::cout << "Dimensions: H=" << dim_H << ", W=" << dim_W << ", T=" << dim_T << "\n";
    std::cout << "Error mode: " << (mode == EbMode::Absolute ? "Absolute" : "Relative") << ", Error bound: " << error_bound << "\n";
    //read data
    size_t num_elements =0;
    float * U_ptr = readfile<float>(u_path.c_str(),num_elements);
    float * V_ptr = readfile<float>(v_path.c_str(),num_elements);
    //调用压缩
    auto start_t = std::chrono::high_resolution_clock::now();
    size_t compressed_size = 0;
    // unsigned char* compressed =
    //     sz_compress_cp_preserve_sos_2p5d_fp<float>(
    // U_ptr, V_ptr, 450, 150, 2001, compressed_size, mode,error_bound);
    
    //把dimension当作filename转换成string
    std::string dim_str = std::to_string(dim_W) + "x" + std::to_string(dim_H) + "x" + std::to_string(dim_T);

    auto* compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap(
    U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T, // 450,150,2001
    compressed_size, error_bound,mode);

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
    cout << "compressed size (after lossless): " << lossless_outsize << " bytes." << "ratio = " << (2*num_elements*sizeof(float))/double(lossless_outsize) << endl;
    
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
    sz_decompress_cp_preserve_sos_2p5d_fp<float>(decompressed, H, W, T, U_dec, V_dec);
    auto dec_end_t = std::chrono::high_resolution_clock::now();
    cout << "decompression time in sec:" << std::chrono::duration<double>(dec_end_t - dec_start_t).count() << endl;

    //verify
    printf("start verify....\n");
    float * U_ori = readfile<float>(u_path.c_str(),num_elements);
    float * V_ori = readfile<float>(v_path.c_str(),num_elements);
    verify(U_ori, V_ori, U_dec, V_dec, H, W, T);
    //check if pre-compute result consistent    
    int64_t *U_ori_fp=(int64_t*)std::malloc(num_elements*sizeof(int64_t));
    int64_t *V_ori_fp=(int64_t*)std::malloc(num_elements*sizeof(int64_t));
    int64_t range_ori=0;
    int64_t scale_ori = convert_to_fixed_point(U_ori,V_ori,num_elements,U_ori_fp,V_ori_fp,range_ori);
    printf("original scale = %ld\n", scale_ori);
    auto cp_faces_ori = compute_cp_2p5d_faces(U_ori_fp, V_ori_fp, H, W, T,dim_str);


    int64_t *U_dec_fp=(int64_t*)std::malloc(num_elements*sizeof(int64_t));
    int64_t *V_dec_fp=(int64_t*)std::malloc(num_elements*sizeof(int64_t));
    convert_to_fixed_point_given_factor(U_dec,V_dec,num_elements,U_dec_fp,V_dec_fp,scale_ori);

    auto cp_faces_dec = compute_cp_2p5d_faces(U_dec_fp, V_dec_fp, H, W, T);

    std::cout << "Total faces with CP dec: " << cp_faces_dec.size() << std::endl;
    // std::cout << (are_unordered_sets_equal(cp_faces_dec, cp_faces_ori) ? "CP face sets are equal." : "CP face sets differ!") << std::endl;
    bool cp_equal = are_unordered_sets_equal(cp_faces_dec, cp_faces_ori);
    std::cout << (cp_equal ? "CP face sets are equal." : "CP face sets differ!") << std::endl;
    if (!cp_equal) {
        print_cp_face_diffs(cp_faces_dec, cp_faces_ori, U_dec, V_dec, U_ori, V_ori,U_dec_fp, V_dec_fp, U_ori_fp, V_ori_fp, H, W, T);
    }
    exit(0);

    // printf("calculating cp_count for decompressed data...\n");
    // const Size3 sz{ (int)H,(int)W,(int)T };
    // int64_t* U_fp=(int64_t*)std::malloc(num_elements*sizeof(int64_t));
    // int64_t* V_fp=(int64_t*)std::malloc(num_elements*sizeof(int64_t));
    // int64_t range=0, scale=convert_to_fixed_point<float,int64_t>(U_dec,V_dec,num_elements,U_fp,V_fp,range);
    // std::vector<int64_t> eb_min(num_elements, 0.01);
    // accumulate_eb_min_global_unique_with_internal(sz, U_fp, V_fp, eb_min);


    return 0;
}
