#define CP_DEBUG_VISIT 0
#define DEBUG_USE 0
#define VERBOSE 0
#define VERIFY 1 //验证
#define DETAIL 0 //详细
#define CPSZ_BASELINE 0 //baseline results for cpsz
#define STREAMING 0 //streaming mode
#define VISUALIZE 0 //visualization mode

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

struct BlockCoreRegion {
  int t_begin = 0, t_end = 0;
  int i_begin = 0, i_end = 0;
  int j_begin = 0, j_end = 0;
};

inline void inv_vid(size_t vid, int H, int W, int &t, int &i, int &j);

struct HaloEvalContext {
  const Size3& sz;
  const BlockCoreRegion* core; // nullptr 表示无需halo判定

  inline bool vertex_within_core(size_t vidx) const {
    if (!core) return true;
    int t,i,j;
    inv_vid(vidx, sz.H, sz.W, t, i, j);
    return t >= core->t_begin && t < core->t_end &&
           i >= core->i_begin && i < core->i_end &&
           j >= core->j_begin && j < core->j_end;
  }

  inline bool touches_halo(std::initializer_list<size_t> vertices) const {
    if (!core) return false;
    for (auto v : vertices) {
      if (!vertex_within_core(v)) return true;
    }
    return false;
  }
};

template<typename T>
inline T derive_cp_abs_eb_guarded(size_t a, size_t b, size_t c,
                                  const T* U_fp, const T* V_fp,
                                  const HaloEvalContext& halo_ctx,
                                  bool& force_lossless)
{
  if (halo_ctx.touches_halo({a, b, c})) {
    force_lossless = true;
    return static_cast<T>(0);
  }
  force_lossless = false;
  return derive_cp_abs_eb_sos_online<T>(U_fp[a], U_fp[b], U_fp[c],
                                        V_fp[a], V_fp[b], V_fp[c]);
}

static inline bool has_cp(const std::unordered_set<FaceKeySZ,FaceKeySZHash>& cp_faces,
                   size_t a,size_t b,size_t c);


template<typename T>
static T compute_required_eb_for_vertex(
    size_t t, size_t i, size_t j,
    const Size3& sz,
    const std::unordered_set<FaceKeySZ, FaceKeySZHash>& cp_faces,
    const HaloEvalContext& halo_ctx,
    T max_eb,
    const T* U_fp, const T* V_fp)
{
  const ptrdiff_t si = static_cast<ptrdiff_t>(sz.W);
  const ptrdiff_t sj = static_cast<ptrdiff_t>(1);
  const ptrdiff_t sk = static_cast<ptrdiff_t>(sz.H) * static_cast<ptrdiff_t>(sz.W);
  const size_t dv = static_cast<size_t>(sz.H) * static_cast<size_t>(sz.W);

  auto has_cp_face = [&](size_t a, size_t b, size_t c) {
    return has_cp(cp_faces, a, b, c);
  };

  auto update_required_eb = [&](size_t a, size_t b, size_t c, T& required_eb) {
    bool force_lossless = false;
    T eb = derive_cp_abs_eb_guarded<T>(a, b, c, U_fp, V_fp, halo_ctx, force_lossless);
    if (force_lossless) {
      required_eb = 0;
      return;
    }
    if (eb < required_eb) required_eb = eb;
  };

  auto early_exit = [&](T required_eb) {
    return required_eb == 0;
  };

  auto in_range_int = [](int pos, int n) {
    return pos >= 0 && pos < n;
  };

  T required_eb = max_eb;

  const size_t v = vid(static_cast<int>(t), static_cast<int>(i), static_cast<int>(j), sz);

  // (A) layer-internal faces (t)
  for (int ci = static_cast<int>(i) - 1; ci <= static_cast<int>(i); ++ci) {
    if (!in_range_int(ci, sz.H - 1)) continue;
    for (int cj = static_cast<int>(j) - 1; cj <= static_cast<int>(j); ++cj) {
      if (!in_range_int(cj, sz.W - 1)) continue;

      size_t v00 = vid(static_cast<int>(t), ci,     cj,     sz);
      size_t v10 = vid(static_cast<int>(t), ci,     cj + 1, sz);
      size_t v01 = vid(static_cast<int>(t), ci + 1, cj,     sz);
      size_t v11 = vid(static_cast<int>(t), ci + 1, cj + 1, sz);

      if (v == v00 || v == v01 || v == v11) {
        if (has_cp_face(v00, v01, v11)) {
          required_eb = 0;
        } else {
          update_required_eb(v11, v01, v00, required_eb);
        }
        if (early_exit(required_eb)) return required_eb;
      }
      if (v == v00 || v == v10 || v == v11) {
        if (has_cp_face(v00, v10, v11)) {
          required_eb = 0;
        } else {
          update_required_eb(v11, v10, v00, required_eb);
        }
        if (early_exit(required_eb)) return required_eb;
      }
    }
  }

  static const int di[6] = {0, 0, 1, -1, 1, -1};
  static const int dj[6] = {1, -1, 0, 0, 1, -1};

  // (B1) side faces between t and t+1
  if (static_cast<int>(t) < sz.T - 1) {
    for (int k = 0; k < 6; ++k) {
      int ni = static_cast<int>(i) + di[k];
      int nj = static_cast<int>(j) + dj[k];
      if (!in_range_int(ni, sz.H) || !in_range_int(nj, sz.W)) continue;
      size_t a = vid(static_cast<int>(t), static_cast<int>(i), static_cast<int>(j), sz);
      size_t b = vid(static_cast<int>(t), ni, nj, sz);
      size_t ap = a + dv;
      size_t bp = b + dv;

      if (k == 0 || k == 3 || k == 5) {
        if (has_cp_face(a, b, bp)) {
          required_eb = 0;
        } else {
          update_required_eb(bp, b, a, required_eb);
        }
        if (early_exit(required_eb)) return required_eb;

        if (has_cp_face(a, bp, ap)) {
          required_eb = 0;
        } else {
          update_required_eb(ap, bp, a, required_eb);
        }
        if (early_exit(required_eb)) return required_eb;
      } else {
        if (has_cp_face(a, b, ap)) {
          required_eb = 0;
        } else {
          update_required_eb(ap, b, a, required_eb);
        }
        if (early_exit(required_eb)) return required_eb;

        if (has_cp_face(b, ap, bp)) {
          required_eb = 0;
        } else {
          update_required_eb(bp, ap, b, required_eb);
        }
        if (early_exit(required_eb)) return required_eb;
      }
    }

    for (int k = 0; k < 6; ++k) {
      int ni = static_cast<int>(i) + di[k];
      int nj = static_cast<int>(j) + dj[k];
      if (!in_range_int(ni, sz.H) || !in_range_int(nj, sz.W)) continue;
      size_t a = vid(static_cast<int>(t), static_cast<int>(i), static_cast<int>(j), sz);
      size_t b = vid(static_cast<int>(t), ni, nj, sz);
      size_t ap = a + dv;
      size_t bp = b + dv;

      if (k == 0 || k == 3 || k == 5) {
        if (has_cp_face(b, a, bp)) {
          required_eb = 0;
        } else {
          update_required_eb(bp, a, b, required_eb);
        }
        if (early_exit(required_eb)) return required_eb;

        if (has_cp_face(bp, a, ap)) {
          required_eb = 0;
        } else {
          update_required_eb(ap, a, bp, required_eb);
        }
        if (early_exit(required_eb)) return required_eb;
      } else {
        if (has_cp_face(ap, a, b)) {
          required_eb = 0;
        } else {
          update_required_eb(b, a, ap, required_eb);
        }
        if (early_exit(required_eb)) return required_eb;

        if (has_cp_face(bp, ap, b)) {
          required_eb = 0;
        } else {
          update_required_eb(b, ap, bp, required_eb);
        }
        if (early_exit(required_eb)) return required_eb;
      }
    }
  }

  // (C1) interior faces for slab [t, t+1]
  if (static_cast<int>(t) < sz.T - 1) {
    if (in_range_int(static_cast<int>(i) - 1, sz.H) && in_range_int(static_cast<int>(j) + 1, sz.W)) {
      size_t f1a = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j),     sz);
      size_t f1b = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j + 1), sz);
      size_t f1c = vid(static_cast<int>(t), static_cast<int>(i - 1), static_cast<int>(j),     sz);
      size_t f1cp = f1c + dv;
      size_t f1bp = f1b + dv;
      if (has_cp_face(f1a, f1cp, f1b)) {
        required_eb = 0;
      } else {
        update_required_eb(f1b, f1cp, f1a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;

      if (has_cp_face(f1a, f1cp, f1bp)) {
        required_eb = 0;
      } else {
        update_required_eb(f1bp, f1cp, f1a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;
    }

    if (in_range_int(static_cast<int>(i) - 1, sz.H) && in_range_int(static_cast<int>(j) - 1, sz.W)) {
      size_t f2a = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j),     sz);
      size_t f2b = vid(static_cast<int>(t), static_cast<int>(i - 1), static_cast<int>(j),     sz);
      size_t f2c = vid(static_cast<int>(t), static_cast<int>(i - 1), static_cast<int>(j - 1), sz);
      size_t f2bp = f2b + dv;
      size_t f2cp = f2c + dv;
      if (has_cp_face(f2a, f2c, f2bp)) {
        required_eb = 0;
      } else {
        update_required_eb(f2bp, f2c, f2a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;

      if (has_cp_face(f2a, f2bp, f2cp)) {
        required_eb = 0;
      } else {
        update_required_eb(f2cp, f2bp, f2a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;
    }

    if (in_range_int(static_cast<int>(i) - 1, sz.H) && in_range_int(static_cast<int>(j) - 1, sz.W)) {
      size_t f3a = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j),     sz);
      size_t f3b = vid(static_cast<int>(t), static_cast<int>(i - 1), static_cast<int>(j - 1), sz);
      size_t f3c = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j - 1), sz);
      size_t f3bp = f3b + dv;
      if (has_cp_face(f3a, f3bp, f3c)) {
        required_eb = 0;
      } else {
        update_required_eb(f3c, f3bp, f3a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;
    }

    if (in_range_int(static_cast<int>(i) + 1, sz.H) && in_range_int(static_cast<int>(j) + 1, sz.W)) {
      size_t f6a = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j),     sz);
      size_t f6b = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j + 1), sz);
      size_t f6c = vid(static_cast<int>(t), static_cast<int>(i + 1), static_cast<int>(j + 1), sz);
      size_t f6bp = f6b + dv;
      if (has_cp_face(f6a, f6bp, f6c)) {
        required_eb = 0;
      } else {
        update_required_eb(f6c, f6bp, f6a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;
    }
  }

  // (C2) slab [t-1, t]
  if (static_cast<int>(t) > 0) {
    if (in_range_int(static_cast<int>(i) - 1, sz.H) && in_range_int(static_cast<int>(j) - 1, sz.W)) {
      size_t f3a = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j),     sz);
      size_t f3b = vid(static_cast<int>(t), static_cast<int>(i - 1), static_cast<int>(j - 1), sz);
      size_t f3c = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j - 1), sz);
      size_t f3cp = f3c - dv;
      if (has_cp_face(f3a, f3b, f3cp)) {
        required_eb = 0;
      } else {
        update_required_eb(f3cp, f3b, f3a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;
    }

    if (in_range_int(static_cast<int>(i) + 1, sz.H) && in_range_int(static_cast<int>(j) - 1, sz.W)) {
      size_t f4a = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j),     sz);
      size_t f4b = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j - 1), sz);
      size_t f4c = vid(static_cast<int>(t), static_cast<int>(i + 1), static_cast<int>(j),     sz);
      size_t f4bp = f4b - dv;
      size_t f4cp = f4c - dv;
      if (has_cp_face(f4a, f4bp, f4cp)) {
        required_eb = 0;
      } else {
        update_required_eb(f4cp, f4bp, f4a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;

      if (has_cp_face(f4a, f4b, f4cp)) {
        required_eb = 0;
      } else {
        update_required_eb(f4cp, f4b, f4a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;
    }

    if (in_range_int(static_cast<int>(i) + 1, sz.H) && in_range_int(static_cast<int>(j) + 1, sz.W)) {
      size_t f5a = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j),     sz);
      size_t f5b = vid(static_cast<int>(t), static_cast<int>(i + 1), static_cast<int>(j),     sz);
      size_t f5c = vid(static_cast<int>(t), static_cast<int>(i + 1), static_cast<int>(j + 1), sz);
      size_t f5bp = f5b - dv;
      size_t f5cp = f5c - dv;
      if (has_cp_face(f5a, f5bp, f5cp)) {
        required_eb = 0;
      } else {
        update_required_eb(f5cp, f5bp, f5a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;

      if (has_cp_face(f5a, f5c, f5bp)) {
        required_eb = 0;
      } else {
        update_required_eb(f5bp, f5c, f5a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;
    }

    if (in_range_int(static_cast<int>(i) + 1, sz.H) && in_range_int(static_cast<int>(j) + 1, sz.W)) {
      size_t f6a = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j),     sz);
      size_t f6b = vid(static_cast<int>(t), static_cast<int>(i),     static_cast<int>(j + 1), sz);
      size_t f6c = vid(static_cast<int>(t), static_cast<int>(i + 1), static_cast<int>(j + 1), sz);
      size_t f6cp = f6c - dv;
      if (has_cp_face(f6a, f6b, f6cp)) {
        required_eb = 0;
      } else {
        update_required_eb(f6cp, f6b, f6a, required_eb);
      }
      if (early_exit(required_eb)) return required_eb;
    }
  }

  return required_eb;
}

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

// 统计混淆矩阵和每个时间戳的细节
struct ConfusionCounts {
  size_t tp = 0;
  size_t tn = 0;
  size_t fp = 0;
  size_t fn = 0;
};

static inline void accumulate_confusion(ConfusionCounts& c, bool in_ori, bool in_dec) {
  if (in_ori && in_dec)        c.tp++;
  else if (!in_ori && !in_dec) c.tn++;
  else if (in_dec)             c.fp++;
  else                         c.fn++;
}

static void summarize_cp_faces_per_layer_and_slab(
    const std::unordered_set<FaceKeySZ, FaceKeySZHash>& cp_faces_dec,
    const std::unordered_set<FaceKeySZ, FaceKeySZHash>& cp_faces_ori,
    int H, int W, int T,
    const std::string& csv_path)
{
  if (H < 2 || W < 2 || T < 1) {
    std::cout << "[VERIFY] CP summary skipped (invalid dimensions).\n";
    return;
  }

  const Size3 sz{H, W, T};
  std::vector<ConfusionCounts> layer_counts((size_t)T);
  std::vector<ConfusionCounts> slab_counts(T > 1 ? (size_t)(T - 1) : size_t(0));

  auto classify_layer = [&](int layer, const FaceKeySZ& face) {
    const bool in_ori = cp_faces_ori.find(face) != cp_faces_ori.end();
    const bool in_dec = cp_faces_dec.find(face) != cp_faces_dec.end();
    accumulate_confusion(layer_counts[(size_t)layer], in_ori, in_dec);
  };

  auto classify_slab = [&](int slab, const FaceKeySZ& face) {
    const bool in_ori = cp_faces_ori.find(face) != cp_faces_ori.end();
    const bool in_dec = cp_faces_dec.find(face) != cp_faces_dec.end();
    accumulate_confusion(slab_counts[(size_t)slab], in_ori, in_dec);
  };

  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < H - 1; ++i) {
      for (int j = 0; j < W - 1; ++j) {
        size_t v00 = vid(t, i,     j,     sz);
        size_t v10 = vid(t, i,     j + 1, sz);
        size_t v01 = vid(t, i + 1, j,     sz);
        size_t v11 = vid(t, i + 1, j + 1, sz);

        classify_layer(t, FaceKeySZ(v00, v01, v11));
        classify_layer(t, FaceKeySZ(v00, v10, v11));
      }
    }
  }

  for (int t = 0; t < T - 1; ++t) {
    const size_t dv = (size_t)H * (size_t)W;

    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W - 1; ++j) {
        size_t a  = vid(t, i, j, sz);
        size_t b  = vid(t, i, j + 1, sz);
        size_t ap = a + dv, bp = b + dv;
        classify_slab(t, FaceKeySZ(a, b, bp));
        classify_slab(t, FaceKeySZ(a, bp, ap));
      }
    }

    for (int i = 0; i < H - 1; ++i) {
      for (int j = 0; j < W; ++j) {
        size_t a  = vid(t, i,     j, sz);
        size_t b  = vid(t, i + 1, j, sz);
        size_t ap = a + dv, bp = b + dv;
        classify_slab(t, FaceKeySZ(a, b, bp));
        classify_slab(t, FaceKeySZ(a, bp, ap));
      }
    }

    for (int i = 0; i < H - 1; ++i) {
      for (int j = 0; j < W - 1; ++j) {
        size_t ax0y0 = vid(t, i,     j,     sz);
        size_t ax1y1 = vid(t, i + 1, j + 1, sz);
        size_t bx0y0 = ax0y0 + dv, bx1y1 = ax1y1 + dv;
        classify_slab(t, FaceKeySZ(ax0y0, ax1y1, bx0y0));
        classify_slab(t, FaceKeySZ(ax1y1, bx0y0, bx1y1));
      }
    }

    for (int i = 0; i < H - 1; ++i) {
      for (int j = 0; j < W - 1; ++j) {
        size_t ax0y0 = vid(t, i,     j,     sz);
        size_t ax0y1 = vid(t, i + 1, j,     sz);
        size_t ax1y1 = vid(t, i + 1, j + 1, sz);
        size_t bx0y0 = ax0y0 + dv;
        size_t bx0y1 = ax0y1 + dv;
        size_t bx1y1 = ax1y1 + dv;

        classify_slab(t, FaceKeySZ(ax0y1, bx0y0, ax1y1));
        classify_slab(t, FaceKeySZ(ax0y1, bx0y0, bx1y1));

        size_t ax1y0 = vid(t, i,     j + 1, sz);
        size_t bx1y0 = ax1y0 + dv;

        classify_slab(t, FaceKeySZ(ax0y0, ax1y1, bx1y0));
        classify_slab(t, FaceKeySZ(ax1y1, bx0y0, bx1y0));
      }
    }
  }

  #if VERBOSE
  std::cout << "[VERIFY] per-layer CP confusion (surface faces):\n";
  for (int t = 0; t < T; ++t) {
    const auto& c = layer_counts[(size_t)t];
    std::cout << "  layer " << t << ": TP=" << c.tp
              << ", TN=" << c.tn
              << ", FP=" << c.fp
              << ", FN=" << c.fn << "\n";
  }

  if (!slab_counts.empty()) {
    std::cout << "[VERIFY] per-slab CP confusion (between t and t+1):\n";
    for (int t = 0; t < T - 1; ++t) {
      const auto& c = slab_counts[(size_t)t];
      std::cout << "  slab " << t << "->" << (t + 1)
                << ": TP=" << c.tp
                << ", TN=" << c.tn
                << ", FP=" << c.fp
                << ", FN=" << c.fn << "\n";
    }
  }
  #endif

  if (!csv_path.empty()) {
    std::ofstream ofs(csv_path);
    if (!ofs) {
      std::cerr << "[VERIFY] failed to open " << csv_path << " for writing.\n";
    } else {
      ofs << "region,index,tp,tn,fp,fn\n";
      for (size_t idx = 0; idx < layer_counts.size(); ++idx) {
        const auto& c = layer_counts[idx];
        ofs << "layer," << idx << "," << c.tp << "," << c.tn << "," << c.fp << "," << c.fn << "\n";
      }
      for (size_t idx = 0; idx < slab_counts.size(); ++idx) {
        const auto& c = slab_counts[idx];
        ofs << "slab," << idx << "," << c.tp << "," << c.tn << "," << c.fp << "," << c.fn << "\n";
      }
    }
    std::cout << "[VERIFY] CP confusion summary written to " << csv_path << "\n";
  }
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
compute_cp_2p5d_faces_parallel(const T_fp* U_fp, const T_fp* V_fp,
                               int H, int W, int T,
                               const Size3& sz,
                               size_t dv,
                               std::vector<int>& cp_per_layer)
{
  std::unordered_set<FaceKeySZ, FaceKeySZHash> faces_with_cp;
  faces_with_cp.reserve((size_t)(H*(size_t)W*(size_t)T / 8));
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
  return faces_with_cp;
}

template<typename T_fp>
static std::unordered_set<FaceKeySZ, FaceKeySZHash>
compute_cp_2p5d_faces_serial(const T_fp* U_fp, const T_fp* V_fp,
                             int H, int W, int T,
                             const Size3& sz,
                             size_t dv)
{
  std::unordered_set<FaceKeySZ, FaceKeySZHash> faces_with_cp;
  faces_with_cp.reserve((size_t)(H*(size_t)W*(size_t)T / 8));
    // ---------------- (1) 层内面：每一层 t ∈ [0..T-1] ----------------
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < H - 1; ++i) {
        for (int j = 0; j < W - 1; ++j) {
          size_t v00 = vid(t, i, j, sz);
          size_t v10 = vid(t, i, j + 1, sz);
          size_t v01 = vid(t, i + 1, j, sz);
          size_t v11 = vid(t, i + 1, j + 1, sz);

          if (face_has_cp_robust(v00, v01, v11, U_fp, V_fp)) {
            faces_with_cp.emplace(v00, v01, v11);
          }
          if (face_has_cp_robust(v00, v10, v11, U_fp, V_fp)) {
            faces_with_cp.emplace(v00, v10, v11);
          }
        }
      }
    }

    // ---------------- (2) 侧面：片 [t, t+1]，t ∈ [0..T-2] ----------------
    for (int t = 0; t < T - 1; ++t) {
      for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W - 1; ++j) {
          const size_t a = vid(t, i, j, sz);
          const size_t b = vid(t, i, j + 1, sz);
          const size_t ap = a + dv;
          const size_t bp = b + dv;
          if (face_has_cp_robust(a, b, bp, U_fp, V_fp)) {
            faces_with_cp.emplace(a, b, bp);
          }
          if (face_has_cp_robust(a, bp, ap, U_fp, V_fp)) {
            faces_with_cp.emplace(a, bp, ap);
          }
        }
      }
    }

    for (int t = 0; t < T - 1; ++t) {
      for (int i = 0; i < H - 1; ++i) {
        for (int j = 0; j < W; ++j) {
          const size_t ax0y0 = vid(t, i, j, sz);
          const size_t ax0y1 = vid(t, i + 1, j, sz);
          const size_t bx0y0 = ax0y0 + dv;
          const size_t bx0y1 = ax0y1 + dv;
          if (face_has_cp_robust(ax0y0, ax0y1, bx0y0, U_fp, V_fp)) {
            faces_with_cp.emplace(ax0y0, ax0y1, bx0y0);
          }
          if (face_has_cp_robust(ax0y1, bx0y0, bx0y1, U_fp, V_fp)) {
            faces_with_cp.emplace(ax0y1, bx0y0, bx0y1);
          }
        }
      }
    }

    for (int t = 0; t < T - 1; ++t) {
      for (int i = 0; i < H - 1; ++i) {
        for (int j = 0; j < W - 1; ++j) {
          const size_t ax0y0 = vid(t, i, j, sz);
          const size_t ax1y1 = vid(t, i + 1, j + 1, sz);
          const size_t bx0y0 = ax0y0 + dv;
          const size_t bx1y1 = ax1y1 + dv;
          if (face_has_cp_robust(ax0y0, ax1y1, bx0y0, U_fp, V_fp)) {
            faces_with_cp.emplace(ax0y0, ax1y1, bx0y0);
          }
          if (face_has_cp_robust(ax1y1, bx0y0, bx1y1, U_fp, V_fp)) {
            faces_with_cp.emplace(ax1y1, bx0y0, bx1y1);
          }
        }
      }
    }

    // ---------------- (3) 内部剖分面：按 3-tet 切分 ----------------
    for (int t = 0; t < T - 1; ++t) {
      for (int i = 0; i < H - 1; ++i) {
        for (int j = 0; j < W - 1; ++j) {
          const size_t ax0y0 = vid(t, i, j, sz);
          const size_t ax0y1 = vid(t, i + 1, j, sz);
          const size_t ax1y1 = vid(t, i + 1, j + 1, sz);
          const size_t ax1y0 = vid(t, i, j + 1, sz);
          const size_t bx0y0 = ax0y0 + dv;
          const size_t bx0y1 = ax0y1 + dv;
          const size_t bx1y1 = ax1y1 + dv;
          const size_t bx1y0 = ax1y0 + dv;

          if (face_has_cp_robust(ax0y1, bx0y0, ax1y1, U_fp, V_fp)) {
            faces_with_cp.emplace(ax0y1, bx0y0, ax1y1);
          }
          if (face_has_cp_robust(ax0y1, bx0y0, bx1y1, U_fp, V_fp)) {
            faces_with_cp.emplace(ax0y1, bx0y0, bx1y1);
          }

          if (face_has_cp_robust(ax0y0, ax1y1, bx1y0, U_fp, V_fp)) {
            faces_with_cp.emplace(ax0y0, ax1y1, bx1y0);
          }
          if (face_has_cp_robust(ax1y1, bx0y0, bx1y0, U_fp, V_fp)) {
            faces_with_cp.emplace(ax1y1, bx0y0, bx1y0);
          }
        }
      }
    }
  return faces_with_cp;
}

template<typename T_fp>
static std::unordered_set<FaceKeySZ, FaceKeySZHash>
compute_cp_2p5d_faces(const T_fp* U_fp, const T_fp* V_fp,
                      int H, int W, int T, std::string filenam="")
{
  bool write_to_file = !filenam.empty();
  const Size3 sz{H, W, T};
  const size_t dv = (size_t)H * (size_t)W;
  std::vector<int> cp_per_layer(T, 0);
  auto faces_with_cp = compute_cp_2p5d_faces_parallel(
      U_fp, V_fp, H, W, T, sz, dv, cp_per_layer);

  // 打印每层 cp 数量（保持原行为）
  // for (int t=0; t<T; t+=10){
  //     printf("  cp in layer %d = %d\n", t, cp_per_layer[t]);
  // }

  if (write_to_file) {
    std::ofstream ofs(filenam);
    if (ofs) {
      for (int t = 0; t < T; ++t) {
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

          }
        }

        // (C) 内部剖分面：两片 ts ∈ {t, t-1}；每相邻 cell 的 Upper/Lower 各 2 面
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
  std::cout << "write unpred cnt = " << unpred_cnt << ",ratio=" << (double)unpred_cnt/(2*N) << "\n";
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


template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  using T = int64_t;
  const Size3 sz{(int)r1, (int)r2, (int)r3};
  const size_t H = r1, W = r2, Tt = r3;
  const size_t N = H * W * Tt;
  const size_t dv = H * W;

  compressed_size = 0;
  if (!U || !V || N == 0) {
    return nullptr;
  }

  T* U_fp = (T*)std::malloc(N * sizeof(T));
  T* V_fp = (T*)std::malloc(N * sizeof(T));
  if (!U_fp || !V_fp) {
    if (U_fp) std::free(U_fp);
    if (V_fp) std::free(V_fp);
    return nullptr;
  }

  T range = 0;
  T scale = convert_to_fixed_point<T_data, T>(U, V, N, U_fp, V_fp, range);

  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  std::vector<int> eb_id(N, 0);
  std::vector<uint8_t> pred_mask(N, 0);
  std::vector<int> dq0(N, 0);
  std::vector<int> dq1(N, 0);

  const int base = 2;
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);
  const T threshold = 1;

  T max_eb = 0;
  if (mode == EbMode::Relative) {
    max_eb = max_pwr_eb * range;
    printf("Compression Using Relative Eb Mode!\n");
  } else if (mode == EbMode::Absolute) {
    max_eb = max_pwr_eb * scale;
    printf("Compression Using Absolute Eb Mode!\n");
  } else {
    std::cerr << "Error: Unsupported EbMode!\n";
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }
  
  const int BlockW = 50;//32
  const int BlockH = 50;//32
  const int BlockT = 200;//16
  const int blocks_t = (int)((Tt + BlockT - 1) / BlockT);
  const int blocks_i = (int)((H + BlockH - 1) / BlockH);
  const int blocks_j = (int)((W + BlockW - 1) / BlockW);

  double enc_max_abs_eb_fp = 0.0;
  double enc_max_real_err_fp = 0.0;

#ifdef _OPENMP
// #  pragma omp parallel for collapse(3) schedule(dynamic) reduction(max:enc_max_abs_eb_fp) reduction(max:enc_max_real_err_fp)
#  pragma omp parallel for collapse(3) schedule(dynamic)
#endif
  for (int bt_idx = 0; bt_idx < blocks_t; ++bt_idx) {
    for (int bi_idx = 0; bi_idx < blocks_i; ++bi_idx) {
      for (int bj_idx = 0; bj_idx < blocks_j; ++bj_idx) {
        const int t0 = bt_idx * BlockT;
        const int t1 = std::min(t0 + BlockT, (int)Tt);
        const int i0 = bi_idx * BlockH;
        const int i1 = std::min(i0 + BlockH, (int)H);
        const int j0 = bj_idx * BlockW;
        const int j1 = std::min(j0 + BlockW, (int)W);
        if (t0 >= t1 || i0 >= i1 || j0 >= j1) continue;

        const int ht0 = std::max(0, t0 - 1);
        const int ht1 = std::min((int)Tt, t1 + 1);
        const int hi0 = std::max(0, i0 - 1);
        const int hi1 = std::min((int)H, i1 + 1);
        const int hj0 = std::max(0, j0 - 1);
        const int hj1 = std::min((int)W, j1 + 1);

        const int lTdim = ht1 - ht0;
        const int lHdim = hi1 - hi0;
        const int lWdim = hj1 - hj0;
        if (lTdim <= 0 || lHdim <= 0 || lWdim <= 0) continue;

        const size_t local_size = (size_t)lTdim * (size_t)lHdim * (size_t)lWdim;
        std::vector<T> U_blk(local_size);
        std::vector<T> V_blk(local_size);

        auto local_index = [&](int t, int i, int j) -> size_t {
          return ((size_t)(t - ht0) * (size_t)lHdim + (size_t)(i - hi0)) * (size_t)lWdim + (size_t)(j - hj0);
        };

        auto local_index_from_vid = [&](size_t v) -> size_t {
          int lt, li, lj;
          inv_vid(v, (int)H, (int)W, lt, li, lj);
          return local_index(lt, li, lj);
        };

        auto readU = [&](size_t v) -> T {
          return U_blk[local_index_from_vid(v)];
        };
        auto readV = [&](size_t v) -> T {
          return V_blk[local_index_from_vid(v)];
        };

        for (int gt = ht0; gt < ht1; ++gt) {
          for (int gi = hi0; gi < hi1; ++gi) {
            for (int gj = hj0; gj < hj1; ++gj) {
              size_t gvid = vid(gt, gi, gj, sz);
              size_t lvid = local_index(gt, gi, gj);
              U_blk[lvid] = U_fp[gvid];
              V_blk[lvid] = V_fp[gvid];
            }
          }
        }

        const ptrdiff_t lsi = (ptrdiff_t)lWdim;
        const ptrdiff_t lsj = 1;
        const ptrdiff_t lsk = (ptrdiff_t)lHdim * (ptrdiff_t)lWdim;

        for (int gt = t0; gt < t1; ++gt) {
          for (int gi = i0; gi < i1; ++gi) {
            for (int gj = j0; gj < j1; ++gj) {
              const size_t gvid = vid(gt, gi, gj, sz);
              const size_t loff = local_index(gt, gi, gj);
              T* curU = U_blk.data() + loff;
              T* curV = V_blk.data() + loff;

              const bool is_boundary = (gt == t0) || (gt == t1 - 1) ||
                                       (gi == i0) || (gi == i1 - 1) ||
                                       (gj == j0) || (gj == j1 - 1);
              if (is_boundary) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              const T curU_val = *curU;
              const T curV_val = *curV;
              T required_eb = max_eb;

              for (int ci = gi - 1; ci <= gi; ++ci) {
                if (!in_range(ci, (int)H - 1)) continue;
                for (int cj = gj - 1; cj <= gj; ++cj) {
                  if (!in_range(cj, (int)W - 1)) continue;

                  size_t v00 = vid(gt, ci, cj, sz);
                  size_t v10 = vid(gt, ci, cj + 1, sz);
                  size_t v01 = vid(gt, ci + 1, cj, sz);
                  size_t v11 = vid(gt, ci + 1, cj + 1, sz);

                  if (gvid == v00 || gvid == v01 || gvid == v11) {
                    if (has_cp(cp_faces, v00, v01, v11)) {
                      required_eb = 0;
                    }
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(v11), readU(v01), readU(v00),
                        readV(v11), readV(v01), readV(v00));
                    if (eb < required_eb) required_eb = eb;
                  }

                  if (gvid == v00 || gvid == v10 || gvid == v11) {
                    if (has_cp(cp_faces, v00, v10, v11)) {
                      required_eb = 0;
                    }
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(v11), readU(v10), readU(v00),
                        readV(v11), readV(v10), readV(v00));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              const int di[6] = {0, 0, 1, -1, 1, -1};
              const int dj[6] = {1, -1, 0, 0, 1, -1};

              if (gt < (int)Tt - 1) {
                for (int k = 0; k < 6; ++k) {
                  int ni = gi + di[k];
                  int nj = gj + dj[k];
                  if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
                  size_t a = gvid;
                  size_t b = vid(gt, ni, nj, sz);
                  size_t ap = a + dv;
                  size_t bp = b + dv;
                  if (k == 0 || k == 3 || k == 5) {
                    if (has_cp(cp_faces, a, b, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(b), readU(a),
                          readV(bp), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                    if (has_cp(cp_faces, a, bp, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(bp), readU(a),
                          readV(ap), readV(bp), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  } else {
                    if (has_cp(cp_faces, a, b, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(b), readU(a),
                          readV(ap), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  }
                }
              }

              if (gt > 0) {
                for (int k = 0; k < 6; ++k) {
                  int ni = gi + di[k];
                  int nj = gj + dj[k];
                  if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
                  size_t a = gvid;
                  size_t b = vid(gt, ni, nj, sz);
                  size_t ap = a - dv;
                  size_t bp = b - dv;
                  if (k == 0 || k == 3 || k == 5) {
                    if (has_cp(cp_faces, a, b, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(b), readU(a),
                          readV(ap), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  } else {
                    if (has_cp(cp_faces, a, b, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(b), readU(a),
                          readV(bp), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                    if (has_cp(cp_faces, a, ap, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(ap), readU(a),
                          readV(bp), readV(ap), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  }
                }
              }

              if (gt < (int)Tt - 1) {
                if (in_range(gi - 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f1a = gvid;
                  size_t f1b = vid(gt, gi, gj + 1, sz);
                  size_t f1c = vid(gt, gi - 1, gj, sz);
                  size_t f1bp = f1b + dv;
                  size_t f1cp = f1c + dv;
                  if (has_cp(cp_faces, f1a, f1b, f1cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f1b), readU(f1cp), readU(f1a),
                        readV(f1b), readV(f1cp), readV(f1a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f1a, f1bp, f1cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f1bp), readU(f1cp), readU(f1a),
                        readV(f1bp), readV(f1cp), readV(f1a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f2a = gvid;
                  size_t f2b = vid(gt, gi - 1, gj, sz);
                  size_t f2c = vid(gt, gi - 1, gj - 1, sz);
                  size_t f2bp = f2b + dv;
                  size_t f2cp = f2c + dv;
                  if (has_cp(cp_faces, f2a, f2bp, f2cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f2bp), readU(f2c), readU(f2a),
                        readV(f2bp), readV(f2c), readV(f2a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f2a, f2cp, f2bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f2cp), readU(f2bp), readU(f2a),
                        readV(f2cp), readV(f2bp), readV(f2a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f3a = gvid;
                  size_t f3b = vid(gt, gi - 1, gj - 1, sz);
                  size_t f3c = vid(gt, gi, gj - 1, sz);
                  size_t f3bp = f3b + dv;
                  size_t f3cp = f3c + dv;
                  if (has_cp(cp_faces, f3a, f3b, f3bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f3c), readU(f3bp), readU(f3a),
                        readV(f3c), readV(f3bp), readV(f3a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f6a = gvid;
                  size_t f6b = vid(gt, gi, gj + 1, sz);
                  size_t f6c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f6bp = f6b + dv;
                  if (has_cp(cp_faces, f6a, f6bp, f6c)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f6c), readU(f6bp), readU(f6a),
                        readV(f6c), readV(f6bp), readV(f6a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              if (gt > 0) {
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f3a = gvid;
                  size_t f3b = vid(gt, gi - 1, gj - 1, sz);
                  size_t f3c = vid(gt, gi, gj - 1, sz);
                  size_t f3bp = f3b - dv;
                  size_t f3cp = f3c - dv;
                  if (has_cp(cp_faces, f3a, f3b, f3cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f3cp), readU(f3b), readU(f3a),
                        readV(f3cp), readV(f3b), readV(f3a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f4a = gvid;
                  size_t f4b = vid(gt, gi, gj - 1, sz);
                  size_t f4c = vid(gt, gi + 1, gj, sz);
                  size_t f4bp = f4b - dv;
                  size_t f4cp = f4c - dv;
                  if (has_cp(cp_faces, f4a, f4bp, f4cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f4cp), readU(f4bp), readU(f4a),
                        readV(f4cp), readV(f4bp), readV(f4a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f4a, f4b, f4cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f4cp), readU(f4b), readU(f4a),
                        readV(f4cp), readV(f4b), readV(f4a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f5a = gvid;
                  size_t f5b = vid(gt, gi + 1, gj, sz);
                  size_t f5c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f5bp = f5b - dv;
                  size_t f5cp = f5c - dv;
                  if (has_cp(cp_faces, f5a, f5bp, f5cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f5cp), readU(f5bp), readU(f5a),
                        readV(f5cp), readV(f5bp), readV(f5a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f5a, f5c, f5bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f5bp), readU(f5c), readU(f5a),
                        readV(f5bp), readV(f5c), readV(f5a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f6a = gvid;
                  size_t f6b = vid(gt, gi, gj + 1, sz);
                  size_t f6c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f6cp = f6c - dv;
                  if (has_cp(cp_faces, f6a, f6b, f6cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f6cp), readU(f6b), readU(f6a),
                        readV(f6cp), readV(f6b), readV(f6a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              T abs_eb = required_eb;
              int id = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);

              if (abs_eb == 0) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              bool unpred_flag = false;
              T dec[2] = {0, 0};
              T abs_err_fp_q[2] = {0, 0};
              for (int p = 0; p < 2; ++p) {
                T* cur = (p == 0) ? curU : curV;
                T curv = (p == 0) ? curU_val : curV_val;

                T d0 = (gt && gi && gj) ? cur[-lsk - lsi - lsj] : 0;
                T d1 = (gt && gi) ? cur[-lsk - lsi] : 0;
                T d2 = (gt && gj) ? cur[-lsk - lsj] : 0;
                T d3 = (gt) ? cur[-lsk] : 0;
                T d4 = (gi && gj) ? cur[-lsi - lsj] : 0;
                T d5 = (gi) ? cur[-lsi] : 0;
                T d6 = (gj) ? cur[-lsj] : 0;
                T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

                T diff = curv - pred;
                T qd = (std::llabs(diff) / abs_eb) + 1;
                if (qd < capacity) {
                  qd = (diff > 0) ? qd : -qd;
                  int qindex = (int)(qd / 2) + intv_radius;
                  if (p == 0) dq0[gvid] = qindex; else dq1[gvid] = qindex;
                  dec[p] = pred + 2 * (qindex - intv_radius) * abs_eb;
                  if (std::llabs(dec[p] - curv) > abs_eb) {
                    unpred_flag = true;
                    break;
                  }
                  abs_err_fp_q[p] = std::llabs(dec[p] - curv);
                } else {
                  unpred_flag = true;
                  break;
                }
              }

              if (unpred_flag) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              eb_id[gvid] = id;
              pred_mask[gvid] = 1;
              *curU = dec[0];
              *curV = dec[1];

              double abs_eb_fp = (double)abs_eb / (double)scale;
              double err_u_fp = (double)abs_err_fp_q[0] / (double)scale;
              double err_v_fp = (double)abs_err_fp_q[1] / (double)scale;
              // enc_max_abs_eb_fp = std::max(enc_max_abs_eb_fp, abs_eb_fp);
              // enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
            }
          }
        }
      }
    }
  }

  std::vector<T_data> unpred;
  unpred.reserve((N / 8 + 1) * 2);
  std::vector<int> dq_stream;
  dq_stream.reserve(2 * N);

  for (size_t v = 0; v < N; ++v) {
    if (pred_mask[v]) {
      dq_stream.push_back(dq0[v]);
      dq_stream.push_back(dq1[v]);
    } else {
      unpred.push_back(U[v]);
      unpred.push_back(V[v]);
    }
  }

  unsigned char* compressed = (unsigned char*)std::malloc((size_t)(2 * N * sizeof(T)));
  if (!compressed) {
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }
  unsigned char* pos = compressed;

  write_variable_to_dst(pos, scale);
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) {
    write_array_to_dst(pos, unpred.data(), unpred_cnt);
  }

  size_t eb_quant_num = N;
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * 1024, eb_id.data(), eb_quant_num, pos);

  size_t data_quant_num = dq_stream.size();
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * capacity, dq_stream.data(), data_quant_num, pos);

  compressed_size = (size_t)(pos - compressed);

  std::free(U_fp);
  std::free(V_fp);
  return compressed;
}

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v2(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  using T = int64_t;
  const Size3 sz{(int)r1, (int)r2, (int)r3};
  const size_t H = r1, W = r2, Tt = r3;
  const size_t N = H * W * Tt;
  const size_t dv = H * W;

  compressed_size = 0;
  if (!U || !V || N == 0) {
    return nullptr;
  }

  T* U_fp = (T*)std::malloc(N * sizeof(T));
  T* V_fp = (T*)std::malloc(N * sizeof(T));
  if (!U_fp || !V_fp) {
    if (U_fp) std::free(U_fp);
    if (V_fp) std::free(V_fp);
    return nullptr;
  }

  T range = 0;
  T scale = convert_to_fixed_point<T_data, T>(U, V, N, U_fp, V_fp, range);

  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  std::vector<int> eb_id(N, 0);
  std::vector<uint8_t> pred_mask(N, 0);
  std::vector<int> dq0(N, 0);
  std::vector<int> dq1(N, 0);

  const int base = 2;
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);
  const T threshold = 1;

  T max_eb = 0;
  if (mode == EbMode::Relative) {
    max_eb = max_pwr_eb * range;
    printf("Compression Using Relative Eb Mode!\n");
  } else if (mode == EbMode::Absolute) {
    max_eb = max_pwr_eb * scale;
    printf("Compression Using Absolute Eb Mode!\n");
  } else {
    std::cerr << "Error: Unsupported EbMode!\n";
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }
  
  const int BlockW = 50;//32
  const int BlockH = 50;//32
  const int BlockT = 200;//16
  const int blocks_t = (int)((Tt + BlockT - 1) / BlockT);
  const int blocks_i = (int)((H + BlockH - 1) / BlockH);
  const int blocks_j = (int)((W + BlockW - 1) / BlockW);

  double enc_max_abs_eb_fp = 0.0;
  double enc_max_real_err_fp = 0.0;

#ifdef _OPENMP
// #  pragma omp parallel for collapse(3) schedule(dynamic) reduction(max:enc_max_abs_eb_fp) reduction(max:enc_max_real_err_fp)
#  pragma omp parallel for collapse(3) schedule(dynamic)
#endif
  for (int bt_idx = 0; bt_idx < blocks_t; ++bt_idx) {
    for (int bi_idx = 0; bi_idx < blocks_i; ++bi_idx) {
      for (int bj_idx = 0; bj_idx < blocks_j; ++bj_idx) {
        const int t0 = bt_idx * BlockT;
        const int t1 = std::min(t0 + BlockT, (int)Tt);
        const int i0 = bi_idx * BlockH;
        const int i1 = std::min(i0 + BlockH, (int)H);
        const int j0 = bj_idx * BlockW;
        const int j1 = std::min(j0 + BlockW, (int)W);
        if (t0 >= t1 || i0 >= i1 || j0 >= j1) continue;

        const int ht0 = std::max(0, t0 - 1);
        const int ht1 = std::min((int)Tt, t1 + 1);
        const int hi0 = std::max(0, i0 - 1);
        const int hi1 = std::min((int)H, i1 + 1);
        const int hj0 = std::max(0, j0 - 1);
        const int hj1 = std::min((int)W, j1 + 1);

        const int lTdim = ht1 - ht0;
        const int lHdim = hi1 - hi0;
        const int lWdim = hj1 - hj0;
        if (lTdim <= 0 || lHdim <= 0 || lWdim <= 0) continue;

        const size_t local_size = (size_t)lTdim * (size_t)lHdim * (size_t)lWdim;
        std::vector<T> U_blk(local_size);
        std::vector<T> V_blk(local_size);

        auto local_index = [&](int t, int i, int j) -> size_t {
          return ((size_t)(t - ht0) * (size_t)lHdim + (size_t)(i - hi0)) * (size_t)lWdim + (size_t)(j - hj0);
        };

        auto local_index_from_vid = [&](size_t v) -> size_t {
          int lt, li, lj;
          inv_vid(v, (int)H, (int)W, lt, li, lj);
          return local_index(lt, li, lj);
        };

        auto readU = [&](size_t v) -> T {
          return U_blk[local_index_from_vid(v)];
        };
        auto readV = [&](size_t v) -> T {
          return V_blk[local_index_from_vid(v)];
        };

        for (int gt = ht0; gt < ht1; ++gt) {
          for (int gi = hi0; gi < hi1; ++gi) {
            for (int gj = hj0; gj < hj1; ++gj) {
              size_t gvid = vid(gt, gi, gj, sz);
              size_t lvid = local_index(gt, gi, gj);
              U_blk[lvid] = U_fp[gvid];
              V_blk[lvid] = V_fp[gvid];
            }
          }
        }

        const ptrdiff_t lsi = (ptrdiff_t)lWdim;
        const ptrdiff_t lsj = 1;
        const ptrdiff_t lsk = (ptrdiff_t)lHdim * (ptrdiff_t)lWdim;

        const int core_height = i1 - i0;
        const int core_width  = j1 - j0;

        for (int gt = t0; gt < t1; ++gt) {
          for (int diag = 0; diag <= (core_height - 1) + (core_width - 1); ++diag) {
            for (int gi = i0; gi < i1; ++gi) {
              int gj = diag - (gi - i0) + j0;
              if (gj < j0 || gj >= j1) {
                continue;
              }

              const size_t gvid = vid(gt, gi, gj, sz);
              const bool is_boundary = (gt == t0) || (gt == t1 - 1) ||
                                       (gi == i0) || (gi == i1 - 1) ||
                                       (gj == j0) || (gj == j1 - 1);
              if (is_boundary) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              const size_t loff = local_index(gt, gi, gj);
              T* curU = U_blk.data() + loff;
              T* curV = V_blk.data() + loff;
              const T curU_val = *curU;
              const T curV_val = *curV;
              T required_eb = max_eb;

              for (int ci = gi - 1; ci <= gi; ++ci) {
                if (!in_range(ci, (int)H - 1)) continue;
                for (int cj = gj - 1; cj <= gj; ++cj) {
                  if (!in_range(cj, (int)W - 1)) continue;

                  size_t v00 = vid(gt, ci, cj, sz);
                  size_t v10 = vid(gt, ci, cj + 1, sz);
                  size_t v01 = vid(gt, ci + 1, cj, sz);
                  size_t v11 = vid(gt, ci + 1, cj + 1, sz);

                  if (gvid == v00 || gvid == v01 || gvid == v11) {
                    if (has_cp(cp_faces, v00, v01, v11)) {
                      required_eb = 0;
                    }
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(v11), readU(v01), readU(v00),
                        readV(v11), readV(v01), readV(v00));
                    if (eb < required_eb) required_eb = eb;
                  }

                  if (gvid == v00 || gvid == v10 || gvid == v11) {
                    if (has_cp(cp_faces, v00, v10, v11)) {
                      required_eb = 0;
                    }
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(v11), readU(v10), readU(v00),
                        readV(v11), readV(v10), readV(v00));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              const int di[6] = {0, 0, 1, -1, 1, -1};
              const int dj[6] = {1, -1, 0, 0, 1, -1};

              if (gt < (int)Tt - 1) {
                for (int k = 0; k < 6; ++k) {
                  int ni = gi + di[k];
                  int nj = gj + dj[k];
                  if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
                  size_t a = gvid;
                  size_t b = vid(gt, ni, nj, sz);
                  size_t ap = a + dv;
                  size_t bp = b + dv;
                  if (k == 0 || k == 3 || k == 5) {
                    if (has_cp(cp_faces, a, b, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(b), readU(a),
                          readV(bp), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                    if (has_cp(cp_faces, a, bp, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(bp), readU(a),
                          readV(ap), readV(bp), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  } else {
                    if (has_cp(cp_faces, a, b, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(b), readU(a),
                          readV(ap), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  }
                }
              }

              if (gt > 0) {
                for (int k = 0; k < 6; ++k) {
                  int ni = gi + di[k];
                  int nj = gj + dj[k];
                  if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
                  size_t a = gvid;
                  size_t b = vid(gt, ni, nj, sz);
                  size_t ap = a - dv;
                  size_t bp = b - dv;
                  if (k == 0 || k == 3 || k == 5) {
                    if (has_cp(cp_faces, a, b, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(b), readU(a),
                          readV(ap), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  } else {
                    if (has_cp(cp_faces, a, b, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(b), readU(a),
                          readV(bp), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                    if (has_cp(cp_faces, a, ap, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(ap), readU(a),
                          readV(bp), readV(ap), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  }
                }
              }

              if (gt < (int)Tt - 1) {
                if (in_range(gi - 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f1a = gvid;
                  size_t f1b = vid(gt, gi, gj + 1, sz);
                  size_t f1c = vid(gt, gi - 1, gj, sz);
                  size_t f1bp = f1b + dv;
                  size_t f1cp = f1c + dv;
                  if (has_cp(cp_faces, f1a, f1b, f1cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f1b), readU(f1cp), readU(f1a),
                        readV(f1b), readV(f1cp), readV(f1a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f1a, f1bp, f1cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f1bp), readU(f1cp), readU(f1a),
                        readV(f1bp), readV(f1cp), readV(f1a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f2a = gvid;
                  size_t f2b = vid(gt, gi - 1, gj, sz);
                  size_t f2c = vid(gt, gi - 1, gj - 1, sz);
                  size_t f2bp = f2b + dv;
                  size_t f2cp = f2c + dv;
                  if (has_cp(cp_faces, f2a, f2bp, f2cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f2bp), readU(f2c), readU(f2a),
                        readV(f2bp), readV(f2c), readV(f2a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f2a, f2cp, f2bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f2cp), readU(f2bp), readU(f2a),
                        readV(f2cp), readV(f2bp), readV(f2a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f3a = gvid;
                  size_t f3b = vid(gt, gi - 1, gj - 1, sz);
                  size_t f3c = vid(gt, gi, gj - 1, sz);
                  size_t f3bp = f3b + dv;
                  size_t f3cp = f3c + dv;
                  if (has_cp(cp_faces, f3a, f3b, f3bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f3c), readU(f3bp), readU(f3a),
                        readV(f3c), readV(f3bp), readV(f3a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f6a = gvid;
                  size_t f6b = vid(gt, gi, gj + 1, sz);
                  size_t f6c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f6bp = f6b + dv;
                  if (has_cp(cp_faces, f6a, f6bp, f6c)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f6c), readU(f6bp), readU(f6a),
                        readV(f6c), readV(f6bp), readV(f6a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              if (gt > 0) {
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f3a = gvid;
                  size_t f3b = vid(gt, gi - 1, gj - 1, sz);
                  size_t f3c = vid(gt, gi, gj - 1, sz);
                  size_t f3bp = f3b - dv;
                  size_t f3cp = f3c - dv;
                  if (has_cp(cp_faces, f3a, f3b, f3cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f3cp), readU(f3b), readU(f3a),
                        readV(f3cp), readV(f3b), readV(f3a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f4a = gvid;
                  size_t f4b = vid(gt, gi, gj - 1, sz);
                  size_t f4c = vid(gt, gi + 1, gj, sz);
                  size_t f4bp = f4b - dv;
                  size_t f4cp = f4c - dv;
                  if (has_cp(cp_faces, f4a, f4bp, f4cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f4cp), readU(f4bp), readU(f4a),
                        readV(f4cp), readV(f4bp), readV(f4a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f4a, f4b, f4cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f4cp), readU(f4b), readU(f4a),
                        readV(f4cp), readV(f4b), readV(f4a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f5a = gvid;
                  size_t f5b = vid(gt, gi + 1, gj, sz);
                  size_t f5c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f5bp = f5b - dv;
                  size_t f5cp = f5c - dv;
                  if (has_cp(cp_faces, f5a, f5bp, f5cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f5cp), readU(f5bp), readU(f5a),
                        readV(f5cp), readV(f5bp), readV(f5a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f5a, f5c, f5bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f5bp), readU(f5c), readU(f5a),
                        readV(f5bp), readV(f5c), readV(f5a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f6a = gvid;
                  size_t f6b = vid(gt, gi, gj + 1, sz);
                  size_t f6c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f6cp = f6c - dv;
                  if (has_cp(cp_faces, f6a, f6b, f6cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f6cp), readU(f6b), readU(f6a),
                        readV(f6cp), readV(f6b), readV(f6a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              T abs_eb = required_eb;
              int id = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);

              if (abs_eb == 0) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              bool unpred_flag = false;
              T dec[2] = {0, 0};
              T abs_err_fp_q[2] = {0, 0};
              for (int p = 0; p < 2; ++p) {
                T* cur = (p == 0) ? curU : curV;
                T curv = (p == 0) ? curU_val : curV_val;

                T d0 = (gt && gi && gj) ? cur[-lsk - lsi - lsj] : 0;
                T d1 = (gt && gi) ? cur[-lsk - lsi] : 0;
                T d2 = (gt && gj) ? cur[-lsk - lsj] : 0;
                T d3 = (gt) ? cur[-lsk] : 0;
                T d4 = (gi && gj) ? cur[-lsi - lsj] : 0;
                T d5 = (gi) ? cur[-lsi] : 0;
                T d6 = (gj) ? cur[-lsj] : 0;
                T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

                T diff = curv - pred;
                T qd = (std::llabs(diff) / abs_eb) + 1;
                if (qd < capacity) {
                  qd = (diff > 0) ? qd : -qd;
                  int qindex = (int)(qd / 2) + intv_radius;
                  if (p == 0) dq0[gvid] = qindex; else dq1[gvid] = qindex;
                  dec[p] = pred + 2 * (qindex - intv_radius) * abs_eb;
                  if (std::llabs(dec[p] - curv) > abs_eb) {
                    unpred_flag = true;
                    break;
                  }
                  abs_err_fp_q[p] = std::llabs(dec[p] - curv);
                } else {
                  unpred_flag = true;
                  break;
                }
              }

              if (unpred_flag) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              eb_id[gvid] = id;
              pred_mask[gvid] = 1;
              *curU = dec[0];
              *curV = dec[1];

              double abs_eb_fp = (double)abs_eb / (double)scale;
              double err_u_fp = (double)abs_err_fp_q[0] / (double)scale;
              double err_v_fp = (double)abs_err_fp_q[1] / (double)scale;
              // enc_max_abs_eb_fp = std::max(enc_max_abs_eb_fp, abs_eb_fp);
              // enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
            }
          }
        }
      }
    }
  }

  std::vector<T_data> unpred;
  unpred.reserve((N / 8 + 1) * 2);
  std::vector<int> dq_stream;
  dq_stream.reserve(2 * N);

  for (size_t v = 0; v < N; ++v) {
    if (pred_mask[v]) {
      dq_stream.push_back(dq0[v]);
      dq_stream.push_back(dq1[v]);
    } else {
      unpred.push_back(U[v]);
      unpred.push_back(V[v]);
    }
  }

  unsigned char* compressed = (unsigned char*)std::malloc((size_t)(2 * N * sizeof(T)));
  if (!compressed) {
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }
  unsigned char* pos = compressed;

  write_variable_to_dst(pos, scale);
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) {
    write_array_to_dst(pos, unpred.data(), unpred_cnt);
  }

  size_t eb_quant_num = N;
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * 1024, eb_id.data(), eb_quant_num, pos);

  size_t data_quant_num = dq_stream.size();
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * capacity, dq_stream.data(), data_quant_num, pos);

  compressed_size = (size_t)(pos - compressed);

  std::free(U_fp);
  std::free(V_fp);
  return compressed;
}

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v3(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  using T = int64_t;
  const Size3 sz{(int)r1, (int)r2, (int)r3};
  const size_t H = r1, W = r2, Tt = r3;
  const size_t N = H * W * Tt;
  const size_t dv = H * W;

  compressed_size = 0;
  if (!U || !V || N == 0) {
    return nullptr;
  }

  T* U_fp = (T*)std::malloc(N * sizeof(T));
  T* V_fp = (T*)std::malloc(N * sizeof(T));
  if (!U_fp || !V_fp) {
    if (U_fp) std::free(U_fp);
    if (V_fp) std::free(V_fp);
    return nullptr;
  }

  T range = 0;
  T scale = convert_to_fixed_point<T_data, T>(U, V, N, U_fp, V_fp, range);

  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  std::vector<int> eb_id(N, 0);
  std::vector<uint8_t> pred_mask(N, 0);
  std::vector<int> dq0(N, 0);
  std::vector<int> dq1(N, 0);

  const int base = 2;
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);
  const T threshold = 1;

  T max_eb = 0;
  if (mode == EbMode::Relative) {
    max_eb = max_pwr_eb * range;
    printf("Compression Using Relative Eb Mode!\n");
  } else if (mode == EbMode::Absolute) {
    max_eb = max_pwr_eb * scale;
    printf("Compression Using Absolute Eb Mode!\n");
  } else {
    std::cerr << "Error: Unsupported EbMode!\n";
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }

  const int BlockH = 96;
  const int BlockW = 128;
  const size_t nbH = (H + BlockH - 1) / BlockH;
  const size_t nbW = (W + BlockW - 1) / BlockW;

  const ptrdiff_t si = (ptrdiff_t)W;
  const ptrdiff_t sj = (ptrdiff_t)1;
  const ptrdiff_t sk = (ptrdiff_t)(H * W);
  const int di[6] = { 0,  0,  1, -1,  1, -1 };
  const int dj[6] = { 1, -1,  0,  0,  1, -1 };

  auto readU = [&](size_t idx) -> T { return U_fp[idx]; };
  auto readV = [&](size_t idx) -> T { return V_fp[idx]; };

  auto process_tile = [&](int t, int bi, int bj) {
    const int i0 = bi * BlockH;
    const int i1 = std::min(i0 + BlockH, (int)H);
    const int j0 = bj * BlockW;
    const int j1 = std::min(j0 + BlockW, (int)W);
    const int Hi = i1 - i0;
    const int Wi = j1 - j0;

    for (int s = 0; s <= Hi + Wi - 2; ++s) {
      int j_lo = std::max(0, s - (Hi - 1));
      int j_hi = std::min(Wi - 1, s);
      for (int joff = j_lo; joff <= j_hi; ++joff) {
        int ioff = s - joff;
        int i = i0 + ioff;
        int j = j0 + joff;
        const size_t v = vid(t, i, j, sz);

        const bool is_boundary = (i == i0) || (i == i1 - 1) ||
                                 (j == j0) || (j == j1 - 1);
        if (is_boundary) {
          eb_id[v] = 0;
          pred_mask[v] = 0;
          dq0[v] = 0;
          dq1[v] = 0;
          continue;
        }

        T* curU = U_fp + v;
        T* curV = V_fp + v;
        const T curU_val = *curU;
        const T curV_val = *curV;
        T required_eb = max_eb;

        for (int ci = i - 1; ci <= i; ++ci) {
          if (!in_range(ci, (int)H - 1)) continue;
          for (int cj = j - 1; cj <= j; ++cj) {
            if (!in_range(cj, (int)W - 1)) continue;

            size_t v00 = vid(t, ci, cj, sz);
            size_t v10 = vid(t, ci, cj + 1, sz);
            size_t v01 = vid(t, ci + 1, cj, sz);
            size_t v11 = vid(t, ci + 1, cj + 1, sz);

            if (v == v00 || v == v01 || v == v11) {
              if (has_cp(cp_faces, v00, v01, v11)) {
                required_eb = 0;
              }
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(v11), readU(v01), readU(v00),
                  readV(v11), readV(v01), readV(v00));
              if (eb < required_eb) required_eb = eb;
            }

            if (v == v00 || v == v10 || v == v11) {
              if (has_cp(cp_faces, v00, v10, v11)) {
                required_eb = 0;
              }
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(v11), readU(v10), readU(v00),
                  readV(v11), readV(v10), readV(v00));
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        if (t < (int)Tt - 1) {
          for (int k = 0; k < 6; ++k) {
            int ni = i + di[k];
            int nj = j + dj[k];
            if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
            size_t a = v;
            size_t b = vid(t, ni, nj, sz);
            size_t ap = a + dv;
            size_t bp = b + dv;
            if (k == 0 || k == 3 || k == 5) {
              if (has_cp(cp_faces, a, b, bp)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(bp), readU(b), readU(a),
                    readV(bp), readV(b), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, a, bp, ap)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(ap), readU(bp), readU(a),
                    readV(ap), readV(bp), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
            } else {
              if (has_cp(cp_faces, a, b, ap)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(ap), readU(b), readU(a),
                    readV(ap), readV(b), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }

        if (t > 0) {
          for (int k = 0; k < 6; ++k) {
            int ni = i + di[k];
            int nj = j + dj[k];
            if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
            size_t a = v;
            size_t b = vid(t, ni, nj, sz);
            size_t ap = a - dv;
            size_t bp = b - dv;
            if (k == 0 || k == 3 || k == 5) {
              if (has_cp(cp_faces, a, b, ap)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(ap), readU(b), readU(a),
                    readV(ap), readV(b), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
            } else {
              if (has_cp(cp_faces, a, b, bp)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(bp), readU(b), readU(a),
                    readV(bp), readV(b), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, a, ap, bp)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(bp), readU(ap), readU(a),
                    readV(bp), readV(ap), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }

        // (C1) and (C2) internal faces
        if (t < (int)Tt - 1) {
          if (in_range(i - 1, (int)H) && in_range(j + 1, (int)W)) {
            size_t f1a = v;
            size_t f1b = vid(t, i, j + 1, sz);
            size_t f1c = vid(t, i - 1, j, sz);
            size_t f1bp = f1b + dv;
            size_t f1cp = f1c + dv;
            if (has_cp(cp_faces, f1a, f1b, f1cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f1b), readU(f1cp), readU(f1a),
                  readV(f1b), readV(f1cp), readV(f1a));
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f1a, f1bp, f1cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f1bp), readU(f1cp), readU(f1a),
                  readV(f1bp), readV(f1cp), readV(f1a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i - 1, (int)H) && in_range(j - 1, (int)W)) {
            size_t f2a = v;
            size_t f2b = vid(t, i - 1, j, sz);
            size_t f2c = vid(t, i - 1, j - 1, sz);
            size_t f2bp = f2b + dv;
            size_t f2cp = f2c + dv;
            if (has_cp(cp_faces, f2a, f2bp, f2cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f2bp), readU(f2c), readU(f2a),
                  readV(f2bp), readV(f2c), readV(f2a));
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f2a, f2cp, f2bp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f2cp), readU(f2bp), readU(f2a),
                  readV(f2cp), readV(f2bp), readV(f2a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i - 1, (int)H) && in_range(j - 1, (int)W)) {
            size_t f3a = v;
            size_t f3b = vid(t, i - 1, j - 1, sz);
            size_t f3c = vid(t, i, j - 1, sz);
            size_t f3bp = f3b + dv;
            size_t f3cp = f3c + dv;
            if (has_cp(cp_faces, f3a, f3b, f3bp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f3c), readU(f3bp), readU(f3a),
                  readV(f3c), readV(f3bp), readV(f3a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i + 1, (int)H) && in_range(j + 1, (int)W)) {
            size_t f6a = v;
            size_t f6b = vid(t, i, j + 1, sz);
            size_t f6c = vid(t, i + 1, j + 1, sz);
            size_t f6bp = f6b + dv;
            if (has_cp(cp_faces, f6a, f6bp, f6c)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f6c), readU(f6bp), readU(f6a),
                  readV(f6c), readV(f6bp), readV(f6a));
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        if (t > 0) {
          if (in_range(i - 1, (int)H) && in_range(j - 1, (int)W)) {
            size_t f3a = v;
            size_t f3b = vid(t, i - 1, j - 1, sz);
            size_t f3c = vid(t, i, j - 1, sz);
            size_t f3bp = f3b - dv;
            size_t f3cp = f3c - dv;
            if (has_cp(cp_faces, f3a, f3b, f3cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f3cp), readU(f3b), readU(f3a),
                  readV(f3cp), readV(f3b), readV(f3a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i + 1, (int)H) && in_range(j - 1, (int)W)) {
            size_t f4a = v;
            size_t f4b = vid(t, i, j - 1, sz);
            size_t f4c = vid(t, i + 1, j, sz);
            size_t f4bp = f4b - dv;
            size_t f4cp = f4c - dv;
            if (has_cp(cp_faces, f4a, f4bp, f4cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f4cp), readU(f4bp), readU(f4a),
                  readV(f4cp), readV(f4bp), readV(f4a));
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f4a, f4b, f4cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f4cp), readU(f4b), readU(f4a),
                  readV(f4cp), readV(f4b), readV(f4a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i + 1, (int)H) && in_range(j + 1, (int)W)) {
            size_t f5a = v;
            size_t f5b = vid(t, i + 1, j, sz);
            size_t f5c = vid(t, i + 1, j + 1, sz);
            size_t f5bp = f5b - dv;
            size_t f5cp = f5c - dv;
            if (has_cp(cp_faces, f5a, f5bp, f5cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f5cp), readU(f5bp), readU(f5a),
                  readV(f5cp), readV(f5bp), readV(f5a));
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f5a, f5c, f5bp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f5bp), readU(f5c), readU(f5a),
                  readV(f5bp), readV(f5c), readV(f5a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i + 1, (int)H) && in_range(j + 1, (int)W)) {
            size_t f6a = v;
            size_t f6b = vid(t, i, j + 1, sz);
            size_t f6c = vid(t, i + 1, j + 1, sz);
            size_t f6cp = f6c - dv;
            if (has_cp(cp_faces, f6a, f6b, f6cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f6cp), readU(f6b), readU(f6a),
                  readV(f6cp), readV(f6b), readV(f6a));
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        T abs_eb = required_eb;
        int id = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);

        if (abs_eb == 0) {
          eb_id[v] = 0;
          pred_mask[v] = 0;
          dq0[v] = 0;
          dq1[v] = 0;
          continue;
        }

        bool unpred_flag = false;
        T dec[2] = {0, 0};
        T abs_err_fp_q[2] = {0, 0};

        for (int p = 0; p < 2; ++p) {
          T* cur = (p == 0) ? curU : curV;
          T curv = (p == 0) ? curU_val : curV_val;

          T d0 = (t && i && j) ? cur[-sk - si - sj] : 0;
          T d1 = (t && i)      ? cur[-sk - si]      : 0;
          T d2 = (t && j)      ? cur[-sk - sj]      : 0;
          T d3 = (t)           ? cur[-sk]           : 0;
          T d4 = (i && j)      ? cur[-si - sj]      : 0;
          T d5 = (i)           ? cur[-si]           : 0;
          T d6 = (j)           ? cur[-sj]           : 0;
          T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

          T diff = curv - pred;
          T qd = (std::llabs(diff) / abs_eb) + 1;
          if (qd < capacity) {
            qd = (diff > 0) ? qd : -qd;
            int qindex = (int)(qd / 2) + intv_radius;
            if (p == 0) {
              dq0[v] = qindex;
            } else {
              dq1[v] = qindex;
            }
            dec[p] = pred + 2 * (qindex - intv_radius) * abs_eb;
            if (std::llabs(dec[p] - curv) > abs_eb) {
              unpred_flag = true;
              break;
            }
            abs_err_fp_q[p] = std::llabs(dec[p] - curv);
          } else {
            unpred_flag = true;
            break;
          }
        }

        if (unpred_flag) {
          eb_id[v] = 0;
          pred_mask[v] = 0;
          dq0[v] = 0;
          dq1[v] = 0;
        } else {
          eb_id[v] = id;
          pred_mask[v] = 1;
          *curU = dec[0];
          *curV = dec[1];
        }
      }
    }
  };

  for (int t = 0; t < (int)Tt; ++t) {
    int Smax = (int)(nbH + nbW - 2);
    for (int s = 0; s <= Smax; ++s) {
      int bi_lo = std::max(0, s - (int)nbW + 1);
      int bi_hi = std::min((int)nbH - 1, s);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int bi = bi_lo; bi <= bi_hi; ++bi) {
        int bj = s - bi;
        process_tile(t, bi, bj);
      }
    }
  }

  std::vector<T_data> unpred;
  unpred.reserve((N / 8 + 1) * 2);
  std::vector<int> data_quant_index;
  data_quant_index.reserve(2 * N);

  size_t unpred_cnt = 0;
  for (size_t v = 0; v < N; ++v) {
    if (!pred_mask[v]) unpred_cnt += 2;
  }

  unsigned char* compressed = (unsigned char*)std::malloc((size_t)(2 * N * sizeof(T)));
  if (!compressed) {
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }
  unsigned char* pos = compressed;

  write_variable_to_dst(pos, scale);
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);

  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) {
    for (size_t v = 0; v < N; ++v) {
      if (!pred_mask[v]) {
        unpred.push_back(U[v]);
        unpred.push_back(V[v]);
      }
    }
    write_array_to_dst(pos, unpred.data(), unpred.size());
  }

  size_t eb_quant_num = N;
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * 1024, eb_id.data(), eb_quant_num, pos);

  for (size_t v = 0; v < N; ++v) {
    if (pred_mask[v]) {
      data_quant_index.push_back(dq0[v]);
      data_quant_index.push_back(dq1[v]);
    }
  }
  size_t data_quant_num = data_quant_index.size();
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * capacity, data_quant_index.data(), data_quant_num, pos);

  compressed_size = (size_t)(pos - compressed);

  std::free(U_fp);
  std::free(V_fp);
  return compressed;
}


template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_warped_lorenzo(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb,                  // 全局绝对误差上限（浮点域）：max_eb(fp)=max_pwr_eb
    EbMode mode 
){
  // Warped-Lorenzo 运行参数（编解码两端需一致；也可写入码流头部）
  const double WARP_ALPHA_X = 0.77;//1.49;   // 像素/帧 = alpha * u,v  （若你的u,v单位即“像素/帧”，取1.0）
  const double WARP_ALPHA_Y = 0.73;//1.49;   // 像素/帧 = alpha * u,v  （若你的u,v单位即“像素/帧”，取1.0）
  const double WARP_DMAX  = 6.0;   // 位移裁剪（像素）
  const double WARP_GATE  = 0.5;  // 小位移门槛（像素）；<0.30像素时退化为0，省去无谓的插值抖动
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

          }
        }

        // (C) 内部剖分面：两片 ts ∈ {t, t-1}；每相邻 cell 的 Upper/Lower 各 2 面
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
        *eb_pos = id;   // 先写代表值 id（abs_eb 已被代表值替换）
        {
          bool unpred_flag=false;
          T dec[2];
          T abs_err_fp_q[2] = {0,0};

          for (int p=0; p<2; ++p){
            T *cur  = (p==0)? curU : curV;
            T  curv = (p==0) ? curU_val : curV_val;

            // ---- Warped‑Lorenzo 预测 ----
            T pred = 0;
            const T* recon_t_plane =
                (p==0) ? (U_fp + (size_t)t*H*W) : (V_fp + (size_t)t*H*W);

            if (t == 0){
              // t=0：退化为 2D Lorenzo
              if (i>0 && j>0) {
                pred = recon_t_plane[(size_t)(i-1)*W + j]
                     + recon_t_plane[(size_t)i*W + (j-1)]
                     - recon_t_plane[(size_t)(i-1)*W + (j-1)];
              } else if (i>0) {
                pred = recon_t_plane[(size_t)(i-1)*W + j];
              } else if (j>0) {
                pred = recon_t_plane[(size_t)i*W + (j-1)];
              } else {
                pred = 0;
              }
            } else {
              const T* recon_tm1_plane =
                  (p==0) ? (U_fp + (size_t)(t-1)*H*W) : (V_fp + (size_t)(t-1)*H*W);
              const T* U_tm1_plane = U_fp + (size_t)(t-1)*H*W; // 位移来源：上一帧速度
              const T* V_tm1_plane = V_fp + (size_t)(t-1)*H*W;

              pred = warp_pred::predict<T>(
                  /*recon_t   =*/recon_t_plane,
                  /*recon_tm1 =*/recon_tm1_plane,
                  /*U_tm1     =*/U_tm1_plane,
                  /*V_tm1     =*/V_tm1_plane,
                  (int)H, (int)W, i, j,
                  /*scale     =*/scale,
                  /*alphaX   =*/WARP_ALPHA_X,
                  /*alphaY   =*/WARP_ALPHA_Y,
                  /*dmax      =*/WARP_DMAX,
                  /*gate_pix  =*/WARP_GATE
              );
            }

            // 量化当前分量
            T diff = curv - pred;
            T qd = (std::llabs(diff)/abs_eb) + 1;
            if (qd < capacity){
              qd = (diff > 0) ? qd : -qd;
              int qindex = (int)(qd/2) + intv_radius;
              dq_pos[p] = qindex;
              dec[p] = pred + 2*(qindex - intv_radius)*abs_eb;

              if (std::llabs(dec[p] - curv) > abs_eb){
                unpred_flag = true; break;
              }
              abs_err_fp_q[p] = std::llabs(dec[p] - curv);
            } else {
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

  std::cerr << "[ENC] max abs_eb(fp) = " << enc_max_abs_eb_fp
            << ", max actual |err|(fp) = " << enc_max_real_err_fp << "\n";

  
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

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_AR2_2D_LORENZO(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb,                 // 全局绝对误差上限（浮点域）
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
  const T threshold = 1;  

  // 2) 预计算：全局 CP 面集合（一次性）
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // 3) 量化/编码缓冲
  int* eb_quant_index  = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index= (int*)std::malloc(2*N*sizeof(int)); // U/V 交错
  double enc_max_abs_eb_fp   = 0.0; // 编码端自检（浮点）
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred; unpred.reserve((N/10)*2);

  // 4) 参数与量化槽
  const int base = 2;
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);

  // 5) 把浮点域绝对误差上限转到定点域（LSB）
  T max_eb = 0;
  if(mode == EbMode::Relative){
    max_eb = (T)std::llround((long double)max_pwr_eb * (long double)range); // 相对 → 绝对
    printf("Compression Using Relative Eb Mode!\n");
  } else if (mode == EbMode::Absolute){
    max_eb = (T)std::llround((long double)max_pwr_eb * (long double)scale); // 浮点 → 定点
    printf("Compression Using Absolute Eb Mode!\n");
  } else{
    std::cerr << "Error: Unsupported EbMode!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  if (max_eb <= 0){
    std::cerr << "Error: max_eb <= 0 after scaling!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }

  // 6) 下标步长 & 邻接
  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W; // 层间位移
  const int di[6] = { 0,  0,  1, -1,  1, -1 }; //y-axis
  const int dj[6] = { 1, -1,  0,  0,  1, -1 }; //x-axis

  // 7) —— A 方案预测器 ——（新增）
  // 时间 AR(2) 预测，输入为“已重构”数据
  auto time_pred_at = [&](const T* base, size_t idx, int t)->T{
    if (t <= 0) return (T)0;            // t==0，用 0 冷启动
    if (t == 1) return base[idx - sk];  // t==1，用 AR(1): x_{t-1}
    return (T)(2*base[idx - sk] - base[idx - 2*sk]); // t>=2, AR(2)
  };
  // 残差的 2D Lorenzo：e_pred = e(i-1,j) + e(i,j-1) - e(i-1,j-1)
  auto e2d_pred_at = [&](const T* base, size_t idx, int t, int i, int j)->T{
    T s = 0;
    if (i > 0){
      size_t up = idx - si;
      s += (base[up] - time_pred_at(base, up, t));
    }
    if (j > 0){
      size_t lf = idx - sj;
      s += (base[lf] - time_pred_at(base, lf, t));
    }
    if (i > 0 && j > 0){
      size_t ul = idx - si - sj;
      s -= (base[ul] - time_pred_at(base, ul, t));
    }
    return s;
  };

  // 8) 主循环
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

        // —— 收集该点的“最小可行 eb”（逐面枚举）——
        T required_eb = max_eb;
        
        // (A) 层内 t：影响 (i-1..i, j-1..j) 的 4 个 cell，每 cell 两三角
        for (int ci=i-1; ci<=i; ++ci){
          if (!in_range(ci, (int)H-1)) continue;
          for (int cj=j-1; cj<=j; ++cj){
            if (!in_range(cj, (int)W-1)) continue;

            size_t v00 = vid(t,ci,  cj,  sz);
            size_t v10 = vid(t,ci,  cj+1,sz);
            size_t v01 = vid(t,ci+1,cj,  sz);
            size_t v11 = vid(t,ci+1,cj+1,sz);

            // Upper: (v00,v01,v11)
            if (v==v00 || v==v01 || v==v11){
              if (has_cp(cp_faces, v00,v01,v11)) { required_eb = 0; }
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v01],U_fp[v00],
                                                    V_fp[v11],V_fp[v01],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
            // Lower: (v00,v10,v11)
            if (v==v00 || v==v10 || v==v11){
              if (has_cp(cp_faces, v00,v10,v11)) { required_eb = 0; }
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v10],U_fp[v00],
                                                    V_fp[v11],V_fp[v10],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
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
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                      V_fp[bp],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[bp],U_fp[a],
                                                      V_fp[ap],V_fp[bp],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
            } else{
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                      V_fp[ap],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
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
            size_t ap = a - dv, bp = b - dv; //上一层
            if (k == 0 || k==3 || k==5){
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                      V_fp[ap],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
            } else{
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                      V_fp[bp],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, a,ap,bp)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[ap],U_fp[a],
                                                      V_fp[bp],V_fp[ap],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }

        // (C1) [t, t+1] 内部剖分面（与原实现一致）
        if (t < (int)Tt-1){
          // tri1
          size_t f1a = vid(t,  i,  j,  sz);
          size_t f1b = vid(t,  i,j+1,  sz);
          size_t f1c = vid(t,  i-1,j,  sz);
          size_t f1ap = f1a + dv, f1bp = f1b + dv, f1cp = f1c + dv;
          if (in_range(i-1,(int)H) && in_range(j+1,(int)W)){
            if (has_cp(cp_faces, f1a,f1cp,f1b)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1b],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1b],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f1a,f1cp,f1bp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1bp],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1bp],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri2
          size_t f2a = vid(t,  i,  j,  sz);
          size_t f2b = vid(t,  i-1,j,  sz);
          size_t f2c = vid(t,  i-1,j-1,sz);
          size_t f2ap = f2a + dv, f2bp = f2b + dv, f2cp = f2c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            if (has_cp(cp_faces, f2a,f2c,f2bp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2bp],U_fp[f2c],U_fp[f2a],
                                                    V_fp[f2bp],V_fp[f2c],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f2a,f2bp,f2cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2cp],U_fp[f2bp],U_fp[f2a],
                                                    V_fp[f2cp],V_fp[f2bp],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri3
          size_t f3a = vid(t,  i,  j,  sz);
          size_t f3b = vid(t,  i-1,j-1,sz);
          size_t f3c = vid(t,  i,j-1,  sz);
          size_t f3ap = f3a + dv, f3bp = f3b + dv, f3cp = f3c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            if (has_cp(cp_faces, f3a,f3bp,f3c)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3c],U_fp[f3bp],U_fp[f3a],
                                                    V_fp[f3c],V_fp[f3bp],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri6
          size_t f6a = vid(t,  i,  j,  sz);
          size_t f6b = vid(t,  i,j+1,  sz);
          size_t f6c = vid(t,  i+1,j+1,sz);
          size_t f6ap = f6a + dv, f6bp = f6b + dv, f6cp = f6c + dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            if (has_cp(cp_faces, f6a,f6bp,f6c)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6c],U_fp[f6bp],U_fp[f6a],
                                                    V_fp[f6c],V_fp[f6bp],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        // (C2) [t-1, t] 内部剖分面
        if (t > 0){
          // tri3
          size_t f3a = vid(t, i,  j,  sz);
          size_t f3b = vid(t, i-1,j-1,sz);
          size_t f3c = vid(t, i,  j-1,sz);
          size_t f3ap = f3a - dv, f3bp = f3b - dv, f3cp = f3c - dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            if (has_cp(cp_faces, f3a,f3b,f3cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3cp],U_fp[f3b],U_fp[f3a],
                                                    V_fp[f3cp],V_fp[f3b],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri4
          size_t f4a = vid(t, i,  j,  sz);
          size_t f4b = vid(t, i,  j-1,sz);
          size_t f4c = vid(t, i+1,j,  sz);
          size_t f4ap = f4a - dv, f4bp = f4b - dv, f4cp = f4c - dv;
          if (in_range(i+1,(int)H) && in_range(j-1,(int)W)){
            if (has_cp(cp_faces, f4a,f4bp,f4cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4bp],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4bp],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f4a,f4b,f4cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4b],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4b],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri5
          size_t f5a = vid(t, i,  j,  sz);
          size_t f5b = vid(t, i+1,j,  sz);
          size_t f5c = vid(t, i+1,j+1,sz);
          size_t f5ap = f5a - dv, f5bp = f5b - dv, f5cp = f5c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            if (has_cp(cp_faces, f5a,f5bp,f5cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5cp],U_fp[f5bp],U_fp[f5a],
                                                    V_fp[f5cp],V_fp[f5bp],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f5a,f5c,f5bp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5bp],U_fp[f5c],U_fp[f5a],
                                                    V_fp[f5bp],V_fp[f5c],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri6
          size_t f6a = vid(t, i,  j,  sz);
          size_t f6b = vid(t, i,  j+1,sz);
          size_t f6c = vid(t, i+1,j+1,sz);
          size_t f6ap = f6a - dv, f6bp = f6b - dv, f6cp = f6c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            if (has_cp(cp_faces, f6a,f6b,f6cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6cp],U_fp[f6b],U_fp[f6a],
                                                    V_fp[f6cp],V_fp[f6b],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        // === 逐点 eb 完成 ===
        T abs_eb = required_eb;                         // 用枚举得到的点级别 eb
        int id = eb_exponential_quantize(abs_eb,        // 注意：与解码端保持一致
                                         base, log_of_base, threshold);
        // 如果 abs_eb==0 → 该点必须无损直存
        if (abs_eb == 0){
          *(eb_pos++) = 0;            // eb-id = 0 → 无损点
          unpred.push_back(U[v]);     // 原始浮点（解码端直接回填）
          unpred.push_back(V[v]);
          continue;
        }
        *eb_pos = id; // 写入该点的 eb-id（Huffman 之后很短）

        // ===== 量化该点 U/V（预测器：AR(2) + 残差 2D Lorenzo）=====
        bool unpred_flag=false;
        T dec[2];
        T abs_err_fp_q[2] = {0,0};

        for (int p=0; p<2; ++p){
          T *base = (p==0)? U_fp : V_fp;
          T  curv = (p==0) ? curU_val : curV_val;

          // 时间预测 p_t(i,j)
          T p_tij = time_pred_at(base, v, t);
          // 残差的 2D Lorenzo 预测
          T epred = e2d_pred_at(base, v, t, i, j);

          // 对“残差的残差”做区间量化
          T diff = (curv - p_tij) - epred;
          T qd = (std::llabs(diff)/abs_eb) + 1;
          if (qd < capacity){
            qd = (diff > 0) ? qd : -qd;
            int qindex = (int)(qd/2) + intv_radius;
            dq_pos[p] = qindex;

            // 解码侧对应：ehat = epred + 2*(qindex-intv_radius)*abs_eb; xhat = p_tij + ehat
            T ehat = epred + (T)2*(qindex - intv_radius)*abs_eb;
            T xhat = p_tij + ehat;

            // 守门：必须满足 |xhat - x| <= abs_eb
            if (std::llabs(xhat - curv) > abs_eb){
              unpred_flag = true; break;
            }
            dec[p] = xhat;
            abs_err_fp_q[p] = std::llabs(xhat - curv);
          }else{
            unpred_flag = true; break;
          }
        }

        if (unpred_flag){
          *(eb_pos++) = 0;                  // 覆盖为无损
          unpred.push_back(U[v]);
          unpred.push_back(V[v]);
        }else{
          ++eb_pos;
          dq_pos += 2;
          *curU = dec[0];                   // 回写“已重构值”，供后续 i,j,t 的预测使用
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

  // 9) 打包码流（与原版一致）
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  std::cout << "write unpred cnt = " << unpred_cnt << ",ratio=" << (double)unpred_cnt/(2*N) << "\n";
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

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_3DL_AR2(
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
  double alpha;
  double beta;
    {
      // 用前 K 帧做估计（3~200 帧），按你数据量设置；K>=3 才能估计到 gamma_2
      const int K = (int)std::min<size_t>(200,Tt);
      if (K >= 3) {
          // 为避免与主循环中的 si/sj/sk 重名，这里用 e 开头
          const ptrdiff_t esi=(ptrdiff_t)W, esj=(ptrdiff_t)1, esk=(ptrdiff_t)(H*W);

          // 纯 3D Lorenzo 预测（基于“原始定点场”），不回写
          auto pred3D_src = [&](const T* cur, int t, int i, int j)->T {
              T d0 = (t&&i&&j)? cur[-esk - esi - esj] : 0;
              T d1 = (t&&i)   ? cur[-esk - esi]      : 0;
              T d2 = (t&&j)   ? cur[-esk - esj]      : 0;
              T d3 = (t)      ? cur[-esk]            : 0;
              T d4 = (i&&j)   ? cur[-esi - esj]      : 0;
              T d5 = (i)      ? cur[-esi]            : 0;
              T d6 = (j)      ? cur[-esj]            : 0;
              return d0 + d3 + d5 + d6 - d1 - d2 - d4;
          };

          // 只需缓存两片历史残差（U/V 各两片）
          std::vector<T> r1U(H*W,(T)0), r2U(H*W,(T)0), rcurU(H*W,(T)0);
          std::vector<T> r1V(H*W,(T)0), r2V(H*W,(T)0), rcurV(H*W,(T)0);

          long double g0 = 0.0L, g1 = 0.0L, g2 = 0.0L; // Σ r_t^2, Σ r_t r_{t-1}, Σ r_t r_{t-2}

          for (int t=0; t<K; ++t) {
              for (int i=0; i<(int)H; ++i){
                  for (int j=0; j<(int)W; ++j){
                      const size_t v = vid(t,i,j,sz);
                      // 残差 = 原始值 - 纯 L3D 预测（均在定点域）
                      const T rU = U_fp[v] - pred3D_src(U_fp + v, t,i,j);
                      const T rV = V_fp[v] - pred3D_src(V_fp + v, t,i,j);
                      const size_t idx2D = (size_t)i*W + (size_t)j;
                      rcurU[idx2D] = rU;
                      rcurV[idx2D] = rV;

                      if (t >= 2){
                          const T r1u = r1U[idx2D], r2u = r2U[idx2D];
                          const T r1v = r1V[idx2D], r2v = r2V[idx2D];

                          // U 分量计入自/互协方差
                          g0 += (long double)rU * (long double)rU;
                          g1 += (long double)rU * (long double)r1u;
                          g2 += (long double)rU * (long double)r2u;
                          // V 分量同样计入（合到一起更稳）
                          g0 += (long double)rV * (long double)rV;
                          g1 += (long double)rV * (long double)r1v;
                          g2 += (long double)rV * (long double)r2v;
                      }
                  }
              }
              // 滚动（t→t-1→t-2）
              std::swap(r2U, r1U); std::swap(r1U, rcurU);
              std::swap(r2V, r1V); std::swap(r1V, rcurV);
          }

          // Yule–Walker 2x2： [g0 g1; g1 g0] [α;β] = [g1; g2]
          double alpha = 0.0, beta = 0.0;
          const long double lam = 1e-9L * (fabsl(g0) + 1.0L); // 轻微正则
          const long double a11 = g0 + lam, a22 = g0 + lam, a12 = g1, a21 = g1;
          const long double b1  = g1,      b2  = g2;
          const long double det = a11*a22 - a12*a21;

          double rho1 = (g0 != 0) ? (double)(g1 / g0) : 0.0;
          double rho2 = (g1 != 0) ? (double)(g2 / g1) : 0.0;
          std::cout << "[AR2-DIAG] rho1="<<rho1<<" rho2="<<rho2
          << " det="<< (double)det << " g0="<<(double)g0
          << " g1="<<(double)g1 << " g2="<<(double)g2 << "\n";

          if (fabsl(det) > 1e-30L) {
              alpha = (double)((b1*a22 - b2*a12)/det);
              beta  = (double)((a11*b2 - a21*b1)/det);
          } else {
              // 退化：用 AR(1) ρ 的双重根近似到 AR(2)
              double rho = 0.0;
              if (g0 > 0) rho = (double)(g1 / g0);
              rho = std::max(-0.99, std::min(0.99, rho));
              alpha = 2.0*rho;
              beta  = -rho*rho;
          }

          // 稳定性投影（AR(2) 的稳定三角： |β|<1, α∈(β-1, 1-β) ）
          const double eps = 1e-6;
          beta = std::max(-1.0 + eps, std::min(1.0 - eps, beta));
          const double amin = beta - 1.0 + eps;
          const double amax = 1.0 - beta - eps;
          if (alpha < amin) alpha = amin;
          if (alpha > amax) alpha = amax;
          std::cout << "[AR2-EST] K="<<K
                    << " alpha="<<alpha<<" beta="<<beta<<"\n";
      }
  }
  // === PATCH [AR2-EST] end ===

  // 参数与量化槽
  const int base = 2;                             // 仍写头，方便向后兼容
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);


  std::vector<T> r1U(H*W, (T)0), r2U(H*W, (T)0), rcurU(H*W, (T)0);
  std::vector<T> r1V(H*W, (T)0), r2V(H*W, (T)0), rcurV(H*W, (T)0);

  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W; // 层间位移（你已有）

  // 3D 一阶 Lorenzo 主预测（时间最慢）
  auto pred3D = [&](const T* cur, int t, int i, int j)->T {
      T d0 = (t&&i&&j)? cur[-sk - si - sj] : 0;
      T d1 = (t&&i)   ? cur[-sk - si]      : 0;
      T d2 = (t&&j)   ? cur[-sk - sj]      : 0;
      T d3 = (t)      ? cur[-sk]           : 0;
      T d4 = (i&&j)   ? cur[-si - sj]      : 0;
      T d5 = (i)      ? cur[-si]           : 0;
      T d6 = (j)      ? cur[-sj]           : 0;
      return d0 + d3 + d5 + d6 - d1 - d2 - d4;
  };

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

          }
        }

        // (C) 内部剖分面：两片 ts ∈ {t, t-1}；每相邻 cell 的 Upper/Lower 各 2 面
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

          // === PATCH [AR2]：L3D 主预测 + 残差 AR(2) ===
          const size_t idx2D = (size_t)i * W + (size_t)j;

          // 主预测（基于已回写邻域）
          const T predL3D_U = pred3D(U_fp + v, t, i, j);
          const T predL3D_V = pred3D(V_fp + v, t, i, j);

          // 残差的 AR(2) 预测
          auto ar_pred = [&](T r1, T r2) -> T {
              double num = alpha * static_cast<double>(r1)
                        + beta * static_cast<double>(r2);
              return static_cast<T>(llround(num)); 
          };
          const T arU = ar_pred(r1U[idx2D], r2U[idx2D]);
          const T arV = ar_pred(r1V[idx2D], r2V[idx2D]);

          const T predU_final = predL3D_U + arU;
          const T predV_final = predL3D_V + arV;

          // eb==0（必保点/退化）仍走无损，但要更新当前帧残差供下一帧使用
          if (abs_eb == 0){
            *(eb_pos++) = 0;
            unpred.push_back(U[v]);
            unpred.push_back(V[v]);
            rcurU[idx2D] = (T)curU_val - predL3D_U;   // 按 L3D 记录残差
            rcurV[idx2D] = (T)curV_val - predL3D_V;
            continue;
          }

          // U 分量量化
          {
            const T diff = (T)curU_val - predU_final;
            T qd = (std::llabs(diff)/abs_eb) + 1;
            if (qd < capacity){
              qd = (diff > 0) ? qd : -qd;
              int qindex = (int)(qd/2) + intv_radius;
              dq_pos[0] = qindex;
              dec[0] = predU_final + 2*(qindex - intv_radius)*abs_eb;
              if (std::llabs(dec[0] - (T)curU_val) > abs_eb) unpred_flag = true;
              else abs_err_fp_q[0] = std::llabs(dec[0] - (T)curU_val);
            }else{
              unpred_flag = true;
            }
          }

          // V 分量量化
          if (!unpred_flag){
            const T diff = (T)curV_val - predV_final;
            T qd = (std::llabs(diff)/abs_eb) + 1;
            if (qd < capacity){
              qd = (diff > 0) ? qd : -qd;
              int qindex = (int)(qd/2) + intv_radius;
              dq_pos[1] = qindex;
              dec[1] = predV_final + 2*(qindex - intv_radius)*abs_eb;
              if (std::llabs(dec[1] - (T)curV_val) > abs_eb) unpred_flag = true;
              else abs_err_fp_q[1] = std::llabs(dec[1] - (T)curV_val);
            }else{
              unpred_flag = true;
            }
          }

          if (unpred_flag){
            *(eb_pos++) = 0;
            unpred.push_back(U[v]);
            unpred.push_back(V[v]);

            // 更新当前帧残差（按 L3D）
            rcurU[idx2D] = (T)curU_val - predL3D_U;
            rcurV[idx2D] = (T)curV_val - predL3D_V;
          }else{
            ++eb_pos;
            dq_pos += 2;
            *curU = dec[0];
            *curV = dec[1];

            // 记录“已重建-主预测(L3D)”残差，供下一帧 AR(2)
            rcurU[idx2D] = dec[0] - predL3D_U;
            rcurV[idx2D] = dec[1] - predL3D_V;

            // 编码端自检（保持不变）
            double abs_eb_fp = (double)abs_eb / (double)scale;
            double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
            double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
            enc_max_abs_eb_fp   = std::max(enc_max_abs_eb_fp, abs_eb_fp);
            enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
          }
        }
      }
    }
    // === PATCH [AR2] 帧末滚动：rcur -> r1 -> r2
    std::swap(r2U, r1U); std::swap(r1U, rcurU);
    std::swap(r2V, r1V); std::swap(r1V, rcurV);
    std::fill(rcurU.begin(), rcurU.end(), (T)0);
    std::fill(rcurV.begin(), rcurV.end(), (T)0);
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
  std::cout << "write unpred cnt = " << unpred_cnt << ",ratio=" << (double)unpred_cnt/(2*N) << "\n";
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



#if 0
template<typename T_data, typename LayerFetcher>
static unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_streaming_impl(
    LayerFetcher&& fetch_layer,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  using T = int64_t;
  const size_t H = r1, W = r2, Tt = r3;
  if (!H || !W || !Tt) {
    compressed_size = 0;
    return nullptr;
  }

  const Size3 sz{(int)H, (int)W, (int)Tt};
  const size_t layer_size = H * W;

  std::vector<T_data> tmpU(layer_size);
  std::vector<T_data> tmpV(layer_size);

  // -------- Pass 1: determine scaling factor --------
  double vector_field_resolution = 0.0;
  for (size_t t = 0; t < Tt; ++t) {
    if (!fetch_layer(t, tmpU.data(), tmpV.data())) {
      compressed_size = 0;
      return nullptr;
    }
    for (size_t idx = 0; idx < layer_size; ++idx) {
      double max_val = std::max(std::fabs((double)tmpU[idx]), std::fabs((double)tmpV[idx]));
      if (max_val > vector_field_resolution) vector_field_resolution = max_val;
    }
  }

  const int type_bits = 63;
  const int nbits = (type_bits - 3) / 2;
  int vbits = 0;
  if (vector_field_resolution > 0.0) {
    vbits = (int)std::ceil(std::log2(vector_field_resolution));
  }
  int shift_bits = nbits - vbits;
  if (shift_bits < 0) shift_bits = 0;
  T scale = (T)1 << shift_bits;

  std::cerr << "resolution=" << vector_field_resolution
            << ", factor=" << (long long)scale
            << ", nbits=" << nbits
            << ", vbits=" << vbits
            << ", shift_bits=" << shift_bits << std::endl;

  T fp_max = std::numeric_limits<T>::min();
  T fp_min = std::numeric_limits<T>::max();
  printf("max = %lld, min = %lld\n", (long long)fp_max, (long long)fp_min);

  auto load_layer_fixed = [&](size_t t,
                              std::vector<T>& outU,
                              std::vector<T>& outV,
                              std::vector<T_data>* storeU,
                              std::vector<T_data>* storeV,
                              bool update_minmax)->bool {
    if (!fetch_layer(t, tmpU.data(), tmpV.data())) {
      return false;
    }
    for (size_t idx = 0; idx < layer_size; ++idx) {
      T u_fp = static_cast<T>(tmpU[idx] * (double)scale);
      T v_fp = static_cast<T>(tmpV[idx] * (double)scale);
      outU[idx] = u_fp;
      outV[idx] = v_fp;
      if (storeU) (*storeU)[idx] = tmpU[idx];
      if (storeV) (*storeV)[idx] = tmpV[idx];
      if (update_minmax) {
        if (u_fp > fp_max) fp_max = u_fp;
        if (v_fp > fp_max) fp_max = v_fp;
        if (u_fp < fp_min) fp_min = u_fp;
        if (v_fp < fp_min) fp_min = v_fp;
      }
    }
    return true;
  };

  auto pre_compute_time = std::chrono::high_resolution_clock::now();

  std::vector<T> u_prev(layer_size, 0), v_prev(layer_size, 0);
  std::vector<T> u_curr(layer_size, 0), v_curr(layer_size, 0);
  std::vector<T> u_next(layer_size, 0), v_next(layer_size, 0);

  if (!load_layer_fixed(0, u_curr, v_curr, nullptr, nullptr, true)) {
    compressed_size = 0;
    return nullptr;
  }
  if (Tt > 1) {
    if (!load_layer_fixed(1, u_next, v_next, nullptr, nullptr, true)) {
      compressed_size = 0;
      return nullptr;
    }
  }

  std::unordered_set<FaceKeySZ, FaceKeySZHash> cp_faces;
  cp_faces.reserve((size_t)(H*(size_t)W*(size_t)Tt / 8));

  auto eval_face = [&](size_t t_base,
                       size_t a,size_t b,size_t c)->bool {
    int idxs[3] = { (int)a, (int)b, (int)c };
    int64_t vf[3][2];
    auto fetch_value = [&](size_t vidx, int pos){
      int vt, vi, vj;
      inv_vid(vidx, (int)H, (int)W, vt, vi, vj);
      size_t off = (size_t)vi * W + (size_t)vj;
      const std::vector<T>* su = nullptr;
      const std::vector<T>* sv = nullptr;
      if (vt == (int)t_base) {
        su = &u_curr; sv = &v_curr;
      } else if (vt == (int)t_base + 1) {
        su = &u_next; sv = &v_next;
      } else if (vt == (int)t_base - 1) {
        su = &u_prev; sv = &v_prev;
      } else {
        su = &u_curr; sv = &v_curr;
      }
      vf[pos][0] = (*su)[off];
      vf[pos][1] = (*sv)[off];
    };
    fetch_value(a, 0);
    fetch_value(b, 1);
    fetch_value(c, 2);
    return ftk::robust_critical_point_in_simplex2(vf, idxs);
  };

  const size_t dv = H * W;

  #ifdef _OPENMP
  const size_t block_rows_cp = 32;
  const size_t block_cols_cp = 32;
  const int cp_threads = omp_get_max_threads();
  std::vector<std::vector<FaceKeySZ>> cp_thread_faces(cp_threads);
  auto flush_thread_faces = [&](std::vector<std::vector<FaceKeySZ>>& buffers){
    for (auto &vec : buffers){
      for (const auto &f : vec){
        cp_faces.emplace(f);
      }
      vec.clear();
    }
  };
  #endif

  for (size_t t = 0; t < Tt; ++t) {
    if (t % 1000 == 0) {
      printf("pre-compute cp lower layer %zu / %zu\n", t, Tt);
    }

    #ifdef _OPENMP
    const size_t layer_i_limit = (H > 0) ? (H - 1) : 0;
    const size_t layer_j_limit = (W > 0) ? (W - 1) : 0;
    const size_t blocks_i_layer = (layer_i_limit + block_rows_cp - 1) / block_rows_cp;
    const size_t blocks_j_layer = (layer_j_limit + block_cols_cp - 1) / block_cols_cp;
    const size_t blocks_i_full  = (H + block_rows_cp - 1) / block_rows_cp;
    const size_t blocks_j_fullm1 = (W > 0) ? ((W - 1 + block_cols_cp - 1) / block_cols_cp) : 0;
    const size_t blocks_i_m1 = (H > 0) ? ((H - 1 + block_rows_cp - 1) / block_rows_cp) : 0;
    const size_t blocks_j_full = (W + block_cols_cp - 1) / block_cols_cp;

    #pragma omp parallel
    {
      auto& local_faces = cp_thread_faces[omp_get_thread_num()];
      local_faces.clear();

      if (layer_i_limit && layer_j_limit) {
        const size_t total_blocks = blocks_i_layer * blocks_j_layer;
        #pragma omp for schedule(static)
        for (size_t block = 0; block < total_blocks; ++block) {
          size_t bi = block / blocks_j_layer;
          size_t bj = block % blocks_j_layer;
          size_t i_begin = bi * block_rows_cp;
          size_t i_end = std::min(i_begin + block_rows_cp, layer_i_limit);
          size_t j_begin = bj * block_cols_cp;
          size_t j_end = std::min(j_begin + block_cols_cp, layer_j_limit);
          for (size_t i = i_begin; i < i_end; ++i) {
            for (size_t j = j_begin; j < j_end; ++j) {
              size_t v00 = vid((int)t,(int)i,(int)j,sz);
              size_t v10 = vid((int)t,(int)i,(int)(j+1),sz);
              size_t v01 = vid((int)t,(int)(i+1),(int)j,sz);
              size_t v11 = vid((int)t,(int)(i+1),(int)(j+1),sz);
              if (eval_face(t, v00, v01, v11)) local_faces.emplace_back(v00, v01, v11);
              if (eval_face(t, v00, v10, v11)) local_faces.emplace_back(v00, v10, v11);
            }
          }
        }
      }

      if (t + 1 < Tt) {
        if (t % 1000 == 0) {
          #pragma omp single
          printf("pre-compute cp side_hor layer %zu / %zu\n", t, Tt);
        }
        if (blocks_i_full && (W > 1)) {
          const size_t total_blocks_h = blocks_i_full * blocks_j_layer;
          #pragma omp for schedule(static)
          for (size_t block = 0; block < total_blocks_h; ++block) {
            size_t bi = block / blocks_j_layer;
            size_t bj = block % blocks_j_layer;
            size_t i_begin = bi * block_rows_cp;
            size_t i_end = std::min(i_begin + block_rows_cp, H);
            size_t j_begin = bj * block_cols_cp;
            size_t j_end = std::min(j_begin + block_cols_cp, W - 1);
            for (size_t i = i_begin; i < i_end; ++i) {
              for (size_t j = j_begin; j < j_end; ++j) {
                size_t a = vid((int)t,(int)i,(int)j,sz);
                size_t b = vid((int)t,(int)i,(int)(j+1),sz);
                size_t ap = a + dv;
                size_t bp = b + dv;
                if (eval_face(t, a, b, bp)) local_faces.emplace_back(a, b, bp);
                if (eval_face(t, a, bp, ap)) local_faces.emplace_back(a, bp, ap);
              }
            }
          }
        }

        if (t % 1000 == 0) {
          #pragma omp single
          printf("pre-compute cp side_ver layer %zu / %zu\n", t, Tt);
        }
        if (blocks_i_m1 && blocks_j_full) {
          const size_t total_blocks_v = blocks_i_m1 * blocks_j_full;
          #pragma omp for schedule(static)
          for (size_t block = 0; block < total_blocks_v; ++block) {
            size_t bi = block / blocks_j_full;
            size_t bj = block % blocks_j_full;
            size_t i_begin = bi * block_rows_cp;
            size_t i_end = std::min(i_begin + block_rows_cp, (H > 0) ? (H - 1) : 0);
            size_t j_begin = bj * block_cols_cp;
            size_t j_end = std::min(j_begin + block_cols_cp, W);
            for (size_t i = i_begin; i < i_end; ++i) {
              for (size_t j = j_begin; j < j_end; ++j) {
                size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
                size_t ax0y1 = vid((int)t,(int)(i+1),(int)j,sz);
                size_t bx0y0 = ax0y0 + dv;
                size_t bx0y1 = ax0y1 + dv;
                if (eval_face(t, ax0y0, ax0y1, bx0y0)) local_faces.emplace_back(ax0y0, ax0y1, bx0y0);
                if (eval_face(t, ax0y1, bx0y0, bx0y1)) local_faces.emplace_back(ax0y1, bx0y0, bx0y1);
              }
            }
          }
        }

        if (t % 1000 == 0) {
          #pragma omp single
          printf("pre-compute cp diag layer %zu / %zu\n", t, Tt);
        }
        if (layer_i_limit && layer_j_limit) {
          const size_t total_blocks_d = blocks_i_layer * blocks_j_layer;
          #pragma omp for schedule(static)
          for (size_t block = 0; block < total_blocks_d; ++block) {
            size_t bi = block / blocks_j_layer;
            size_t bj = block % blocks_j_layer;
            size_t i_begin = bi * block_rows_cp;
            size_t i_end = std::min(i_begin + block_rows_cp, layer_i_limit);
            size_t j_begin = bj * block_cols_cp;
            size_t j_end = std::min(j_begin + block_cols_cp, layer_j_limit);
            for (size_t i = i_begin; i < i_end; ++i) {
              for (size_t j = j_begin; j < j_end; ++j) {
                size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
                size_t ax1y1 = vid((int)t,(int)(i+1),(int)(j+1),sz);
                size_t bx0y0 = ax0y0 + dv;
                size_t bx1y1 = ax1y1 + dv;
                if (eval_face(t, ax0y0, ax1y1, bx0y0)) local_faces.emplace_back(ax0y0, ax1y1, bx0y0);
                if (eval_face(t, ax1y1, bx0y0, bx1y1)) local_faces.emplace_back(ax1y1, bx0y0, bx1y1);
              }
            }
          }
        }

        if (t % 1000 == 0) {
          #pragma omp single
          printf("pre-compute cp inside layer %zu / %zu\n", t, Tt);
        }
        if (layer_i_limit && layer_j_limit) {
          const size_t total_blocks_in = blocks_i_layer * blocks_j_layer;
          #pragma omp for schedule(static)
          for (size_t block = 0; block < total_blocks_in; ++block) {
            size_t bi = block / blocks_j_layer;
            size_t bj = block % blocks_j_layer;
            size_t i_begin = bi * block_rows_cp;
            size_t i_end = std::min(i_begin + block_rows_cp, layer_i_limit);
            size_t j_begin = bj * block_cols_cp;
            size_t j_end = std::min(j_begin + block_cols_cp, layer_j_limit);
            for (size_t i = i_begin; i < i_end; ++i) {
              for (size_t j = j_begin; j < j_end; ++j) {
                size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
                size_t ax0y1 = vid((int)t,(int)(i+1),(int)j,sz);
                size_t ax1y1 = vid((int)t,(int)(i+1),(int)(j+1),sz);
                size_t ax1y0 = vid((int)t,(int)i,(int)(j+1),sz);
                size_t bx0y0 = ax0y0 + dv;
                size_t bx0y1 = ax0y1 + dv;
                size_t bx1y1 = ax1y1 + dv;
                size_t bx1y0 = ax1y0 + dv;
                if (eval_face(t, ax0y1, bx0y0, ax1y1)) local_faces.emplace_back(ax0y1, bx0y0, ax1y1);
                if (eval_face(t, ax0y1, bx0y0, bx1y1)) local_faces.emplace_back(ax0y1, bx0y0, bx1y1);
                if (eval_face(t, ax0y0, ax1y1, bx1y0)) local_faces.emplace_back(ax0y0, ax1y1, bx1y0);
                if (eval_face(t, ax1y1, bx0y0, bx1y0)) local_faces.emplace_back(ax1y1, bx0y0, bx1y0);
              }
            }
          }
        }
      }
    }

    flush_thread_faces(cp_thread_faces);
    #else
    if (t % 1000 == 0) {
      printf("pre-compute cp lower layer %zu / %zu\n", t, Tt);
    }
    for (size_t i = 0; i + 1 < H; ++i) {
      for (size_t j = 0; j + 1 < W; ++j) {
        size_t v00 = vid((int)t,(int)i,(int)j,sz);
        size_t v10 = vid((int)t,(int)i,(int)(j+1),sz);
        size_t v01 = vid((int)t,(int)(i+1),(int)j,sz);
        size_t v11 = vid((int)t,(int)(i+1),(int)(j+1),sz);
        if (eval_face(t, v00, v01, v11)) cp_faces.emplace(v00, v01, v11);
        if (eval_face(t, v00, v10, v11)) cp_faces.emplace(v00, v10, v11);
      }
    }
    if (t + 1 < Tt) {
      if (t % 1000 == 0) {
        printf("pre-compute cp side_hor layer %zu / %zu\n", t, Tt);
      }
      for (size_t i = 0; i < H; ++i) {
        for (size_t j = 0; j + 1 < W; ++j) {
          size_t a = vid((int)t,(int)i,(int)j,sz);
          size_t b = vid((int)t,(int)i,(int)(j+1),sz);
          size_t ap = a + dv;
          size_t bp = b + dv;
          if (eval_face(t, a, b, bp)) cp_faces.emplace(a, b, bp);
          if (eval_face(t, a, bp, ap)) cp_faces.emplace(a, bp, ap);
        }
      }
      if (t % 1000 == 0) {
        printf("pre-compute cp side_ver layer %zu / %zu\n", t, Tt);
      }
      for (size_t i = 0; i + 1 < H; ++i) {
        for (size_t j = 0; j < W; ++j) {
          size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
          size_t ax0y1 = vid((int)t,(int)(i+1),(int)j,sz);
          size_t bx0y0 = ax0y0 + dv;
          size_t bx0y1 = ax0y1 + dv;
          if (eval_face(t, ax0y0, ax0y1, bx0y0)) cp_faces.emplace(ax0y0, ax0y1, bx0y0);
          if (eval_face(t, ax0y1, bx0y0, bx0y1)) cp_faces.emplace(ax0y1, bx0y0, bx0y1);
        }
      }
      if (t % 1000 == 0) {
        printf("pre-compute cp diag layer %zu / %zu\n", t, Tt);
      }
      for (size_t i = 0; i + 1 < H; ++i) {
        for (size_t j = 0; j + 1 < W; ++j) {
          size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
          size_t ax1y1 = vid((int)t,(int)(i+1),(int)(j+1),sz);
          size_t bx0y0 = ax0y0 + dv;
          size_t bx1y1 = ax1y1 + dv;
          if (eval_face(t, ax0y0, ax1y1, bx0y0)) cp_faces.emplace(ax0y0, ax1y1, bx0y0);
          if (eval_face(t, ax1y1, bx0y0, bx1y1)) cp_faces.emplace(ax1y1, bx0y0, bx1y1);
        }
      }
      if (t % 1000 == 0) {
        printf("pre-compute cp inside layer %zu / %zu\n", t, Tt);
      }
      for (size_t i = 0; i + 1 < H; ++i) {
        for (size_t j = 0; j + 1 < W; ++j) {
          size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
          size_t ax0y1 = vid((int)t,(int)(i+1),(int)j,sz);
          size_t ax1y1 = vid((int)t,(int)(i+1),(int)(j+1),sz);
          size_t ax1y0 = vid((int)t,(int)i,(int)(j+1),sz);
          size_t bx0y0 = ax0y0 + dv;
          size_t bx0y1 = ax0y1 + dv;
          size_t bx1y1 = ax1y1 + dv;
          size_t bx1y0 = ax1y0 + dv;
          if (eval_face(t, ax0y1, bx0y0, ax1y1)) cp_faces.emplace(ax0y1, bx0y0, ax1y1);
          if (eval_face(t, ax0y1, bx0y0, bx1y1)) cp_faces.emplace(ax0y1, bx0y0, bx1y1);
          if (eval_face(t, ax0y0, ax1y1, bx1y0)) cp_faces.emplace(ax0y0, ax1y1, bx1y0);
          if (eval_face(t, ax1y1, bx0y0, bx1y0)) cp_faces.emplace(ax1y1, bx0y0, bx1y0);
        }
      }
    }
    #endif

    // rotate buffers
    if (t + 1 < Tt) {
      u_prev.swap(u_curr);
      v_prev.swap(v_curr);
      u_curr.swap(u_next);
      v_curr.swap(v_next);
      if (t + 2 < Tt) {
        if (!load_layer_fixed(t + 2, u_next, v_next, nullptr, nullptr, true)) {
          compressed_size = 0;
          return nullptr;
        }
      }
    }
  }

  printf("max = %lld, min = %lld\n", (long long)fp_max, (long long)fp_min);
  const T range = fp_max - fp_min;

  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // -------- Pass 3: quantization in streaming manner --------
  int* eb_quant_index  = (int*)std::malloc((size_t)(H*W*Tt) * sizeof(int));
  int* data_quant_index= (int*)std::malloc((size_t)(2*H*W*Tt) * sizeof(int));
  if (!eb_quant_index || !data_quant_index) {
    if (eb_quant_index) std::free(eb_quant_index);
    if (data_quant_index) std::free(data_quant_index);
    compressed_size = 0;
    return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred;
  unpred.reserve((H*W*Tt/10)*2);

  double enc_max_abs_eb_fp   = 0.0;
  double enc_max_real_err_fp = 0.0;

  const int base = 2;
  const int capacity = 65536;
  const double log_of_base = std::log2(base);
  const int intv_radius = (capacity >> 1);

  T max_eb = 0;
  if (mode == EbMode::Relative) {
    printf("Compression Using Relative Eb Mode!\n");
    max_eb = (T)(max_pwr_eb * (double)range);
  } else if (mode == EbMode::Absolute) {
    printf("Compression Using Absolute Eb Mode!\n");
    max_eb = (T)(max_pwr_eb * (double)scale);
  } else {
    std::cerr << "Error: Unsupported EbMode!\n";
    std::free(eb_quant_index);
    std::free(data_quant_index);
    compressed_size = 0;
    return nullptr;
  }
  const T threshold = 1;

  std::vector<T> dec_prev_u(layer_size, 0), dec_prev_v(layer_size, 0);
  std::vector<T> dec_curr_u(layer_size, 0), dec_curr_v(layer_size, 0);
  std::vector<T> required_eb_layer(layer_size, max_eb);

  std::vector<T_data> float_prev_u(layer_size, (T_data)0);
  std::vector<T_data> float_prev_v(layer_size, (T_data)0);
  std::vector<T_data> float_curr_u(layer_size, (T_data)0);
  std::vector<T_data> float_curr_v(layer_size, (T_data)0);
  std::vector<T_data> float_next_u(layer_size, (T_data)0);
  std::vector<T_data> float_next_v(layer_size, (T_data)0);

  std::fill(u_prev.begin(), u_prev.end(), 0);
  std::fill(v_prev.begin(), v_prev.end(), 0);

  if (!load_layer_fixed(0, u_curr, v_curr, &float_curr_u, &float_curr_v, false)) {
    std::free(eb_quant_index);
    std::free(data_quant_index);
    compressed_size = 0;
    return nullptr;
  }
  if (Tt > 1) {
    if (!load_layer_fixed(1, u_next, v_next, &float_next_u, &float_next_v, false)) {
      std::free(eb_quant_index);
      std::free(data_quant_index);
      compressed_size = 0;
      return nullptr;
    }
  }

  const int di[6] = { 0,  0,  1, -1,  1, -1 };
  const int dj[6] = { 1, -1,  0,  0,  1, -1 };

  auto fetch_fixed_pair = [&](size_t t_base, size_t vidx)->std::pair<T,T> {
    int vt, vi, vj;
    inv_vid(vidx, (int)H, (int)W, vt, vi, vj);
    size_t off = (size_t)vi * W + (size_t)vj;
    if (vt == (int)t_base) {
      return { u_curr[off], v_curr[off] };
    } else if (vt == (int)t_base + 1) {
      return { u_next[off], v_next[off] };
    } else if (vt == (int)t_base - 1) {
      return { u_prev[off], v_prev[off] };
    }
    return { u_curr[off], v_curr[off] };
  };

  auto compute_required_eb = [&](size_t t_idx, size_t i_idx, size_t j_idx)->T {
    size_t global_idx = vid((int)t_idx,(int)i_idx,(int)j_idx,sz);
    T required_eb = max_eb;

    for (int ci = (int)i_idx - 1; ci <= (int)i_idx; ++ci) {
      if (!in_range(ci, (int)H - 1)) continue;
      for (int cj = (int)j_idx - 1; cj <= (int)j_idx; ++cj) {
        if (!in_range(cj, (int)W - 1)) continue;
        size_t v00 = vid((int)t_idx, ci,   cj,   sz);
        size_t v10 = vid((int)t_idx, ci,   cj+1, sz);
        size_t v01 = vid((int)t_idx, ci+1, cj,   sz);
        size_t v11 = vid((int)t_idx, ci+1, cj+1, sz);

        auto u01 = fetch_fixed_pair(t_idx, v01);
        auto u11 = fetch_fixed_pair(t_idx, v11);
        auto u10 = fetch_fixed_pair(t_idx, v10);
        auto u00 = fetch_fixed_pair(t_idx, v00);

        if (global_idx == v00 || global_idx == v01 || global_idx == v11) {
          if (has_cp(cp_faces, v00, v01, v11)) required_eb = 0;
          T eb = derive_cp_abs_eb_sos_online<T>(u11.first, u01.first, u00.first,
                                                u11.second, u01.second, u00.second);
          if (eb < required_eb) required_eb = eb;
        }
        if (global_idx == v00 || global_idx == v10 || global_idx == v11) {
          if (has_cp(cp_faces, v00, v10, v11)) required_eb = 0;
          T eb = derive_cp_abs_eb_sos_online<T>(u11.first, u10.first, u00.first,
                                                u11.second, u10.second, u00.second);
          if (eb < required_eb) required_eb = eb;
        }
      }
    }

    if (t_idx + 1 < Tt) {
      for (int k = 0; k < 6; ++k) {
        int ni = (int)i_idx + di[k];
        int nj = (int)j_idx + dj[k];
        if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
        size_t a = vid((int)t_idx, (int)i_idx, (int)j_idx, sz);
        size_t b = vid((int)t_idx, ni, nj, sz);
        size_t ap = a + dv;
        size_t bp = b + dv;
        auto val_a  = fetch_fixed_pair(t_idx, a);
        auto val_b  = fetch_fixed_pair(t_idx, b);
        auto val_ap = fetch_fixed_pair(t_idx, ap);
        auto val_bp = fetch_fixed_pair(t_idx, bp);

        if (k == 0 || k == 3 || k == 5) {
          if (has_cp(cp_faces, a, b, bp)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_bp.first, val_b.first, val_a.first,
                                                  val_bp.second, val_b.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
          if (has_cp(cp_faces, a, bp, ap)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_ap.first, val_bp.first, val_a.first,
                                                  val_ap.second, val_bp.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
        } else {
          if (has_cp(cp_faces, a, b, ap)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_ap.first, val_b.first, val_a.first,
                                                  val_ap.second, val_b.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
        }
      }
    }

    if (t_idx > 0) {
      for (int k = 0; k < 6; ++k) {
        int ni = (int)i_idx + di[k];
        int nj = (int)j_idx + dj[k];
        if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
        size_t a = vid((int)t_idx, (int)i_idx, (int)j_idx, sz);
        size_t b = vid((int)t_idx, ni, nj, sz);
        size_t ap = a - dv;
        size_t bp = b - dv;
        auto val_a  = fetch_fixed_pair(t_idx, a);
        auto val_b  = fetch_fixed_pair(t_idx, b);
        auto val_ap = fetch_fixed_pair(t_idx, ap);
        auto val_bp = fetch_fixed_pair(t_idx, bp);

        if (k == 0 || k == 3 || k == 5) {
          if (has_cp(cp_faces, a, b, ap)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_ap.first, val_b.first, val_a.first,
                                                  val_ap.second, val_b.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
          if (has_cp(cp_faces, a, ap, bp)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_bp.first, val_ap.first, val_a.first,
                                                  val_bp.second, val_ap.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
        } else {
          if (has_cp(cp_faces, a, b, bp)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_bp.first, val_b.first, val_a.first,
                                                  val_bp.second, val_b.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
          if (has_cp(cp_faces, a, bp, ap)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_ap.first, val_bp.first, val_a.first,
                                                  val_ap.second, val_bp.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
        }
      }
    }

    auto valid = [&](int x, int limit){ return x >= 0 && x < limit; };

    if (t_idx + 1 < Tt) {
      int i_int = (int)i_idx;
      int j_int = (int)j_idx;
      if (valid(i_int - 1, (int)H) && valid(j_int + 1, (int)W)) {
        size_t f1a  = vid((int)t_idx, i_int,     j_int,     sz);
        size_t f1b  = vid((int)t_idx, i_int,     j_int + 1, sz);
        size_t f1c  = vid((int)t_idx, i_int - 1, j_int,     sz);
        size_t f1ap = f1a + dv;
        size_t f1bp = f1b + dv;
        size_t f1cp = f1c + dv;
        auto va = fetch_fixed_pair(t_idx, f1a);
        auto vb = fetch_fixed_pair(t_idx, f1b);
        auto vcp = fetch_fixed_pair(t_idx, f1cp);
        auto vbp = fetch_fixed_pair(t_idx, f1bp);
        if (has_cp(cp_faces, f1a, f1cp, f1b)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vb.first, vcp.first, va.first,
                                                vb.second, vcp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
        if (has_cp(cp_faces, f1a, f1cp, f1bp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vbp.first, vcp.first, va.first,
                                                vbp.second, vcp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int - 1, (int)H) && valid(j_int - 1, (int)W)) {
        size_t f2a  = vid((int)t_idx, i_int,     j_int,     sz);
        size_t f2b  = vid((int)t_idx, i_int - 1, j_int,     sz);
        size_t f2c  = vid((int)t_idx, i_int - 1, j_int - 1, sz);
        size_t f2ap = f2a + dv;
        size_t f2bp = f2b + dv;
        size_t f2cp = f2c + dv;
        auto va = fetch_fixed_pair(t_idx, f2a);
        auto vc = fetch_fixed_pair(t_idx, f2c);
        auto vbp = fetch_fixed_pair(t_idx, f2bp);
        auto vcp = fetch_fixed_pair(t_idx, f2cp);
        if (has_cp(cp_faces, f2a, f2c, f2bp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vbp.first, vc.first, va.first,
                                                vbp.second, vc.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
        if (has_cp(cp_faces, f2a, f2bp, f2cp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vcp.first, vbp.first, va.first,
                                                vcp.second, vbp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int - 1, (int)H) && valid(j_int - 1, (int)W)) {
        size_t f3a  = vid((int)t_idx, i_int,     j_int,     sz);
        size_t f3b  = vid((int)t_idx, i_int - 1, j_int - 1, sz);
        size_t f3c  = vid((int)t_idx, i_int,     j_int - 1, sz);
        size_t f3bp = f3b + dv;
        auto va = fetch_fixed_pair(t_idx, f3a);
        auto vb = fetch_fixed_pair(t_idx, f3b);
        auto vbp = fetch_fixed_pair(t_idx, f3bp);
        auto vc = fetch_fixed_pair(t_idx, f3c);
        if (has_cp(cp_faces, f3a, f3bp, f3c)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vc.first, vbp.first, va.first,
                                                vc.second, vbp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int + 1, (int)H) && valid(j_int + 1, (int)W)) {
        size_t f6a  = vid((int)t_idx, i_int,     j_int,     sz);
        size_t f6b  = vid((int)t_idx, i_int,     j_int + 1, sz);
        size_t f6c  = vid((int)t_idx, i_int + 1, j_int + 1, sz);
        size_t f6bp = f6b + dv;
        auto va = fetch_fixed_pair(t_idx, f6a);
        auto vb = fetch_fixed_pair(t_idx, f6b);
        auto vbp = fetch_fixed_pair(t_idx, f6bp);
        auto vc = fetch_fixed_pair(t_idx, f6c);
        if (has_cp(cp_faces, f6a, f6bp, f6c)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vc.first, vbp.first, va.first,
                                                vc.second, vbp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }
    }

    if (t_idx > 0) {
      int i_int = (int)i_idx;
      int j_int = (int)j_idx;
      if (valid(i_int - 1, (int)H) && valid(j_int - 1, (int)W)) {
        size_t f3a  = vid((int)t_idx,     i_int,     j_int,     sz);
        size_t f3b  = vid((int)t_idx - 1, i_int - 1, j_int - 1, sz);
        size_t f3c  = vid((int)t_idx,     i_int,     j_int - 1, sz);
        size_t f3cp = f3c - dv;
        auto va = fetch_fixed_pair(t_idx, f3a);
        auto vb = fetch_fixed_pair(t_idx, f3b);
        auto vc = fetch_fixed_pair(t_idx, f3c);
        auto vcp = fetch_fixed_pair(t_idx, f3cp);
        if (has_cp(cp_faces, f3a, f3c, f3cp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vcp.first, vc.first, va.first,
                                                vcp.second, vc.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int + 1, (int)H) && valid(j_int - 1, (int)W)) {
        size_t f4a  = vid((int)t_idx,     i_int,     j_int,     sz);
        size_t f4b  = vid((int)t_idx - 1, i_int,     j_int - 1, sz);
        size_t f4c  = vid((int)t_idx - 1, i_int + 1, j_int - 1, sz);
        size_t f4bp = f4b + dv;
        size_t f4cp = f4c + dv;
        auto va = fetch_fixed_pair(t_idx, f4a);
        auto vb = fetch_fixed_pair(t_idx, f4b);
        auto vc = fetch_fixed_pair(t_idx, f4c);
        auto vbp = fetch_fixed_pair(t_idx, f4bp);
        auto vcp = fetch_fixed_pair(t_idx, f4cp);
        if (has_cp(cp_faces, f4a, f4bp, f4cp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vcp.first, vbp.first, va.first,
                                                vcp.second, vbp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
        if (has_cp(cp_faces, f4a, f4cp, f4b)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vb.first, vcp.first, va.first,
                                                vb.second, vcp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int + 1, (int)H) && valid(j_int + 1, (int)W)) {
        size_t f5a  = vid((int)t_idx,     i_int,     j_int,     sz);
        size_t f5b  = vid((int)t_idx - 1, i_int + 1, j_int,     sz);
        size_t f5c  = vid((int)t_idx - 1, i_int + 1, j_int + 1, sz);
        size_t f5bp = f5b + dv;
        size_t f5cp = f5c + dv;
        auto va = fetch_fixed_pair(t_idx, f5a);
        auto vb = fetch_fixed_pair(t_idx, f5b);
        auto vc = fetch_fixed_pair(t_idx, f5c);
        auto vbp = fetch_fixed_pair(t_idx, f5bp);
        auto vcp = fetch_fixed_pair(t_idx, f5cp);
        if (has_cp(cp_faces, f5a, f5bp, f5cp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vcp.first, vbp.first, va.first,
                                                vcp.second, vbp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
        if (has_cp(cp_faces, f5a, f5c, f5bp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vbp.first, vc.first, va.first,
                                                vbp.second, vc.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int + 1, (int)H) && valid(j_int + 1, (int)W)) {
        size_t f6a  = vid((int)t_idx,     i_int,     j_int,     sz);
        size_t f6b  = vid((int)t_idx,     i_int,     j_int + 1, sz);
        size_t f6c  = vid((int)t_idx - 1, i_int + 1, j_int + 1, sz);
        auto va = fetch_fixed_pair(t_idx, f6a);
        auto vb = fetch_fixed_pair(t_idx, f6b);
        auto vc = fetch_fixed_pair(t_idx, f6c);
        if (has_cp(cp_faces, f6a, f6b, f6c)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vc.first, vb.first, va.first,
                                                vc.second, vb.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }
    }

    return required_eb;
  };

  for (size_t t = 0; t < Tt; ++t) {
    if (t % 100 == 0) {
      printf("processing slice %zu / %zu\n", t, Tt);
    }

#ifdef _OPENMP
    const size_t block_rows_eb = 8;
    const size_t block_cols_eb = 8;
    const size_t blocks_i_eb = (H + block_rows_eb - 1) / block_rows_eb;
    const size_t blocks_j_eb = (W + block_cols_eb - 1) / block_cols_eb;
    #pragma omp parallel for schedule(static)
    for (size_t block = 0; block < blocks_i_eb * blocks_j_eb; ++block) {
      size_t bi = block / blocks_j_eb;
      size_t bj = block % blocks_j_eb;
      size_t i_begin = bi * block_rows_eb;
      size_t i_end = std::min(i_begin + block_rows_eb, H);
      size_t j_begin = bj * block_cols_eb;
      size_t j_end = std::min(j_begin + block_cols_eb, W);
      for (size_t i = i_begin; i < i_end; ++i) {
        for (size_t j = j_begin; j < j_end; ++j) {
          size_t off = i * W + j;
          required_eb_layer[off] = compute_required_eb(t, i, j);
        }
      }
    }
#else
    for (size_t i = 0; i < H; ++i) {
      for (size_t j = 0; j < W; ++j) {
        size_t off = i * W + j;
        required_eb_layer[off] = compute_required_eb(t, i, j);
      }
    }
#endif

    for (size_t i = 0; i < H; ++i) {
      for (size_t j = 0; j < W; ++j) {
        size_t off = i * W + j;
        const T curU_val = u_curr[off];
        const T curV_val = v_curr[off];
        T required_eb = required_eb_layer[off];

        T abs_eb = required_eb;
        int id = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
        if (abs_eb == 0) {
          *(eb_pos++) = 0;
          unpred.push_back(float_curr_u[off]);
          unpred.push_back(float_curr_v[off]);
          dec_curr_u[off] = curU_val;
          dec_curr_v[off] = curV_val;
          continue;
        }

        *eb_pos = id;
        bool unpred_flag = false;
        T dec_val[2];
        T abs_err_fp_q[2] = {0,0};

        for (int comp = 0; comp < 2; ++comp) {
          const T curv = (comp == 0) ? curU_val : curV_val;
          const std::vector<T>& prev_buf = (comp == 0) ? dec_prev_u : dec_prev_v;
          const std::vector<T>& curr_buf = (comp == 0) ? dec_curr_u : dec_curr_v;
          auto sample_prev = [&](size_t row, size_t col)->T {
            return prev_buf[row * W + col];
          };
          auto sample_curr = [&](size_t row, size_t col)->T {
            return curr_buf[row * W + col];
          };

          T d0 = (t && i && j) ? sample_prev(i - 1, j - 1) : 0;
          T d1 = (t && i)     ? sample_prev(i - 1, j)     : 0;
          T d2 = (t && j)     ? sample_prev(i,     j - 1) : 0;
          T d3 = (t)          ? sample_prev(i,     j)     : 0;
          T d4 = (i && j)     ? sample_curr(i - 1, j - 1) : 0;
          T d5 = (i)          ? sample_curr(i - 1, j)     : 0;
          T d6 = (j)          ? sample_curr(i,     j - 1) : 0;
          T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

          T diff = curv - pred;
          T qd = (std::llabs(diff)/abs_eb) + 1;
          if (qd < capacity) {
            qd = (diff > 0) ? qd : -qd;
            int qindex = (int)(qd/2) + intv_radius;
            dq_pos[comp] = qindex;
            dec_val[comp] = pred + 2 * ( (T)qindex - (T)intv_radius ) * abs_eb;
            if (std::llabs(dec_val[comp] - curv) > abs_eb) {
              unpred_flag = true;
              break;
            }
            abs_err_fp_q[comp] = std::llabs(dec_val[comp] - curv);
          } else {
            unpred_flag = true;
            break;
          }
        }

        if (unpred_flag) {
          *(eb_pos++) = 0;
          unpred.push_back(float_curr_u[off]);
          unpred.push_back(float_curr_v[off]);
          dec_curr_u[off] = curU_val;
          dec_curr_v[off] = curV_val;
        } else {
          ++eb_pos;
          dq_pos += 2;
          dec_curr_u[off] = dec_val[0];
          dec_curr_v[off] = dec_val[1];
          double abs_eb_fp = (double)abs_eb / (double)scale;
          double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
          double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
          if (abs_eb_fp > enc_max_abs_eb_fp) enc_max_abs_eb_fp = abs_eb_fp;
          double real_err = std::max(err_u_fp, err_v_fp);
          if (real_err > enc_max_real_err_fp) enc_max_real_err_fp = real_err;
        }
      }
    }

    // rotate buffers for next timestep
    dec_prev_u.swap(dec_curr_u);
    dec_prev_v.swap(dec_curr_v);
    std::fill(dec_curr_u.begin(), dec_curr_u.end(), 0);
    std::fill(dec_curr_v.begin(), dec_curr_v.end(), 0);

    u_prev.swap(u_curr);
    v_prev.swap(v_curr);
    float_prev_u.swap(float_curr_u);
    float_prev_v.swap(float_curr_v);

    if (t + 1 < Tt) {
      u_curr.swap(u_next);
      v_curr.swap(v_next);
      float_curr_u.swap(float_next_u);
      float_curr_v.swap(float_next_v);
      if (t + 2 < Tt) {
        if (!load_layer_fixed(t + 2, u_next, v_next, &float_next_u, &float_next_v, false)) {
          std::free(eb_quant_index);
          std::free(data_quant_index);
          compressed_size = 0;
          return nullptr;
        }
      }
    }
  }

  unsigned char *compressed = (unsigned char*)std::malloc((size_t)(2 * H * W * Tt * sizeof(T)));
  if (!compressed) {
    std::free(eb_quant_index);
    std::free(data_quant_index);
    compressed_size = 0;
    return nullptr;
  }
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) {
    write_array_to_dst(pos, unpred.data(), unpred_cnt);
  }

  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::free(eb_quant_index);

  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  return compressed;
}

template<typename T_data, typename LayerFetcher>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_streaming(
    LayerFetcher&& fetch_layer,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  return sz_compress_cp_preserve_sos_2p5d_online_fp_streaming_impl<T_data>(
      std::forward<LayerFetcher>(fetch_layer), r1, r2, r3,
      compressed_size, max_pwr_eb, mode);
}

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_streaming(
    const T_data* U,
    const T_data* V,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  const size_t H = r1, W = r2;
  const size_t layer_size = H * W;
  auto loader = [&](size_t t, T_data* dstU, T_data* dstV) {
    const size_t offset = t * layer_size;
    std::copy_n(U + offset, layer_size, dstU);
    std::copy_n(V + offset, layer_size, dstV);
    return true;
  };
  return sz_compress_cp_preserve_sos_2p5d_online_fp_streaming_impl<T_data>(
      loader, r1, r2, r3, compressed_size, max_pwr_eb, mode);
}
#endif

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
bool sz_decompress_cp_preserve_sos_2p5d_fp_warped_Lorenzo(
    const unsigned char* compressed,
    size_t r1, size_t r2, size_t r3,
    T_data*& U, T_data*& V)
{
    using T = int64_t;
    const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

    if (U) std::free(U);
    if (V) std::free(V);
    const double WARP_ALPHA_X = 0.77;//1.49;   // 像素/帧 = alpha * u,v  （若你的u,v单位即“像素/帧”，取1.0）
    const double WARP_ALPHA_Y = 0.73;//1.49;   // 像素/帧 = alpha * u,v  （若你的u,v单位即“像素/帧”，取1.0）
    const double WARP_DMAX  = 6.0;   // 位移裁剪（像素）
    const double WARP_GATE  = 0.5;  // 小位移门槛（像素）；<0.30像素时退化为0，省去无谓的插值抖动
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
    if(!U_fp || !V_fp){
        if(U_fp) std::free(U_fp);
        if(V_fp) std::free(V_fp);
        std::free(eb_idx); std::free(dq);
        return false;
    }

    T* U_pos=U_fp; T* V_pos=V_fp;
    int* eb_pos=eb_idx; int* dq_pos=dq;
    std::vector<size_t> unpred_indices; unpred_indices.reserve(n_unpred_points);

    // 步长（与编码端一致）
    const ptrdiff_t si = (ptrdiff_t)W;             // 行步长（列数）
    const ptrdiff_t sj = (ptrdiff_t)1;             // 列步长
    const ptrdiff_t sk = (ptrdiff_t)(H*W);         // 层步长（整平面）

    // ===== 主循环：按 t(慢)->i->j 扫描，边解码边重建 =========================
    for (int t=0; t<(int)Tt; ++t){
        for (int i=0; i<(int)H; ++i){
            for (int j=0; j<(int)W; ++j){
                int ebid = *eb_pos++;

                if (ebid == 0){
                    // 无损点：直接回填原始浮点（后一阶段 convert_to_floating_point 会被此覆盖）
                    size_t off = (size_t)(U_pos - U_fp);
                    unpred_indices.push_back(off);
                    *U_pos = *(unpred_pos++) * scale;
                    *V_pos = *(unpred_pos++) * scale;
                } else {
                    // 幂指数代表误差
                    T abs_eb = pow(base,ebid) * threshold;
                    // 两分量依次重建
                    for (int pcomp=0; pcomp<2; ++pcomp){
                        T* cur = (pcomp==0) ? U_pos : V_pos;  // 当前写入位置
                        T pred = 0;

                        if (t == 0){
                            // ---- 首帧：2D Lorenzo 退化（与编码端一致） ----
                            const T* recon_t_plane = (pcomp==0)
                                ? (U_fp + (size_t)t*H*W) : (V_fp + (size_t)t*H*W);
                            if (i>0 && j>0){
                                pred = recon_t_plane[(size_t)(i-1)*W + j]
                                     + recon_t_plane[(size_t)i*W + (j-1)]
                                     - recon_t_plane[(size_t)(i-1)*W + (j-1)];
                            } else if (i>0){
                                pred = recon_t_plane[(size_t)(i-1)*W + j];
                            } else if (j>0){
                                pred = recon_t_plane[(size_t)i*W + (j-1)];
                            } else {
                                pred = 0;
                            }
                        } else {
                            // ---- P 帧：Warped‑Lorenzo（与编码端相同） ----
                            const T* recon_t_plane   = (pcomp==0)
                                ? (U_fp + (size_t)t*H*W)   : (V_fp + (size_t)t*H*W);
                            const T* recon_tm1_plane = (pcomp==0)
                                ? (U_fp + (size_t)(t-1)*H*W) : (V_fp + (size_t)(t-1)*H*W);
                            const T* U_tm1_plane = U_fp + (size_t)(t-1)*H*W; // 位移来源：上一帧速度
                            const T* V_tm1_plane = V_fp + (size_t)(t-1)*H*W;

                            pred = warp_pred::predict<T>(
                                /*recon_t   =*/recon_t_plane,
                                /*recon_tm1 =*/recon_tm1_plane,
                                /*U_tm1     =*/U_tm1_plane,
                                /*V_tm1     =*/V_tm1_plane,
                                (int)H, (int)W, i, j,
                                /*scale     =*/scale,
                                /*alphaX     =*/WARP_ALPHA_X,
                                /*alphaY     =*/WARP_ALPHA_Y,
                                /*dmax      =*/WARP_DMAX,
                                /*gate_pix  =*/WARP_GATE
                            );
                        }

                        // 反量化并写入
                        int qidx = *dq_pos++;
                        *cur = pred + 2* (qidx - intv_radius) * abs_eb;
                    }
                }
                ++U_pos; ++V_pos;
            }
        }
    }

    // ===== 回浮点并覆盖不可预测点 ============================================
    U = (T_data*)std::malloc(N*sizeof(T_data));
    V = (T_data*)std::malloc(N*sizeof(T_data));
    if(!U || !V){
        std::free(U); std::free(V);
        std::free(U_fp); std::free(V_fp);
        std::free(eb_idx); std::free(dq);
        return false;
    }
    convert_to_floating_point(U_fp, V_fp, N, U, V, scale);

    // 覆盖无损点（原始浮点）
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
bool sz_decompress_cp_preserve_sos_2p5d_fp_AR2_2D_Lorenzo(
    const unsigned char* compressed,
    size_t r1, size_t r2, size_t r3,
    T_data*& U, T_data*& V)
{
    using T = int64_t;
    const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

    if (U) std::free(U);
    if (V) std::free(V);

    // —— 读头部（与编码端一致）——
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

    // —— 定点重建缓冲（与编码端的“回写已重构值”一致）——
    T* U_fp=(T*)std::malloc(N*sizeof(T));
    T* V_fp=(T*)std::malloc(N*sizeof(T));
    if(!U_fp || !V_fp){
        if(U_fp) std::free(U_fp);
        if(V_fp) std::free(V_fp);
        std::free(eb_idx); std::free(dq);
        return false;
    }

    T* U_pos=U_fp; T* V_pos=V_fp;
    int* eb_pos=eb_idx; int* dq_pos=dq;
    std::vector<size_t> unpred_indices; unpred_indices.reserve(n_unpred_points);

    // 扫描步长（时间最慢：j 最快、i 次之、t 最慢）
    const ptrdiff_t si=W, sj=1, sk=(ptrdiff_t)(H*W);

    for (int t=0; t<(int)Tt; ++t){
        for (int i=0; i<(int)H; ++i){
            for (int j=0; j<(int)W; ++j){

                int ebid = *eb_pos++;

                if (ebid == 0){
                    // 无损点：直接读回原始浮点并转为定点存入，
                    // 这样后续邻点/后续帧的预测能与编码端一致
                    size_t off = (size_t)(U_pos - U_fp);
                    unpred_indices.push_back(off);
                    *U_pos = *(unpred_pos++) * scale;
                    *V_pos = *(unpred_pos++) * scale;
                } else {
                    // 幂指数重建该点的步长（LSB）
                    // 与编码端 eb_exponential_quantize 的代表值严格对应
                    T abs_eb = pow(base,ebid) * threshold;

                    // —— 按 A 方案重建：AR(2) + 残差 2D Lorenzo —— //
                    // 注意：这里的“已重构值”与编码端回写的一致

                    for (int pcomp=0; pcomp<2; ++pcomp){
                        T* cur = (pcomp==0) ? U_pos : V_pos;

                        // 1) 时间 AR(2) 预测 p_t(i,j)
                        T pt = 0;
                        if (t <= 0){
                            pt = (T)0;
                        } else if (t == 1){
                            pt = cur[-sk];
                        } else {
                            pt = (T)(2*cur[-sk] - cur[-2*sk]);
                        }

                        // 2) 残差的 2D Lorenzo 预测 epred
                        T epred = 0;
                        if (i > 0){
                            // 邻居 (i-1,j) 的时间预测
                            T pt_up = 0;
                            if      (t <= 0) pt_up = 0;
                            else if (t == 1) pt_up = cur[-si - sk];
                            else             pt_up = (T)(2*cur[-si - sk] - cur[-si - 2*sk]);
                            epred += (cur[-si] - pt_up);
                        }
                        if (j > 0){
                            // 邻居 (i,j-1) 的时间预测
                            T pt_lf = 0;
                            if      (t <= 0) pt_lf = 0;
                            else if (t == 1) pt_lf = cur[-sj - sk];
                            else             pt_lf = (T)(2*cur[-sj - sk] - cur[-sj - 2*sk]);
                            epred += (cur[-sj] - pt_lf);
                        }
                        if (i > 0 && j > 0){
                            // 邻居 (i-1,j-1) 的时间预测
                            T pt_ul = 0;
                            if      (t <= 0) pt_ul = 0;
                            else if (t == 1) pt_ul = cur[-si - sj - sk];
                            else             pt_ul = (T)(2*cur[-si - sj - sk] - cur[-si - sj - 2*sk]);
                            epred -= (cur[-si - sj] - pt_ul);
                        }

                        // 3) 读区间量化索引并反量化到 ehat
                        int qidx = *dq_pos++;
                        T ehat = epred + (T)2 * ( (T)qidx - (T)intv_radius ) * abs_eb;

                        // 4) 得到 xhat = p_t + ehat
                        *cur = pt + ehat;
                    }
                }

                ++U_pos; ++V_pos;
            }
        }
    }

    // —— 定点→浮点，并用原始无损数据覆盖（保证 bit-exact）——
    U = (T_data*)std::malloc(N*sizeof(T_data));
    V = (T_data*)std::malloc(N*sizeof(T_data));
    if(!U || !V){
        std::free(U); std::free(V);
        std::free(U_fp); std::free(V_fp);
        std::free(eb_idx); std::free(dq);
        return false;
    }
    convert_to_floating_point(U_fp, V_fp, N, U, V, scale);

    // 覆盖无损点（原封不动）
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
bool sz_decompress_cp_preserve_sos_2p5d_fp_3DL_AR2(
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

    const double RHO = 0.85;                    // 可后续自适应估计，这里先固定
    const double ALPHA = 2.0 * RHO; //= 1.7;  // 可后续自适应估计，这里先固定
    const double BETA = -(RHO * RHO); //= -0.72; // 可后续自适应估计，这里先固定

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
    auto pred3D = [&](const T* cur, int t, int i, int j)->T {
        T d0 = (t&&i&&j)? cur[-sk - si - sj]:0;
        T d1 = (t&&i)   ? cur[-sk - si]     : 0;
        T d2 = (t&&j)   ? cur[-sk - sj]     : 0;
        T d3 = (t)      ? cur[-sk]          : 0;
        T d4 = (i&&j)   ? cur[-si - sj]     : 0;
        T d5 = (i)      ? cur[-si]          : 0;
        T d6 = (j)      ? cur[-sj]          : 0;
        return d0 + d3 + d5 + d6 - d1 - d2 - d4;
    };
    // PATCH [AR(2) residual buffers]
    std::vector<T> r1U(H*W, (T)0), r2U(H*W, (T)0), rcurU(H*W, (T)0);
    std::vector<T> r1V(H*W, (T)0), r2V(H*W, (T)0), rcurV(H*W, (T)0);

    for (int t=0; t<(int)Tt; ++t){
        for (int i=0; i<(int)H; ++i){
            for (int j=0; j<(int)W; ++j){
                const size_t idx2D = (size_t)i * W + (size_t)j;
                // 先算主预测+AR(2)（注意：要用 *pos 未写入当前点* 的邻域）
                const T predL3D_U = pred3D(U_pos, t, i, j);
                const T predL3D_V = pred3D(V_pos, t, i, j);
                const T arU = std::llround( ALPHA * r1U[idx2D] + BETA * r2U[idx2D] );
                const T arV = std::llround( ALPHA * r1V[idx2D] + BETA * r2V[idx2D] );
                const T predU_final = predL3D_U + arU;
                const T predV_final = predL3D_V + arV;

                int ebid = *eb_pos++;

                if (ebid == 0){
                    size_t off = (size_t)(U_pos - U_fp);
                    unpred_indices.push_back(off);
                    *U_pos = *(unpred_pos++) * scale;
                    *V_pos = *(unpred_pos++) * scale;

                    // PATCH: 记录当前帧残差（相对 L3D 主预测）
                    rcurU[idx2D] = *U_pos - predL3D_U;
                    rcurV[idx2D] = *V_pos - predL3D_V;
                }

                else {
                    T abs_eb = pow(base,ebid) * threshold;
                    {
                        int qidx = *dq_pos++;
                        *U_pos = predU_final + 2*(qidx - intv_radius)*abs_eb;
                    }
                    // V
                    {
                        int qidx = *dq_pos++;
                        *V_pos = predV_final + 2*(qidx - intv_radius)*abs_eb;
                    }
                    // PATCH: 记录当前帧残差
                    rcurU[idx2D] = *U_pos - predL3D_U;
                    rcurV[idx2D] = *V_pos - predL3D_V;

                }
                ++U_pos; ++V_pos;
            }
        }
        // 帧末滚动
        std::swap(r2U, r1U); std::swap(r1U, rcurU);
        std::swap(r2V, r1V); std::swap(r1V, rcurV);
        std::fill(rcurU.begin(), rcurU.end(), (T)0);
        std::fill(rcurV.begin(), rcurV.end(), (T)0);
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
    // printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap...\n");  
    // auto* compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap(
    //     U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
    //     compressed_size, error_bound, mode);

    // printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel...\n");
    // auto* compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel(
    //     U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
    //     compressed_size, error_bound, mode);

    // printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_V2...\n");
    // auto* compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v2(
    //     U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
    //     compressed_size, error_bound, mode);

    // printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_V3...\n");
    // auto* compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v3(
    //     U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
    //     compressed_size, error_bound, mode);

    // printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_warped_lorenzo...\n");
    // auto* compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_warped_lorenzo(
    //     U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
    //     compressed_size, error_bound, mode);
    // printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_warped_AR2_2D_LORENZO...\n");
    // auto* compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_AR2_2D_LORENZO(
    //   U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
    //   compressed_size, error_bound, mode);

        printf("sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_3DL_AR2...\n");
    auto* compressed = sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_3DL_AR2(
        U_ptr, V_ptr, /*H=*/dim_H, /*W=*/dim_W, /*T=*/dim_T,
        compressed_size, error_bound, mode);
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
    
    exit(0);
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

    printf("sz_decompress_cp_preserve_sos_2p5d_fp...\n");
    sz_decompress_cp_preserve_sos_2p5d_fp<float>(decompressed, H, W, T, U_dec, V_dec);

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
    verify_elements = num_elements;
    U_ori = U_ptr;
    V_ori = V_ptr;
    #endif

    verify(U_ori, V_ori, U_dec, V_dec, H, W, T);

    int64_t* U_ori_fp = (int64_t*)std::malloc(verify_elements * sizeof(int64_t));
    int64_t* V_ori_fp = (int64_t*)std::malloc(verify_elements * sizeof(int64_t));
    int64_t range_ori = 0;
    int64_t scale_ori = convert_to_fixed_point(U_ori, V_ori, verify_elements, U_ori_fp, V_ori_fp, range_ori);
    printf("original scale = %ld\n", scale_ori);
    auto cp_faces_ori = compute_cp_2p5d_faces(U_ori_fp, V_ori_fp, H, W, T, dim_str);
    std::cout << "Total faces with CP ori: " << cp_faces_ori.size() << std::endl;

    int64_t* U_dec_fp = (int64_t*)std::malloc(verify_elements * sizeof(int64_t));
    int64_t* V_dec_fp = (int64_t*)std::malloc(verify_elements * sizeof(int64_t));
    convert_to_fixed_point_given_factor(U_dec, V_dec, verify_elements, U_dec_fp, V_dec_fp, scale_ori);

    auto cp_faces_dec = compute_cp_2p5d_faces(U_dec_fp, V_dec_fp, H, W, T);

    std::cout << "Total faces with CP dec: " << cp_faces_dec.size() << std::endl;
    // std::cout << (are_unordered_sets_equal(cp_faces_dec, cp_faces_ori) ? "CP face sets are equal." : "CP face sets differ!") << std::endl;
    bool cp_equal = are_unordered_sets_equal(cp_faces_dec, cp_faces_ori);
    std::cout << (cp_equal ? "CP face sets are equal." : "CP face sets differ!") << std::endl;
    if (!cp_equal) {
        print_cp_face_diffs(cp_faces_dec, cp_faces_ori, U_dec, V_dec, U_ori, V_ori,U_dec_fp, V_dec_fp, U_ori_fp, V_ori_fp, H, W, T);
    }
    exit(0);

    #if DETAIL
      const std::string csv_name =
      std::to_string(H) + "_" + std::to_string(W) + "_" + std::to_string(T) + ".csv";
      summarize_cp_faces_per_layer_and_slab(cp_faces_dec, cp_faces_ori, H, W, T, csv_name);
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
        auto cp_faces_cpsz = compute_cp_2p5d_faces(U_dec_cpsz_fp, V_dec_cpsz_fp, H, W, T);
        std::cout << "Total faces with CP dec CPSZ baseline: " << cp_faces_cpsz.size() << std::endl;

        auto cp_faces_ori_cpsz = compute_cp_2p5d_faces(U_ori_cpsz_fp, V_ori_cpsz_fp, H, W, T);
        const std::string csv_name_cpsz =
            std::to_string(H) + "_" + std::to_string(W) + "_" + std::to_string(T) + "_" + std::to_string(error_bound) + "_cpsz.csv";
        summarize_cp_faces_per_layer_and_slab(cp_faces_cpsz, cp_faces_ori_cpsz, H, W, T, csv_name_cpsz);
        std::cout << "CPSZ baseline summary written to " << csv_name_cpsz << std::endl;
    }

    if (U_dec_cpsz && V_dec_cpsz) {
        printf("start verify CPSZ baseline....\n");
        verify(U_ptr, V_ptr, U_dec_cpsz, V_dec_cpsz, H, W, T);
    }

    std::free(U_dec_cpsz);
    std::free(V_dec_cpsz);
    std::free(U_dec_cpsz_fp);
    std::free(V_dec_cpsz_fp);
    std::free(U_ori_cpsz_fp);
    std::free(V_ori_cpsz_fp);

#endif
    
    //释放返回的缓冲区（由函数 malloc 分配）
    std::free(decompressed);
    std::free(U_ptr);
    std::free(V_ptr);
    std::free(U_dec);
    std::free(V_dec);

    return 0;
}
