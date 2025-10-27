#pragma once

#ifndef CP_DEBUG_VISIT
#define CP_DEBUG_VISIT 0
#endif
#ifndef DEBUG_USE
#define DEBUG_USE 0
#endif
#ifndef VERBOSE
#define VERBOSE 0
#endif
#ifndef VERIFY
#define VERIFY 0
#endif
#ifndef DETAIL
#define DETAIL 0
#endif
#ifndef CPSZ_BASELINE
#define CPSZ_BASELINE 0
#endif
#ifndef STREAMING
#define STREAMING 0
#endif
#ifndef VISUALIZE
#define VISUALIZE 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <array>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <utility>
#include <initializer_list>
#include <tuple>
#include <queue>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <ftk/numeric/critical_point_type.hh>
#include <ftk/numeric/critical_point_test.hh>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <ftk/numeric/clamp.hh>
#include <ftk/numeric/linear_interpolation.hh>
#include <ftk/geometry/cc2curves.hh>

#include "sz_cp_preserve_utils.hpp"


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
        // FTK 版本的切分
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

    //横向面 x-axis
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W - 1; ++j) {
        size_t a  = vid(t, i, j, sz);
        size_t b  = vid(t, i, j + 1, sz);
        size_t ap = a + dv, bp = b + dv;
        classify_slab(t, FaceKeySZ(a, b, bp));
        classify_slab(t, FaceKeySZ(a, bp, ap));
      }
    }

    //纵向面 y-axis
    for (int i = 0; i < H - 1; ++i) {
      for (int j = 0; j < W; ++j) {
        size_t a  = vid(t, i,     j, sz);
        size_t b  = vid(t, i + 1, j, sz);
        size_t ap = a + dv, bp = b + dv;
        classify_slab(t, FaceKeySZ(a, b, bp));
        classify_slab(t, FaceKeySZ(a, bp, ap));
      }
    }
    //对角线面
    for (int i = 0; i < H - 1; ++i) {
      for (int j = 0; j < W - 1; ++j) {
        size_t ax0y0 = vid(t, i,     j,     sz);
        size_t ax1y1 = vid(t, i + 1, j + 1, sz);
        size_t bx0y0 = ax0y0 + dv, bx1y1 = ax1y1 + dv;
        classify_slab(t, FaceKeySZ(ax0y0, ax1y1, bx1y1));
        classify_slab(t, FaceKeySZ(ax0y0, bx0y0, bx1y1));
      }
    }

    for (int i = 0; i < H - 1; ++i) {
      for (int j = 0; j < W - 1; ++j) {
        size_t ax0y0 = vid(t, i,     j,     sz);
        size_t ax0y1 = vid(t, i + 1, j,     sz);
        size_t ax1y1 = vid(t, i + 1, j + 1, sz);
        size_t ax1y0 = vid(t, i,     j + 1, sz);
        size_t bx0y0 = ax0y0 + dv;
        size_t bx0y1 = ax0y1 + dv;
        size_t bx1y1 = ax1y1 + dv;
        size_t bx1y0 = ax1y0 + dv;
        //upper triangle
        classify_slab(t, FaceKeySZ(ax0y0,bx1y1,ax0y1));
        classify_slab(t, FaceKeySZ(ax0y0,bx1y1,bx0y1));
        //lower triangle
        classify_slab(t, FaceKeySZ(ax0y0,bx1y0,bx1y1));
        classify_slab(t, FaceKeySZ(ax0y0,ax1y0,bx1y1));
      }
    }
  }
  //total for slabs
  if (!slab_counts.empty()) {
    ConfusionCounts total_slab;
    for (const auto& c : slab_counts) {
      total_slab.tp += c.tp;
      total_slab.tn += c.tn;
      total_slab.fp += c.fp;
      total_slab.fn += c.fn;
    }
    std::cout << "[VERIFY] total slab CP confusion (between layers): "
              << "TP=" << total_slab.tp
              << ", TN=" << total_slab.tn
              << ", FP=" << total_slab.fp
              << ", FN=" << total_slab.fn << "\n";
  }
  //total for layers
  {
    ConfusionCounts total_layer;
    for (const auto& c : layer_counts) {
      total_layer.tp += c.tp;
      total_layer.tn += c.tn;
      total_layer.fp += c.fp;
      total_layer.fn += c.fn;
    }
    std::cout << "[VERIFY] total layer CP confusion (surface faces): "
              << "TP=" << total_layer.tp
              << ", TN=" << total_layer.tn
              << ", FP=" << total_layer.fp
              << ", FN=" << total_layer.fn << "\n";
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
#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
  printf("Using OpenMP with %d threads\n", nthreads);
#else
  const int nthreads = 1;
#endif
  std::vector<std::vector<FaceKeySZ>> thread_faces((size_t)nthreads);
    #pragma omp parallel
    {
      const int tid =
#ifdef _OPENMP
        omp_get_thread_num();
#else
        0;
#endif
      auto &local = thread_faces[(size_t)tid];

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
compute_cp_2p5d_faces_parallel_new(const T_fp* U_fp, const T_fp* V_fp,
                               int H, int W, int T,
                               const Size3& sz,
                               size_t dv,
                               std::vector<int>& cp_per_layer)
{
  std::unordered_set<FaceKeySZ, FaceKeySZHash> faces_with_cp;
  faces_with_cp.reserve((size_t)(H*(size_t)W*(size_t)T / 8));
  // 使用 OpenMP 并行：每个线程使用本地缓冲，最后统一去重合并
#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
  printf("Using OpenMP with %d threads\n", nthreads);
#else
  const int nthreads = 1;
#endif
  std::vector<std::vector<FaceKeySZ>> thread_faces((size_t)nthreads);
    #pragma omp parallel
    {
      const int tid =
#ifdef _OPENMP
        omp_get_thread_num();
#else
        0;
#endif
      auto &local = thread_faces[(size_t)tid];

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
            if (face_has_cp_robust(ax0y0, ax0y1,bx0y1, U_fp,V_fp))  local.emplace_back(ax0y0, ax0y1,bx0y1); //ftk version
            if (face_has_cp_robust(ax0y0,bx0y0,bx0y1, U_fp,V_fp)) local.emplace_back(ax0y0,bx0y0,bx0y1); //ftk version
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
            if (face_has_cp_robust(ax0y0,bx0y0,bx1y1, U_fp,V_fp))  local.emplace_back(ax0y0,bx0y0,bx1y1); //ftk version
            if (face_has_cp_robust(ax0y0,ax1y1,bx1y1, U_fp,V_fp)) local.emplace_back(ax0y0,ax1y1,bx1y1); //ftk version
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
              if (face_has_cp_robust(ax0y0,bx1y1,bx0y1, U_fp,V_fp)) local.emplace_back(ax0y0,bx1y1,bx0y1); //ftk version
              if (face_has_cp_robust(ax0y0,bx1y1,ax0y1, U_fp,V_fp)) local.emplace_back(ax0y0,bx1y1,ax0y1); //ftkl version
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
              if (face_has_cp_robust(ax0y0,bx1y0,bx1y1, U_fp,V_fp)) local.emplace_back(ax0y0,bx1y0,bx1y1);//ftk version
              if (face_has_cp_robust(ax0y0,ax1y0,bx1y1, U_fp,V_fp)) local.emplace_back(ax0y0,ax1y0,bx1y1);//ftk version
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

template<typename T_fp>
static std::unordered_set<FaceKeySZ, FaceKeySZHash>
compute_cp_2p5d_faces_new(const T_fp* U_fp, const T_fp* V_fp,
                      int H, int W, int T, std::string filenam="")
{
  bool write_to_file = !filenam.empty();
  const Size3 sz{H, W, T};
  const size_t dv = (size_t)H * (size_t)W;
  std::vector<int> cp_per_layer(T, 0);
  auto faces_with_cp = compute_cp_2p5d_faces_parallel_new(
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

// =====================================================================
// ===============  2.5D 全局 compute_cp （返回哈希集合）FTK划分  ==============
// =====================================================================

template<typename T_fp>
static inline void accumulate_face_layer_count(
    const FaceKeySZ& face,
    const Size3& sz,
    std::vector<int>& cp_per_layer)
{
  if (cp_per_layer.empty())
    return;

  int t0, i0, j0;
  inv_vid(face.v[0], sz.H, sz.W, t0, i0, j0);
  bool same_layer = true;
  for (int k = 1; k < 3; ++k) {
    int tk, ik, jk;
    inv_vid(face.v[k], sz.H, sz.W, tk, ik, jk);
    if (tk != t0) {
      same_layer = false;
      break;
    }
  }

  if (same_layer && t0 >= 0 && t0 < (int)cp_per_layer.size())
    cp_per_layer[(size_t)t0]++;
}

inline std::array<FaceKeySZ,4> tetra_face_keys(const Tet& tet)
{
  return {{
      FaceKeySZ(tet.v[0], tet.v[1], tet.v[2]),
      FaceKeySZ(tet.v[0], tet.v[1], tet.v[3]),
      FaceKeySZ(tet.v[1], tet.v[2], tet.v[3]),
      FaceKeySZ(tet.v[2], tet.v[0], tet.v[3])
  }};
}

template<typename T_fp>
static std::unordered_set<FaceKeySZ, FaceKeySZHash>
ftk_compute_cp_2p5d_faces(const T_fp* U_fp, const T_fp* V_fp,
                      int H, int W, int T, std::string filenam="")
{
  //TODO:之后要把hit_hist的相关统计去掉，算准确时间
  std::unordered_set<FaceKeySZ, FaceKeySZHash> faces_with_cp;
  if (H < 2 || W < 2 || T < 1)
    return faces_with_cp;

  const Size3 sz{H, W, T};
  const size_t dv = static_cast<size_t>(H) * static_cast<size_t>(W);
  std::vector<int> cp_per_layer((size_t)T, 0);

  auto record_face = [&](const FaceKeySZ& face) {
    auto res = faces_with_cp.emplace(face);
    if (res.second)
      accumulate_face_layer_count<T_fp>(face, sz, cp_per_layer);
  };

  std::vector<size_t> hit_hist(5, 0);
  size_t tetra_total = 0;

  struct BadTetInfo {
    int t;
    int i;
    int j;
    TriInCell tri_type;
    int local_index;
    size_t hits;
  };
  std::vector<BadTetInfo> bad_tets;

  auto process_tetra = [&](int t, int i, int j,
                           TriInCell tri_type,
                           int local_index,
                           const Tet& tet) {
    ++tetra_total;
    size_t hits = 0;
    const auto face_keys = tetra_face_keys(tet);
    for (const auto& face : face_keys) {
      if (face_has_cp_robust(face.v[0], face.v[1], face.v[2], U_fp, V_fp)) {
        ++hits;
        record_face(face);
      }
    }

    if (hits >= hit_hist.size())
      hit_hist.resize(hits + 1, 0);
    hit_hist[hits]++;

    if (hits != 0 && hits != 2 && bad_tets.size() < 16) {
      bad_tets.push_back({t, i, j, tri_type, local_index, hits});
    }
  };

  if (T == 1) {
    for (int i = 0; i < H - 1; ++i) {
      for (int j = 0; j < W - 1; ++j) {
        const size_t v00 = vid(0, i,     j,     sz);
        const size_t v01 = vid(0, i,     j + 1, sz);
        const size_t v10 = vid(0, i + 1, j,     sz);
        const size_t v11 = vid(0, i + 1, j + 1, sz);

        if (face_has_cp_robust(v00, v01, v11, U_fp, V_fp))
          record_face(FaceKeySZ(v00, v01, v11));
        if (face_has_cp_robust(v00, v10, v11, U_fp, V_fp))
          record_face(FaceKeySZ(v00, v10, v11));
      }
    }
  } else {
    for (int t = 0; t < T - 1; ++t) {
      for (int i = 0; i < H - 1; ++i) {
        for (int j = 0; j < W - 1; ++j) {
          auto process_prism = [&](TriInCell which) {
            const auto base = tri_vertices_2d(i, j, which, t, sz);
            const std::array<size_t,3> top{
                base[0] + dv,
                base[1] + dv,
                base[2] + dv
            };
            const auto tets = prism_split_3tets(base, top);
            for (int local = 0; local < (int)tets.size(); ++local)
              process_tetra(t, i, j, which, local, tets[local]);
          };

          process_prism(TriInCell::Upper);
          process_prism(TriInCell::Lower);
        }
      }
    }
  }

  if (tetra_total > 0) {
    std::cout << "[compute_cp_2p5d_faces] tetra face-hit histogram:";
    for (size_t h = 0; h < hit_hist.size(); ++h)
      std::cout << ' ' << h << '=' << hit_hist[h];
    std::cout << std::endl;

    if (!bad_tets.empty()) {
      std::cout << "[compute_cp_2p5d_faces] warning: "
                << bad_tets.size()
                << " tetrahedra have a face-hit count other than 0 or 2 (showing up to 16)."
                << std::endl;
      for (size_t idx = 0; idx < bad_tets.size(); ++idx) {
        const auto& bt = bad_tets[idx];
        std::cout << "  tet(t=" << bt.t
                  << ", i=" << bt.i
                  << ", j=" << bt.j
                  << ", tri=" << (bt.tri_type == TriInCell::Upper ? "Upper" : "Lower")
                  << ", local=" << bt.local_index
                  << ", hits=" << bt.hits << ")" << std::endl;
      }
    } else {
      std::cout << "[compute_cp_2p5d_faces] all tetrahedra satisfy the 0/2 face-hit rule." << std::endl;
    }
  }

  if (!filenam.empty()) {
    std::ofstream ofs(filenam);
    if (ofs) {
      for (int t = 0; t < T; ++t)
        ofs << cp_per_layer[t] << '\n';
      ofs.close();
      std::cout << "Wrote cp per layer to " << filenam << std::endl;
    } else {
      std::cout << "Failed to open file " << filenam << " for writing cp per layer" << std::endl;
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

struct TracedCriticalPointSampleSZ {
  FaceKeySZ face;
  size_t face_id = 0;
  std::array<double,3> position{{0.0, 0.0, 0.0}}; // x, y, t
  std::array<double,3> barycentric{{0.0, 0.0, 0.0}};
  double cond = 0.0;
  bool ordinal = false;
  bool solver_success = false;
};

struct TracedCriticalPointCurveSZ {
  size_t id = 0;
  bool loop = false;
  std::vector<TracedCriticalPointSampleSZ> samples;
};

inline bool face_is_ordinal(const FaceKeySZ& face, const Size3& sz)
{
  int t0, i0, j0;
  inv_vid(face.v[0], sz.H, sz.W, t0, i0, j0);
  int t1, i1, j1;
  inv_vid(face.v[1], sz.H, sz.W, t1, i1, j1);
  int t2, i2, j2;
  inv_vid(face.v[2], sz.H, sz.W, t2, i2, j2);
  return (t0 == t1) && (t1 == t2);
}

template<typename T_data>
inline TracedCriticalPointSampleSZ compute_face_cp_sample(
    const FaceKeySZ& face,
    size_t face_id,
    const T_data* U,
    const T_data* V,
    const Size3& sz)
{
  TracedCriticalPointSampleSZ sample;
  sample.face = face;
  sample.face_id = face_id;
  sample.ordinal = face_is_ordinal(face, sz);

  double vecs[3][2];
  double coords[3][4];

  for (int k = 0; k < 3; ++k) {
    const size_t vidx = face.v[(size_t)k];
    vecs[k][0] = static_cast<double>(U[vidx]);
    vecs[k][1] = static_cast<double>(V[vidx]);

    int t, i, j;
    inv_vid(vidx, sz.H, sz.W, t, i, j);
    coords[k][0] = static_cast<double>(j);
    coords[k][1] = static_cast<double>(i);
    coords[k][2] = 0.0;
    coords[k][3] = static_cast<double>(t);
  }

  double mu[3];
  double cond = 0.0;
  bool succ = ftk::inverse_lerp_s2v2(vecs, mu, &cond);
  if (!succ)
    ftk::clamp_barycentric<3>(mu);

  double pos[4] = {0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k)
    for (int d = 0; d < 4; ++d)
      pos[d] += coords[k][d] * mu[k];

  sample.position = {pos[0], pos[1], pos[3]};
  sample.barycentric = {mu[0], mu[1], mu[2]};
  sample.cond = cond;
  sample.solver_success = succ;
  return sample;
}

inline void add_tetra_adjacency(
    const Tet& tet,
    const std::unordered_map<FaceKeySZ, size_t, FaceKeySZHash>& face_to_id,
    std::vector<std::set<size_t>>& adjacency)
{
  static const int face_idx[4][3] = {
      {1, 2, 3},
      {0, 2, 3},
      {0, 1, 3},
      {0, 1, 2}
  };

  std::array<size_t,4> hits{};
  size_t hit_count = 0;

  for (int f = 0; f < 4; ++f) {
    const FaceKeySZ key(
        tet.v[face_idx[f][0]],
        tet.v[face_idx[f][1]],
        tet.v[face_idx[f][2]]);
    auto it = face_to_id.find(key);
    if (it != face_to_id.end())
      hits[hit_count++] = it->second;
  }

  for (size_t i = 0; i < hit_count; ++i) {
    for (size_t j = i + 1; j < hit_count; ++j) {
      adjacency[hits[i]].insert(hits[j]);
      adjacency[hits[j]].insert(hits[i]);
    }
  }
}

template<typename T_data>
inline std::vector<TracedCriticalPointCurveSZ> ftk_trace_critical_point_curves(
    const T_data* U,
    const T_data* V,
    const Size3& sz,
    const std::unordered_set<FaceKeySZ, FaceKeySZHash>& cp_faces)
{
  std::vector<TracedCriticalPointCurveSZ> curves;
  if (cp_faces.empty())
    return curves;

  std::unordered_map<FaceKeySZ, size_t, FaceKeySZHash> face_to_id;
  face_to_id.reserve(cp_faces.size() * 2);

  std::vector<TracedCriticalPointSampleSZ> samples;
  samples.reserve(cp_faces.size());

  size_t next_id = 0;
  for (const auto& face : cp_faces) {
    face_to_id.emplace(face, next_id);
    samples.emplace_back(compute_face_cp_sample(face, next_id, U, V, sz));
    ++next_id;
  }

  std::vector<std::set<size_t>> adjacency(samples.size());
  const size_t dv = static_cast<size_t>(sz.H) * static_cast<size_t>(sz.W);

  if (sz.T >= 2) {
    for (int t = 0; t < sz.T - 1; ++t) {
      for (int i = 0; i < sz.H - 1; ++i) {
        for (int j = 0; j < sz.W - 1; ++j) {
          auto base_upper = tri_vertices_2d(i, j, TriInCell::Upper, t, sz);
          std::array<size_t,3> top_upper{
              base_upper[0] + dv,
              base_upper[1] + dv,
              base_upper[2] + dv
          };
          for (const auto& tet : prism_split_3tets(base_upper, top_upper))
            add_tetra_adjacency(tet, face_to_id, adjacency);

          auto base_lower = tri_vertices_2d(i, j, TriInCell::Lower, t, sz);
          std::array<size_t,3> top_lower{
              base_lower[0] + dv,
              base_lower[1] + dv,
              base_lower[2] + dv
          };
          for (const auto& tet : prism_split_3tets(base_lower, top_lower))
            add_tetra_adjacency(tet, face_to_id, adjacency);
        }
      }
    }
  }

  std::function<std::set<size_t>(size_t)> neighbor_func =
      [&](size_t node) -> std::set<size_t> {
        if (node >= adjacency.size())
          return std::set<size_t>();
        return adjacency[node];
      };

  std::vector<char> visited(samples.size(), 0);
  for (size_t start = 0; start < samples.size(); ++start) {
    if (visited[start])
      continue;

    std::set<size_t> component;
    std::queue<size_t> q;
    q.push(start);
    visited[start] = 1;

    while (!q.empty()) {
      size_t node = q.front();
      q.pop();
      component.insert(node);

      for (size_t nb : adjacency[node]) {
        if (!visited[nb]) {
          visited[nb] = 1;
          q.push(nb);
        }
      }
    }

    auto linear_graphs =
        ftk::connected_component_to_linear_components<size_t>(component, neighbor_func);
    for (auto& graph : linear_graphs) {
      if (graph.empty())
        continue;

      TracedCriticalPointCurveSZ curve;
      curve.id = curves.size();
      curve.loop = ftk::is_loop<size_t>(graph, neighbor_func);
      curve.samples.reserve(graph.size());

      for (size_t node_id : graph)
        curve.samples.push_back(samples[node_id]);

      curves.push_back(std::move(curve));
    }
  }

  return curves;
}

template<typename T_data>
inline std::vector<TracedCriticalPointCurveSZ> ftk_trace_critical_point_curves(
    const T_data* U,
    const T_data* V,
    int H, int W, int T,
    const std::unordered_set<FaceKeySZ, FaceKeySZHash>& cp_faces)
{
  const Size3 sz{H, W, T};
  return ftk_trace_critical_point_curves(U, V, sz, cp_faces);
}


// ----------------------------------------------------------------------------
// FTK use different way to split tetrahedra: Write traced critical points to VTK PolyData (.vtp) without linking VTK.
// Produces a PolyData with Points=(x,y,z=t). Each curve is a PolyLine cell.
// Adds useful PointData and CellData arrays for analysis.
// ----------------------------------------------------------------------------
template<typename T_data>
inline bool ftk_write_traced_critical_point_vtp(
    const std::string& vtp_path,
    const T_data* U,
    const T_data* V,
    const Size3& sz,
    const std::unordered_set<FaceKeySZ, FaceKeySZHash>& cp_faces)
{
  auto curves = ftk_trace_critical_point_curves(U, V, sz, cp_faces);

  // Gather points and attributes
  std::vector<double> pts; pts.reserve(3 * 1024);
  std::vector<int>    pt_curve_id;
  std::vector<int>    pt_seq_idx;
  std::vector<long long> pt_face_id;
  std::vector<double> pt_bary; // 3 components
  std::vector<double> pt_cond;
  std::vector<int>    pt_ordinal;
  std::vector<int>    pt_solver_succ;

  std::vector<int> line_conn;   line_conn.reserve(1024);
  std::vector<int> line_offsets; line_offsets.reserve(256);
  std::vector<int> line_curve_id;
  std::vector<int> line_is_loop;

  std::vector<int> vert_conn;   // singleton points (curves with 1 sample)
  std::vector<int> vert_offsets;

  int point_counter = 0;
  for (const auto& curve : curves) {
    const int cid = static_cast<int>(curve.id);
    const int n   = static_cast<int>(curve.samples.size());

    if (n <= 0) continue;

    // Record points and per-point attributes
    const int start_idx = point_counter;
    for (int k = 0; k < n; ++k) {
      const auto& s = curve.samples[(size_t)k];
      pts.push_back(s.position[0]);
      pts.push_back(s.position[1]);
      pts.push_back(s.position[2]); // use time as z

      pt_curve_id.push_back(cid);
      pt_seq_idx.push_back(k);
      pt_face_id.push_back((long long)s.face_id);
      pt_bary.push_back(s.barycentric[0]);
      pt_bary.push_back(s.barycentric[1]);
      pt_bary.push_back(s.barycentric[2]);
      pt_cond.push_back(s.cond);
      pt_ordinal.push_back(s.ordinal ? 1 : 0);
      pt_solver_succ.push_back(s.solver_success ? 1 : 0);

      ++point_counter;
    }

    // Make a PolyLine (if >=2 points) or a Vert (if single point)
    if (n >= 2) {
      for (int k = 0; k < n; ++k)
        line_conn.push_back(start_idx + k);
      line_offsets.push_back((line_offsets.empty() ? 0 : line_offsets.back()) + n);
      line_curve_id.push_back(cid);
      line_is_loop.push_back(curve.loop ? 1 : 0);
    } else {
      vert_conn.push_back(start_idx);
      vert_offsets.push_back((vert_offsets.empty() ? 0 : vert_offsets.back()) + 1);
    }
  }

  // Write XML .vtp (ASCII)
  std::ofstream ofs(vtp_path);
  if (!ofs) {
    std::cerr << "Failed to open " << vtp_path << " for writing .vtp" << std::endl;
    return false;
  }

  ofs.setf(std::ios::fixed);
  ofs << std::setprecision(17);

  const int nPoints = static_cast<int>(pts.size() / 3);
  const int nLines  = static_cast<int>(line_offsets.size());
  const int nVerts  = static_cast<int>(vert_offsets.size());
  const int nCells  = nVerts + nLines; // CellData arrays must cover all cell types

  ofs << "<?xml version=\"1.0\"?>\n";
  ofs << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
  ofs << "  <PolyData>\n";
  ofs << "    <Piece NumberOfPoints=\"" << nPoints
      << "\" NumberOfVerts=\"" << nVerts
      << "\" NumberOfLines=\"" << nLines
      << "\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

  // PointData
  ofs << "      <PointData>\n";
  // curve_id
  ofs << "        <DataArray Name=\"curve_id\" type=\"Int32\" format=\"ascii\">\n          ";
  for (size_t i = 0; i < pt_curve_id.size(); ++i) {
    ofs << pt_curve_id[i] << (i+1==pt_curve_id.size()?"\n":" ");
  }
  ofs << "        </DataArray>\n";
  // seq_idx
  ofs << "        <DataArray Name=\"seq_idx\" type=\"Int32\" format=\"ascii\">\n          ";
  for (size_t i = 0; i < pt_seq_idx.size(); ++i) {
    ofs << pt_seq_idx[i] << (i+1==pt_seq_idx.size()?"\n":" ");
  }
  ofs << "        </DataArray>\n";
  // face_id
  ofs << "        <DataArray Name=\"face_id\" type=\"Int64\" format=\"ascii\">\n          ";
  for (size_t i = 0; i < pt_face_id.size(); ++i) {
    ofs << pt_face_id[i] << (i+1==pt_face_id.size()?"\n":" ");
  }
  ofs << "        </DataArray>\n";
  // barycentric (3 components)
  ofs << "        <DataArray Name=\"barycentric\" type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
  for (size_t i = 0; i < pt_bary.size(); ++i) {
    ofs << pt_bary[i] << (i+1==pt_bary.size()?"\n":" ");
  }
  ofs << "        </DataArray>\n";
  // cond
  ofs << "        <DataArray Name=\"cond\" type=\"Float64\" format=\"ascii\">\n          ";
  for (size_t i = 0; i < pt_cond.size(); ++i) {
    ofs << pt_cond[i] << (i+1==pt_cond.size()?"\n":" ");
  }
  ofs << "        </DataArray>\n";
  // ordinal
  ofs << "        <DataArray Name=\"ordinal\" type=\"Int32\" format=\"ascii\">\n          ";
  for (size_t i = 0; i < pt_ordinal.size(); ++i) {
    ofs << pt_ordinal[i] << (i+1==pt_ordinal.size()?"\n":" ");
  }
  ofs << "        </DataArray>\n";
  // solver_success
  ofs << "        <DataArray Name=\"solver_success\" type=\"Int32\" format=\"ascii\">\n          ";
  for (size_t i = 0; i < pt_solver_succ.size(); ++i) {
    ofs << pt_solver_succ[i] << (i+1==pt_solver_succ.size()?"\n":" ");
  }
  ofs << "        </DataArray>\n";
  ofs << "      </PointData>\n";

  // CellData for all cells (Verts + Lines): pad verts with defaults
  ofs << "      <CellData>\n";
  if (nCells > 0) {
    std::vector<int> cell_curve_id;
    std::vector<int> cell_is_loop;
    cell_curve_id.reserve(nCells);
    cell_is_loop.reserve(nCells);
    // First Verts
    for (int i = 0; i < nVerts; ++i) {
      cell_curve_id.push_back(-1);
      cell_is_loop.push_back(0);
    }
    // Then Lines
    for (size_t i = 0; i < line_curve_id.size(); ++i) {
      cell_curve_id.push_back(line_curve_id[i]);
    }
    for (size_t i = 0; i < line_is_loop.size(); ++i) {
      cell_is_loop.push_back(line_is_loop[i]);
    }

    ofs << "        <DataArray Name=\"curve_id\" type=\"Int32\" format=\"ascii\">\n          ";
    for (size_t i = 0; i < cell_curve_id.size(); ++i) {
      ofs << cell_curve_id[i] << (i+1==cell_curve_id.size()?"\n":" ");
    }
    ofs << "        </DataArray>\n";

    ofs << "        <DataArray Name=\"is_loop\" type=\"Int32\" format=\"ascii\">\n          ";
    for (size_t i = 0; i < cell_is_loop.size(); ++i) {
      ofs << cell_is_loop[i] << (i+1==cell_is_loop.size()?"\n":" ");
    }
    ofs << "        </DataArray>\n";
  }
  ofs << "      </CellData>\n";

  // Points
  ofs << "      <Points>\n";
  ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
  for (size_t i = 0; i < pts.size(); ++i) {
    ofs << pts[i] << (i+1==pts.size()?"\n":" ");
  }
  ofs << "        </DataArray>\n";
  ofs << "      </Points>\n";

  // Verts (for curves with a single point)
  if (nVerts > 0) {
    ofs << "      <Verts>\n";
    ofs << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n          ";
    for (size_t i = 0; i < vert_conn.size(); ++i) {
      ofs << vert_conn[i] << (i+1==vert_conn.size()?"\n":" ");
    }
    ofs << "        </DataArray>\n";
    ofs << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n          ";
    for (size_t i = 0; i < vert_offsets.size(); ++i) {
      ofs << vert_offsets[i] << (i+1==vert_offsets.size()?"\n":" ");
    }
    ofs << "        </DataArray>\n";
    ofs << "      </Verts>\n";
  }

  // Lines
  if (nLines > 0) {
    ofs << "      <Lines>\n";
    ofs << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n          ";
    for (size_t i = 0; i < line_conn.size(); ++i) {
      ofs << line_conn[i] << (i+1==line_conn.size()?"\n":" ");
    }
    ofs << "        </DataArray>\n";
    ofs << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n          ";
    for (size_t i = 0; i < line_offsets.size(); ++i) {
      ofs << line_offsets[i] << (i+1==line_offsets.size()?"\n":" ");
    }
    ofs << "        </DataArray>\n";
    ofs << "      </Lines>\n";
  }

  ofs << "    </Piece>\n";
  ofs << "  </PolyData>\n";
  ofs << "</VTKFile>\n";

  ofs.close();
  return true;
}

template<typename T_data>
inline bool ftk_write_traced_critical_point_vtp(
    const std::string& vtp_path,
    const T_data* U,
    const T_data* V,
    int H, int W, int T,
    const std::unordered_set<FaceKeySZ, FaceKeySZHash>& cp_faces)
{
  const Size3 sz{H, W, T};
  return ftk_write_traced_critical_point_vtp(vtp_path, U, V, sz, cp_faces);
}


struct Metrics {
    double min_orig  = +std::numeric_limits<double>::infinity();
    double max_orig  = -std::numeric_limits<double>::infinity();
    double min_recon = +std::numeric_limits<double>::infinity();
    double max_recon = -std::numeric_limits<double>::infinity();
    double max_abs_err = 0.0;
    double mse   = 0.0;
    double rmse  = 0.0;
    double nrmse = 0.0;
    double psnr  = 0.0;
};

template<typename T>
static inline Metrics compute_metrics(const T* orig, const T* recon, size_t N)
{
    Metrics m;
    long double sse = 0.0L; // 用长双精度累加，降低数值误差

    for (size_t i = 0; i < N; ++i) {
        const double o = static_cast<double>(orig[i]);
        const double r = static_cast<double>(recon[i]);
        const double e = std::abs(o - r);

        m.min_orig  = std::min(m.min_orig,  o);
        m.max_orig  = std::max(m.max_orig,  o);
        m.min_recon = std::min(m.min_recon, r);
        m.max_recon = std::max(m.max_recon, r);

        m.max_abs_err = std::max(m.max_abs_err, e);

        const double de = o - r;
        sse += static_cast<long double>(de) * static_cast<long double>(de);
    }

    const double denom = static_cast<double>(N > 0 ? N : 1);
    const double mse  = static_cast<double>(sse / denom);
    const double rmse = std::sqrt(mse);
    m.mse  = mse;
    m.rmse = rmse;

    const double range = m.max_orig - m.min_orig; // 原始数据动态范围
    m.nrmse = (range > 0.0) ? rmse / range : rmse;

    if (rmse == 0.0) {
        m.psnr = std::numeric_limits<double>::infinity();
    } else if (range > 0.0) {
        m.psnr = 20.0 * std::log10(range) - 10.0 * std::log10(mse);
    } else {
        const double peak = std::abs(m.max_orig);
        m.psnr = (peak > 0.0)
                 ? 20.0 * std::log10(peak) - 10.0 * std::log10(mse)
                 : -std::numeric_limits<double>::infinity();
    }

    return m;
}

static inline void print_metrics(const char* name, const Metrics& m)
{
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6);
    std::cout << "[" << name << "]\n"
              << "  orig min/max : " << m.min_orig  << " / " << m.max_orig  << "\n"
              << "  recon min/max: " << m.min_recon << " / " << m.max_recon << "\n"
              << "  max |err|    : " << m.max_abs_err << "\n"
              << "  MSE          : " << m.mse   << "\n"
              << "  RMSE         : " << m.rmse  << "\n"
              << "  NRMSE        : " << m.nrmse << "  (normalized by orig range)\n"
              << "  PSNR [dB]    : " << m.psnr  << "\n";
}

template<typename T>
static inline void verify(const T* U_orig, const T* V_orig,
                          const T* U_recon, const T* V_recon,
                          size_t r1, size_t r2, size_t r3)
{
    const size_t N = r1 * r2 * r3;

    Metrics mu = compute_metrics(U_orig, U_recon, N);
    Metrics mv = compute_metrics(V_orig, V_recon, N);

    Metrics mall;
    {
        mall.min_orig  = std::min(mu.min_orig, mv.min_orig);
        mall.max_orig  = std::max(mu.max_orig, mv.max_orig);
        mall.min_recon = std::min(mu.min_recon, mv.min_recon);
        mall.max_recon = std::max(mu.max_recon, mv.max_recon);
        mall.max_abs_err = std::max(mu.max_abs_err, mv.max_abs_err);

        // 合并 MSE、RMSE
        const long double sse_u = static_cast<long double>(mu.mse) * N;
        const long double sse_v = static_cast<long double>(mv.mse) * N;
        const size_t Nall = 2 * N;
        const double mse_all  = static_cast<double>((sse_u + sse_v) / (Nall > 0 ? Nall : 1));
        const double rmse_all = std::sqrt(mse_all);
        mall.mse  = mse_all;
        mall.rmse = rmse_all;

        const double range_all = mall.max_orig - mall.min_orig;
        mall.nrmse = (range_all > 0.0) ? (rmse_all / range_all) : rmse_all;

        if (rmse_all == 0.0) {
            mall.psnr = std::numeric_limits<double>::infinity();
        } else if (range_all > 0.0) {
            mall.psnr = 20.0 * std::log10(range_all) - 10.0 * std::log10(mse_all);
        } else {
            const double peak = std::max(std::abs(mall.max_orig), std::abs(mall.min_orig));
            mall.psnr = (peak > 0.0)
                        ? 20.0 * std::log10(peak) - 10.0 * std::log10(mse_all)
                        : -std::numeric_limits<double>::infinity();
        }
    }

    print_metrics("U", mu);
    print_metrics("V", mv);
    print_metrics("U+V (combined)", mall);
}
struct Rot2D {
    double c; // cos(theta)
    double s; // sin(theta)
};

// 子采样 + Welford 在线估计 U/V 的协方差，并给出 PCA 对角化旋转角
template<typename T_data>
Rot2D estimate_channel_rotation(const T_data* U, const T_data* V, size_t N){
    const size_t Kmax = 200000;
    size_t stride = (N > Kmax) ? (N / Kmax) : 1;

    double mean_u = 0.0, mean_v = 0.0;
    double Cuu = 0.0, Cvv = 0.0, Cuv = 0.0;
    size_t k = 0;

    for (size_t idx=0; idx<N; idx += stride){
        double u = (double)U[idx];
        double v = (double)V[idx];
        ++k;
        double du = u - mean_u;
        double dv = v - mean_v;
        mean_u += du / k;
        mean_v += dv / k;
        Cuu += du * (u - mean_u);
        Cvv += dv * (v - mean_v);
        Cuv += du * (v - mean_v);
    }
    if (k > 1){
        Cuu /= (k-1);
        Cvv /= (k-1);
        Cuv /= (k-1);
    }

    // 相关性极弱时不旋转
    if (std::abs(Cuv) < 1e-15 && std::abs(Cuu - Cvv) < 1e-15){
        return Rot2D{1.0, 0.0};
    }
    double theta = 0.5 * std::atan2(2.0*Cuv, Cuu - Cvv);
    return Rot2D{ std::cos(theta), std::sin(theta) };
}

// 前向旋转 (u,v) -> (a,b)
template<typename T_data>
inline void apply_rot_forward_point(T_data u, T_data v, const Rot2D& R,
                                    T_data& a_out, T_data& b_out){
    double a =  R.c * (double)u + R.s * (double)v;
    double b = -R.s * (double)u + R.c * (double)v;
    a_out = (T_data)a;
    b_out = (T_data)b;
}
