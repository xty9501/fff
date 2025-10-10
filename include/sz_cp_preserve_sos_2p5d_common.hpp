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
#include <algorithm>
#include <functional>
#include <utility>
#include <initializer_list>
#include <tuple>
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
