#ifndef _sz_cp_preserve_utils_hpp
#define _sz_cp_preserve_utils_hpp

#include <cstddef>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <bit>
#include <algorithm>

template<typename T>
using unpred_vec = std::vector<T>;

inline int eb_exponential_quantize(double& eb, const int base, const double log_of_base, const double threshold=std::numeric_limits<float>::epsilon()){
	if(eb <= threshold){
		eb = 0;
		return 0;
	}
	int id = log2(eb / threshold)/log_of_base;
	eb = pow(base, id) * threshold;
	return id;
}

inline int eb_exponential_quantize(int64_t& eb, const int base, const double log_of_base, const int64_t threshold=1){
	if(eb <= threshold){
		eb = 0;
		return 0;
	}
	int id = log2(eb / threshold)/log_of_base;
	eb = pow(base, id) * threshold;
	return id;
}

// inline int eb_exponential_quantize_new(int64_t& eb, int64_t threshold=1) {
//   if (eb <= threshold) { eb = 0; return 0; } // 0: unpred
//   // base = 2
//   uint64_t ratio = static_cast<uint64_t>(eb / threshold);
//   int id = 63 - std::countl_zero(ratio);      // floor(log2(ratio))
//   eb = threshold << id;                       // dequant step
//   return id + 1;                              // 偏移 1 后写流
// }

// inline bool eb_dequant_from_id(int eid, int64_t threshold, int64_t& eb) {
//   if (eid == 0) { eb = 0; return false; }     // unpred
//   int id = eid - 1;
//   if (id < 0 || id > 62) { id = std::clamp(id,0,62); }  // 保护
//   eb = threshold << id;
//   return true;                                // 预测路径
// }

// inline int eb_exponential_quantize_new(int64_t &eb, int64_t threshold){
//     // eb <= threshold 时，用 id=1 表示“步长 = threshold”
//     if (eb <= threshold){ eb = threshold; return 1; }
//     // 向下取整到 {threshold, 2*threshold, 4*threshold, ...}
//     uint64_t q = (uint64_t)(eb / threshold);   // floor(eb/th)
//     int id = 0;                                // 得到最高位位置
//     while (q){ ++id; q >>= 1; }
//     eb = threshold << (id-1);                  // 代表值
//     return id;                                 // id>=1
// }

// inline bool eb_dequant_from_id(int ebid, int64_t threshold, int64_t &eb){
//     if (ebid <= 0){ eb = 0; return false; }    // 0 → 不可预测
//     eb = threshold << (ebid-1);                // 与 eb_exponential_quantize_new 对应
//     return true;
// }

// ---- eb 幂指数量化（向下取整到 {th, 2th, 4th, ...}；id>=1）+ 反量化 ----
inline int eb_exponential_quantize_new(int64_t &eb, int64_t threshold){
    // eb <= threshold 归到 id=1（代表值=threshold）
    if (eb <= threshold){ eb = threshold; return 1; }

    // 计算 floor(eb / threshold) 的最高位位置
    uint64_t q = (uint64_t)(eb / threshold);
#if __cpp_lib_bitops >= 201907L
    int id = std::bit_width(q);          // C++20，有就用
#else
    int id = 0;                           // 退化实现
    while (q){ ++id; q >>= 1; }
#endif
    // 代表值为 threshold << (id-1)
    int shift = id - 1;
    if (shift >= 62) {                    // 饱和保护
        eb = (int64_t)std::numeric_limits<int64_t>::max();
        return 63;
    }
    eb = threshold << shift;
    return id;                            // id>=1
}

inline bool eb_dequant_from_id(int ebid, int64_t threshold, int64_t &eb){
    if (ebid <= 0){ eb = 0; return false; }           // 0 → 无损/不可预测点
    int shift = ebid - 1;
    if (shift >= 62){ eb = (int64_t)std::numeric_limits<int64_t>::max(); return true; }
    eb = threshold << shift;                           // 与量化严格配对
    return true;
}

inline int eb_linear_quantize(double& eb, double threshold=1e-5){
	int id = eb / threshold;
	eb = id * threshold;
	return id;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// maximal error bound to keep the sign of postive*(1+e)^d - negative*(1-e)^d
template<typename T>
inline double max_eb_to_keep_sign(const T positive, const T negative, int degree){
  if((negative < 0) || (positive < 0)){
    printf("%.4f, %.4f\n", negative, positive);
    exit(0);
  }
  if((negative == 0) || (positive == 0)){
    return 1;
  }
  // double c = fabs(positive - negative) / (positive + negative);
  double P = 0, N = 0;
  switch(degree){
    case 1:
    	P = positive;
    	N = negative;
		break;
    case 2:
    	P = sqrt(positive);
    	N = sqrt(negative);
    	break;
    case 3:
    	P = cbrt(positive);
    	N = cbrt(negative);
    	break;
    default:
		printf("Degree higher than 3 not supported yet\n");
		exit(0);
  }
  return fabs(P - N)/(P + N);
}

/* ----------------------------- asqrt1 ----------------------------- */

/* This is asqrt with an additional step of the Newton iteration, for
increased accuracy.
   The relative error ranges from 0 to +0.00000023. */

inline float asqrt1(float x0) {
   union {int ix; float x;};

   x = x0;                      // x can be viewed as int.
   ix = 0x1fbb3f80 + (ix >> 1); // Initial guess.
   x = 0.5f*(x + x0/x);         // Newton step.
   x = 0.5f*(x + x0/x);         // Newton step again.
   return x;
}

inline float max_eb_to_keep_sign_2d_offline(const float positive, const float negative){
	float P = asqrt1(positive);
	float N = asqrt1(negative);
	return fabs(P - N)/(P + N);
}

/* ----------------------------- acbrt1 ----------------------------- */

/* This is acbrt with an additional step of the Newton iteration, for
increased accuracy.
   The relative error ranges from 0 to +0.00000116. */

inline float acbrt1(float x0) {
   union {int ix; float x;};

   x = x0;                      // x can be viewed as int.
   ix = ix/4 + ix/16;           // Approximate divide by 3.
   ix = ix + ix/16;
   ix = ix + ix/256;
   ix = 0x2a5137a0 + ix;        // Initial guess.
   x = 0.33333333f*(2.0f*x + x0/(x*x));  // Newton step.
   x = 0.33333333f*(2.0f*x + x0/(x*x));  // Newton step again.
   return x;
}

inline float max_eb_to_keep_sign_3d_offline(const float positive, const float negative){
	float P = acbrt1(positive);
	float N = acbrt1(negative);
	return fabs(P - N)/(P + N);
}

template<typename T>
inline void accumulate(const T value, double& positive, double& negative){
	if(value >= 0) positive += value;
	else negative += - value;
}

template<typename T>
T *
log_transform(const T * data, unsigned char * sign, size_t n, bool verbose=false){
	T * log_data = (T *) malloc(n*sizeof(T));
	for(int i=0; i<n; i++){
		sign[i] = 0;
		if(data[i] != 0){
			sign[i] = (data[i] > 0);
			log_data[i] = (data[i] > 0) ? log2f(data[i]) : log2f(-data[i]); 
		}
		else{
			sign[i] = 0;
			log_data[i] = -100; //TODO???
		}
	}
	return log_data;
}

template<typename T>
T *
log_transform(const T * data, unsigned char * sign, unsigned char * zero, size_t n){
	T * log_data = (T *) malloc(n*sizeof(T));
	for(int i=0; i<n; i++){
		sign[i] = 0;
		if(data[i] != 0){
			sign[i] = (data[i] > 0);
			log_data[i] = (data[i] > 0) ? log2(data[i]) : log2(-data[i]); 
			zero[i] = 0;
		}
		else{
			sign[i] = 0;
			log_data[i] = -100; //TODO???
			zero[i] = 1;
		}
	}
	return log_data;
}

template<typename T>
inline bool in_range(T pos, T n){
	return (pos >= 0) && (pos < n);
}

#endif