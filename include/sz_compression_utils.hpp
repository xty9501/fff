#ifndef _sz_compression_utils_hpp
#define _sz_compression_utils_hpp

#include "sz_def.hpp"
#include <limits>

template <typename T>
inline T
out_of_range_data_encode(const T data, int residue_len, unsigned char *& sign_pos, int *& type_pos, unsigned char *& dst, int& pos){
	*(sign_pos ++) = (data>0);
	*(type_pos ++) = getExponent(data);
	fp_int<T> fp;
	fp.fval = data;
	int discard_len = mantissa_len<T>() - residue_len;
	fp.ival = (fp.ival >> discard_len) << discard_len;
	T decompressed = fp.fval;
	fp.ival &= 0x000FFFFFFFFFFFFF;
	fp.ival >>= discard_len;
	while(residue_len){
		int byte_rest_len = 8 - pos;
		if(residue_len >= byte_rest_len){
			*dst = (*dst) | (0xFF & (fp.ival >> (residue_len - byte_rest_len)));
			residue_len -= byte_rest_len;
			dst ++;
			pos = 0;
		}
		else{
			*dst = (*dst) | (((0xFF >> (8 - residue_len)) & fp.ival) << (byte_rest_len - residue_len));
			pos += residue_len;
			break;
		}
	}
	return decompressed;
}

template <typename T>
inline void
write_variable_to_dst(unsigned char *& dst, const T& var){
    memcpy(dst, &var, sizeof(T));
    dst += sizeof(T);
}

template <typename T>
inline void
write_array_to_dst(unsigned char *& dst, const T * array, size_t length){
    memcpy(dst, array, length*sizeof(T));
    dst += length*sizeof(T);
}

// quantize with decompression data (Lorenzo)
template<typename T>
inline int
quantize(float pred, T cur_data, double precision, int capacity, int intv_radius, T *& unpredictable_data_pos, T * decompressed){
	double diff = cur_data - pred;
	double quant_diff = fabs(diff) / precision + 1;
	if(quant_diff < capacity){
		quant_diff = (diff > 0) ? quant_diff : -quant_diff;
		int quant_index = (int)(quant_diff/2) + intv_radius;
		T decompressed_data = pred + 2 * (quant_index - intv_radius) * precision; 
		*decompressed = decompressed_data;
		if(fabs(decompressed_data - cur_data) <= precision) return quant_index;
 	}
 	*decompressed = cur_data;
 	*(unpredictable_data_pos++) = cur_data;
 	return 0;
}

// return quantization index, no decompression data (regression)
template<typename T>
inline int
quantize(float pred, T cur_data, double precision, int capacity, int intv_radius, T *& unpredictable_data_pos){
	double diff = cur_data - pred;
	double quant_diff = fabs(diff) / precision + 1;
	if(quant_diff < capacity){
		quant_diff = (diff > 0) ? quant_diff : -quant_diff;
		int quant_index = (int)(quant_diff/2) + intv_radius;
		T decompressed_data = pred + 2 * (quant_index - intv_radius) * precision; 
		if(fabs(decompressed_data - cur_data) <= precision) return quant_index;
 	}
 	*(unpredictable_data_pos++) = cur_data;
 	return 0;
}

inline void
compress_regression_coefficient_2d(const double * precisions, float * reg_params_pos, int * reg_params_type_pos, float *& reg_unpredictable_data_pos){
	float * prev_reg_params = reg_params_pos - RegCoeffNum2d;
	for(int i=0; i<RegCoeffNum2d; i++){
		*(reg_params_type_pos ++) = quantize(*prev_reg_params, *reg_params_pos, precisions[i], RegCoeffCapacity, RegCoeffRadius, reg_unpredictable_data_pos, reg_params_pos);
		prev_reg_params ++, reg_params_pos ++; 
	}
}

inline void
compress_regression_coefficient_3d(const double * precisions, float * reg_params_pos, int * reg_params_type_pos, float *& reg_unpredictable_data_pos){
	float * prev_reg_params = reg_params_pos - RegCoeffNum3d;
	for(int i=0; i<RegCoeffNum3d; i++){
		*(reg_params_type_pos ++) = quantize(*prev_reg_params, *reg_params_pos, precisions[i], RegCoeffCapacity, RegCoeffRadius, reg_unpredictable_data_pos, reg_params_pos);
		prev_reg_params ++, reg_params_pos ++; 
	}
}

void
encode_regression_coefficients_2d(const int * reg_params_type, const float * reg_unpredictable_data, size_t reg_count, size_t reg_unpredictable_count, unsigned char *& compressed_pos);

void
encode_regression_coefficients(const int * reg_params_type, const float * reg_unpredictable_data, size_t reg_count, size_t reg_unpredictable_count, unsigned char *& compressed_pos);

// copied from conf.c
unsigned int 
round_up_power_of_2(unsigned int base);

// modified from TypeManager.c
// change return value and increment byteArray
void 
convertIntArray2ByteArray_fast_1b_to_result_sz(const unsigned char* intArray, size_t intArrayLength, unsigned char *& compressed_pos);

// for test meta-compressor. sz implementation can remove this header.
HuffmanTree *
build_Huffman_tree(const int * type, size_t num_elements, size_t state_num);

void
Huffman_encode_tree_and_data(size_t state_num, const int * type, size_t num_elements, unsigned char*& compressed_pos);

// variation with speculative compression on derived eb
template<typename T>
T relax_eb(T eb, T factor){
	return eb * factor;
}

template<typename T>
T relax_eb(T eb){
	return eb * 2;
}

template<typename T>
T restrict_eb(T eb){
	return eb / 2;
}


// --------------------- Zero-RLE Encode ---------------------
inline int* zero_rle_encode(const int* data, size_t length, size_t& encoded_length)
{
    // 最坏情况：没有0 -> 压缩结果与原数据等长
    int* encoded = (int*)std::malloc(length * sizeof(int));
    if (!encoded) {
        std::cerr << "Memory allocation failed for encoded array.\n";
        std::exit(EXIT_FAILURE);
    }

    size_t out_idx = 0;
    size_t i = 0;

    while (i < length) {
        if (data[i] == 0) {
            // 统计连续0的长度
            size_t run = 0;
            while (i < length && data[i] == 0) {
                run++;
                i++;
            }
            encoded[out_idx++] = -static_cast<int>(run);
        } else {
            encoded[out_idx++] = data[i];
            i++;
        }
    }

    encoded_length = out_idx;
    return encoded;
}

// --------------------- Zero-RLE Decode ---------------------
inline int* zero_rle_decode(const int* encoded, size_t encoded_length, size_t& decoded_length)
{
    // 预估原始长度：不超过若干倍压缩长度
    size_t capacity = encoded_length * 4;
    int* decoded = (int*)std::malloc(capacity * sizeof(int));
    if (!decoded) {
        std::cerr << "Memory allocation failed for decoded array.\n";
        std::exit(EXIT_FAILURE);
    }

    size_t out_idx = 0;
    for (size_t i = 0; i < encoded_length; ++i) {
        int v = encoded[i];
        if (v < 0) {
            size_t run = static_cast<size_t>(-v);
            // 扩容检测（确保不会越界）
            if (out_idx + run > capacity) {
                capacity = (out_idx + run) * 2;
                decoded = (int*)std::realloc(decoded, capacity * sizeof(int));
            }
            std::memset(decoded + out_idx, 0, run * sizeof(int));
            out_idx += run;
        } else {
            if (out_idx + 1 > capacity) {
                capacity *= 2;
                decoded = (int*)std::realloc(decoded, capacity * sizeof(int));
            }
            decoded[out_idx++] = v;
        }
    }

    decoded_length = out_idx;
    return decoded;
}


// ---------------------正负1，0-RLE Encode-----------------
constexpr int RUN_MARKER = std::numeric_limits<int>::min();       // [RUN_MARKER, value(-1/0/+1), run_length]
constexpr int LIT_MARKER = std::numeric_limits<int>::min() + 1;   // [LIT_MARKER, k, v1..vk]
constexpr int INT_MAX_C  = std::numeric_limits<int>::max();

static inline bool is_run_value(int v) noexcept {
    return v == -1 || v == 0 || v == 1;
}

// 追加一个 int 到输出缓冲（自动扩容）
static inline void push_int(int*& buf, size_t& sz, size_t& cap, int v) {
    if (sz >= cap) {
        cap = cap ? cap * 2 : 1024;
        buf = (int*)std::realloc(buf, cap * sizeof(int));
        if (!buf) { std::fprintf(stderr, "realloc failed\n"); std::exit(EXIT_FAILURE); }
    }
    buf[sz++] = v;
}


inline int* tri_rle_encode(const int* data, size_t length, size_t& encoded_length) {
    encoded_length = 0;
    if (!data || length == 0) return nullptr;

    int* out = nullptr;
    size_t out_sz = 0, out_cap = 0;

    size_t i = 0;
    while (i < length) {
        if (is_run_value(data[i])) {
            // 统计同值（-1/0/+1）游程
            const int val = data[i];
            size_t run = 1;
            ++i;
            while (i < length && data[i] == val) { ++run; ++i; }

            // 可能超 INT_MAX，分片写出
            while (run > 0) {
                const int chunk = (run > static_cast<size_t>(INT_MAX_C)) ? INT_MAX_C
                                                                         : static_cast<int>(run);
                push_int(out, out_sz, out_cap, RUN_MARKER);
                push_int(out, out_sz, out_cap, val);     // -1 / 0 / +1
                push_int(out, out_sz, out_cap, chunk);   // run_length
                run -= static_cast<size_t>(chunk);
            }
        } else {
            // 累积一段非(-1/0/+1)的“字面块”
            size_t start = i;
            while (i < length && !is_run_value(data[i])) ++i;
            size_t k = i - start;

            // 可能超 INT_MAX，同样分片
            while (k > 0) {
                const int chunk = (k > static_cast<size_t>(INT_MAX_C)) ? INT_MAX_C
                                                                       : static_cast<int>(k);
                push_int(out, out_sz, out_cap, LIT_MARKER);
                push_int(out, out_sz, out_cap, chunk);   // 字面值数量
                for (int t = 0; t < chunk; ++t) {
                    push_int(out, out_sz, out_cap, data[start + static_cast<size_t>(t)]);
                }
                start += static_cast<size_t>(chunk);
                k     -= static_cast<size_t>(chunk);
            }
        }
    }

    encoded_length = out_sz;
    return out; // 调用者负责 free()
}


inline int* tri_rle_decode(const int* enc, size_t enc_len, size_t& decoded_length) {
    decoded_length = 0;
    if (!enc || enc_len == 0) return nullptr;

    size_t cap = 1024;
    int* out = (int*)std::malloc(cap * sizeof(int));
    if (!out) { std::fprintf(stderr, "malloc failed\n"); std::exit(EXIT_FAILURE); }
    size_t out_sz = 0;

    auto ensure = [&](size_t need) {
        if (need > cap) {
            while (cap < need) cap *= 2;
            out = (int*)std::realloc(out, cap * sizeof(int));
            if (!out) { std::fprintf(stderr, "realloc failed\n"); std::exit(EXIT_FAILURE); }
        }
    };

    size_t i = 0;
    while (i < enc_len) {
        const int tag = enc[i++];
        if (tag == RUN_MARKER) {
            if (i + 1 >= enc_len) { std::fprintf(stderr, "corrupt RUN record\n"); std::exit(EXIT_FAILURE); }
            const int val = enc[i++];
            const int run = enc[i++];
            if (!is_run_value(val) || run < 0) {
                std::fprintf(stderr, "invalid RUN record\n"); std::exit(EXIT_FAILURE);
            }
            const size_t need = out_sz + static_cast<size_t>(run);
            ensure(need);
            for (int k = 0; k < run; ++k) out[out_sz++] = val;

        } else if (tag == LIT_MARKER) {
            if (i >= enc_len) { std::fprintf(stderr, "corrupt LIT header\n"); std::exit(EXIT_FAILURE); }
            const int cnt = enc[i++];
            if (cnt < 0 || i + static_cast<size_t>(cnt) > enc_len) {
                std::fprintf(stderr, "invalid LIT count\n"); std::exit(EXIT_FAILURE);
            }
            const size_t need = out_sz + static_cast<size_t>(cnt);
            ensure(need);
            for (int k = 0; k < cnt; ++k) out[out_sz++] = enc[i++];

        } else {
            std::fprintf(stderr, "unknown tag in stream\n"); std::exit(EXIT_FAILURE);
        }
    }

    decoded_length = out_sz;
    return out; // 调用者负责 free()
}
#endif
