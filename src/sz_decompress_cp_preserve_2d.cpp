#include "sz_decompress_3d.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_decompress_block_processing.hpp"
#include <limits>
#include <unordered_set>

template<typename T>
void
sz_decompress_cp_preserve_2d_offline(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V){
	if(U) free(U);
	if(V) free(V);
	size_t num_elements = r1 * r2;
	const unsigned char * compressed_pos = compressed;
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	printf("base = %d\n", base);
	double threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	size_t compressed_eb_size = 0;
	read_variable_from_src(compressed_pos, compressed_eb_size);
	size_t compressed_u_size = 0;
	read_variable_from_src(compressed_pos, compressed_u_size);
	size_t compressed_v_size = 0;
	read_variable_from_src(compressed_pos, compressed_v_size);
	printf("eb_size = %ld, u_size = %ld, v_size = %ld\n", compressed_eb_size, compressed_u_size, compressed_v_size);
	int * type = Huffman_decode_tree_and_data(2*1024, 2*num_elements, compressed_pos);
	double * eb = (double *) malloc(num_elements*sizeof(double));
	// const double threshold=std::numeric_limits<double>::epsilon();
	for(int i=0; i<num_elements; i++){
		if(type[i] == 0) eb[i] = 0;
		else eb[i] = pow(base, type[i]) * threshold;
		// else eb[i] = type[i] * 1e-2;
	}
	U = sz_decompress_2d_with_eb<T>(compressed_pos, eb, r1, r2);
	compressed_pos += compressed_u_size;
	for(int i=0; i<num_elements; i++){
		if(type[num_elements + i] == 0) eb[i] = 0;
		else eb[i] = pow(base, type[num_elements + i]) * threshold;
		// else eb[i] = type[num_elements + i] * 1e-2;
	}
	V = sz_decompress_2d_with_eb<T>(compressed_pos, eb, r1, r2);
	free(eb);
}

template
void
sz_decompress_cp_preserve_2d_offline<float>(const unsigned char * compressed, size_t r1, size_t r2, float *& U, float *& V);

template
void
sz_decompress_cp_preserve_2d_offline<double>(const unsigned char * compressed, size_t r1, size_t r2, double *& U, double *& V);

template<typename T>
void
sz_decompress_cp_preserve_2d_offline_log(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V){
	if(U) free(U);
	if(V) free(V);
	size_t num_elements = r1 * r2;
	const unsigned char * compressed_pos = compressed;
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	// printf("base = %d\n", base);
	double threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	size_t compressed_eb_size = 0;
	read_variable_from_src(compressed_pos, compressed_eb_size);
	size_t compressed_u_size = 0;
	read_variable_from_src(compressed_pos, compressed_u_size);
	size_t compressed_v_size = 0;
	read_variable_from_src(compressed_pos, compressed_v_size);
	// printf("eb_size = %ld, u_size = %ld, v_size = %ld\n", compressed_eb_size, compressed_u_size, compressed_v_size);
	int * type = Huffman_decode_tree_and_data(2*1024, num_elements, compressed_pos);
	double * eb = (double *) malloc(num_elements*sizeof(double));
	// const double threshold=std::numeric_limits<float>::epsilon();
	for(int i=0; i<num_elements; i++){
		if(type[i] == 0) eb[i] = 0;
		else eb[i] = pow(base, type[i]) * threshold;
		// else eb[i] = type[i] * 5e-3;
	}
	size_t sign_map_size = (num_elements - 1)/8 + 1;
	unsigned char * sign_map_u = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);	
	unsigned char * sign_map_v = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);	
	// printf("before data: %ld\n", compressed_pos - compressed);
	U = sz_decompress_2d_with_eb<T>(compressed_pos, eb, r1, r2);
	compressed_pos += compressed_u_size;
	for(int i=0; i<num_elements; i++){
		if(U[i] < -99) U[i] = 0;
		else U[i] = sign_map_u[i] ? exp2(U[i]) : -exp2(U[i]);
	}
	V = sz_decompress_2d_with_eb<T>(compressed_pos, eb, r1, r2);
	for(int i=0; i<num_elements; i++){
		if(V[i] < -99) V[i] = 0;
		else V[i] = sign_map_v[i] ? exp2(V[i]) : -exp2(V[i]);
	}
	free(sign_map_u);
	free(sign_map_v);
	free(eb);
}

template
void
sz_decompress_cp_preserve_2d_offline_log<float>(const unsigned char * compressed, size_t r1, size_t r2, float *& U, float *& V);

template
void
sz_decompress_cp_preserve_2d_offline_log<double>(const unsigned char * compressed, size_t r1, size_t r2, double *& U, double *& V);

template<typename T>
void
sz_decompress_cp_preserve_2d_online(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V){
	if(U) free(U);
	if(V) free(V);
	size_t num_elements = r1 * r2;
	const unsigned char * compressed_pos = compressed;
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	printf("base = %d\n", base);
	double threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	const int capacity = (intv_radius << 1);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	const T * unpred_data_pos = (T *) compressed_pos;
	compressed_pos += unpred_data_count*sizeof(T);
	int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, 2*num_elements, compressed_pos);
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, 2*num_elements, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	U = (T *) malloc(num_elements*sizeof(T));
	V = (T *) malloc(num_elements*sizeof(T));
	T * U_pos = U;
	T * V_pos = V;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// const double threshold=std::numeric_limits<float>::epsilon();
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
			// get eb
			if(*eb_quant_index_pos == 0){
				*U_pos = *(unpred_data_pos ++);
				*V_pos = *(unpred_data_pos ++);
				eb_quant_index_pos += 2;
			}
			else{
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? U_pos : V_pos;					
					double eb = pow(base, *eb_quant_index_pos ++) * threshold;
					// double eb = *(eb_quant_index_pos ++) * 1e-3;
					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
					T d1 = (i) ? cur_data_pos[-r2] : 0;
					T d2 = (j) ? cur_data_pos[-1] : 0;
					T pred = d1 + d2 - d0;
					*cur_data_pos = pred + 2 * (data_quant_index_pos[k] - intv_radius) * eb;
				}
			}
			U_pos ++;
			V_pos ++;
			data_quant_index_pos += 2;
		}
	}
	free(eb_quant_index);
	free(data_quant_index);
}

template
void
sz_decompress_cp_preserve_2d_online<float>(const unsigned char * compressed, size_t r1, size_t r2, float *& U, float *& V);

template
void
sz_decompress_cp_preserve_2d_online<double>(const unsigned char * compressed, size_t r1, size_t r2, double *& U, double *& V);

template<typename T, typename T_fp>
static void 
convert_to_floating_point(const T_fp * U_fp, const T_fp * V_fp, size_t num_elements, T * U, T * V, int64_t vector_field_scaling_factor){
	for(int i=0; i<num_elements; i++){
		U[i] = U_fp[i] * (T)1.0 / vector_field_scaling_factor;
		V[i] = V_fp[i] * (T)1.0 / vector_field_scaling_factor;
	}
}

template<typename T_data>
void
sz_decompress_cp_preserve_2d_online_fp(const unsigned char * compressed, size_t r1, size_t r2, T_data *& U, T_data *& V){
	if(U) free(U);
	if(V) free(V);
	using T = int64_t;
	size_t num_elements = r1 * r2;
	const unsigned char * compressed_pos = compressed;
	T vector_field_scaling_factor = 0;
	read_variable_from_src(compressed_pos, vector_field_scaling_factor);
	T range = 0;
	read_variable_from_src(compressed_pos, range);
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	//printf("base = %d\n", base);
	T threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	const int capacity = (intv_radius << 1);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	const T_data * unpred_data = (T_data *) compressed_pos;
	const T_data * unpred_data_pos = unpred_data;
	compressed_pos += unpred_data_count*sizeof(T_data);
	size_t eb_quant_num = 0;
	read_variable_from_src(compressed_pos, eb_quant_num);
	int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, eb_quant_num, compressed_pos);
	size_t data_quant_num = 0;
	read_variable_from_src(compressed_pos, data_quant_num);
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, data_quant_num, compressed_pos);
	//printf("pos = %ld\n", compressed_pos - compressed);
	T * U_fp = (T *) malloc(num_elements*sizeof(T));
	T * V_fp = (T *) malloc(num_elements*sizeof(T));
	T * U_pos = U_fp;
	T * V_pos = V_fp;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	std::vector<int> unpred_data_indices;
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
			// get eb
			if(*eb_quant_index_pos == 0){
				size_t offset = U_pos - U_fp;
				unpred_data_indices.push_back(offset);
				*U_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
				*V_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
				eb_quant_index_pos ++;
			}
			else{
				T eb = pow(base, *eb_quant_index_pos ++) * threshold;
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? U_pos : V_pos;					
					// double eb = *(eb_quant_index_pos ++) * 1e-3;
					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
					T d1 = (i) ? cur_data_pos[-r2] : 0;
					T d2 = (j) ? cur_data_pos[-1] : 0;
					T pred = d1 + d2 - d0;
					*cur_data_pos = pred + 2 * (data_quant_index_pos[k] - intv_radius) * eb;
				}
				data_quant_index_pos += 2;
			}
			U_pos ++;
			V_pos ++;
		}
	}
	free(eb_quant_index);
	free(data_quant_index);
	U = (T_data *) malloc(num_elements*sizeof(T_data));
	V = (T_data *) malloc(num_elements*sizeof(T_data));
	convert_to_floating_point(U_fp, V_fp, num_elements, U, V, vector_field_scaling_factor);
	unpred_data_pos = unpred_data;
	for(const auto& index:unpred_data_indices){
		U[index] = *(unpred_data_pos++);
		V[index] = *(unpred_data_pos++);
	}
	free(U_fp);
	free(V_fp);
}

template
void
sz_decompress_cp_preserve_2d_online_fp<float>(const unsigned char * compressed, size_t r1, size_t r2, float *& U, float *& V);

template
void
sz_decompress_cp_preserve_2d_online_fp<double>(const unsigned char * compressed, size_t r1, size_t r2, double *& U, double *& V);

template<typename T>
void
sz_decompress_cp_preserve_2d_online_log(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V){
	if(U) free(U);
	if(V) free(V);
	size_t num_elements = r1 * r2;
	const unsigned char * compressed_pos = compressed;
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	// printf("base = %d\n", base);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	size_t sign_map_size = (num_elements - 1)/8 + 1;
	unsigned char * sign_map_u = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);	
	unsigned char * sign_map_v = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);	
	const int capacity = (intv_radius << 1);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	const T * unpred_data = (T *) compressed_pos;
	const T * unpred_data_pos = unpred_data;
	compressed_pos += unpred_data_count*sizeof(T);
	int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, num_elements, compressed_pos);
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, 2*num_elements, compressed_pos);
	U = (T *) malloc(num_elements*sizeof(T));
	V = (T *) malloc(num_elements*sizeof(T));
	T * U_pos = U;
	T * V_pos = V;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	const double threshold=std::numeric_limits<float>::epsilon();
	std::unordered_set<int> unpred_data_indices;
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
			// get eb
			if(*eb_quant_index_pos == 0){
				unpred_data_indices.insert(i*r2 + j);
				T data_U = *(unpred_data_pos ++);
				T data_V = *(unpred_data_pos ++);
				*U_pos = (data_U == 0) ? -100 : log2f(fabs(data_U));
				*V_pos = (data_V == 0) ? -100 : log2f(fabs(data_V));
				eb_quant_index_pos ++;
			}
			else{
				double eb = (*eb_quant_index_pos == 0) ? 0 : pow(base, *eb_quant_index_pos) * threshold;
				// double eb = (*eb_quant_index_pos == 0) ? 0 : *eb_quant_index_pos * 1e-2;
				eb_quant_index_pos ++;
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? U_pos : V_pos;					
					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
					T d1 = (i) ? cur_data_pos[-r2] : 0;
					T d2 = (j) ? cur_data_pos[-1] : 0;
					T pred = d1 + d2 - d0;
					*cur_data_pos = pred + 2 * (data_quant_index_pos[k] - intv_radius) * eb;
				}
			}
			U_pos ++;
			V_pos ++;
			data_quant_index_pos += 2;
		}
	}
	unpred_data_pos = unpred_data;
	for(int i=0; i<num_elements; i++){
		if(unpred_data_indices.count(i)){
			U[i] = *(unpred_data_pos++);
			V[i] = *(unpred_data_pos++);
		}
		else{
			if(U[i] < -99) U[i] = 0;
			else U[i] = sign_map_u[i] ? exp2(U[i]) : -exp2(U[i]);
			if(V[i] < -99) V[i] = 0;
			else V[i] = sign_map_v[i] ? exp2(V[i]) : -exp2(V[i]);
		}
	}
	free(sign_map_u);
	free(sign_map_v);
	free(eb_quant_index);
	free(data_quant_index);
}

template
void
sz_decompress_cp_preserve_2d_online_log<float>(const unsigned char * compressed, size_t r1, size_t r2, float *& U, float *& V);

template
void
sz_decompress_cp_preserve_2d_online_log<double>(const unsigned char * compressed, size_t r1, size_t r2, double *& U, double *& V);


template<typename T_data>
void
sz_decompress_cp_preserve_2d_time_online_fp(const unsigned char * compressed, size_t& r1_time_dim, size_t& r2, size_t& r3, T_data *& U, T_data *& V){
	if(U) free(U);
	if(V) free(V);
	const unsigned char * compressed_pos = compressed;
	size_t time_dim = 0;
	read_variable_from_src(compressed_pos, time_dim);
	size_t dim_r2 = 0;
	read_variable_from_src(compressed_pos, dim_r2);
	size_t dim_r3 = 0;
	read_variable_from_src(compressed_pos, dim_r3);
	std::vector<const unsigned char *> frame_ptrs;
	frame_ptrs.reserve(time_dim);
	for(size_t t = 0; t < time_dim; t++){
		size_t frame_size = 0;
		read_variable_from_src(compressed_pos, frame_size);
		frame_ptrs.push_back(compressed_pos);
		compressed_pos += frame_size;
	}
	size_t frame_element_count = dim_r2 * dim_r3;
	size_t total_elements = time_dim * frame_element_count;
	if(total_elements == 0){
		r1_time_dim = time_dim;
		r2 = dim_r2;
		r3 = dim_r3;
		U = nullptr;
		V = nullptr;
		return;
	}
	U = (T_data *) malloc(total_elements * sizeof(T_data));
	V = (T_data *) malloc(total_elements * sizeof(T_data));
	T_data * cur_U = U;
	T_data * cur_V = V;
	for(size_t t = 0; t < time_dim; t++){
		T_data * frame_U = nullptr;
		T_data * frame_V = nullptr;
		sz_decompress_cp_preserve_2d_online_fp(frame_ptrs[t], dim_r2, dim_r3, frame_U, frame_V);
		for(size_t idx = 0; idx < frame_element_count; idx++){
			cur_U[idx] = frame_U[idx];
			cur_V[idx] = frame_V[idx];
		}
		free(frame_U);
		free(frame_V);
		cur_U += frame_element_count;
		cur_V += frame_element_count;
	}
	r1_time_dim = time_dim;
	r2 = dim_r2;
	r3 = dim_r3;
}


template
void
sz_decompress_cp_preserve_2d_time_online_fp(const unsigned char * compressed, size_t& r1_time_dim, size_t& r2, size_t& r3, float *& U, float *& V);

template
void
sz_decompress_cp_preserve_2d_time_online_fp(const unsigned char * compressed, size_t& r1_time_dim, size_t& r2, size_t& r3, double *& U, double *& V);
