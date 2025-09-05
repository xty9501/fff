#include "sz_cp_preserve_utils.hpp"
#include "sz_compress_cp_preserve_2d.hpp"
#include "sz_def.hpp"
#include "sz_compression_utils.hpp"
#include <ftk/numeric/critical_point_type.hh>
#include <ftk/numeric/critical_point_test.hh>
#include <ftk/numeric/fixed_point.hh>
#include <ftk/numeric/gradient.hh>
#include <ftk/ndarray.hh>

static const int exturded_coordinates[24][4][3] = {
	// offset = 0, 0, 0
	{
		{0, 0, 1},
		{0, 1, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{0, 1, 0},
		{0, 1, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{0, 0, 1},
		{1, 0, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{0, 1, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{1, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 0, 0}
	},
	// offset = -1, 0, 0
	{
		{0, 0, 0},
		{1, 0, 1},
		{1, 1, 1},
		{1, 0, 0}
	},
	{
		{0, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{1, 0, 0}
	},
	// offset = 0, -1, 0
	{
		{0, 0, 0},
		{0, 1, 1},
		{1, 1, 1},
		{0, 1, 0}
	},
	{
		{0, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 1, 0}
	},
	// offset = -1, -1, 0
	{
		{0, 0, 0},
		{0, 1, 0},
		{1, 1, 1},
		{1, 1, 0}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 1},
		{1, 1, 0}
	},
	// offset = 0, 0, -1
	{
		{0, 0, 0},
		{0, 1, 1},
		{1, 1, 1},
		{0, 0, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 1},
		{1, 1, 1},
		{0, 0, 1}
	},
	// offset = -1, 0, -1
	{
		{0, 0, 0},
		{0, 0, 1},
		{1, 1, 1},
		{1, 0, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 1},
		{1, 0, 1}
	},
	// offset = 0, -1, -1
	{
		{0, 0, 0},
		{0, 0, 1},
		{1, 1, 1},
		{0, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 1, 0},
		{1, 1, 1},
		{0, 1, 1}
	},
	// offset = -1, -1, -1
	{
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 1, 0},
		{0, 1, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 0, 1},
		{1, 0, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 1, 0},
		{1, 1, 0},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 0},
		{1, 1, 1}
	}
};

// template<typename T_fp>
// static vector<bool> 
// compute_cp_2d_time(const T_fp * U_fp, const T_fp * V_fp, int r1, int r2, int r3){ //r1 is time dimension
// 	// check cp for all cells
// 	vector<bool> cp_exist(6*(r1-1)*(r2-1)*(r3-1), 0);
// 	ptrdiff_t dim0_offset = r2*r3;
// 	ptrdiff_t dim1_offset = r3;
// 	ptrdiff_t cell_dim0_offset = (r2-1)*(r3-1);
// 	ptrdiff_t cell_dim1_offset = r3-1;
// 	int indices[4] = {0};
// 	// __int128 vf[4][3] = {0};
// 	int64_t vf[4][3] = {0};
// 	for(int i=0; i<r1-1; i++){
// 		for(int j=0; j<r2-1; j++){
// 			for(int k=0; k<r3-1; k++){
// 				bool verbose = false;
// 				// order (reserved, z->x):
// 				ptrdiff_t cell_offset = 6*(i*cell_dim0_offset + j*cell_dim1_offset + k);
// 				ptrdiff_t tmp_offset = 6*(i*dim0_offset + j*dim1_offset + k);
// 				// if(tmp_offset/6 == 4001374/6) verbose = true;
// 				// if(verbose){
// 				// 	std::cout << i << " " << j << " " << k << std::endl;
// 				// }
// 				// (ftk-0) 000, 001, 011, 111
// 				update_index(vf, indices, 0, i*dim0_offset + j*dim1_offset + k, U_fp, V_fp, W_fp);
// 				update_index(vf, indices, 1, (i+1)*dim0_offset + j*dim1_offset + k, U_fp, V_fp, W_fp);
// 				update_index(vf, indices, 2, (i+1)*dim0_offset + (j+1)*dim1_offset + k, U_fp, V_fp, W_fp);
// 				update_index(vf, indices, 3, (i+1)*dim0_offset + (j+1)*dim1_offset + (k+1), U_fp, V_fp, W_fp);
// 				cp_exist[cell_offset] = (check_cp(vf, indices) == 1);
// 				// if(verbose){
// 				// 	auto offset = cell_offset;
// 				// 	std::cout << "cell id = " << offset << ", cp = " << +cp_exist[offset] << std::endl;
// 				// 	std::cout << "indices: ";
// 				// 	for(int i=0; i<4; i++){
// 				// 		std::cout << indices[i] << " "; 
// 				// 	}
// 				// 	std::cout << std::endl;
// 				// 	T_fp tmp[4][3];
// 				// 	for(int i=0; i<4; i++){
// 				// 		for(int j=0; j<3; j++){
// 				// 			tmp[i][j] = vf[i][j];
// 				// 		}
// 				// 	}
// 				// 	ftk::print4x3("M:", tmp);
// 				// }				
// 				// (ftk-2) 000, 010, 011, 111
// 				update_index(vf, indices, 1, i*dim0_offset + (j+1)*dim1_offset + k, U_fp, V_fp, W_fp);
// 				cp_exist[cell_offset + 1] = (check_cp(vf, indices) == 1);
// 				// (ftk-1) 000, 001, 101, 111
// 				update_index(vf, indices, 1, (i+1)*dim0_offset + j*dim1_offset + k, U_fp, V_fp, W_fp);
// 				update_index(vf, indices, 2, (i+1)*dim0_offset + j*dim1_offset + k+1, U_fp, V_fp, W_fp);
// 				cp_exist[cell_offset + 2] = (check_cp(vf, indices) == 1);
// 				// (ftk-4) 000, 100, 101, 111
// 				update_index(vf, indices, 1, i*dim0_offset + j*dim1_offset + k+1, U_fp, V_fp, W_fp);
// 				cp_exist[cell_offset + 3] = (check_cp(vf, indices) == 1);
// 				// if(verbose){
// 				// 	auto offset = cell_offset + 3;
// 				// 	std::cout << "cell id = " << offset << ", cp = " << +cp_exist[offset] << std::endl;
// 				// 	std::cout << "indices: ";
// 				// 	for(int i=0; i<4; i++){
// 				// 		std::cout << indices[i] << " "; 
// 				// 	}
// 				// 	std::cout << std::endl;
// 				// 	T_fp tmp[4][3];
// 				// 	for(int i=0; i<4; i++){
// 				// 		for(int j=0; j<3; j++){
// 				// 			tmp[i][j] = vf[i][j];
// 				// 		}
// 				// 	}

// 				// 	ftk::print4x3("M:", tmp);
// 				// }
// 				// (ftk-3) 000, 010, 110, 111
// 				update_index(vf, indices, 1, i*dim0_offset + (j+1)*dim1_offset + k, U_fp, V_fp, W_fp);
// 				update_index(vf, indices, 2, i*dim0_offset + (j+1)*dim1_offset + k+1, U_fp, V_fp, W_fp);
// 				cp_exist[cell_offset + 4] = (check_cp(vf, indices) == 1);
// 				// if(verbose){
// 				// 	auto offset = cell_offset + 4;
// 				// 	std::cout << "cell id = " << offset << ", cp = " << +cp_exist[offset] << std::endl;
// 				// 	std::cout << "indices: ";
// 				// 	for(int i=0; i<4; i++){
// 				// 		std::cout << indices[i] << " "; 
// 				// 	}
// 				// 	std::cout << std::endl;
// 				// 	T_fp tmp[4][3];
// 				// 	for(int i=0; i<4; i++){
// 				// 		for(int j=0; j<3; j++){
// 				// 			tmp[i][j] = vf[i][j];
// 				// 		}
// 				// 	}
// 				// 	ftk::print4x3("M:", tmp);
// 				// }				
// 				// (ftk-5) 000, 100, 110, 111
// 				update_index(vf, indices, 1, i*dim0_offset + j*dim1_offset + k+1, U_fp, V_fp, W_fp);
// 				cp_exist[cell_offset + 5] = (check_cp(vf, indices) == 1);
// 			}
// 		}
// 	}
// 	return cp_exist;	
// }


template <typename T> 
static bool 
same_direction(T u0, T u1, T u2) {
	//判断u的正负号是否相同
    int sgn0 = sgn(u0);
    if(sgn0 == 0) return false;
    if((sgn0 == sgn(u1)) && (sgn0 == sgn(u2))) return true;
    return false;
}

/*
triangle mesh x0, x1, x2, derive cp-preserving eb for x2 given x0, x1
using SoS method
*/
// template<typename T>
// static T 
// derive_cp_abs_eb_sos_online(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2){
// 	T M0 = u2*v0 - u0*v2;
// 	T M1 = u1*v2 - u2*v1;
// 	T M2 = u0*v1 - u1*v0;
// 	T M = M0 + M1 + M2;
// 	if(M == 0) return 0;
// 	// keep sign for the original simplex
// 	T eb = std::abs(M) / (std::abs(u1 - u0) + std::abs(v0 - v1));
// 	{
// 		// keep sign for replacing the first and second vertices
// 		if(std::abs(u1) + std::abs(v1) != 0){
// 			eb = MINF(eb, std::abs(u1*v2 - u2*v1) / (std::abs(u1) + std::abs(v1)));
// 		}
// 		else return 0;
// 		if(std::abs(u0) + std::abs(v0) != 0){
// 			eb = MINF(eb, std::abs(u0*v2 - u2*v0) / (std::abs(u0) + std::abs(v0)));			
// 		}
// 		else return 0;
// 		// T cur_eb = MINF(std::abs(u1*v2 - u2*v1) / (std::abs(u1) + std::abs(v1)), std::abs(u0*v2 - u2*v0) / (std::abs(u0) + std::abs(v0)));
// 		// eb = MINF(eb, cur_eb);
// 	}
// 	if(same_direction(u0, u1, u2)){			
// 		eb = MAX(eb, std::abs(u2));
// 	}
// 	if(same_direction(v0, v1, v2)){			
// 		eb = MAX(eb, std::abs(v2));
// 	}
// 	return eb;
// }

template<typename T>
static T 
derive_cp_abs_eb_sos_online(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2){
	T M0 = u2*v0 - u0*v2;
	T M1 = u1*v2 - u2*v1;
	T M2 = u0*v1 - u1*v0;
	T M = M0 + M1 + M2;
	if(M == 0) return 0;
	// keep sign for the original simplex
	T eb = std::abs(M) / (std::abs(u1 - u0) + std::abs(v0 - v1));
	{
		// keep sign for replacing the first and second vertices
		if(std::abs(u1) + std::abs(v1) != 0){
			eb = MINF(eb, std::abs(u1*v2 - u2*v1) / (std::abs(u1) + std::abs(v1)));
		}
		else return 0;
		if(std::abs(u0) + std::abs(v0) != 0){
			eb = MINF(eb, std::abs(u0*v2 - u2*v0) / (std::abs(u0) + std::abs(v0)));			
		}
		else return 0;
		// T cur_eb = MINF(std::abs(u1*v2 - u2*v1) / (std::abs(u1) + std::abs(v1)), std::abs(u0*v2 - u2*v0) / (std::abs(u0) + std::abs(v0)));
		// eb = MINF(eb, cur_eb);
	}
	if(same_direction(u0, u1, u2)){			
		eb = MAX(eb, std::abs(u2));
	}
	if(same_direction(v0, v1, v2)){			
		eb = MAX(eb, std::abs(v2));
	}
	return eb;
}
template<typename T_fp>
static int 
check_cp(T_fp vf[3][2], int indices[3]){
	// robust critical point test
	bool succ = ftk::robust_critical_point_in_simplex2(vf, indices);
	if (!succ) return -1;
	return 1;
}

template<typename T_fp>
static vector<bool> 
compute_cp(const T_fp * U_fp, const T_fp * V_fp, int r1, int r2){
	// check cp for all cells
	vector<bool> cp_exist(2*(r1-1)*(r2-1), 0);
	for(int i=0; i<r1-1; i++){
		for(int j=0; j<r2-1; j++){
			int indices[3];
			indices[0] = i*r2 + j;
			indices[1] = (i+1)*r2 + j;
			indices[2] = (i+1)*r2 + (j+1); 
			T_fp vf[3][2];
			// cell index 0
			for(int p=0; p<3; p++){
				vf[p][0] = U_fp[indices[p]];
				vf[p][1] = V_fp[indices[p]];
			}
			cp_exist[2*(i * (r2-1) + j)] = (check_cp(vf, indices) == 1);
			// cell index 1
			indices[1] = i*r2 + (j+1);
			vf[1][0] = U_fp[indices[1]];
			vf[1][1] = V_fp[indices[1]];
			cp_exist[2*(i * (r2-1) + j) + 1] = (check_cp(vf, indices) == 1);
		}
	}
	return cp_exist;	
}

#define SINGULAR 0
#define ATTRACTING 1 // 2 real negative eigenvalues
#define REPELLING 2 // 2 real positive eigenvalues
#define SADDLE 3// 1 real negative and 1 real positive
#define ATTRACTING_FOCUS 4 // complex with negative real
#define REPELLING_FOCUS 5 // complex with positive real
#define CENTER 6 // complex with 0 real

template<typename T_fp, typename T>
static int 
check_cp_type(T_fp vf[3][2], T v[3][2], double X[3][2], int indices[3]){
	// robust critical point test
	bool succ = ftk::robust_critical_point_in_simplex2(vf, indices);
	if (!succ) return -1;

	double J[2][2]; // jacobian
	double v_d[3][2];
	for(int i=0; i<3; i++){
		for(int j=0; j<2; j++){
			v_d[i][j] = v[i][j];
		}
	}
	ftk::jacobian_2dsimplex2(X, v_d, J);  
	int cp_type = 0;
	std::complex<double> eig[2];
	double delta = ftk::solve_eigenvalues2x2(J, eig);
	// if(fabs(delta) < std::numeric_limits<double>::epsilon())
	if (delta >= 0) { // two real roots
	if (eig[0].real() * eig[1].real() < 0) {
		cp_type = SADDLE;
	} else if (eig[0].real() < 0) {
		cp_type = ATTRACTING;
	}
	else if (eig[0].real() > 0){
		cp_type = REPELLING;
	}
		else cp_type = SINGULAR;
	} else { // two conjugate roots
		if (eig[0].real() < 0) {
			cp_type = ATTRACTING_FOCUS;
		} else if (eig[0].real() > 0) {
			cp_type = REPELLING_FOCUS;
		} else 
		cp_type = CENTER;
	}
	return cp_type;
}

template<typename T_data, typename T_fp>
static vector<int> 
compute_cp_and_type(const T_fp * U_fp, const T_fp * V_fp, const T_data * U, const T_data * V, int r1, int r2){
	/*
		X3	X2
		X0	X1
	*/
	// order: x then y, X320 then X210
	double X1[3][2] = {
		{0, 0},
		{0, 1},
		{1, 1}
	};
	double X2[3][2] = {
		{0, 0},
		{1, 0},
		{1, 1}
	};
	vector<int> cp_type(2*(r1-1)*(r2-1), -1);
	for(int i=0; i<r1-1; i++){
		for(int j=0; j<r2-1; j++){
			int indices[3];
			indices[0] = i*r2 + j;
			indices[1] = (i+1)*r2 + j;
			indices[2] = (i+1)*r2 + (j+1); 
			T_fp vf[3][2];
			T_data v[3][2];
			// cell index 0
			for(int p=0; p<3; p++){
				vf[p][0] = U_fp[indices[p]];
				vf[p][1] = V_fp[indices[p]];
				v[p][0] = U[indices[p]];
				v[p][1] = V[indices[p]];
			}
			cp_type[2*(i * (r2-1) + j)] = check_cp_type(vf, v, X1, indices);
			// cell index 1
			indices[1] = i*r2 + (j+1);
			vf[1][0] = U_fp[indices[1]], vf[1][1] = V_fp[indices[1]];
			v[1][0] = U[indices[1]], v[1][1] = V[indices[1]];			
			cp_type[2*(i * (r2-1) + j) + 1] = check_cp_type(vf, v, X2, indices);
		}
	}
	return cp_type;
}

// template<typename T>
// unsigned char *
// sz_compress_cp_preserve_sos_2d_online(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb){
// 	size_t num_elements = r1 * r2;
// 	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
// 	memcpy(decompressed_U, U, num_elements*sizeof(T));
// 	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
// 	memcpy(decompressed_V, V, num_elements*sizeof(T));
// 	int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int));
// 	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
// 	int * eb_quant_index_pos = eb_quant_index;
// 	int * data_quant_index_pos = data_quant_index;
// 	// next, row by row
// 	const int base = 2;
// 	const double log_of_base = log2(base);
// 	const int capacity = 65536;
// 	const int intv_radius = (capacity >> 1);
// 	unpred_vec<T> unpred_data;
// 	T * U_pos = decompressed_U;
// 	T * V_pos = decompressed_V;
// 	// offsets to get six adjacent triangle indices
// 	// the 7-th rolls back to T0
// 	/*
// 			T3	T4
// 		T2	X 	T5
// 		T1	T0(T6)
// 	*/
// 	const int offsets[7] = {
// 		-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
// 	};
// 	const T x[6][3] = {
// 		{1, 0, 1},
// 		{0, 0, 1},
// 		{0, 1, 1},
// 		{0, 1, 0},
// 		{1, 1, 0},
// 		{1, 0, 0}
// 	};
// 	const T y[6][3] = {
// 		{0, 0, 1},
// 		{0, 1, 1},
// 		{0, 1, 0},
// 		{1, 1, 0},
// 		{1, 0, 0},
// 		{1, 0, 1}
// 	};
// 	int index_offset[6][2][2];
// 	for(int i=0; i<6; i++){
// 		for(int j=0; j<2; j++){
// 			index_offset[i][j][0] = x[i][j] - x[i][2];
// 			index_offset[i][j][1] = y[i][j] - y[i][2];
// 		}
// 	}
// 	double threshold = std::numeric_limits<double>::epsilon();
// 	// conditions_2d cond;
// 	for(int i=0; i<r1; i++){
// 		// printf("start %d row\n", i);
// 		T * cur_U_pos = U_pos;
// 		T * cur_V_pos = V_pos;
// 		for(int j=0; j<r2; j++){
// 			T required_eb = max_pwr_eb;
// 			// derive eb given six adjacent triangles
// 			for(int k=0; k<6; k++){
// 				bool in_mesh = true;
// 				for(int p=0; p<2; p++){
// 					// reserved order!
// 					if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
// 						in_mesh = false;
// 						break;
// 					}
// 				}
// 				if(in_mesh){
// 					required_eb = MINF(required_eb, (T) derive_cp_abs_eb_sos_online(cur_U_pos[offsets[k]], cur_U_pos[offsets[k+1]], cur_U_pos[0],
// 						cur_V_pos[offsets[k]], cur_V_pos[offsets[k+1]], cur_V_pos[0]));
// 				}
// 			}
// 			if(required_eb > 0){
// 				bool unpred_flag = false;
// 				T decompressed[2];
// 				// compress U and V
// 				for(int k=0; k<2; k++){
// 					T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
// 					T cur_data = *cur_data_pos;
// 					double abs_eb = required_eb;
// 					eb_quant_index_pos[k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
// 					// eb_quant_index_pos[k] = eb_linear_quantize(abs_eb, 1e-3);
// 					if(eb_quant_index_pos[k] > 0){
// 						// get adjacent data and perform Lorenzo
// 						/*
// 							d2 X
// 							d0 d1
// 						*/
// 						T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
// 						T d1 = (i) ? cur_data_pos[-r2] : 0;
// 						T d2 = (j) ? cur_data_pos[-1] : 0;
// 						T pred = d1 + d2 - d0;
// 						double diff = cur_data - pred;
// 						double quant_diff = fabs(diff) / abs_eb + 1;
// 						if(quant_diff < capacity){
// 							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
// 							int quant_index = (int)(quant_diff/2) + intv_radius;
// 							data_quant_index_pos[k] = quant_index;
// 							decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
// 							// check original data
// 							if(fabs(decompressed[k] - cur_data) >= abs_eb){
// 								unpred_flag = true;
// 								break;
// 							}
// 						}
// 						else{
// 							unpred_flag = true;
// 							break;
// 						}
// 					}
// 					else unpred_flag = true;
// 				}
// 				if(unpred_flag){
// 					// recover quant index
// 					*(eb_quant_index_pos ++) = 0;
// 					*(eb_quant_index_pos ++) = 0;
// 					*(data_quant_index_pos ++) = intv_radius;
// 					*(data_quant_index_pos ++) = intv_radius;
// 					unpred_data.push_back(*cur_U_pos);
// 					unpred_data.push_back(*cur_V_pos);
// 				}
// 				else{
// 					eb_quant_index_pos += 2;
// 					data_quant_index_pos += 2;
// 					// assign decompressed data
// 					*cur_U_pos = decompressed[0];
// 					*cur_V_pos = decompressed[1];
// 				}
// 			}
// 			else{
// 				// record as unpredictable data
// 				*(eb_quant_index_pos ++) = 0;
// 				*(eb_quant_index_pos ++) = 0;
// 				*(data_quant_index_pos ++) = intv_radius;
// 				*(data_quant_index_pos ++) = intv_radius;
// 				unpred_data.push_back(*cur_U_pos);
// 				unpred_data.push_back(*cur_V_pos);
// 			}
// 			cur_U_pos ++, cur_V_pos ++;
// 		}
// 		U_pos += r2;
// 		V_pos += r2;
// 	}
// 	free(decompressed_U);
// 	free(decompressed_V);
// 	printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
// 	unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
// 	unsigned char * compressed_pos = compressed;
// 	write_variable_to_dst(compressed_pos, base);
// 	write_variable_to_dst(compressed_pos, threshold);
// 	write_variable_to_dst(compressed_pos, intv_radius);
// 	size_t unpredictable_count = unpred_data.size();
// 	write_variable_to_dst(compressed_pos, unpredictable_count);
// 	write_array_to_dst(compressed_pos, (T *)&unpred_data[0], unpredictable_count);	
// 	Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_pos);
// 	free(eb_quant_index);
// 	Huffman_encode_tree_and_data(2*capacity, data_quant_index, 2*num_elements, compressed_pos);
// 	printf("pos = %ld\n", compressed_pos - compressed);
// 	free(data_quant_index);
// 	compressed_size = compressed_pos - compressed;
// 	return compressed;	
// }

// template
// unsigned char *
// sz_compress_cp_preserve_sos_2d_online(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb);

// template
// unsigned char *
// sz_compress_cp_preserve_sos_2d_online(const double * U, const double * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb);

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

/// process 2 time-frame vector field data U, V

// template<typename T_data>
// unsigned char *
// process_2_time_frame_vector_field(const T_data * U_prev, const T_data * V_prev,
// 								  const T_data * U, const T_data * V,
// 								  size_t r1, size_t r2, size_t& compressed_size,
// 								  bool transpose, double max_pwr_eb) {
// 	// 给定2个时间帧的向量场数据U_prev, V_prev和U, V，处理它们并返回压缩后的数据，保证压缩前后cp和streamline不变
// 	size_t num_elements = r1 * r2;
// 								  }

// fff version
template<typename T_data>
unsigned char *
sz_compress_cp_preserve_sos_2d_time_online_fp(const T_data * U, const T_data * V, size_t r1_time_dim, size_t r2,size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb){
	using T = int64_t;
	unsigned char * compressed = (unsigned char *) malloc(2*r1_time_dim*r2*r3*sizeof(int64_t));
	unsigned char * compressed_pos = compressed;
	unsigned char * prev_compressed_pos = compressed_pos;
	T * U_fp = (T *) malloc(r1_time_dim*r2*r3*sizeof(int64_t));
	T * V_fp = (T *) malloc(r1_time_dim*r2*r3*sizeof(int64_t));
	T range = 0;
	T vector_field_scaling_factor = convert_to_fixed_point(U, V, r1_time_dim*r2*r3, U_fp, V_fp, range); //将u，v转化为定点数存在U_fp，V_fp中,同时返回缩放因子
	int * eb_quant_index = (int *) malloc(2*r1_time_dim*r2*r3*sizeof(int));
	int * data_quant_index = (int *) malloc(2*r1_time_dim*r2*r3*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	const int base = 2;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	T max_eb = range * max_pwr_eb;
	unpred_vec<T_data> unpred_data;
	ptrdiff_t dim0_offset = r2*r3;
	ptrdiff_t dim1_offset = r3;
	ptrdiff_t cell_dim0_offset = (r2-1) * (r3-1);
	ptrdiff_t cell_dim1_offset = r3-1;
	int simplex_offset[24];
	int index_offset[24][3][3];
	int offset[24][3];
	//TODO: compute_offset function
	T * cur_U_pos = U_fp;
	T * cur_V_pos = V_fp;
	T threshold = 1;
	//TODO: compute cp for all 2D cells



	

	// compressed_size = compressed_pos - compressed;
	// printf("compressed size = %ld\n", compressed_size);
	// return compressed;
	

	// using T = int64_t;
	// size_t num_elements = r1 * r2;
	// T * U_fp = (T *) malloc(num_elements*sizeof(T));
	// T * V_fp = (T *) malloc(num_elements*sizeof(T));
	// T range = 0;
	// T vector_field_scaling_factor = convert_to_fixed_point(U, V, num_elements, U_fp, V_fp, range); //将u，v转化为定点数存在U_fp，V_fp中,同时返回缩放因子
	// printf("fixed point range = %lld\n", range);
	// int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
	// int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	// int * eb_quant_index_pos = eb_quant_index;
	// int * data_quant_index_pos = data_quant_index;
	// // next, row by row
	// const int base = 2;
	// const double log_of_base = log2(base);
	// const int capacity = 65536;
	// const int intv_radius = (capacity >> 1);
	// T max_eb = range * max_pwr_eb;
	// unpred_vec<T_data> unpred_data;
	// // offsets to get six adjacent triangle indices
	// // the 7-th rolls back to T0
	// /*
	// 		T3	T4
	// 	T2	X 	T5
	// 	T1	T0(T6)
	// */
	// const int offsets[7] = {
	// 	-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
	// };
	// const int x[6][3] = {
	// 	{1, 0, 1},
	// 	{0, 0, 1},
	// 	{0, 1, 1},
	// 	{0, 1, 0},
	// 	{1, 1, 0},
	// 	{1, 0, 0}
	// };
	// const int y[6][3] = {
	// 	{0, 0, 1},
	// 	{0, 1, 1},
	// 	{0, 1, 0},
	// 	{1, 1, 0},
	// 	{1, 0, 0},
	// 	{1, 0, 1}
	// };
	// int index_offset[6][2][2];
	// for(int i=0; i<6; i++){
	// 	for(int j=0; j<2; j++){
	// 		index_offset[i][j][0] = x[i][j] - x[i][2];
	// 		index_offset[i][j][1] = y[i][j] - y[i][2];
	// 	}
	// }
	// int cell_offset[6] = {
	// 	-2*((int)r2-1)-1, -2*((int)r2-1)-2, -1, 0, 1, -2*((int)r2-1)
	// };
	// T * U_pos = U_fp;
	// T * V_pos = V_fp;
	// T threshold = 1;
	// // check cp for all cells
	// vector<bool> cp_exist = compute_cp(U_fp, V_fp, r1, r2);
	// //print number of 1 in cp_exist
	// printf("cp_exist: %ld\n", std::count(cp_exist.begin(), cp_exist.end(), true));
	// for(int i=0; i<r1; i++){
	// 	// printf("start %d row\n", i);
	// 	T * cur_U_pos = U_pos;
	// 	T * cur_V_pos = V_pos;
	// 	for(int j=0; j<r2; j++){
	// 		T required_eb = max_eb;
	// 		// derive eb given six adjacent triangles
	// 		for(int k=0; k<6; k++){
	// 			//计算周围6个simplex的eb
	// 			bool in_mesh = true;
	// 			for(int p=0; p<2; p++){
	// 				// reserved order!
	// 				if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
	// 					in_mesh = false;
	// 					break;
	// 				}
	// 			}
	// 			if(in_mesh){
	// 				bool original_has_cp = cp_exist[2*(i*(r2-1) + j) + cell_offset[k]];
	// 				if(original_has_cp) required_eb = 0;//if exist cp, req_eb =0
	// 				else required_eb = MINF(required_eb, (T) derive_cp_abs_eb_sos_online(cur_U_pos[offsets[k]], cur_U_pos[offsets[k+1]], cur_U_pos[0],
	// 					cur_V_pos[offsets[k]], cur_V_pos[offsets[k+1]], cur_V_pos[0]));
	// 			}
	// 		}
	// 		T abs_eb = required_eb;
	// 		*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold); //返回quant之后的位置int
	// 		if(abs_eb > 0){
	// 			bool unpred_flag = false;
	// 			T decompressed[2];
	// 			// compress U and V
	// 			for(int k=0; k<2; k++){
	// 				T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
	// 				T cur_data = *cur_data_pos;
	// 				// get adjacent data and perform Lorenzo
	// 				/*
	// 					d2 X
	// 					d0 d1
	// 				*/
	// 				T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0; //边界为0
	// 				T d1 = (i) ? cur_data_pos[-r2] : 0;//边界为0
	// 				T d2 = (j) ? cur_data_pos[-1] : 0;//边界为0
	// 				T pred = d1 + d2 - d0;
	// 				T diff = cur_data - pred;
	// 				T quant_diff = std::abs(diff) / abs_eb + 1;
	// 				if(quant_diff < capacity){
	// 					quant_diff = (diff > 0) ? quant_diff : -quant_diff;
	// 					int quant_index = (int)(quant_diff/2) + intv_radius;
	// 					data_quant_index_pos[k] = quant_index;
	// 					decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb; //还原
	// 					// check original data
	// 					if(std::abs(decompressed[k] - cur_data) >= required_eb){
	// 						unpred_flag = true;
	// 						break;
	// 					}
	// 				}
	// 				else{
	// 					// lorenzo无法准确表示（quant超过了capacity），直接记录为unpred
	// 					unpred_flag = true;
	// 					break;
	// 				}
	// 			}
	// 			if(unpred_flag){
	// 				// recover quant index
	// 				*(eb_quant_index_pos ++) = 0; // eqv to: set pointer eb_quant_index_pos's value to 0, then eb_quant_index_pos ++
	// 				ptrdiff_t offset = cur_U_pos - U_fp; // 计算指针cur_U_pos和U_fp的差值，为了得到当前位置的index
	// 				unpred_data.push_back(U[offset]);
	// 				unpred_data.push_back(V[offset]);
	// 			}
	// 			else{
	// 				eb_quant_index_pos ++;
	// 				data_quant_index_pos += 2;
	// 				// assign decompressed data
	// 				*cur_U_pos = decompressed[0];
	// 				*cur_V_pos = decompressed[1];
	// 			}
	// 		}
	// 		else{
	// 			// record as unpredictable data
	// 			*(eb_quant_index_pos ++) = 0;
	// 			ptrdiff_t offset = cur_U_pos - U_fp;
	// 			unpred_data.push_back(U[offset]);
	// 			unpred_data.push_back(V[offset]);
	// 		}
	// 		cur_U_pos ++, cur_V_pos ++;
	// 	}
	// 	U_pos += r2;
	// 	V_pos += r2;
	// }
	// free(U_fp);
	// free(V_fp);
	// printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	// unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
	// unsigned char * compressed_pos = compressed;
	// write_variable_to_dst(compressed_pos, vector_field_scaling_factor);
	// write_variable_to_dst(compressed_pos, base);
	// write_variable_to_dst(compressed_pos, threshold);
	// write_variable_to_dst(compressed_pos, intv_radius);
	// size_t unpredictable_count = unpred_data.size();
	// write_variable_to_dst(compressed_pos, unpredictable_count);
	// write_array_to_dst(compressed_pos, (T_data *)&unpred_data[0], unpredictable_count);	
	// size_t eb_quant_num = eb_quant_index_pos - eb_quant_index;
	// write_variable_to_dst(compressed_pos, eb_quant_num);
	// Huffman_encode_tree_and_data(2*1024, eb_quant_index, eb_quant_num, compressed_pos);
	// free(eb_quant_index);
	// size_t data_quant_num = data_quant_index_pos - data_quant_index;
	// write_variable_to_dst(compressed_pos, data_quant_num);
	// Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);
	// printf("pos = %ld\n", compressed_pos - compressed);
	// free(data_quant_index);
	// compressed_size = compressed_pos - compressed;
	// return compressed;	

}


template
unsigned char *
sz_compress_cp_preserve_sos_2d_time_online_fp(const float * U, const float * V, size_t r1, size_t r2, size_t time_dim, size_t& compressed_size, bool transpose, double max_pwr_eb);

template
unsigned char *
sz_compress_cp_preserve_sos_2d_time_online_fp(const double * U, const double * V, size_t r1, size_t r2, size_t time_dim, size_t& compressed_size, bool transpose, double max_pwr_eb);


// fff new
template<typename T_data>
unsigned char *
fff_compress_2d_fp(const T_data * U, const T_data * V, size_t r1, size_t r2, size_t time_dim, size_t& compressed_size, bool transpose, double max_pwr_eb){
	printf("fff_compress_2d_fp called with r1=%zu, r2=%zu, time_dim=%zu\n", r1, r2, time_dim);
	ftk::ndarray<float> vector_field;
    vector_field.reshape({2, static_cast<size_t>(r1), static_cast<size_t>(r2), static_cast<size_t>(time_dim)});
    
    std::cout << "Created FTK ndarray with shape: ";
    vector_field.print_shape(std::cout);
	for (int t = 0; t < time_dim; t++) {
        for (int h = 0; h < r2; h++) {
            for (int w = 0; w < r1; w++) {
                size_t input_idx = t * r2 * r1 + h * r1 + w;  // (T,H,W)顺序
                vector_field(0, w, h, t) = U[input_idx]; // u分量
                vector_field(1, w, h, t) = V[input_idx]; // v分量
            }
        }
    }
	printf("vector_field shape: %zu %zu %zu %zu\n", vector_field.shape(0), vector_field.shape(1), vector_field.shape(2), vector_field.shape(3));

	
	


}

template
unsigned char *
fff_compress_2d_fp(const float * U, const float * V, size_t r1, size_t r2, size_t time_dim, size_t& compressed_size, bool transpose, double max_pwr_eb);

template
unsigned char *
fff_compress_2d_fp(const double * U, const double * V, size_t r1, size_t r2, size_t time_dim, size_t& compressed_size, bool transpose, double max_pwr_eb);


template<typename T_data>
unsigned char *
sz_compress_cp_preserve_sos_2d_online_fp_spec_eb(const T_data * U, const T_data * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb){

	using T = int64_t;
	size_t num_elements = r1 * r2;
	T * U_fp = (T *) malloc(num_elements*sizeof(T));
	T * V_fp = (T *) malloc(num_elements*sizeof(T));
	T range = 0;
	T vector_field_scaling_factor = convert_to_fixed_point(U, V, num_elements, U_fp, V_fp, range);
	printf("fixed point range = %lld\n", range);
	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 2;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	T max_eb = range * max_pwr_eb;
	unpred_vec<T_data> unpred_data;
	// offsets to get six adjacent triangle indices
	// the 7-th rolls back to T0
	/*
			T3	T4
		T2	X 	T5
		T1	T0(T6)
	*/
	const int offsets[7] = {
		-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
	};
	const int x[6][3] = {
		{1, 0, 1},
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0}
	};
	const int y[6][3] = {
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0},
		{1, 0, 1}
	};
	int index_offset[6][2][2];
	for(int i=0; i<6; i++){
		for(int j=0; j<2; j++){
			index_offset[i][j][0] = x[i][j] - x[i][2];
			index_offset[i][j][1] = y[i][j] - y[i][2];
		}
	}
	// intermediate variable
	T decompressed[2];
	T data_ori[2];
	T pred[2];
	T pred_residue[2];
	int cell_offset[6] = {
		-2*((int)r2-1)-1, -2*((int)r2-1)-2, -1, 0, 1, -2*((int)r2-1)
	};
	T * U_pos = U_fp;
	T * V_pos = V_fp;
	T threshold = 1;
	// conditions_2d cond;
	// check cp for all cells
	vector<bool> cp_exist = compute_cp(U_fp, V_fp, r1, r2);
	for(int i=0; i<r1; i++){
		// printf("start %d row\n", i);
		T * cur_U_pos = U_pos;
		T * cur_V_pos = V_pos;
		for(int j=0; j<r2; j++){
			T required_eb = max_eb;
			// derive eb given six adjacent triangles
			for(int k=0; k<6; k++){
				bool in_mesh = true;
				for(int p=0; p<2; p++){
					// reserved order!
					if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
						in_mesh = false;
						break;
					}
				}
				if(in_mesh){
					bool original_has_cp = cp_exist[2*(i*(r2-1) + j) + cell_offset[k]];
					if(original_has_cp) required_eb = 0;
					else required_eb = MINF(required_eb, (T) derive_cp_abs_eb_sos_online(cur_U_pos[offsets[k]], cur_U_pos[offsets[k+1]], cur_U_pos[0],
						cur_V_pos[offsets[k]], cur_V_pos[offsets[k+1]], cur_V_pos[0]));
				}
			}
			T abs_eb = required_eb;
			// relax error bound
			abs_eb = relax_eb(abs_eb, (T) 8);
			abs_eb = MINF(abs_eb, max_eb);
			*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
			if(abs_eb > 0){
				bool unpred_flag = false;
				// predict U and V
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
					data_ori[k] = *cur_data_pos;
					// get adjacent data and perform Lorenzo
					/*
						d2 X
						d0 d1
					*/
					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
					T d1 = (i) ? cur_data_pos[-r2] : 0;
					T d2 = (j) ? cur_data_pos[-1] : 0;
					pred[k] = d1 + d2 - d0;
					pred_residue[k] = data_ori[k] - pred[k];
				}
				bool repeated = false;
				while(true){
					bool verification_flag = true;
					// quantize using relaxed eb
					for(int k=0; k<2; k++){
						T diff = pred_residue[k];
						T quant_diff = std::abs(diff) / abs_eb + 1;
						if(quant_diff < capacity){
							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
							int quant_index = (int)(quant_diff/2) + intv_radius;
							data_quant_index_pos[k] = quant_index;
							decompressed[k] = pred[k] + 2 * (quant_index - intv_radius) * abs_eb; 
							// check original data
							if(std::abs(decompressed[k] - data_ori[k]) >= required_eb){
								verification_flag = false;
								break;
							}
						}
						else{
							unpred_flag = true;
							verification_flag = true;
							break;
						}					
					}
					if(verification_flag) break;
					if(repeated){
						// special case when decompressed[p] - data_ori[p] == required_eb
						// will lead to finite loop
						unpred_flag = true;
						break;
					}
					abs_eb = restrict_eb(abs_eb);
					if(abs_eb < required_eb){
						abs_eb = required_eb;
						repeated = true;
					}
					*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
				}			
				if(unpred_flag){
					// recover quant index
					*(eb_quant_index_pos ++) = 0;
					ptrdiff_t offset = cur_U_pos - U_fp;
					unpred_data.push_back(U[offset]);
					unpred_data.push_back(V[offset]);
				}
				else{
					eb_quant_index_pos ++;
					data_quant_index_pos += 2;
					// assign decompressed data
					*cur_U_pos = decompressed[0];
					*cur_V_pos = decompressed[1];
				}
			}
			else{
				// record as unpredictable data
				*(eb_quant_index_pos ++) = 0;
				ptrdiff_t offset = cur_U_pos - U_fp;
				unpred_data.push_back(U[offset]);
				unpred_data.push_back(V[offset]);
			}
			cur_U_pos ++, cur_V_pos ++;
		}
		U_pos += r2;
		V_pos += r2;
	}
	free(U_fp);
	free(V_fp);
	printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, vector_field_scaling_factor);
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T_data *)&unpred_data[0], unpredictable_count);	
	size_t eb_quant_num = eb_quant_index_pos - eb_quant_index;
	write_variable_to_dst(compressed_pos, eb_quant_num);
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, eb_quant_num, compressed_pos);
	free(eb_quant_index);
	size_t data_quant_num = data_quant_index_pos - data_quant_index;
	write_variable_to_dst(compressed_pos, data_quant_num);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	

}

template
unsigned char *
sz_compress_cp_preserve_sos_2d_online_fp_spec_eb(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb);

template
unsigned char *
sz_compress_cp_preserve_sos_2d_online_fp_spec_eb(const double * U, const double * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb);

// variation with speculative compression on cp detection
template<typename T_data>
unsigned char *
sz_compress_cp_preserve_sos_2d_online_fp_spec_exec_fn(const T_data * U, const T_data * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb, double max_factor){

	using T = int64_t;
	size_t num_elements = r1 * r2;
	T * U_fp = (T *) malloc(num_elements*sizeof(T));
	T * V_fp = (T *) malloc(num_elements*sizeof(T));
	T range = 0;
	T vector_field_scaling_factor = convert_to_fixed_point(U, V, num_elements, U_fp, V_fp, range);
	printf("fixed point range = %lld\n", range);
	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 2;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	T max_eb = range * max_pwr_eb;
	unpred_vec<T_data> unpred_data;
	// offsets to get six adjacent triangle indices
	// the 7-th rolls back to T0
	/*
			T3	T4
		T2	X 	T5
		T1	T0(T6)
	*/
	const int offsets[7] = {
		-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
	};
	const int x[6][3] = {
		{1, 0, 1},
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0}
	};
	const int y[6][3] = {
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0},
		{1, 0, 1}
	};
	int index_offset[6][2][2];
	for(int i=0; i<6; i++){
		for(int j=0; j<2; j++){
			index_offset[i][j][0] = x[i][j] - x[i][2];
			index_offset[i][j][1] = y[i][j] - y[i][2];
		}
	}
	// offset relative to 2*(i*r2 + j)
	// note: width for cells is 2*(r2-1)
	int cell_offset[6] = {
		-2*((int)r2-1)-1, -2*((int)r2-1)-2, -1, 0, 1, -2*((int)r2-1)
	};
	// check cp for all cells
	vector<bool> cp_exist = compute_cp(U_fp, V_fp, r1, r2);
	T * U_pos = U_fp;
	T * V_pos = V_fp;
	T threshold = 1;
	// conditions_2d cond;
	for(int i=0; i<r1; i++){
		// printf("start %d row\n", i);
		T * cur_U_pos = U_pos;
		T * cur_V_pos = V_pos;
		for(int j=0; j<r2; j++){
			T abs_eb = max_eb;
			bool unpred_flag = false;
			bool verification_flag = false;
			// check if cp exists in adjacent cells
			if((*cur_U_pos == 0) && (*cur_V_pos == 0))
			{
				verification_flag = true;
				unpred_flag = true;
			}
			else{
				for(int k=0; k<6; k++){
					bool in_mesh = true;
					for(int p=0; p<2; p++){
						// reserved order!
						if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
							in_mesh = false;
							break;
						}
					}
					if(in_mesh){
						bool original_has_cp = cp_exist[2*(i*(r2-1) + j) + cell_offset[k]];
						if(original_has_cp){
							unpred_flag = true;
							verification_flag = true;
							break;
						}
					}
				}
			}
			T decompressed[2];
			// compress data and then verify
			while(!verification_flag){
				*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
				unpred_flag = false;
				// compress U and V
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
					T cur_data = *cur_data_pos;
					// get adjacent data and perform Lorenzo
					/*
						d2 X
						d0 d1
					*/
					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
					T d1 = (i) ? cur_data_pos[-r2] : 0;
					T d2 = (j) ? cur_data_pos[-1] : 0;
					T pred = d1 + d2 - d0;
					T diff = cur_data - pred;
					T quant_diff = std::abs(diff) / abs_eb + 1;
					if(quant_diff < capacity){
						quant_diff = (diff > 0) ? quant_diff : -quant_diff;
						int quant_index = (int)(quant_diff/2) + intv_radius;
						data_quant_index_pos[k] = quant_index;
						decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
					}
					else{
						unpred_flag = true;
						break;
					}
				}
				if(unpred_flag) break;
				// verify cp in six adjacent triangles
				verification_flag = true;
				for(int k=0; k<6; k++){
					bool in_mesh = true;
					for(int p=0; p<2; p++){
						// reserved order!
						if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
							in_mesh = false;
							break;
						}
					}
					if(in_mesh){
						int indices[3];
						for(int p=0; p<2; p++){
							indices[p] = (i + index_offset[k][p][1])*r2 + (j + index_offset[k][p][0]);
						}
						indices[2] = i*r2 + j;
						// TODO: change stragegy and consider types
						T vf[3][2];
						for(int p=0; p<2; p++){
							vf[p][0] = U_fp[indices[p]];
							vf[p][1] = V_fp[indices[p]];
						}
						vf[2][0] = decompressed[0], vf[2][1] = decompressed[1];
						bool decompressed_has_cp = (check_cp(vf, indices) == 1);
						if(decompressed_has_cp){
							verification_flag = false;
							break;
						}
					}
				}
				// relax error bound
				abs_eb /= 2;
				if((!verification_flag) && (abs_eb <= max_eb * 1.0/max_factor)){
					unpred_flag = true;
					verification_flag = true;					
				}
			}
			if(unpred_flag){
				// recover quant index
				*(eb_quant_index_pos ++) = 0;
				ptrdiff_t offset = cur_U_pos - U_fp;
				unpred_data.push_back(U[offset]);
				unpred_data.push_back(V[offset]);
			}
			else{
				eb_quant_index_pos ++;
				data_quant_index_pos += 2;
				// assign decompressed data
				*cur_U_pos = decompressed[0];
				*cur_V_pos = decompressed[1];
			}
			cur_U_pos ++, cur_V_pos ++;
		}
		U_pos += r2;
		V_pos += r2;
	}
	free(U_fp);
	free(V_fp);
	printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, vector_field_scaling_factor);
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T_data *)&unpred_data[0], unpredictable_count);	
	size_t eb_quant_num = eb_quant_index_pos - eb_quant_index;
	write_variable_to_dst(compressed_pos, eb_quant_num);
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, eb_quant_num, compressed_pos);
	free(eb_quant_index);
	size_t data_quant_num = data_quant_index_pos - data_quant_index;
	write_variable_to_dst(compressed_pos, data_quant_num);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	

}

template
unsigned char *
sz_compress_cp_preserve_sos_2d_online_fp_spec_exec_fn(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb, double max_factor);

template
unsigned char *
sz_compress_cp_preserve_sos_2d_online_fp_spec_exec_fn(const double * U, const double * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb, double max_factor);

template<typename T_data, typename T_fp>
static inline T_data convert_fp_to_float(T_fp fp, T_fp vector_field_scaling_factor){
	return fp * (T_data) 1.0 / vector_field_scaling_factor;
}
// variation with speculative compression on cp detection
// execute both fn and fp
template<typename T_data>
unsigned char *
sz_compress_cp_preserve_sos_2d_online_fp_spec_exec_all(const T_data * U, const T_data * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb, double max_factor){
	using T = int64_t;
	size_t num_elements = r1 * r2;
	T * U_fp = (T *) malloc(num_elements*sizeof(T));
	T * V_fp = (T *) malloc(num_elements*sizeof(T));
	T range = 0;
	T vector_field_scaling_factor = convert_to_fixed_point(U, V, num_elements, U_fp, V_fp, range);
	printf("fixed point range = %lld\n", range);
	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 2;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	T max_eb = range * max_pwr_eb;
	unpred_vec<T_data> unpred_data;
	// offsets to get six adjacent triangle indices
	// the 7-th rolls back to T0
	/*
			T3	T4
		T2	X 	T5
		T1	T0(T6)
	*/
	const int offsets[7] = {
		-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
	};
	// x for r2
	const int x[6][3] = {
		{1, 0, 1},
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0}
	};
	// y for r1
	const int y[6][3] = {
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0},
		{1, 0, 1}
	};
	int index_offset[6][2][2];
	for(int i=0; i<6; i++){
		for(int j=0; j<2; j++){
			index_offset[i][j][0] = x[i][j] - x[i][2];
			index_offset[i][j][1] = y[i][j] - y[i][2];
		}
	}
	// offset relative to 2*(i*r2 + j)
	// note: width for cells is 2*(r2-1)
	int cell_offset[6] = {
		-2*((int)r2-1)-1, -2*((int)r2-1)-2, -1, 0, 1, -2*((int)r2-1)
	};
	T * U_pos = U_fp;
	T * V_pos = V_fp;
	// dec_data
	T_data * dec_U = (T_data *) malloc(num_elements*sizeof(T_data));
	T_data * dec_V = (T_data *) malloc(num_elements*sizeof(T_data));
	memcpy(dec_U, U, num_elements*sizeof(T_data));
	memcpy(dec_V, V, num_elements*sizeof(T_data));
	T threshold = 1;
	// conditions_2d cond;
	// check cp and type for all cells
	auto cp_type = compute_cp_and_type(U_fp, V_fp, U, V, r1, r2);
	for(int i=0; i<r1; i++){
		// printf("start %d row\n", i);
		T * cur_U_pos = U_pos;
		T * cur_V_pos = V_pos;
		for(int j=0; j<r2; j++){
			T abs_eb = max_eb;
			bool unpred_flag = false;
			bool verification_flag = false;
			T decompressed[2];
			// compress data and then verify
			while(!verification_flag){
				*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
				unpred_flag = false;
				// compress U and V
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
					T cur_data = *cur_data_pos;
					// get adjacent data and perform Lorenzo
					/*
						d2 X
						d0 d1
					*/
					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
					T d1 = (i) ? cur_data_pos[-r2] : 0;
					T d2 = (j) ? cur_data_pos[-1] : 0;
					T pred = d1 + d2 - d0;
					T diff = cur_data - pred;
					T quant_diff = std::abs(diff) / abs_eb + 1;
					if(quant_diff < capacity){
						quant_diff = (diff > 0) ? quant_diff : -quant_diff;
						int quant_index = (int)(quant_diff/2) + intv_radius;
						data_quant_index_pos[k] = quant_index;
						decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
					}
					else{
						unpred_flag = true;
						break;
					}
				}
				if(unpred_flag) break;
				// verify cp in six adjacent triangles
				verification_flag = true;
				for(int k=0; k<6; k++){
					bool in_mesh = true;
					for(int p=0; p<2; p++){
						// reserved order!
						if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
							in_mesh = false;
							break;
						}
					}
					if(in_mesh){
						int indices[3];
						for(int p=0; p<2; p++){
							indices[p] = (i + index_offset[k][p][1])*r2 + (j + index_offset[k][p][0]);
						}
						indices[2] = i*r2 + j;
						double X[3][2];
						X[0][0] = x[k][0], X[0][1] = y[k][0];
						X[1][0] = x[k][1], X[1][1] = y[k][1];
						X[2][0] = x[k][2], X[2][1] = y[k][2];
						// get vf and v
						T vf[3][2];
						for(int p=0; p<2; p++){
							vf[p][0] = U_fp[indices[p]];
							vf[p][1] = V_fp[indices[p]];
						}
						vf[2][0] = decompressed[0], vf[2][1] = decompressed[1];
						T_data v[3][2];
						// use decompressed/original data for other vertices
						for(int p=0; p<2; p++){
							v[p][0] = dec_U[indices[p]];
							v[p][1] = dec_V[indices[p]];
						}
						// compute decompressed data for current vertex
						for(int p=0; p<2; p++){
							v[2][p] = convert_fp_to_float<T_data>(decompressed[p], vector_field_scaling_factor);
						}
						// sort indices
						for(int p=0; p<3; p++){
							int min_ind = p;
							for(int q=p+1; q<3; q++){
								if(indices[q] < indices[min_ind]){
									min_ind = q;
								}
							}
							if(min_ind != p){
								// swap indices and X, v, vf
								std::swap(indices[p], indices[min_ind]);
								std::swap(X[p][0], X[min_ind][0]);
								std::swap(X[p][1], X[min_ind][1]);
								std::swap(v[p][0], v[min_ind][0]);
								std::swap(v[p][1], v[min_ind][1]);
								std::swap(vf[p][0], vf[min_ind][0]);
								std::swap(vf[p][1], vf[min_ind][1]);
							}
						}
						int decompressed_cp_type = check_cp_type(vf, v, X, indices);
						if(decompressed_cp_type != cp_type[2*(i*(r2-1) + j) + cell_offset[k]]){
							verification_flag = false;
							break;
						}
					}
				}
				// relax error bound
				abs_eb /= 2;
				if((!verification_flag) && (abs_eb <= max_eb * 1.0/max_factor)){
					unpred_flag = true;
					verification_flag = true;					
				}
			}
			ptrdiff_t offset = cur_U_pos - U_fp;
			if(unpred_flag){
				// recover quant index
				*(eb_quant_index_pos ++) = 0;
				unpred_data.push_back(U[offset]);
				unpred_data.push_back(V[offset]);
			}
			else{
				eb_quant_index_pos ++;
				data_quant_index_pos += 2;
				// assign decompressed data
				*cur_U_pos = decompressed[0];
				*cur_V_pos = decompressed[1];
				dec_U[offset] = convert_fp_to_float<T_data>(decompressed[0], vector_field_scaling_factor);
				dec_V[offset] = convert_fp_to_float<T_data>(decompressed[1], vector_field_scaling_factor);
			}
			cur_U_pos ++, cur_V_pos ++;
		}
		U_pos += r2;
		V_pos += r2;
	}
	free(U_fp);
	free(V_fp);
	printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, vector_field_scaling_factor);
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T_data *)&unpred_data[0], unpredictable_count);	
	size_t eb_quant_num = eb_quant_index_pos - eb_quant_index;
	write_variable_to_dst(compressed_pos, eb_quant_num);
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, eb_quant_num, compressed_pos);
	free(eb_quant_index);
	size_t data_quant_num = data_quant_index_pos - data_quant_index;
	write_variable_to_dst(compressed_pos, data_quant_num);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	

}

template
unsigned char *
sz_compress_cp_preserve_sos_2d_online_fp_spec_exec_all(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb, double max_factor);

template
unsigned char *
sz_compress_cp_preserve_sos_2d_online_fp_spec_exec_all(const double * U, const double * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb, double max_factor);
