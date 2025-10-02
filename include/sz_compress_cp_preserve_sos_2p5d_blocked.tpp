#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cstdint>
#include <cstdlib>
#include "sz_cp_preserve_utils.hpp"
#include "sz_compression_utils.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cpsz {
namespace detail {

inline std::pair<size_t, size_t> select_tile_shape(size_t rows, size_t cols, int num_threads) {
  const size_t minimum = 8;
  if (rows == 0 || cols == 0) {
    return {1, 1};
  }
  size_t target_tiles = num_threads > 0 ? static_cast<size_t>(num_threads) : 1;
  if (target_tiles == 0) target_tiles = 1;
  double elems = static_cast<double>(rows) * static_cast<double>(cols);
  double elems_per_tile = elems / static_cast<double>(target_tiles);
  if (elems_per_tile < 1.0) elems_per_tile = 1.0;
  double tile_edge = std::sqrt(elems_per_tile);
  size_t tile_r = static_cast<size_t>(tile_edge);
  size_t tile_c = tile_r;
  size_t lower_r = std::min(minimum, rows);
  size_t lower_c = std::min(minimum, cols);
  if (tile_r < lower_r) tile_r = lower_r;
  if (tile_c < lower_c) tile_c = lower_c;
  if (tile_r > rows) tile_r = rows;
  if (tile_c > cols) tile_c = cols;
  if (tile_r == 0) tile_r = 1;
  if (tile_c == 0) tile_c = 1;
  return {tile_r, tile_c};
}

inline size_t to_index(size_t i, size_t j, size_t stride) {
  return i * stride + j;
}

} // namespace detail

template<typename T_data, typename LayerFetcher, typename EbFetcher>
unsigned char* omp_sz_compress_cp_preserve_sos_2p5d_streaming_blocked(
    LayerFetcher&& fetch_layer,
    EbFetcher&& fetch_required_eb,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode,
    int num_threads)
{
  using T = int64_t;
  compressed_size = 0;
  if (!r1 || !r2 || !r3) {
    return nullptr;
  }

  const size_t H = r1;
  const size_t W = r2;
  const size_t Tt = r3;
  const size_t layer_size = H * W;

  std::vector<T_data> tmpU(layer_size);
  std::vector<T_data> tmpV(layer_size);

  double vector_field_resolution = 0.0;
  for (size_t t = 0; t < Tt; ++t) {
    if (!fetch_layer(t, tmpU.data(), tmpV.data())) {
      return nullptr;
    }
    for (size_t idx = 0; idx < layer_size; ++idx) {
      double mag = std::max(std::fabs(static_cast<double>(tmpU[idx])),
                            std::fabs(static_cast<double>(tmpV[idx])));
      if (mag > vector_field_resolution) vector_field_resolution = mag;
    }
  }

  const int type_bits = 63;
  const int nbits = (type_bits - 3) / 2;
  int vbits = 0;
  if (vector_field_resolution > 0.0) {
    vbits = static_cast<int>(std::ceil(std::log2(vector_field_resolution)));
  }
  int shift_bits = nbits - vbits;
  if (shift_bits < 0) shift_bits = 0;
  const T scale = static_cast<T>(1) << shift_bits;

  std::vector<T> u_prev(layer_size, static_cast<T>(0));
  std::vector<T> v_prev(layer_size, static_cast<T>(0));
  std::vector<T> u_curr(layer_size, static_cast<T>(0));
  std::vector<T> v_curr(layer_size, static_cast<T>(0));
  std::vector<T> u_next(layer_size, static_cast<T>(0));
  std::vector<T> v_next(layer_size, static_cast<T>(0));

  std::vector<T_data> float_prev_u(layer_size, static_cast<T_data>(0));
  std::vector<T_data> float_prev_v(layer_size, static_cast<T_data>(0));
  std::vector<T_data> float_curr_u(layer_size, static_cast<T_data>(0));
  std::vector<T_data> float_curr_v(layer_size, static_cast<T_data>(0));
  std::vector<T_data> float_next_u(layer_size, static_cast<T_data>(0));
  std::vector<T_data> float_next_v(layer_size, static_cast<T_data>(0));

  std::vector<T> dec_prev_u(layer_size, static_cast<T>(0));
  std::vector<T> dec_prev_v(layer_size, static_cast<T>(0));
  std::vector<T> dec_curr_u(layer_size, static_cast<T>(0));
  std::vector<T> dec_curr_v(layer_size, static_cast<T>(0));

  auto load_layer_fixed = [&](size_t t, std::vector<T>& outU, std::vector<T>& outV,
                              std::vector<T_data>* storeU,
                              std::vector<T_data>* storeV) -> bool {
    if (!fetch_layer(t, tmpU.data(), tmpV.data())) {
      return false;
    }
    for (size_t idx = 0; idx < layer_size; ++idx) {
      T u_fp = static_cast<T>(tmpU[idx] * static_cast<double>(scale));
      T v_fp = static_cast<T>(tmpV[idx] * static_cast<double>(scale));
      outU[idx] = u_fp;
      outV[idx] = v_fp;
      if (storeU) (*storeU)[idx] = tmpU[idx];
      if (storeV) (*storeV)[idx] = tmpV[idx];
    }
    return true;
  };

  if (!load_layer_fixed(0, u_curr, v_curr, &float_curr_u, &float_curr_v)) {
    return nullptr;
  }
  if (Tt > 1) {
    if (!load_layer_fixed(1, u_next, v_next, &float_next_u, &float_next_v)) {
      return nullptr;
    }
  }

  std::vector<int> eb_codes;
  eb_codes.reserve(layer_size * Tt);
  std::vector<int> data_codes;
  data_codes.reserve(2 * layer_size * Tt);
  std::vector<T_data> unpred;
  unpred.reserve((layer_size * Tt / 8 + 1) * 2);

  std::vector<T> required_layer(layer_size, static_cast<T>(0));

  const int base = 2;
  const int capacity = 65536;
  const double log_of_base = std::log2(base);
  const int intv_radius = capacity >> 1;
  const T threshold = static_cast<T>(1);
  (void)max_pwr_eb;
  (void)mode;

  double enc_max_abs_eb_fp = 0.0;
  double enc_max_real_err_fp = 0.0;

  const int effective_threads = num_threads > 0 ? num_threads : 1;
  auto [block_rows, block_cols] = detail::select_tile_shape(H, W, effective_threads);
  const size_t blocks_i = (H + block_rows - 1) / block_rows;
  const size_t blocks_j = (W + block_cols - 1) / block_cols;
  const size_t total_waves = blocks_i + blocks_j - 1;

  for (size_t t = 0; t < Tt; ++t) {
    if (!fetch_required_eb(t, required_layer.data())) {
      return nullptr;
    }

    std::vector<int> layer_eb(layer_size, 0);
    std::vector<int> layer_q0(layer_size, 0);
    std::vector<int> layer_q1(layer_size, 0);
    std::vector<uint8_t> layer_pred(layer_size, 0);
    std::vector<T_data> layer_unpred_u(layer_size, static_cast<T_data>(0));
    std::vector<T_data> layer_unpred_v(layer_size, static_cast<T_data>(0));

    double layer_max_abs_fp = 0.0;
    double layer_max_real_err_fp = 0.0;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(effective_threads) reduction(max:layer_max_abs_fp) reduction(max:layer_max_real_err_fp)
    for (long wave = 0; wave < static_cast<long>(total_waves); ++wave)
#else
    for (size_t wave = 0; wave < total_waves; ++wave)
#endif
    {
#ifdef _OPENMP
      size_t wave_idx = static_cast<size_t>(wave);
#else
      size_t wave_idx = wave;
#endif
      for (size_t bi = 0; bi < blocks_i; ++bi) {
        if (bi > wave_idx) continue;
        size_t bj = wave_idx - bi;
        if (bj >= blocks_j) continue;

        size_t i_begin = bi * block_rows;
        size_t i_end = std::min(i_begin + block_rows, H);
        size_t j_begin = bj * block_cols;
        size_t j_end = std::min(j_begin + block_cols, W);

        for (size_t i = i_begin; i < i_end; ++i) {
          for (size_t j = j_begin; j < j_end; ++j) {
            const size_t idx = detail::to_index(i, j, W);

            T curU_val = u_curr[idx];
            T curV_val = v_curr[idx];
            T abs_eb = required_layer[idx];
            if (abs_eb < static_cast<T>(0)) abs_eb = -abs_eb;

            int eb_id = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
            if (abs_eb == 0) {
              layer_eb[idx] = 0;
              layer_pred[idx] = 0;
              layer_unpred_u[idx] = float_curr_u[idx];
              layer_unpred_v[idx] = float_curr_v[idx];
              dec_curr_u[idx] = curU_val;
              dec_curr_v[idx] = curV_val;
              continue;
            }

            bool unpred_flag = false;
            T dec_val[2] = {0, 0};
            T abs_err_fp_q[2] = {0, 0};

            for (int comp = 0; comp < 2; ++comp) {
              const T curv = (comp == 0) ? curU_val : curV_val;
              const std::vector<T>& prev_buf = (comp == 0) ? dec_prev_u : dec_prev_v;
              const std::vector<T>& curr_buf = (comp == 0) ? dec_curr_u : dec_curr_v;

              T d0 = (t > 0 && i > 0 && j > 0) ? prev_buf[detail::to_index(i - 1, j - 1, W)] : static_cast<T>(0);
              T d1 = (t > 0 && i > 0) ? prev_buf[detail::to_index(i - 1, j, W)] : static_cast<T>(0);
              T d2 = (t > 0 && j > 0) ? prev_buf[detail::to_index(i, j - 1, W)] : static_cast<T>(0);
              T d3 = (t > 0) ? prev_buf[detail::to_index(i, j, W)] : static_cast<T>(0);
              T d4 = (i > 0 && j > 0) ? curr_buf[detail::to_index(i - 1, j - 1, W)] : static_cast<T>(0);
              T d5 = (i > 0) ? curr_buf[detail::to_index(i - 1, j, W)] : static_cast<T>(0);
              T d6 = (j > 0) ? curr_buf[detail::to_index(i, j - 1, W)] : static_cast<T>(0);

              T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
              T diff = curv - pred;
              T qd = static_cast<T>(std::llabs(diff) / (abs_eb ? abs_eb : static_cast<T>(1))) + static_cast<T>(1);
              if (qd < static_cast<T>(capacity)) {
                qd = (diff > 0) ? qd : -qd;
                int qindex = static_cast<int>(qd / 2) + intv_radius;
                if (comp == 0) {
                  layer_q0[idx] = qindex;
                } else {
                  layer_q1[idx] = qindex;
                }
                T recon = pred + static_cast<T>(2) * (static_cast<T>(qindex) - static_cast<T>(intv_radius)) * abs_eb;
                dec_val[comp] = recon;
                if (std::llabs(recon - curv) > abs_eb) {
                  unpred_flag = true;
                  break;
                }
                abs_err_fp_q[comp] = std::llabs(recon - curv);
              } else {
                unpred_flag = true;
                break;
              }
            }

            if (unpred_flag) {
              layer_eb[idx] = 0;
              layer_pred[idx] = 0;
              layer_unpred_u[idx] = float_curr_u[idx];
              layer_unpred_v[idx] = float_curr_v[idx];
              dec_curr_u[idx] = curU_val;
              dec_curr_v[idx] = curV_val;
            } else {
              layer_eb[idx] = eb_id;
              layer_pred[idx] = 1;
              dec_curr_u[idx] = dec_val[0];
              dec_curr_v[idx] = dec_val[1];

              double abs_eb_fp = static_cast<double>(abs_eb) / static_cast<double>(scale);
              double err_u_fp = static_cast<double>(abs_err_fp_q[0]) / static_cast<double>(scale);
              double err_v_fp = static_cast<double>(abs_err_fp_q[1]) / static_cast<double>(scale);
              if (abs_eb_fp > layer_max_abs_fp) layer_max_abs_fp = abs_eb_fp;
              double real_err = std::max(err_u_fp, err_v_fp);
              if (real_err > layer_max_real_err_fp) layer_max_real_err_fp = real_err;
            }
          }
        }
      }
    }

    enc_max_abs_eb_fp = std::max(enc_max_abs_eb_fp, layer_max_abs_fp);
    enc_max_real_err_fp = std::max(enc_max_real_err_fp, layer_max_real_err_fp);

    eb_codes.insert(eb_codes.end(), layer_eb.begin(), layer_eb.end());
    for (size_t idx = 0; idx < layer_size; ++idx) {
      if (layer_pred[idx]) {
        data_codes.push_back(layer_q0[idx]);
        data_codes.push_back(layer_q1[idx]);
      } else {
        unpred.push_back(layer_unpred_u[idx]);
        unpred.push_back(layer_unpred_v[idx]);
      }
    }

    dec_prev_u.swap(dec_curr_u);
    dec_prev_v.swap(dec_curr_v);
    std::fill(dec_curr_u.begin(), dec_curr_u.end(), static_cast<T>(0));
    std::fill(dec_curr_v.begin(), dec_curr_v.end(), static_cast<T>(0));

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
        if (!load_layer_fixed(t + 2, u_next, v_next, &float_next_u, &float_next_v)) {
          return nullptr;
        }
      }
    }
  }

  const size_t reserve_bytes = std::max<size_t>(1, (sizeof(T) * 4 * layer_size * Tt) +
                                                    (sizeof(T_data) * unpred.size()) + 1024);
  unsigned char* compressed = static_cast<unsigned char*>(std::malloc(reserve_bytes));
  if (!compressed) {
    compressed_size = 0;
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

  size_t eb_num = eb_codes.size();
  write_variable_to_dst(pos, eb_num);
  if (eb_num) {
    Huffman_encode_tree_and_data(2 * 1024, eb_codes.data(), eb_num, pos);
  }

  size_t dq_num = data_codes.size();
  write_variable_to_dst(pos, dq_num);
  if (dq_num) {
    Huffman_encode_tree_and_data(2 * capacity, data_codes.data(), dq_num, pos);
  }

  compressed_size = static_cast<size_t>(pos - compressed);
  return compressed;
}

template<typename T_data>
unsigned char* omp_sz_compress_cp_preserve_sos_2p5d_streaming_blocked(
    const T_data* U,
    const T_data* V,
    const int64_t* required_eb,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode,
    int num_threads)
{
  const size_t H = r1;
  const size_t W = r2;
  const size_t layer_size = H * W;

  auto fetch_layer = [U, V, layer_size](size_t t, T_data* dstU, T_data* dstV) -> bool {
    const size_t offset = t * layer_size;
    std::memcpy(dstU, U + offset, layer_size * sizeof(T_data));
    std::memcpy(dstV, V + offset, layer_size * sizeof(T_data));
    return true;
  };

  auto fetch_required_eb = [required_eb, layer_size](size_t t, int64_t* dst) -> bool {
    const size_t offset = t * layer_size;
    std::memcpy(dst, required_eb + offset, layer_size * sizeof(int64_t));
    return true;
  };

  return omp_sz_compress_cp_preserve_sos_2p5d_streaming_blocked<T_data>(
      fetch_layer, fetch_required_eb, r1, r2, r3, compressed_size,
      max_pwr_eb, mode, num_threads);
}

template<typename T_data>
bool omp_sz_decompress_cp_preserve_sos_2p5d_streaming_blocked(
    const unsigned char* compressed,
    size_t r1, size_t r2, size_t r3,
    T_data*& U,
    T_data*& V)
{
  using T = int64_t;
  const size_t H = r1;
  const size_t W = r2;
  const size_t Tt = r3;
  const size_t N = H * W * Tt;

  if (U) std::free(U);
  if (V) std::free(V);
  U = nullptr;
  V = nullptr;

  const unsigned char* p = compressed;
  T scale = 0; read_variable_from_src(p, scale);
  int base = 0; read_variable_from_src(p, base);
  T threshold = 0; read_variable_from_src(p, threshold);
  int intv_radius = 0; read_variable_from_src(p, intv_radius);
  const int capacity = intv_radius << 1;

  size_t unpred_cnt = 0; read_variable_from_src(p, unpred_cnt);
  if (unpred_cnt % 2 != 0) {
    return false;
  }

  const T_data* unpred_data = reinterpret_cast<const T_data*>(p);
  const T_data* unpred_pos = unpred_data;
  p += unpred_cnt * sizeof(T_data);

  size_t eb_num = 0; read_variable_from_src(p, eb_num);
  int* eb_idx = Huffman_decode_tree_and_data(2 * 1024, eb_num, p);
  if (!eb_idx) {
    return false;
  }

  size_t dq_num = 0; read_variable_from_src(p, dq_num);
  int* dq = Huffman_decode_tree_and_data(2 * capacity, dq_num, p);
  if (!dq) {
    std::free(eb_idx);
    return false;
  }

  if (eb_num != N) {
    std::free(eb_idx);
    std::free(dq);
    return false;
  }
  const size_t n_unpred_points = unpred_cnt / 2;
  if (dq_num != 2 * (N - n_unpred_points)) {
    std::free(eb_idx);
    std::free(dq);
    return false;
  }

  T* U_fp = static_cast<T*>(std::malloc(N * sizeof(T)));
  T* V_fp = static_cast<T*>(std::malloc(N * sizeof(T)));
  if (!U_fp || !V_fp) {
    if (U_fp) std::free(U_fp);
    if (V_fp) std::free(V_fp);
    std::free(eb_idx);
    std::free(dq);
    return false;
  }

  T* U_pos = U_fp;
  T* V_pos = V_fp;
  int* eb_pos = eb_idx;
  int* dq_pos = dq;
  std::vector<size_t> unpred_indices;
  unpred_indices.reserve(n_unpred_points);

  const ptrdiff_t si = static_cast<ptrdiff_t>(W);
  const ptrdiff_t sj = 1;
  const ptrdiff_t sk = static_cast<ptrdiff_t>(H * W);

  for (int t = 0; t < static_cast<int>(Tt); ++t) {
    for (int i = 0; i < static_cast<int>(H); ++i) {
      for (int j = 0; j < static_cast<int>(W); ++j) {
        int ebid = *eb_pos++;
        if (ebid == 0) {
          size_t off = static_cast<size_t>(U_pos - U_fp);
          unpred_indices.push_back(off);
          *U_pos = static_cast<T>(*(unpred_pos++) * static_cast<double>(scale));
          *V_pos = static_cast<T>(*(unpred_pos++) * static_cast<double>(scale));
        } else {
          T abs_eb = static_cast<T>(std::pow(static_cast<double>(base), ebid) * static_cast<double>(threshold));
          for (int comp = 0; comp < 2; ++comp) {
            T* cur = (comp == 0) ? U_pos : V_pos;
            T d0 = (t && i && j) ? cur[-sk - si - sj] : static_cast<T>(0);
            T d1 = (t && i) ? cur[-sk - si] : static_cast<T>(0);
            T d2 = (t && j) ? cur[-sk - sj] : static_cast<T>(0);
            T d3 = (t) ? cur[-sk] : static_cast<T>(0);
            T d4 = (i && j) ? cur[-si - sj] : static_cast<T>(0);
            T d5 = (i) ? cur[-si] : static_cast<T>(0);
            T d6 = (j) ? cur[-sj] : static_cast<T>(0);
            T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
            int qidx = *dq_pos++;
            *cur = pred + static_cast<T>(2) * (qidx - intv_radius) * abs_eb;
          }
        }
        ++U_pos;
        ++V_pos;
      }
    }
  }

  U = static_cast<T_data*>(std::malloc(N * sizeof(T_data)));
  V = static_cast<T_data*>(std::malloc(N * sizeof(T_data)));
  if (!U || !V) {
    if (U) std::free(U);
    if (V) std::free(V);
    std::free(U_fp);
    std::free(V_fp);
    std::free(eb_idx);
    std::free(dq);
    return false;
  }

  // Inline float conversion to avoid needing external helper declaration
  for (size_t idx = 0; idx < N; ++idx) {
    U[idx] = static_cast<T_data>(static_cast<double>(U_fp[idx]) / static_cast<double>(scale));
    V[idx] = static_cast<T_data>(static_cast<double>(V_fp[idx]) / static_cast<double>(scale));
  }
  unpred_pos = unpred_data;
  for (size_t off : unpred_indices) {
    U[off] = *unpred_pos++;
    V[off] = *unpred_pos++;
  }

  std::free(U_fp);
  std::free(V_fp);
  std::free(eb_idx);
  std::free(dq);
  return true;
}

} // namespace cpsz
