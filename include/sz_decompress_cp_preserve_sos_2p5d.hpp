#pragma once

#include "sz_cp_preserve_sos_2p5d_common.hpp"
#include "sz_decompression_utils.hpp"
#include "sz_prediction.hpp"

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


template<typename T_data>
bool sz_decompress_cp_preserve_sos_2p5d_fp_chanrot(
    const unsigned char* compressed,
    size_t r1, size_t r2, size_t r3,
    T_data*& U, T_data*& V){
    using T = int64_t;
    const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

    if (U) std::free(U);
    if (V) std::free(V);

    // === 头部 ===
    const unsigned char* p = compressed;
    T scale=0; read_variable_from_src(p, scale);
    printf("read scale = %ld\n", scale);
    int base=0; read_variable_from_src(p, base);
    printf("read base = %d\n", base);
    T threshold=0; read_variable_from_src(p, threshold);
    printf("read threshold = %ld\n", threshold);
    int intv_radius=0; read_variable_from_src(p, intv_radius);
    printf("read intv_radius = %d\n", intv_radius);
    const int capacity = (intv_radius<<1); (void)capacity;

    // [CHANROT] 读取通道旋转系数（与压缩端一致的写入顺序）
    double rot_c = 1.0, rot_s = 0.0;
    read_variable_from_src(p, rot_c);
    read_variable_from_src(p, rot_s);
    printf("read chanrot cos = %.6f, sin = %.6f\n", rot_c, rot_s);

    // 无损点（在压缩端以旋转域 (a,b) 写入）
    size_t unpred_cnt=0; read_variable_from_src(p, unpred_cnt);
    if (unpred_cnt % 2 != 0) return false;

    const T_data* unpred_data = reinterpret_cast<const T_data*>(p);
    const T_data* unpred_pos  = unpred_data;
    p += unpred_cnt * sizeof(T_data);

    // eb / dq
    size_t eb_num=0; read_variable_from_src(p, eb_num);
    int* eb_idx = Huffman_decode_tree_and_data(/*state_num=*/2*1024, eb_num, p);
    if (!eb_idx) return false;

    size_t dq_num=0; read_variable_from_src(p, dq_num);
    int* dq = Huffman_decode_tree_and_data(/*state_num=*/2*65536, dq_num, p);
    if (!dq){ std::free(eb_idx); return false; }

    if (eb_num != N) { std::free(eb_idx); std::free(dq); return false; }
    const size_t n_unpred_points = unpred_cnt/2;
    if (dq_num != 2*(N - n_unpred_points)) { std::free(eb_idx); std::free(dq); return false; }

    // === 定点缓冲（旋转域上的 A_fp / B_fp，沿用变量名 U_fp/V_fp 以最小改动） ===
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

    const ptrdiff_t si=W, sj=1, sk=(ptrdiff_t)(H*W);

    // === 解码重建到旋转域定点缓存 ===
    for (int t=0; t<(int)Tt; ++t){
        for (int i=0; i<(int)H; ++i){
            for (int j=0; j<(int)W; ++j){
                int ebid = *eb_pos++;

                if (ebid == 0){
                    size_t off = (size_t)(U_pos - U_fp);
                    unpred_indices.push_back(off);
                    // 注意：unpred 按 (a,b) 写入，这里直接回填到旋转域定点缓存（先放浮点，稍后统一回浮点）
                    *U_pos = (T)( *(unpred_pos++) * scale );
                    *V_pos = (T)( *(unpred_pos++) * scale );
                } else {
                    T abs_eb = (T)( std::pow((double)base,(double)ebid) * (double)threshold );

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
                        *cur = pred + 2* (qidx - intv_radius) * abs_eb;
                    }
                }
                ++U_pos; ++V_pos;
            }
        }
    }

    // === 回浮点到旋转域 (A,B) ===
    T_data* A = (T_data*)std::malloc(N*sizeof(T_data));
    T_data* B = (T_data*)std::malloc(N*sizeof(T_data));
    if(!A || !B){
        std::free(A); std::free(B);
        std::free(U_fp); std::free(V_fp);
        std::free(eb_idx); std::free(dq);
        return false;
    }
    convert_to_floating_point(U_fp, V_fp, N, A, B, scale);

    // 覆盖无损点（无损点在压缩端以旋转域 (a,b) 存储）
    unpred_pos = unpred_data;
    for (size_t off : unpred_indices){
        A[off] = *unpred_pos++;
        B[off] = *unpred_pos++;
    }

    // === 逆旋转 (a,b) -> (u,v) 并输出 ===
    U = (T_data*)std::malloc(N*sizeof(T_data));
    V = (T_data*)std::malloc(N*sizeof(T_data));
    if(!U || !V){
        std::free(U); std::free(V);
        std::free(A); std::free(B);
        std::free(U_fp); std::free(V_fp);
        std::free(eb_idx); std::free(dq);
        return false;
    }

    for (size_t idx=0; idx<N; ++idx){
        double a = (double)A[idx], b = (double)B[idx];
        double u =  rot_c * a - rot_s * b;
        double v =  rot_s * a + rot_c * b;
        U[idx] = (T_data)u;
        V[idx] = (T_data)v;
    }

    // 释放
    std::free(A); std::free(B);
    std::free(U_fp); std::free(V_fp);
    std::free(eb_idx); std::free(dq);
    return true;
    }


template<typename T_data>
bool sz_decompress_cp_preserve_sos_2p5d_fp_mop(
    const unsigned char* compressed,
    size_t r1, size_t r2, size_t r3,
    T_data*& U, T_data*& V)
{
    using T = int64_t;
    const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

    if (U) std::free(U);
    if (V) std::free(V);

    // 头部（与 encoder 顺序一致）
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

    // 读取 MOP2 扩展头
    char magic[4];
    std::memcpy(magic, p, 4); p += 4;
    if (!(magic[0]=='M' && magic[1]=='O' && magic[2]=='P' && magic[3]=='2')){
        fprintf(stderr, "[DEC] ERROR: MOP2 magic not found.\n");
        return false;
    }
    uint32_t BH=0, BW=0, blocks_i=0, blocks_j=0;
    read_variable_from_src(p, BH);
    printf("read BH = %u\n", BH);
    read_variable_from_src(p, BW);
    printf("read BW = %u\n", BW);
    read_variable_from_src(p, blocks_i);
    printf("read blocks_i = %u\n", blocks_i);
    read_variable_from_src(p, blocks_j);
    printf("read blocks_j = %u\n", blocks_j);
    uint64_t pm_len=0;
    read_variable_from_src(p, pm_len);
    printf("read pm_len = %llu\n", (unsigned long long)pm_len);
    const uint8_t* pm_bytes = reinterpret_cast<const uint8_t*>(p);
    p += (size_t)pm_len;

    // 解包 block modes
    const size_t blocks_per_frame = (size_t)blocks_i * (size_t)blocks_j;
    const size_t n_modes = (size_t)Tt * blocks_per_frame;
    if (pm_len < ( (n_modes + 3) / 4 )){
        fprintf(stderr, "[DEC] ERROR: pm_len too small (pm_len=%llu, expect >= %zu)\n",
                (unsigned long long)pm_len, (n_modes+3)/4);
        return false;
    }
    std::vector<uint8_t> modes;
    unpack_modes_2bit(pm_bytes, n_modes, modes);
    // Optional: Debug
    printf("read MOP2: BH=%u BW=%u blocks_i=%u blocks_j=%u n_modes=%zu\n",
           BH,BW,blocks_i,blocks_j,n_modes);

    // 读取 unpred / eb / dq（保持和 encoder 一致）
    size_t unpred_cnt=0; read_variable_from_src(p, unpred_cnt);
    printf("read unpred_cnt = %zu\n", unpred_cnt);
    if (unpred_cnt % 2 != 0) return false;
    const T_data* unpred_data = reinterpret_cast<const T_data*>(p);
    const T_data* unpred_pos  = unpred_data;
    p += unpred_cnt * sizeof(T_data);
    //print sum of unpred_data
    {
        double sumU=0.0, sumV=0.0;
        for (size_t idx=0; idx<unpred_cnt/2; ++idx){
            sumU += (double)(unpred_data[2*idx+0]);
            sumV += (double)(unpred_data[2*idx+1]);
        }
        printf("sum of unpred U = %.6f, V = %.6f, sumU+V = %.6f\n", sumU, sumV, sumU+sumV);
    }

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
        if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp);
        std::free(eb_idx); std::free(dq); return false;
    }

    //临时写法
    double dt_dx, dt_dy;
    if (W ==640 && H == 80){
        dt_dx = 0.8;
        dt_dy = 0.8;
    }
    else if (W == 150 && H == 450){
        dt_dx = 0.01 / 0.66666666;
        dt_dy = 0.01 / 0.66666666;
    }
    else if (W == 450 && H == 200){
        dt_dx = (4.428 / 499) / (10.0 / 450);
        dt_dy = (4.428 / 499) / (4.0 / 200);
    }
    else if (W == 512 && H == 512){
    dt_dx = (10.0 / 1000.0) / ( 1.0 / 512.0);
    dt_dy = (10.0 / 1000.0) / ( 1.0 / 512.0);
    }
    else{
        std::cout << "Please provide dt/dx and dt/dy for this dataset!" << std::endl;
        exit(0);
    }
    pred_advect_set_params((int)H, (int)W, dt_dx, dt_dy, (double)scale, /*max_disp*/-1.0);

    int* eb_pos = eb_idx;
    int* dq_pos = dq;
    std::vector<size_t> unpred_indices; unpred_indices.reserve(n_unpred_points);

    // 步长（与编码端一致）
    const ptrdiff_t si = (ptrdiff_t)W;
    const ptrdiff_t sj = (ptrdiff_t)1;
    const ptrdiff_t sk = (ptrdiff_t)(H*W);

    // 逐块逐点重建：解码顺序与编码端严格一致（t -> 块 -> 像素）
    for (int t = 0; t < (int)Tt; ++t) {
        const size_t frame_mode_off = (size_t)t * blocks_per_frame;
        // ===每帧绑定 t-1 的 U/V 速度平面，供 ADVECT 使用 ===
        const size_t plane = H * W;
        if (t >= 1) {
            const int64_t* u_prev_plane = (const int64_t*)(U_fp + (size_t)(t-1) * plane);
            const int64_t* v_prev_plane = (const int64_t*)(V_fp + (size_t)(t-1) * plane);
            // si/sj 用元素步长（与你的预测器一致：si=W, sj=1）
            pred_advect_bind_prev_uv(u_prev_plane, v_prev_plane, /*si_uv=*/(ptrdiff_t)W, /*sj_uv=*/(ptrdiff_t)1);
        } else {
            // t==0 时确保 ADVECT 不会使用未绑定内存（建议你的实现里提供 reset）
            // 如果没有 reset，可传空指针并在 pred_advect_bilinear 内部判 ready_uv=0 → 退化到 L3D
            pred_advect_bind_prev_uv(nullptr, nullptr, 0, 0);
        }
        for (int bi = 0; bi < (int)H; bi += (int)BH) {
            const int b_i = bi / (int)BH;
            for (int bj = 0; bj < (int)W; bj += (int)BW) {
                const int b_j = bj / (int)BW;
                const PM pm = (PM)(modes[ frame_mode_off + (size_t)b_i * (size_t)blocks_j + (size_t)b_j ] & 0x3);

                const int i_end = std::min(bi + (int)BH, (int)H);
                const int j_end = std::min(bj + (int)BW, (int)W);
                for (int i = bi; i < i_end; ++i) {
                    for (int j = bj; j < j_end; ++j) {
                        const size_t off = (size_t)t * H * W + (size_t)i * W + (size_t)j;
                        int ebid = *eb_pos++;
                        if (ebid == 0) {
                            // 无损点：直接回填浮点（按编码端顺序消费 unpred 队列）
                            unpred_indices.push_back(off);
                            U_fp[off] = (T)(*(unpred_pos++) * scale);
                            V_fp[off] = (T)(*(unpred_pos++) * scale);
                        } else {
                            T abs_eb = (T)(pow(base, ebid) * (double)threshold);
                            // U 分量
                            {
                                T* cur = U_fp + off;
                                const T pred = predict_dispatch(pm, cur, t, i, j, si, sj, sk);
                                const int qidx = *dq_pos++;
                                *cur = pred + (T)2 * (qidx - intv_radius) * abs_eb;
                            }
                            // V 分量
                            {
                                T* cur = V_fp + off;
                                const T pred = predict_dispatch(pm, cur, t, i, j, si, sj, sk);
                                const int qidx = *dq_pos++;
                                *cur = pred + (T)2 * (qidx - intv_radius) * abs_eb;
                            }
                        }
                    }
                }
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
