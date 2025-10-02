#ifndef _sz_prediction_hpp
#define _sz_prediction_hpp

template<typename T>
inline T
regression_predict_2d(const float * reg_params_pos, int x, int y){
	return reg_params_pos[0] * x + reg_params_pos[1] * y + reg_params_pos[2];
}

template<typename T>
inline T
lorenzo_predict_2d(const T * data_pos, size_t dim0_offset){
	return data_pos[-1] + data_pos[-dim0_offset] - data_pos[-dim0_offset - 1];
}

template<typename T>
inline T
regression_predict_3d(const float * reg_params_pos, int x, int y, int z){
	return reg_params_pos[0] * x + reg_params_pos[1] * y + reg_params_pos[2] * z + reg_params_pos[3];
}

template<typename T>
inline T
lorenzo_predict_3d(const T * data_pos, size_t dim0_offset, size_t dim1_offset){
	return data_pos[-1] + data_pos[-dim1_offset] + data_pos[-dim0_offset] 
	- data_pos[-dim1_offset - 1] - data_pos[-dim0_offset - 1] 
	- data_pos[-dim0_offset - dim1_offset] + data_pos[-dim0_offset - dim1_offset - 1];
}

// ===== Warped-Lorenzo helpers (both encoder & decoder) ======================
namespace warp_pred {

// --- 工具：有界访问（行主序 [i*W + j]）---
template<typename T>
inline double at_d(const T* img, int H, int W, int i, int j){
    if (i < 0) i = 0; else if (i >= H) i = H-1;
    if (j < 0) j = 0; else if (j >= W) j = W-1;
    return (double)img[(size_t)i*W + j];
}

// 3x3 高斯平滑的中心值（double）
template<typename T>
inline double gauss3x3_c(const T* img, int H, int W, int i, int j){
    // 权重: 1 2 1 / 2 4 2 / 1 2 1，归一化除以16
    double s =
        1*at_d(img,H,W,i-1,j-1) + 2*at_d(img,H,W,i-1,j) + 1*at_d(img,H,W,i-1,j+1) +
        2*at_d(img,H,W,i,  j-1) + 4*at_d(img,H,W,i,  j) + 2*at_d(img,H,W,i,  j+1) +
        1*at_d(img,H,W,i+1,j-1) + 2*at_d(img,H,W,i+1,j) + 1*at_d(img,H,W,i+1,j+1);
    return s * (1.0/16.0);
}

inline double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// 双线性采样（img 按 [i*W + j] 存，x->列 j，y->行 i），返回定点值
template<typename T>
inline T bilerp_fixed_clamped(const T* img, int H, int W, double x /*j*/, double y /*i*/) {
    x = clampd(x, 0.0, (double)(W - 1));
    y = clampd(y, 0.0, (double)(H - 1));
    int j0 = (int)std::floor(x), i0 = (int)std::floor(y);
    int j1 = (j0 < W-1) ? j0 + 1 : j0;
    int i1 = (i0 < H-1) ? i0 + 1 : i0;
    double fx = x - j0, fy = y - i0;

    // 固定乘加顺序，配合 llround，跨平台可复现
    double a = (1.0 - fx) * (1.0 - fy) * (double)img[(size_t)i0*W + j0];
    double b = (      fx) * (1.0 - fy) * (double)img[(size_t)i0*W + j1];
    double c = (1.0 - fx) * (      fy) * (double)img[(size_t)i1*W + j0];
    double d = (      fx) * (      fy) * (double)img[(size_t)i1*W + j1];
    return (T) (long long) std::llround(a + b + c + d);
}

// 单分量 Warped‑Lorenzo 预测
// recon_t    : 当前帧(该分量)的已重建平面起始指针
// recon_tm1  : 前一帧(该分量)的已重建平面起始指针
// U_tm1/V_tm1: 前一帧速度场(已重建)平面起始指针（用于位移）
// scale      : 浮点->定点比例
// alpha      : 像素/帧 = alpha * u,v   （若 u,v 单位即像素/帧，alpha=1）
// dmax       : 位移裁剪（像素）
// gate_pix   : 小位移退化门槛（像素）
template<typename T>
inline T predict(const T* recon_t, const T* recon_tm1,
                 const T* U_tm1, const T* V_tm1,
                 int H, int W, int i, int j,
                 T scale, double alpha_x, double alpha_y, double dmax, double gate_pix)
{
    // 当前帧空间邻域（左/上/左上），已重建
    T L  = (j>0) ? recon_t[(size_t)i*W + (j-1)] : (T)0;
    T U  = (i>0) ? recon_t[(size_t)(i-1)*W + j] : (T)0;
    T LU = (i>0 && j>0) ? recon_t[(size_t)(i-1)*W + (j-1)] : (T)0;

    // 位移：取自 t-1 的速度，换回浮点，再乘 alpha 得“像素位移”
    size_t idx = (size_t)i*W + (size_t)j;
	// 3x3 高斯平滑
	// double u = gauss3x3_c(U_tm1, H, W, i, j) / (double)scale;
	// double v = gauss3x3_c(V_tm1, H, W, i, j) / (double)scale;
    double u = (double)U_tm1[idx] / (double)scale;
    double v = (double)V_tm1[idx] / (double)scale;
    double dx = alpha_x * u;  // 列方向（x/j）
    double dy = alpha_y * v;  // 行方向（y/i）

    // 门控与裁剪
    if (dx*dx + dy*dy < gate_pix*gate_pix) { dx = 0.0; dy = 0.0; }
    if (dx > dmax) dx = dmax; else if (dx < -dmax) dx = -dmax;
    if (dy > dmax) dy = dmax; else if (dy < -dmax) dy = -dmax;

    // 反向映射到 t-1 采样
    double x = (double)j - dx;
    double y = (double)i - dy;

    T  w   = bilerp_fixed_clamped<T>(recon_tm1, H, W, x,       y);
    T  wx  = bilerp_fixed_clamped<T>(recon_tm1, H, W, x-1.0,   y);
    T  wy  = bilerp_fixed_clamped<T>(recon_tm1, H, W, x,       y-1.0);
    T  wxy = bilerp_fixed_clamped<T>(recon_tm1, H, W, x-1.0,   y-1.0);

    if (i==0 && j==0) return w;
    if (i==0 && j>0)  return L  + w - wx;
    if (i>0  && j==0) return U  + w - wy;
    return L + U + w - LU - wx - wy + wxy;
}

} // namespace warp_pred


// =====MoPred helpers (both encoder & decoder)==============================

enum class PM { L3D, L3D_S2, TLE_2DL };

// ---- 3D Lorenzo（时间最慢）----
inline int64_t pred_l3d(const int64_t* cur, int t,int i,int j,
                        ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk)
{
    int64_t d0 = (t&&i&&j)? cur[-sk - si - sj] : 0;
    int64_t d1 = (t&&i)   ? cur[-sk - si]      : 0;
    int64_t d2 = (t&&j)   ? cur[-sk - sj]      : 0;
    int64_t d3 = (t)      ? cur[-sk]           : 0;
    int64_t d4 = (i&&j)   ? cur[-si - sj]      : 0;
    int64_t d5 = (i)      ? cur[-si]           : 0;
    int64_t d6 = (j)      ? cur[-sj]           : 0;
    return d0 + d3 + d5 + d6 - d1 - d2 - d4;
}

// ---- stride=2 的 3D Lorenzo（应对交替/半步错相）----
inline int64_t pred_l3d_stride2(const int64_t* cur, int t,int i,int j,
                                ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk)
{
    if (t < 2) return pred_l3d(cur,t,i,j,si,sj,sk);
    int64_t d0 = (t>=2 && i && j) ? cur[-2*sk - si - sj] : 0;
    int64_t d1 = (t>=2 && i)      ? cur[-2*sk - si]      : 0;
    int64_t d2 = (t>=2 && j)      ? cur[-2*sk - sj]      : 0;
    int64_t d3 = (t>=2)           ? cur[-2*sk]           : 0;
    int64_t d4 = (i && j)         ? cur[-si - sj]        : 0;
    int64_t d5 = (i)              ? cur[-si]             : 0;
    int64_t d6 = (j)              ? cur[-sj]             : 0;
    return d0 + d3 + d5 + d6 - d1 - d2 - d4;
}

// ---- 时间线性外推 + 2D Lorenzo 残差（TLE_2DL）----
inline int64_t pred_time_lin_plus_2d_lorenzo(
    const int64_t* cur, int t,int i,int j,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk)
{
    if (t < 2) return pred_l3d(cur,t,i,j,si,sj,sk);

    // base: 时间线性外推
    const int64_t base = 2*cur[-sk] - cur[-2*sk];

    // 在 residual 平面做 2D Lorenzo
    int64_t rim1j=0, rijm1=0, rim1jm1=0;
    if (i){
        const int64_t base_im1j = 2*cur[-sk - si] - cur[-2*sk - si];
        rim1j = cur[-si] - base_im1j;
    }
    if (j){
        const int64_t base_ijm1 = 2*cur[-sk - sj] - cur[-2*sk - sj];
        rijm1 = cur[-sj] - base_ijm1;
    }
    if (i && j){
        const int64_t base_im1jm1 = 2*cur[-sk - si - sj] - cur[-2*sk - si - sj];
        rim1jm1 = cur[-si - sj] - base_im1jm1;
    }
    const int64_t r_pred = rim1j + rijm1 - rim1jm1;
    return base + r_pred;
}

// ---- 总调度 ----
inline int64_t predict_dispatch(PM pm, const int64_t* cur, int t,int i,int j,
                                ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk)
{
    switch (pm){
        case PM::L3D:     return pred_l3d(cur,t,i,j,si,sj,sk);
        case PM::L3D_S2:  return pred_l3d_stride2(cur,t,i,j,si,sj,sk);
        case PM::TLE_2DL: return pred_time_lin_plus_2d_lorenzo(cur,t,i,j,si,sj,sk);
    }
    return pred_l3d(cur,t,i,j,si,sj,sk);
}

// ====== 分块选择器：只用历史已解码帧（t-1,t-2,t-3），零副信息 ======
// 说明：对块 (bi:bi+BH-1, bj:bj+BW-1) 内抽样；统计：
//   S1 = 一阶时间差 L1 ；S2 = 二阶时间差 L1（线性偏离）；ALT/SMO = 交替指标
// 规则：
//   1) 若 S2 <= tau_lin * S1  → TLE_2DL （线性外推更优）
//   2) 否则若 ALT/SMO < tau_alt → L3D_S2
//   3) 否则 → L3D
#ifndef SZ_MOP_BLOCK_BH
#define SZ_MOP_BLOCK_BH 64
#endif
#ifndef SZ_MOP_BLOCK_BW
#define SZ_MOP_BLOCK_BW 64
#endif
#ifndef SZ_MOP_SUBS
#define SZ_MOP_SUBS 16   // 抽样步长
#endif

template<typename T>
inline PM choose_pred_mode_block(const T* U_fp, const T* V_fp,
                                 int H,int W,int t,
                                 int bi,int bj,int BH,int BW,
                                 ptrdiff_t sk, int subs=SZ_MOP_SUBS,
                                 long double tau_lin=0.20L,
                                 long double tau_alt=0.50L)
{
    if (t < 2) return PM::L3D;

    const int i_lo = bi, i_hi = std::min(bi+BH, H);
    const int j_lo = bj, j_hi = std::min(bj+BW, W);

    const T* Ut1 = U_fp + (ptrdiff_t)(t-1)*sk;
    const T* Ut2 = U_fp + (ptrdiff_t)(t-2)*sk;
    const T* Vt1 = V_fp + (ptrdiff_t)(t-1)*sk;
    const T* Vt2 = V_fp + (ptrdiff_t)(t-2)*sk;

    const bool has_t3 = (t >= 3);
    const T* Ut3 = has_t3 ? (U_fp + (ptrdiff_t)(t-3)*sk) : nullptr;
    const T* Vt3 = has_t3 ? (V_fp + (ptrdiff_t)(t-3)*sk) : nullptr;

    long double S1=0.0L, S2=0.0L, ALT=0.0L, SMO=0.0L;
    size_t cnt = 0;

    for (int i=i_lo; i<i_hi; i+=subs){
        const int row = i*W;
        for (int j=j_lo; j<j_hi; j+=subs){
            const int idx = row + j;
            long long du1 = (long long)Ut1[idx] - (long long)Ut2[idx];
            long long dv1 = (long long)Vt1[idx] - (long long)Vt2[idx];
            S1 += std::llabs(du1) + std::llabs(dv1);

            if (has_t3){
                long long du2 = (long long)Ut1[idx] - 2LL*(long long)Ut2[idx] + (long long)Ut3[idx];
                long long dv2 = (long long)Vt1[idx] - 2LL*(long long)Vt2[idx] + (long long)Vt3[idx];
                S2 += std::llabs(du2) + std::llabs(dv2);

                long long du_b = (long long)Ut2[idx] - (long long)Ut3[idx];
                long long dv_b = (long long)Vt2[idx] - (long long)Vt3[idx];
                ALT += std::llabs(du1 + du_b) + std::llabs(dv1 + dv_b);
                SMO += std::llabs(du1 - du_b) + std::llabs(dv1 - dv_b);
            }
            ++cnt;
        }
    }
    if (cnt==0) return PM::L3D;
    S1 /= (long double)cnt;
    S2 = has_t3 ? (S2/(long double)cnt) : S1; // 没 t-3 时不触发线性外推

    // 规则 1：线性外推检测
    if (has_t3 && S2 <= tau_lin * (S1 + 1e-9L))
        return PM::TLE_2DL;

    // 规则 2：交替性检测（仅当有 t-3）
    if (has_t3){
        long double alt_score = ALT / (SMO + 1e-9L);
        if (alt_score < tau_alt)
            return PM::L3D_S2;
    }

    return PM::L3D;
}


#endif