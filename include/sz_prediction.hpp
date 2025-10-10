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

// ---------------------------------------------
// 小工具：C(n,k)（我们只用到 n<=3，k<=3 的很小范围）
inline int binom_small(int n, int k){
    static const int C[4][4] = {
        {1,0,0,0},
        {1,1,0,0},
        {1,2,1,0},
        {1,3,3,1}
    };
    return C[n][k];
}

// 统一索引步长：sk=t 步长, si=i 步长, sj=j 步长
// 已在你的代码里：sk = H*W, si=W, sj=1

// ---------------------------------------------
// 混合阶 3D Lorenzo 预测器（T=int64_t 定点值）
// kt,ki,kj 为各维阶数；边界自动降阶（用 min(kt,t) 等）
template<typename T>
inline T lorenzo3d_predict_mixed_order(
    const T* cur, int t, int i, int j,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk,
    int kt, int ki, int kj)
{
    const int at = std::min(kt, t);
    const int ai = std::min(ki, i);
    const int aj = std::min(kj, j);

    T sum = (T)0;
    for (int a=0; a<=at; ++a){
        for (int b=0; b<=ai; ++b){
            for (int c=0; c<=aj; ++c){
                if (a==0 && b==0 && c==0) continue; // 不用自身
                // (-1)^(a+b+c+1) 的符号：奇数为 +1，偶数为 -1
                const int sign = ((a+b+c)&1) ? +1 : -1;
                const int coef = binom_small(kt,a) * binom_small(ki,b) * binom_small(kj,c);
                const T  val  = *(cur - a*sk - b*si - c*sj);
                sum += (sign>0) ? (T)( (long long)coef * val )
                                : (T)(- (long long)coef * val );
            }
        }
    }
    return sum;
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

// 线性索引（避免依赖 vid/Size3）
inline size_t idx3(int t,int i,int j, int H,int W){ return (size_t)t*(size_t)H*(size_t)W + (size_t)i*(size_t)W + (size_t)j; }

inline bool inR(int x, int lo, int hi_exclusive){ return (x>=lo) && (x<hi_exclusive); }

// 近似“码长代价”——我们用 qd 作为 proxy（越小越好），可选用 log2(qd) 更接近 Huffman。
// 这里给一个可切换的实现：用 qd 或 log2(qd)
inline long double q_cost_from_qd(int64_t qd, bool use_log=false){
    if (qd <= 0) return 0.0L;
    if (!use_log) return (long double)qd;                 // L1 级别
    return (long double)std::log2((long double)qd);       // 更接近熵
}

// ---------- 预测器枚举 ----------
// enum class PM : uint8_t { L3D=0, L3D_S2=1, TLE_2DL=2, TAR2C=3 };
// enum class PM : uint8_t { L3D=0, L3D_S2=1, DTL_AR1=2, TAR2C=3 };
enum class PM : uint8_t { L3D=0, ADVECT=1, DTL_AR1=2, TAR2C=3 };
inline std::ostream& operator<<(std::ostream& os, PM pm) {
    switch (pm) {
        case PM::L3D:      return os << "L3D";
        case PM::ADVECT:   return os << "ADVECT";
        case PM::DTL_AR1:  return os << "DTL_AR1";
        case PM::TAR2C:    return os << "TAR2C";
        default:           return os << "Unknown";
    }
}

// MODE 0:---------- 3D Lorenzo（时间最慢）----------
inline int64_t pred_l3d(const int64_t* cur, int t,int i,int j,
                        ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk){
    int64_t d0 = (t&&i&&j)? cur[-sk - si - sj] : 0;
    int64_t d1 = (t&&i)   ? cur[-sk - si]      : 0;
    int64_t d2 = (t&&j)   ? cur[-sk - sj]      : 0;
    int64_t d3 = (t)      ? cur[-sk]           : 0;
    int64_t d4 = (i&&j)   ? cur[-si - sj]      : 0;
    int64_t d5 = (i)      ? cur[-si]           : 0;
    int64_t d6 = (j)      ? cur[-sj]           : 0;
    return d0 + d3 + d5 + d6 - d1 - d2 - d4;
}


// ---------- 时间线性外推 + 2D Lorenzo 残差 ----------
inline int64_t pred_time_lin_plus_2d_lorenzo( //this predictor is not common to pick in dataset tested
    const int64_t* cur, int t,int i,int j,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk){
    if (t < 2) return pred_l3d(cur,t,i,j,si,sj,sk);
    const int64_t base = 2*cur[-sk] - cur[-2*sk];
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


// MODE 1:---------- stride=2 的 3D Lorenzo ----------
inline int64_t pred_l3d_stride2(const int64_t* cur, int t,int i,int j,
                                ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk){
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

// MODE 2:---------- DTL_AR1：时间差分 + 2D Lorenzo + AR(1) ----------

// inline int64_t pred_dt2d_ar1(const int64_t* cur, int t,int i,int j,
//                              ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk)
// {
//     // 没有充足的时间历史时，退化为 L3D，避免边界条件分叉
//     if (t < 1) return pred_l3d(cur,t,i,j,si,sj,sk);
//     // 基准值：上一帧同点（x(t-1)）
//     const int64_t x_im1   = cur[-sk];
//     // AR(1) 的时间差分项 Δx(t-1) = x(t-1) - x(t-2)
//     const int64_t dt_prev = (t >= 2) ? (cur[-sk] - cur[-2*sk]) : 0;
//     // 当前帧在“左/上/左上”的时间差分（需要同位置的上一帧，已存在）
//     int64_t dt_l = 0, dt_u = 0, dt_ul = 0;
//     if (j)         dt_l  = cur[-sj]        - cur[-sk - sj];
//     if (i)         dt_u  = cur[-si]        - cur[-sk - si];
//     if (i && j)    dt_ul = cur[-si - sj]   - cur[-sk - si - sj];
//     // 对 Δx 做 2D Lorenzo
//     const int64_t dt_lor = dt_l + dt_u - dt_ul;
//     // Δx 的预测：2D Lorenzo + AR(1)
//     const int64_t dt_pred = dt_lor + dt_prev;
//     // x 的预测
//     return x_im1 + dt_pred;
// }

static inline int64_t pred_dt2d_ar1(const int64_t* cur, int t,int i,int j,
                                         ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk)
{
    if (t < 1) return pred_l3d(cur,t,i,j,si,sj,sk);

    // Δx(t,·) = x(t,·) - x(t-1,·) 在同一时间 t 的邻域做 2D Lorenzo
    int64_t dt_l  = 0, dt_u  = 0, dt_ul = 0;
    if (j)       dt_l  = cur[-sj]      - cur[-sk - sj];
    if (i)       dt_u  = cur[-si]      - cur[-sk - si];
    if (i && j)  dt_ul = cur[-si - sj] - cur[-sk - si - sj];

    const int64_t dt_pred = dt_l + dt_u - dt_ul;
    return cur[-sk] + dt_pred;  // x(t-1) + 预测的 Δx(t)
}

// MODE 3:---------- 时间 AR(2) + 护栏 ----------
inline int64_t pred_time_ar2_clamped(
    const int64_t* cur, int t,int i,int j,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk)
{
    if (t < 2) return pred_l3d(cur,t,i,j,si,sj,sk);

    // 纯时间 AR(2)
    int64_t base = 2*cur[-sk] - cur[-2*sk];

    // 在 t-1 & t-2 的 2x2 局部邻域做 min/max 作为护栏
    int64_t lo = base, hi = base;

    auto upd = [&](int dt, ptrdiff_t off){
        int64_t v = cur[-dt*sk + off];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    };

    // 中心点
    upd(1, 0); upd(2, 0);
    // 左、上、左上（若存在）
    if (i) { upd(1, -si); upd(2, -si); }
    if (j) { upd(1, -sj); upd(2, -sj); }
    if (i && j) { upd(1, -si - sj); upd(2, -si - sj); }

    if (base < lo) base = lo;
    else if (base > hi) base = hi;
    return base;
}

//backup1: 半拉格朗日（运动补偿）预测：pred_advect_bilinear

#if defined(__cplusplus)
static thread_local
#else
// C11: _Thread_local；若不支持，退化为 static（非线程安全）
static
#endif
struct {
    // 上一帧 U/V 的“平面基址”（即 t-1 帧的 (i=0,j=0) 处指针）
    const int64_t* u_prev_plane;
    const int64_t* v_prev_plane;
    // 其在内存中的行/列步长（元素为单位，与被压字段可能相同）
    ptrdiff_t si_uv, sj_uv;

    // 网格行列数（H=rows=Y，W=cols=X），用于边界裁剪
    int H, W;

    // 将物理速度转换为“栅格/帧”的比例（CFL）：Δt/Δx, Δt/Δy
    double dt_over_dx, dt_over_dy;

    // 单次回溯的最大位移（格）：避免极端速度导致过大外推；<=0 表示不限制
    double max_disp;

    int64_t scale;

    // 是否已配置
    int ready;
} g_mcctx = {0};

// 在进入一帧（或一个块）前绑定 U/V 的 t-1 平面与其步长
static inline void pred_advect_bind_prev_uv(const int64_t* u_prev_plane,
                                            const int64_t* v_prev_plane,
                                            ptrdiff_t si_uv, ptrdiff_t sj_uv)
{
    g_mcctx.u_prev_plane = u_prev_plane;
    g_mcctx.v_prev_plane = v_prev_plane;
    g_mcctx.si_uv = si_uv;
    g_mcctx.sj_uv = sj_uv;
    g_mcctx.ready = 1;
}

// 设置网格尺寸与 dt/dx, dt/dy，以及可选的位移限幅
static inline void pred_advect_set_params(int H, int W,
                                          double dt_over_dx, double dt_over_dy, int64_t scale,
                                          double max_disp /* e.g. 2.0；<=0 关闭 */)
{
    g_mcctx.H = H;
    g_mcctx.W = W;
    g_mcctx.dt_over_dx = dt_over_dx;
    g_mcctx.dt_over_dy = dt_over_dy;
    g_mcctx.max_disp = max_disp;
    g_mcctx.scale = scale;
}
#if 0
// 主函数：半拉格朗日预测（双线性）
static inline int64_t pred_advect_bilinear(
    const int64_t* cur, int t,int i,int j,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk)
{
    // 冷启动/越界：回退
    if (t < 1 || !g_mcctx.ready || g_mcctx.H<=0 || g_mcctx.W<=0) {
        return pred_l3d(cur,t,i,j,si,sj,sk);
    }

    // 1) 取 t-1 帧的速度（定点）
    const int64_t u_fp = g_mcctx.u_prev_plane[(ptrdiff_t)i*g_mcctx.si_uv + (ptrdiff_t)j*g_mcctx.sj_uv];
    const int64_t v_fp = g_mcctx.v_prev_plane[(ptrdiff_t)i*g_mcctx.si_uv + (ptrdiff_t)j*g_mcctx.sj_uv];

    // 2) 定点 → 物理速度（或网格/秒），再换算像素位移
    //    位移(像素) = (u_fp / scale) * (Δt/Δx)
    //               = u_fp * ( (Δt/Δx) / scale )
    const double invS = 1.0 / (double)g_mcctx.scale; 
    const double disp_j = (double)u_fp * (g_mcctx.dt_over_dx * invS); // 列方向像素位移
    const double disp_i = (double)v_fp * (g_mcctx.dt_over_dy * invS); // 行方向像素位移

    // 3) 反向回溯到 t-1 的连续坐标
    double jstar = (double)j - disp_j;
    double istar = (double)i - disp_i;

    // 位移限幅（可选，单位像素）
    if (g_mcctx.max_disp > 0.0) {
        const double jl = (double)j - g_mcctx.max_disp, jh = (double)j + g_mcctx.max_disp;
        const double il = (double)i - g_mcctx.max_disp, ih = (double)i + g_mcctx.max_disp;
        if (jstar < jl) jstar = jl; else if (jstar > jh) jstar = jh;
        if (istar < il) istar = il; else if (istar > ih) istar = ih;
    }

    // 4) clamp 到图像范围
    if (jstar < 0.0) jstar = 0.0; else if (jstar > (double)(g_mcctx.W-1)) jstar = (double)(g_mcctx.W-1);
    if (istar < 0.0) istar = 0.0; else if (istar > (double)(g_mcctx.H-1)) istar = (double)(g_mcctx.H-1);

    // 5) 双线性插值索引与权重
    const int j0 = (int)floor(jstar);
    const int i0 = (int)floor(istar);
    const int j1 = (j0 + 1 < g_mcctx.W) ? j0 + 1 : j0;
    const int i1 = (i0 + 1 < g_mcctx.H) ? i0 + 1 : i0;

    const double ax = jstar - (double)j0;
    const double ay = istar - (double)i0;

    // 6) 读取 t-1 平面四邻域并双线性
    const int64_t* prev_plane = cur - sk - (ptrdiff_t)i*si - (ptrdiff_t)j*sj;

    const int64_t p00 = prev_plane[(ptrdiff_t)i0*si + (ptrdiff_t)j0*sj];
    const int64_t p10 = prev_plane[(ptrdiff_t)i0*si + (ptrdiff_t)j1*sj];
    const int64_t p01 = prev_plane[(ptrdiff_t)i1*si + (ptrdiff_t)j0*sj];
    const int64_t p11 = prev_plane[(ptrdiff_t)i1*si + (ptrdiff_t)j1*sj];

    const double w00 = (1.0-ax)*(1.0-ay);
    const double w10 = (     ax)*(1.0-ay);
    const double w01 = (1.0-ax)*(     ay);
    const double w11 = (     ax)*(     ay);

    const double val = w00*(double)p00 + w10*(double)p10
                     + w01*(double)p01 + w11*(double)p11;

    // 7) 与解码端一致的舍入
    return (int64_t)llround(val);
}

#endif
// ---------Rk2+自适应子步长的半拉格朗日预测---------

static inline double bilinear_sample_fp(
    const int64_t* plane, ptrdiff_t si, ptrdiff_t sj,
    int H, int W, double i_f, double j_f)
{
    if (j_f < 0.0) j_f = 0.0; else if (j_f > (double)(W-1)) j_f = (double)(W-1);
    if (i_f < 0.0) i_f = 0.0; else if (i_f > (double)(H-1)) i_f = (double)(H-1);

    const int j0 = (int)floor(j_f);
    const int i0 = (int)floor(i_f);
    const int j1 = (j0 + 1 < W) ? j0 + 1 : j0;
    const int i1 = (i0 + 1 < H) ? i0 + 1 : i0;

    const double ax = j_f - (double)j0;
    const double ay = i_f - (double)i0;

    const int64_t p00 = plane[(ptrdiff_t)i0*si + (ptrdiff_t)j0*sj];
    const int64_t p10 = plane[(ptrdiff_t)i0*si + (ptrdiff_t)j1*sj];
    const int64_t p01 = plane[(ptrdiff_t)i1*si + (ptrdiff_t)j0*sj];
    const int64_t p11 = plane[(ptrdiff_t)i1*si + (ptrdiff_t)j1*sj];

    const double w00 = (1.0-ax)*(1.0-ay);
    const double w10 =      ax *(1.0-ay);
    const double w01 = (1.0-ax)*     ay ;
    const double w11 =      ax *     ay ;

    return w00*(double)p00 + w10*(double)p10 + w01*(double)p01 + w11*(double)p11;
}

// —— 主函数：半拉格朗日预测（RK2 + 自适应子步，无需 max_disp）——
static inline int64_t pred_advect_bilinear(
    const int64_t* cur, int t,int i,int j,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk)
{
    if (t < 1 || !g_mcctx.ready || g_mcctx.H<=0 || g_mcctx.W<=0) {
        return pred_l3d(cur,t,i,j,si,sj,sk);
    }

    // CFL 系数（定点域）：cflx = (Δt/Δx) / scale, cfly = (Δt/Δy) / scale
    const double invS = 1.0 / (double)g_mcctx.scale;
    const double cflx  = g_mcctx.dt_over_dx * invS;
    const double cfly  = g_mcctx.dt_over_dy * invS;

    // 初始位置
    const double i0 = (double)i, j0 = (double)j;

    // 用“格点速度”先估计一把总位移大小，决定是否子步
    const int64_t u_fp0 = g_mcctx.u_prev_plane[(ptrdiff_t)i*g_mcctx.si_uv + (ptrdiff_t)j*g_mcctx.sj_uv];
    const int64_t v_fp0 = g_mcctx.v_prev_plane[(ptrdiff_t)i*g_mcctx.si_uv + (ptrdiff_t)j*g_mcctx.sj_uv];
    const double disp_j_est = (double)u_fp0 * cflx;
    const double disp_i_est = (double)v_fp0 * cfly;
    const double disp_inf   = fmax(fabs(disp_j_est), fabs(disp_i_est));

    // 阈值与上限：每小步 ≤ dmax 像素，最多 N_MAX 小步
    const double dmax = 2.0;     // 每小步最大位移（像素）
    const int    N_MAX = 32;     // 子步上限（过大可考虑回退 L3D）

    if (disp_inf <= dmax) {
        // —— 小位移：用 RK2（mid-point）提高精度，成本几乎不变 ——
        // 半步位置
        const double uh0 = bilinear_sample_fp(g_mcctx.u_prev_plane, g_mcctx.si_uv, g_mcctx.sj_uv,
                                              g_mcctx.H, g_mcctx.W, i0, j0);
        const double vh0 = bilinear_sample_fp(g_mcctx.v_prev_plane, g_mcctx.si_uv, g_mcctx.sj_uv,
                                              g_mcctx.H, g_mcctx.W, i0, j0);
        double jh = j0 - 0.5 * uh0 * cflx;
        double ih = i0 - 0.5 * vh0 * cfly;

        // 半步速度
        const double uh = bilinear_sample_fp(g_mcctx.u_prev_plane, g_mcctx.si_uv, g_mcctx.sj_uv,
                                             g_mcctx.H, g_mcctx.W, ih, jh);
        const double vh = bilinear_sample_fp(g_mcctx.v_prev_plane, g_mcctx.si_uv, g_mcctx.sj_uv,
                                             g_mcctx.H, g_mcctx.W, ih, jh);

        // 终点
        double jstar = j0 - uh * cflx;
        double istar = i0 - vh * cfly;

        // clamp 到图像范围
        if (jstar < 0.0) jstar = 0.0; else if (jstar > (double)(g_mcctx.W-1)) jstar = (double)(g_mcctx.W-1);
        if (istar < 0.0) istar = 0.0; else if (istar > (double)(g_mcctx.H-1)) istar = (double)(g_mcctx.H-1);

        // 从 t-1 的目标分量平面采样
        const int64_t* prev_plane = cur - sk - (ptrdiff_t)i*si - (ptrdiff_t)j*sj;
        const double val = bilinear_sample_fp(prev_plane, si, sj, g_mcctx.H, g_mcctx.W, istar, jstar);
        return (int64_t)llround(val);
    } else {
        // —— 大位移：自适应子步，每步 ≤ dmax px；每步用局部速度回溯 ——
        int N = (int)ceil(disp_inf / dmax);
        if (N > N_MAX) {
            // 过大：你可以选择回退 L3D，或把 N clamp 到 N_MAX
            N = N_MAX; // 或：return pred_l3d(cur,t,i,j,si,sj,sk);
        }
        const double cflx_step = cflx / (double)N;
        const double cfly_step = cfly / (double)N;

        double ip = i0, jp = j0; // 累积位置
        for (int s=0; s<N; ++s) {
            const double u = bilinear_sample_fp(g_mcctx.u_prev_plane, g_mcctx.si_uv, g_mcctx.sj_uv,
                                                g_mcctx.H, g_mcctx.W, ip, jp);
            const double v = bilinear_sample_fp(g_mcctx.v_prev_plane, g_mcctx.si_uv, g_mcctx.sj_uv,
                                                g_mcctx.H, g_mcctx.W, ip, jp);
            jp -= u * cflx_step;
            ip -= v * cfly_step;
            // 每小步 clamp，避免数值漂移跑出域
            if (jp < 0.0) jp = 0.0; else if (jp > (double)(g_mcctx.W-1)) jp = (double)(g_mcctx.W-1);
            if (ip < 0.0) ip = 0.0; else if (ip > (double)(g_mcctx.H-1)) ip = (double)(g_mcctx.H-1);
        }

        const int64_t* prev_plane = cur - sk - (ptrdiff_t)i*si - (ptrdiff_t)j*sj;
        const double val = bilinear_sample_fp(prev_plane, si, sj, g_mcctx.H, g_mcctx.W, ip, jp);
        return (int64_t)llround(val);
    }
}

inline int64_t predict_dispatch(PM pm, const int64_t* cur, int t,int i,int j,
                                ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk){
    switch (pm){
        case PM::L3D:     return pred_l3d(cur,t,i,j,si,sj,sk); //mode 0
        // case PM::L3D_S2:  return pred_l3d_stride2(cur,t,i,j,si,sj,sk); //mode 1
        case PM::ADVECT:  return pred_advect_bilinear(cur,t,i,j,si,sj,sk); //mode 1
        // case PM::TLE_2DL: return pred_time_lin_plus_2d_lorenzo(cur,t,i,j,si,sj,sk);
        case PM::DTL_AR1: return pred_dt2d_ar1(cur,t,i,j,si,sj,sk); //mode 2
        case PM::TAR2C:   return pred_time_ar2_clamped(cur,t,i,j,si,sj,sk); //mode 3
    }
    return pred_l3d(cur,t,i,j,si,sj,sk);
}

// ---------- 2bit 模式打包/解包 ----------
inline void pack_modes_2bit(const std::vector<uint8_t>& modes, std::vector<uint8_t>& out_bytes){
    out_bytes.clear(); out_bytes.reserve((modes.size()+3)/4);
    uint8_t acc=0; int fill=0;
    for (uint8_t m : modes){
        acc |= (m & 0x3u) << fill;
        fill += 2;
        if (fill == 8){ out_bytes.push_back(acc); acc=0; fill=0; }
    }
    if (fill) out_bytes.push_back(acc);
}
inline void unpack_modes_2bit(const uint8_t* in_bytes, size_t n_modes, std::vector<uint8_t>& modes){
    modes.resize(n_modes);
    size_t o=0; size_t i=0;
    while (o < n_modes){
        uint8_t b = in_bytes[i++];
        for (int k=0; k<4 && o<n_modes; ++k){
            modes[o++] = (b >> (2*k)) & 0x3u;
        }
    }
}

inline void print_mode_counts(const std::vector<uint8_t>& modes) {
    std::vector<size_t> counts(4, 0);
    for (uint8_t m : modes) {
        if (m < 4) {
            counts[m]++;
        } else {
            std::cerr << "Warning: invalid mode value " << static_cast<int>(m) << std::endl;
        }
    }

    for (size_t i = 0; i < 4; ++i) {
        std::cout << "Mode " << static_cast<PM>(i) << ": " << counts[i] << " times\n";
    }
}
// ---------- 编码端“前视抽样打分”选择器（需要当前帧原始定点值） ----------
#ifndef SZ_MOP_BH
#define SZ_MOP_BH 32
#endif
#ifndef SZ_MOP_BW
#define SZ_MOP_BW 32
#endif
#ifndef SZ_MOP_EVAL_SUBS
#define SZ_MOP_EVAL_SUBS 1  // 抽样步长,越小越能代表总体
#endif
#ifndef SZ_MOP_EVAL_GUARD
#define SZ_MOP_EVAL_GUARD 1  // 只在块的顶行+左列抽样，避免“尚未回写”的邻居偏差
#endif
#ifndef SZ_MOP_REL_IMPROVE_MIN
#define SZ_MOP_REL_IMPROVE_MIN 0.003L   // 0.003L,0.3% 改进才切换
#endif

// template<typename T>
// inline PM choose_pred_mode_block_lookahead(
//     const T* U_fp, const T* V_fp,
//     int H,int W,int t, int bi,int bj,int BH,int BW,
//     ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk,
//     int subs=SZ_MOP_EVAL_SUBS)
// {
//     if (t < 2) return PM::L3D;
//     const int i_end = std::min(bi+BH, H);
//     const int j_end = std::min(bj+BW, W);
//     long double score[3] = {0.0L,0.0L,0.0L};
//     size_t cnt = 0;
//     // 仅抽样“顶行 + 左列”的栅格点，保证 2D 邻居均已解码（块级光栅顺序）
//     for (int j=bj; j<j_end; j+=subs){
//         int i = bi;
//         size_t v = (size_t)t*H*W + (size_t)i*W + (size_t)j;
//         const T *cu = U_fp + v, *cv = V_fp + v;
//         int64_t p0 = pred_l3d(cu,t,i,j,si,sj,sk);
//         int64_t p1 = pred_l3d_stride2(cu,t,i,j,si,sj,sk);
//         int64_t p2 = pred_time_lin_plus_2d_lorenzo(cu,t,i,j,si,sj,sk);
//         score[0] += std::llabs((long long)(*cu - p0));
//         score[1] += std::llabs((long long)(*cu - p1));
//         score[2] += std::llabs((long long)(*cu - p2));
//         p0 = pred_l3d(cv,t,i,j,si,sj,sk);
//         p1 = pred_l3d_stride2(cv,t,i,j,si,sj,sk);
//         p2 = pred_time_lin_plus_2d_lorenzo(cv,t,i,j,si,sj,sk);
//         score[0] += std::llabs((long long)(*cv - p0));
//         score[1] += std::llabs((long long)(*cv - p1));
//         score[2] += std::llabs((long long)(*cv - p2));
//         ++cnt;
//     }
//     for (int i=bi+SZ_MOP_EVAL_GUARD; i<i_end; i+=subs){
//         int j = bj;
//         size_t v = (size_t)t*H*W + (size_t)i*W + (size_t)j;
//         const T *cu = U_fp + v, *cv = V_fp + v;
//         int64_t p0 = pred_l3d(cu,t,i,j,si,sj,sk);
//         int64_t p1 = pred_l3d_stride2(cu,t,i,j,si,sj,sk);
//         int64_t p2 = pred_time_lin_plus_2d_lorenzo(cu,t,i,j,si,sj,sk);
//         score[0] += std::llabs((long long)(*cu - p0));
//         score[1] += std::llabs((long long)(*cu - p1));
//         score[2] += std::llabs((long long)(*cu - p2));
//         p0 = pred_l3d(cv,t,i,j,si,sj,sk);
//         p1 = pred_l3d_stride2(cv,t,i,j,si,sj,sk);
//         p2 = pred_time_lin_plus_2d_lorenzo(cv,t,i,j,si,sj,sk);
//         score[0] += std::llabs((long long)(*cv - p0));
//         score[1] += std::llabs((long long)(*cv - p1));
//         score[2] += std::llabs((long long)(*cv - p2));
//         ++cnt;
//     }
//     if (cnt==0) return PM::L3D; // 极小块退化
//     int best = 0; long double bestv = score[0];
//     for (int k=1; k<3; ++k){ if (score[k] < bestv){ best=k; bestv=score[k]; } }
//     // return (best==0)? PM::L3D : (best==1? PM::L3D_S2 : PM::TLE_2DL);
//     return (best==0)? PM::L3D : (best==1? PM::L3D_S2 : PM::DTL_AR1);
// }

#if 0
template<typename T>
inline PM choose_pred_mode_block_lookahead_q(
    const T* U_fp, const T* V_fp,
    int H,int W,int t, int bi,int bj,int BH,int BW,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk,
    T eb_lsb_cap,              // 传入 max((T)1, max_eb)
    int subs=SZ_MOP_EVAL_SUBS)
{
    if (t < 1) return PM::L3D;  // 第一帧没有时间差分，直接用 L3D

    const int i_end = std::min(bi+BH, H);
    const int j_end = std::min(bj+BW, W);
    if (eb_lsb_cap <= 0) eb_lsb_cap = 1;

    long double score[3] = {0.0L,0.0L,0.0L}; // L3D, L3D_S2, DTL_AR1
    size_t cnt = 0;

    auto q_cost = [&](long long diff)-> long double {
        long long q = (std::llabs(diff) / (long long)eb_lsb_cap) + 1;
        if (q > 32767) q = 32767; // 避免极端值影响
        return (long double)q;
    };
        
    // --- 绑定 t-1 帧的 U/V 平面，供 ADVECT 使用 ---
    const int64_t* u_prev_plane = (const int64_t*)(U_fp + (size_t)(t-1) * (size_t)H * (size_t)W);
    const int64_t* v_prev_plane = (const int64_t*)(V_fp + (size_t)(t-1) * (size_t)H * (size_t)W);
    pred_advect_bind_prev_uv(u_prev_plane, v_prev_plane, si, sj);

    // 顶行
    for (int j=bj; j<j_end; j+=subs){
        int i = bi;
        const size_t v = (size_t)t*H*W + (size_t)i*W + (size_t)j;
        const T *cu = U_fp + v, *cv = V_fp + v;

        int64_t p0 = pred_l3d(cu,t,i,j,si,sj,sk);
        // int64_t p1 = pred_l3d_stride2(cu,t,i,j,si,sj,sk);
        int64_t p1 = pred_advect_bilinear(cu,t,i,j,si,sj,sk);
        int64_t p2 = pred_dt2d_ar1(cu,t,i,j,si,sj,sk);
        int64_t p3 = pred_time_ar2_clamped(cu,t,i,j,si,sj,sk);
        score[0] += q_cost((long long)(*cu - p0));
        score[1] += q_cost((long long)(*cu - p1));
        score[2] += q_cost((long long)(*cu - p2));
        score[3] += q_cost((long long)(*cu - p3));

        p0 = pred_l3d(cv,t,i,j,si,sj,sk);
        // p1 = pred_l3d_stride2(cv,t,i,j,si,sj,sk);
        p1 = pred_advect_bilinear(cv,t,i,j,si,sj,sk);
        p2 = pred_dt2d_ar1(cv,t,i,j,si,sj,sk);
        p3 = pred_time_ar2_clamped(cv,t,i,j,si,sj,sk);
        score[0] += q_cost((long long)(*cv - p0));
        score[1] += q_cost((long long)(*cv - p1));
        score[2] += q_cost((long long)(*cv - p2));
        score[3] += q_cost((long long)(*cv - p3));
        ++cnt;
    }
    // 左列（避开顶行第一个交点）
    for (int i=bi+SZ_MOP_EVAL_GUARD; i<i_end; i+=subs){
        int j = bj;
        const size_t v = (size_t)t*H*W + (size_t)i*W + (size_t)j;
        const T *cu = U_fp + v, *cv = V_fp + v;

        int64_t p0 = pred_l3d(cu,t,i,j,si,sj,sk);
        // int64_t p1 = pred_l3d_stride2(cu,t,i,j,si,sj,sk);
        int64_t p1 = pred_advect_bilinear(cu,t,i,j,si,sj,sk);
        int64_t p2 = pred_dt2d_ar1(cu,t,i,j,si,sj,sk);
        int64_t p3 = pred_time_ar2_clamped(cu,t,i,j,si,sj,sk);
        score[0] += q_cost((long long)(*cu - p0));
        score[1] += q_cost((long long)(*cu - p1));
        score[2] += q_cost((long long)(*cu - p2));
        score[3] += q_cost((long long)(*cu - p3));

        p0 = pred_l3d(cv,t,i,j,si,sj,sk);
        // p1 = pred_l3d_stride2(cv,t,i,j,si,sj,sk);
        p1 = pred_advect_bilinear(cv,t,i,j,si,sj,sk);
        p2 = pred_dt2d_ar1(cv,t,i,j,si,sj,sk);
        p3 = pred_time_ar2_clamped(cv,t,i,j,si,sj,sk);
        score[0] += q_cost((long long)(*cv - p0));
        score[1] += q_cost((long long)(*cv - p1));
        score[2] += q_cost((long long)(*cv - p2));
        score[3] += q_cost((long long)(*cv - p3));
        ++cnt;
    }

    if (cnt==0) return PM::L3D;

    int best = 0; long double bestv = score[0];
    for (int k=1; k<4; ++k){ if (score[k] < bestv){ best=k; bestv=score[k]; } }

    // 只有当相对 L3D 改进超过阈值才切换，避免在不同 eb 下“略差”的情况
    if (best != 0){
        const long double rel_gain = (score[0] - bestv) / (score[0] + 1e-12L);
        if (rel_gain < SZ_MOP_REL_IMPROVE_MIN) best = 0;
    }

    // return (best==0)? PM::L3D : (best==1? PM::L3D_S2 : (best==2? PM::DTL_AR1 : PM::TAR2C));
    return (best==0)? PM::L3D : (best==1? PM::ADVECT : (best==2? PM::DTL_AR1 : PM::TAR2C));
}
#endif

#ifndef SZ_MOP_OFLOW_PENALTY_BITS
// q 溢出（qd >= capacity）按每个样本附加的惩罚比特，经验值 8~24 之间可调
#define SZ_MOP_OFLOW_PENALTY_BITS 16.0
#endif

#if 0
//以下函数采样只考虑块的顶行+左列，避免“尚未回写”的邻居偏差
template<typename T>
inline PM choose_pred_mode_block_lookahead_bitspersample(
    const T* U_fp, const T* V_fp,
    int H,int W,int t, int bi,int bj,int BH,int BW,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk,
    T eb_lsb_cap,              // 传入 max((T)1, max_eb)
    int intv_radius,           // 与编码端一致：capacity >> 1
    int capacity,              // 与编码端一致
    int subs=SZ_MOP_EVAL_SUBS)
{
    // 前两帧尽量保守（特别是 AR2 需要 t>=2）
    if (t < 1) return PM::L3D;
    if (eb_lsb_cap <= 0) eb_lsb_cap = 1;
    const int bins = 2 * capacity;
    // 四个候选器的直方图 + 计数 + 溢出统计
    std::vector<uint32_t> h0(bins,0), h1(bins,0), h2(bins,0), h3(bins,0);
    uint64_t n0=0, n1=0, n2=0, n3=0;
    uint64_t o0=0, o1=0, o2=0, o3=0;
    auto push_qidx = [&](std::vector<uint32_t>& h, uint64_t& n, uint64_t& of, long long diff){
        long long q = ( (diff>=0? diff : -diff) / (long long)eb_lsb_cap ) + 1; // q>=1
        if (q >= capacity){ ++of; return; } // 溢出：编码端往往会走“不可预测/大符号”，这里加罚
        if (diff < 0) q = -q;
        long long qidx = (long long)(q/2) + intv_radius;  // 与编码端一致：向零截断
        if (qidx < 0) qidx = 0;
        else if (qidx >= (long long)bins) qidx = bins-1;
        ++h[(size_t)qidx];
        ++n;
    };
    auto entropy_bits_per_sample = [&](const std::vector<uint32_t>& h, uint64_t n)-> long double {
        if (n == 0) return 1e9L; // 极端情况（几乎不抽样），回退大值
        long double H = 0.0L;
        for (size_t k=0; k<h.size(); ++k){
            uint32_t c = h[k];
            if (!c) continue;
            long double p = (long double)c / (long double)n;
            H += -p * std::log2(p);
        }
        return H; // bits / sample（这里“sample”=一个分量的一次量化索引）
    };
    const int i_end = std::min(bi+BH, H);
    const int j_end = std::min(bj+BW, W);

    // --- 绑定 ADVECT 所需的 t-1 帧 U/V 平面（O(1）） ---
    // 假设 U_fp / V_fp 与被压字段同布局（元素步长 si/sj/sk 对齐）
    const int64_t* u_prev_plane = (const int64_t*)(U_fp + (size_t)(t-1) * (size_t)H * (size_t)W);
    const int64_t* v_prev_plane = (const int64_t*)(V_fp + (size_t)(t-1) * (size_t)H * (size_t)W);
    pred_advect_bind_prev_uv(u_prev_plane, v_prev_plane, /*si_uv*/si, /*sj_uv*/sj);

    // 顶行
    for (int j=bj; j<j_end; j+=subs){
        const int i = bi;
        const size_t v = (size_t)t*H*W + (size_t)i*W + (size_t)j;
        const T *cu = U_fp + v, *cv = V_fp + v;
        // U 分量
        {
            long long d0 = (long long)(*cu - pred_l3d            (cu,t,i,j,si,sj,sk));
            // long long d1 = (long long)(*cu - pred_l3d_stride2    (cu,t,i,j,si,sj,sk));
            long long d1 = (long long)(*cu - pred_advect_bilinear(cu,t,i,j,si,sj,sk));
            long long d2 = (long long)(*cu - pred_dt2d_ar1       (cu,t,i,j,si,sj,sk));
            long long d3 = (long long)(*cu - pred_time_ar2_clamped(cu,t,i,j,si,sj,sk));
            push_qidx(h0,n0,o0,d0);
            push_qidx(h1,n1,o1,d1);
            push_qidx(h2,n2,o2,d2);
            push_qidx(h3,n3,o3,d3);
        }
        // V 分量
        {
            long long d0 = (long long)(*cv - pred_l3d            (cv,t,i,j,si,sj,sk));
            // long long d1 = (long long)(*cv - pred_l3d_stride2    (cv,t,i,j,si,sj,sk));
            long long d1 = (long long)(*cv - pred_advect_bilinear(cv,t,i,j,si,sj,sk));
            long long d2 = (long long)(*cv - pred_dt2d_ar1       (cv,t,i,j,si,sj,sk));
            long long d3 = (long long)(*cv - pred_time_ar2_clamped(cv,t,i,j,si,sj,sk));
            push_qidx(h0,n0,o0,d0);
            push_qidx(h1,n1,o1,d1);
            push_qidx(h2,n2,o2,d2);
            push_qidx(h3,n3,o3,d3);
        }
    }
    // 左列（避开顶行第一个交点）
    for (int i=bi+SZ_MOP_EVAL_GUARD; i<i_end; i+=subs){
        const int j = bj;
        const size_t v = (size_t)t*H*W + (size_t)i*W + (size_t)j;
        const T *cu = U_fp + v, *cv = V_fp + v;
        // U 分量
        {
            long long d0 = (long long)(*cu - pred_l3d            (cu,t,i,j,si,sj,sk));
            // long long d1 = (long long)(*cu - pred_l3d_stride2    (cu,t,i,j,si,sj,sk));
            long long d1 = (long long)(*cu - pred_advect_bilinear(cu,t,i,j,si,sj,sk));
            long long d2 = (long long)(*cu - pred_dt2d_ar1       (cu,t,i,j,si,sj,sk));
            long long d3 = (long long)(*cu - pred_time_ar2_clamped(cu,t,i,j,si,sj,sk));
            push_qidx(h0,n0,o0,d0);
            push_qidx(h1,n1,o1,d1);
            push_qidx(h2,n2,o2,d2);
            push_qidx(h3,n3,o3,d3);
        }
        // V 分量
        {
            long long d0 = (long long)(*cv - pred_l3d            (cv,t,i,j,si,sj,sk));
            // long long d1 = (long long)(*cv - pred_l3d_stride2    (cv,t,i,j,si,sj,sk));
            long long d1 = (long long)(*cv - pred_advect_bilinear(cv,t,i,j,si,sj,sk));
            long long d2 = (long long)(*cv - pred_dt2d_ar1       (cv,t,i,j,si,sj,sk));
            long long d3 = (long long)(*cv - pred_time_ar2_clamped(cv,t,i,j,si,sj,sk));
            push_qidx(h0,n0,o0,d0);
            push_qidx(h1,n1,o1,d1);
            push_qidx(h2,n2,o2,d2);
            push_qidx(h3,n3,o3,d3);
        }
    }
    // bits/sample（越小越好）+ 溢出罚项 + 模式比特开销（折算到每像素）
    long double bits[4] = {
        entropy_bits_per_sample(h0,n0),
        entropy_bits_per_sample(h1,n1),
        entropy_bits_per_sample(h2,n2),
        entropy_bits_per_sample(h3,n3)
    };
    auto add_penalty = [&](long double& b, uint64_t n, uint64_t of){
        const long double denom = (long double)(n + of) + 1e-12L;
        const long double of_ratio = (long double)of / denom;
        b += of_ratio * (long double)SZ_MOP_OFLOW_PENALTY_BITS; // 溢出越多，罚越多
        // 模式开销：每块 2 bit，折算到“每像素、每分量”的尺度（近似）
        b += (2.0L / ( (long double)BH * (long double)BW * 2.0L ));
    };
    add_penalty(bits[0],n0,o0);
    add_penalty(bits[1],n1,o1);
    add_penalty(bits[2],n2,o2);
    add_penalty(bits[3],n3,o3);
    // AR2 在 t<2 时无效：强行回退一个大值，保证不会被选中
    if (t < 2) bits[3] = 1e9L;
    // 选 bits 最小
    int best = 0; long double bestv = bits[0];
    for (int k=1; k<4; ++k){ if (bits[k] < bestv){ best=k; bestv=bits[k]; } }
    // 只有当相对 L3D 的熵改进够大才切换，避免“有时略差一点”
    if (best != 0){
        const long double rel_gain = (bits[0] - bestv) / (bits[0] + 1e-12L);
        if (rel_gain < (long double)SZ_MOP_REL_IMPROVE_MIN) best = 0;
    }
    switch (best){
        case 0: return PM::L3D;
        // case 1: return PM::L3D_S2;
        case 1: return PM::ADVECT;
        case 2: return PM::DTL_AR1;
        default:return PM::TAR2C;
    }
}

#endif

//以下是使用block的全局采样
// 估计香农熵（bits/sample）
static inline long double entropy_bits_per_sample_vec(const std::vector<uint32_t>& h, uint64_t n){
    if (n == 0) return 1e9L;
    long double H = 0.0L;
    for (size_t k=0; k<h.size(); ++k){
        uint32_t c = h[k];
        if (!c) continue;
        long double p = (long double)c / (long double)n;
        H += -p * std::log2(p);
    }
    return H;
}
template<typename T>
inline PM choose_pred_mode_block_lookahead_bitspersample_entire_block_sample(
    const T* U_fp, const T* V_fp,
    int H,int W,int t, int bi,int bj,int BH,int BW,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk,
    T eb_lsb_cap,              // max((T)1, max_eb)
    int intv_radius,           // capacity >> 1
    int capacity,              // 量化桶容量（正半轴）
    int subs                   // 采样步长（>=1）
){
    if (t < 1) return PM::L3D;
    if (eb_lsb_cap <= 0) eb_lsb_cap = 1;
    if (subs < 1) subs = 1;

    const int i_end = std::min(bi+BH, H);
    const int j_end = std::min(bj+BW, W);
    const size_t plane = (size_t)H * (size_t)W;
    const int bins = 2 * capacity;

    // 若 subs 过大导致该块没有任何采样点，则回退为 subs=1
    const int ns_i = ( (i_end - bi) + subs - 1 ) / subs;
    const int ns_j = ( (j_end - bj) + subs - 1 ) / subs;
    if (ns_i <= 0 || ns_j <= 0) subs = 1;

    // 绑定 ADVECT 的 U/V(t-1) 速度平面（与真实一致）
    const int64_t* u_prev_plane = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
    const int64_t* v_prev_plane = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
    pred_advect_bind_prev_uv(u_prev_plane, v_prev_plane, /*si_uv*/si, /*sj_uv*/sj);

    // 4 候选：0=L3D, 1=ADVECT, 2=DTL_AR1, 3=TAR2C
    std::vector<uint32_t> hist[4] = {
        std::vector<uint32_t>(bins,0),
        std::vector<uint32_t>(bins,0),
        std::vector<uint32_t>(bins,0),
        std::vector<uint32_t>(bins,0)
    };
    uint64_t n[4]  = {0,0,0,0};   // 仅统计“被抽样点”的 U+V 样本数
    uint64_t of[4] = {0,0,0,0};   // 仅统计“被抽样点”的溢出（按分量计）

    auto push_qidx = [&](std::vector<uint32_t>& h, uint64_t& ncnt,
                         long long diff)-> bool {
        long long ad = (diff>=0? diff : -diff);
        long long qd = ad / (long long)eb_lsb_cap + 1; // q>=1
        if (qd >= capacity) return false;
        if (diff < 0) qd = -qd;
        long long qidx = (long long)(qd/2) + intv_radius; // 与编码端一致：向零截断
        if (qidx < 0) qidx = 0;
        else if (qidx >= (long long)bins) qidx = bins-1;
        ++h[(size_t)qidx];
        ++ncnt;
        return true;
    };

    // 对单个模式做“整块微编码仿真”，但直方图只在 subs 网格上计入
    auto run_mode = [&](int mode_idx){
        // 候选缓冲：[t-1 | t] 两帧相邻，保证 -sk 指向 t-1
        std::vector<int64_t> bufU(plane*2), bufV(plane*2);
        const int64_t* U_prev = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
        const int64_t* V_prev = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
        const int64_t* U_cur0 = (const int64_t*)(U_fp + (size_t)t*plane); // 原始/已写回的 t 帧
        const int64_t* V_cur0 = (const int64_t*)(V_fp + (size_t)t*plane);

        std::memcpy(bufU.data(),       U_prev, plane*sizeof(int64_t));
        std::memcpy(bufU.data()+plane, U_cur0, plane*sizeof(int64_t));
        std::memcpy(bufV.data(),       V_prev, plane*sizeof(int64_t));
        std::memcpy(bufV.data()+plane, V_cur0, plane*sizeof(int64_t));

        for (int i=bi; i<i_end; ++i){
            for (int j=bj; j<j_end; ++j){
                const size_t idx2d = (size_t)i*W + (size_t)j;
                const int64_t xU = (int64_t)U_cur0[idx2d];
                const int64_t xV = (int64_t)V_cur0[idx2d];

                int64_t* curU = bufU.data() + plane + idx2d;
                int64_t* curV = bufV.data() + plane + idx2d;

                int64_t pU, pV;
                switch (mode_idx){
                    case 0: pU = pred_l3d(curU,t,i,j,si,sj,sk);
                            pV = pred_l3d(curV,t,i,j,si,sj,sk); break;
                    case 1: pU = pred_advect_bilinear(curU,t,i,j,si,sj,sk);
                            pV = pred_advect_bilinear(curV,t,i,j,si,sj,sk); break;
                    // case 1: pU = pred_l3d_stride2(curU,t,i,j,si,sj,sk);
                    //         pV = pred_l3d_stride2(curV,t,i,j,si,sj,sk); break;
                    case 2: pU = pred_dt2d_ar1(curU,t,i,j,si,sj,sk);
                            pV = pred_dt2d_ar1(curV,t,i,j,si,sj,sk); break;
                    default:pU = pred_time_ar2_clamped(curU,t,i,j,si,sj,sk);
                            pV = pred_time_ar2_clamped(curV,t,i,j,si,sj,sk); break;
                }

                const long long dU = (long long)(xU - pU);
                const long long dV = (long long)(xV - pV);

                // 编码端一致的量化+反量化尝试（本像素一定要执行，以便写回候选缓冲，供后续邻居使用）
                auto quantize_once = [&](long long diff, int64_t pred, int& qindex, int64_t& recon)-> bool {
                    long long ad = (diff>=0? diff : -diff);
                    long long qd = ad / (long long)eb_lsb_cap + 1;
                    if (qd >= capacity) return false;
                    if (diff < 0) qd = -qd;
                    qindex = (int)(qd/2) + intv_radius;
                    recon  = pred + 2LL * ( (long long)qindex - intv_radius ) * (long long)eb_lsb_cap;
                    long long err = recon - (diff + pred); // recon - x
                    if ( (err>=0? err : -err) > (long long)eb_lsb_cap ) return false;
                    return true;
                };

                int qU=0, qV=0; int64_t rU=0, rV=0;
                bool okU = quantize_once(dU, pU, qU, rU);
                bool okV = quantize_once(dV, pV, qV, rV);

                // 写回候选缓冲：任何一分量失败 → 都不写回（与编码端一致：unpred 保持原值）
                if (okU && okV){
                    *curU = rU; *curV = rV;
                }

                // 只在 subs 网格上统计直方图/溢出
                if ( ((i - bi) % subs == 0) && ((j - bj) % subs == 0) ){
                    if (okU){
                        long long diff_check = (long long)(xU - pU);
                        if (!push_qidx(hist[mode_idx], n[mode_idx], diff_check)) { of[mode_idx]++; }
                    } else {
                        of[mode_idx]++; // U 溢出/失败
                    }
                    if (okV){
                        long long diff_check_v = (long long)(xV - pV);
                        if (!push_qidx(hist[mode_idx], n[mode_idx], diff_check_v)) { of[mode_idx]++; }
                    } else {
                        of[mode_idx]++; // V 溢出/失败
                    }
                }
            }
        }
    };

    // 依次评估 4 个模式
    run_mode(0); // L3D
    run_mode(1); // L3D_S2
    run_mode(2); // DTL_AR1
    run_mode(3); // TAR2C

    long double bits[4] = {
        entropy_bits_per_sample_vec(hist[0], n[0]),
        entropy_bits_per_sample_vec(hist[1], n[1]),
        entropy_bits_per_sample_vec(hist[2], n[2]),
        entropy_bits_per_sample_vec(hist[3], n[3])
    };
    auto add_penalty = [&](int k){
        const long double denom = (long double)(n[k] + of[k]) + 1e-12L;
        const long double of_ratio = (long double)of[k] / denom;
        bits[k] += of_ratio * (long double)SZ_MOP_OFLOW_PENALTY_BITS;
        // 4 路模式 → 2 bit/块；折算到“每像素每分量”
        bits[k] += (2.0L / ( (long double)BH * (long double)BW * 2.0L ));
    };
    add_penalty(0); add_penalty(1); add_penalty(2); add_penalty(3);

    if (t < 2) bits[3] = 1e9L; // TAR2C 屏蔽首两帧

    int best = 0; long double bestv = bits[0];
    for (int k=1; k<4; ++k){ if (bits[k] < bestv){ best=k; bestv=bits[k]; } }
    if (best != 0){
        const long double rel_gain = (bits[0] - bestv) / (bits[0] + 1e-12L);
        if (rel_gain < (long double)SZ_MOP_REL_IMPROVE_MIN) best = 0;
    }
    switch (best){
        case 0: return PM::L3D;
        case 1: return PM::ADVECT;   // 你当前把 L3D_S2 替换为 ADVECT
        // case 1: return PM::L3D_S2;
        case 2: return PM::DTL_AR1;
        default:return PM::TAR2C;
    }
}
#endif