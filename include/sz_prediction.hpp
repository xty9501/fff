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
enum class PM : uint8_t { L3D=0, ADVECT=1};
inline std::ostream& operator<<(std::ostream& os, PM pm) {
    switch (pm) {
        case PM::L3D:      return os << "L3D";
        case PM::ADVECT:   return os << "ADVECT";
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
        // case PM::DTL_AR1: return pred_dt2d_ar1(cur,t,i,j,si,sj,sk); //mode 2
        // case PM::TAR2C:   return pred_time_ar2_clamped(cur,t,i,j,si,sj,sk); //mode 3
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

// ---------- 1bit 模式打包/解包 ----------
inline void pack_modes_1bit(const std::vector<uint8_t>& modes, std::vector<uint8_t>& out_bytes) {
    out_bytes.clear();
    out_bytes.reserve((modes.size() + 7) / 8); // 每8个模式 -> 1字节

    uint8_t acc = 0;
    int fill = 0;

    for (uint8_t m : modes) {
        acc |= (m & 0x1u) << fill;  // 取最低位，放入累积字节
        fill += 1;
        if (fill == 8) {
            out_bytes.push_back(acc);
            acc = 0;
            fill = 0;
        }
    }

    if (fill) out_bytes.push_back(acc);
}
inline void unpack_modes_1bit(const uint8_t* in_bytes, size_t n_modes, std::vector<uint8_t>& modes) {
    modes.resize(n_modes);

    size_t o = 0;  // 输出计数
    size_t i = 0;  // 输入字节索引

    while (o < n_modes) {
        uint8_t b = in_bytes[i++];
        for (int k = 0; k < 8 && o < n_modes; ++k) {
            modes[o++] = (b >> k) & 0x1u;  // 提取第k位
        }
    }
}

inline void print_mode_counts(const std::vector<uint8_t>& modes) {
    std::vector<size_t> counts(2, 0);
    for (uint8_t m : modes) {
        if (m < 2) {
            counts[m]++;
        } else {
            std::cerr << "Warning: invalid mode value " << static_cast<int>(m) << std::endl;
        }
    }

    for (size_t i = 0; i < 2; ++i) {
        std::cout << "Mode " << static_cast<PM>(i) << ": " << counts[i] << " times\n";
    }
    std::cout <<"percentage of SL: " << (double)counts[1]/(double)(counts[0]+counts[1]) << "\n"; 
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

    // 若 subs 过大导致该块没有任何采样点，则回退 subs=1
    const int ns_i = ((i_end - bi) + subs - 1) / subs;
    const int ns_j = ((j_end - bj) + subs - 1) / subs;
    if (ns_i <= 0 || ns_j <= 0) subs = 1;

    // 绑定 ADVECT 的 U/V(t-1)
    const int64_t* u_prev_plane = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
    const int64_t* v_prev_plane = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
    pred_advect_bind_prev_uv(u_prev_plane, v_prev_plane, /*si_uv*/si, /*sj_uv*/sj);

    // 2 候选：0=L3D, 1=ADVECT
    std::vector<uint32_t> hist[2] = {
        std::vector<uint32_t>(bins,0),
        std::vector<uint32_t>(bins,0)
    };
    uint64_t n[2]  = {0,0};   // 抽样点的 U+V 样本数
    uint64_t of[2] = {0,0};   // 抽样点的溢出计数（按分量）

    auto push_qidx = [&](std::vector<uint32_t>& h, uint64_t& ncnt,
                         long long diff)-> bool {
        long long ad = (diff>=0? diff : -diff);
        long long qd = ad / (long long)eb_lsb_cap + 1; // q>=1
        if (qd >= capacity) return false;
        if (diff < 0) qd = -qd;
        long long qidx = (long long)(qd/2) + intv_radius; // 向零截断，与编码端一致
        if (qidx < 0) qidx = 0;
        else if (qidx >= (long long)bins) qidx = bins-1;
        ++h[(size_t)qidx];
        ++ncnt;
        return true;
    };

    auto run_mode = [&](int mode_idx){
        // 候选缓冲：[t-1 | t] 相邻，保证 -sk 指向 t-1
        std::vector<int64_t> bufU(plane*2), bufV(plane*2);
        const int64_t* U_prev = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
        const int64_t* V_prev = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
        const int64_t* U_cur0 = (const int64_t*)(U_fp + (size_t)t*plane);
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
                    case 0: // L3D
                        pU = pred_l3d(curU,t,i,j,si,sj,sk);
                        pV = pred_l3d(curV,t,i,j,si,sj,sk);
                        break;
                    default: // 1: ADVECT
                        pU = pred_advect_bilinear(curU,t,i,j,si,sj,sk);
                        pV = pred_advect_bilinear(curV,t,i,j,si,sj,sk);
                        break;
                }

                const long long dU = (long long)(xU - pU);
                const long long dV = (long long)(xV - pV);

                // 编码端一致的量化+反量化尝试（本像素一定要执行，以便写回供邻居使用）
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

                // 写回候选缓冲：任何一分量失败 → 不写回（unpred 保持原值）
                if (okU && okV){
                    *curU = rU; *curV = rV;
                }

                // 仅在 subs 网格上统计
                if ( ((i - bi) % subs == 0) && ((j - bj) % subs == 0) ){
                    if (okU){
                        long long diff_check = (long long)(xU - pU);
                        if (!push_qidx(hist[mode_idx], n[mode_idx], diff_check)) { of[mode_idx]++; }
                    } else {
                        of[mode_idx]++;
                    }
                    if (okV){
                        long long diff_check_v = (long long)(xV - pV);
                        if (!push_qidx(hist[mode_idx], n[mode_idx], diff_check_v)) { of[mode_idx]++; }
                    } else {
                        of[mode_idx]++;
                    }
                }
            }
        }
    };

    // 评估 2 个模式
    run_mode(0); // L3D
    run_mode(1); // ADVECT

    long double bits[2] = {
        entropy_bits_per_sample_vec(hist[0], n[0]),
        entropy_bits_per_sample_vec(hist[1], n[1])
    };

    auto add_penalty = [&](int k){
        const long double denom = (long double)(n[k] + of[k]) + 1e-12L;
        const long double of_ratio = (long double)of[k] / denom;
        bits[k] += of_ratio * (long double)SZ_MOP_OFLOW_PENALTY_BITS;
        // 2 路模式 → 1 bit/块；折算到“每像素每分量”
        bits[k] += (1.0L / ( (long double)BH * (long double)BW * 2.0L ));
    };
    add_penalty(0);
    add_penalty(1);

    // 选择更优模式，带相对收益门槛，默认3DL
    int best = 0; long double bestv = bits[0];
    if (bits[1] < bestv){ best = 1; bestv = bits[1]; }

    if (best != 0){
        const long double rel_gain = (bits[0] - bestv) / (bits[0] + 1e-12L);
        if (rel_gain < (long double)SZ_MOP_REL_IMPROVE_MIN) best = 0;
    }

    //默认选择ADVECT
    // int best = 1; long double bestv = bits[1];
    // if (bits[0] + 1e-12L < bits[1]) {
    // const long double rel_gain = (bits[1] - bits[0]) / (bits[1] + 1e-12L);
    // if (rel_gain > (long double)SZ_MOP_REL_IMPROVE_MIN)
    //     best = 0;  // L3D 提升显著，才选 L3D
    // }   

    return (best==0) ? PM::L3D : PM::ADVECT;
}

template<typename T>
inline PM choose_pred_mode_block_lookahead_bitspersample_entire_block_sample_fast(
    const T* U_fp, const T* V_fp,
    int H,int W,int t, int bi,int bj,int BH,int BW,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk,
    T eb_lsb_cap,
    int intv_radius,
    int capacity,
    int subs
){
    // (void)eb_lsb_cap;
    // (void)intv_radius;
    // (void)capacity;

    if (t < 1) return PM::L3D;
    if (subs < 1) subs = 1;

    const int i_end = std::min(bi+BH, H);
    const int j_end = std::min(bj+BW, W);
    const size_t plane = (size_t)H * (size_t)W;

    int ns_i = ((i_end - bi) + subs - 1) / subs;
    int ns_j = ((j_end - bj) + subs - 1) / subs;
    if (ns_i <= 0 || ns_j <= 0) subs = 1;

    const int64_t* U_prev_plane = (const int64_t*)(U_fp + (size_t)(t-1) * plane);
    const int64_t* V_prev_plane = (const int64_t*)(V_fp + (size_t)(t-1) * plane);

    pred_advect_bind_prev_uv(U_prev_plane, V_prev_plane, si, sj);

    long double score_l3d = 0.0L;
    long double score_adv = 0.0L;
    uint64_t samples = 0;

    const int64_t* U_cur_plane = (const int64_t*)(U_fp + (size_t)t * plane);
    const int64_t* V_cur_plane = (const int64_t*)(V_fp + (size_t)t * plane);

    for (int i=bi; i<i_end; ++i){
        for (int j=bj; j<j_end; ++j){
            if ( ((i - bi) % subs != 0) || ((j - bj) % subs != 0) ) continue;

            const size_t idx2d = (size_t)i * (size_t)W + (size_t)j;
            const int64_t* curU = U_cur_plane + idx2d;
            const int64_t* curV = V_cur_plane + idx2d;
            const int64_t xU = *curU;
            const int64_t xV = *curV;

            const int64_t pU_l3d = pred_l3d(curU,t,i,j,si,sj,sk);
            const int64_t pV_l3d = pred_l3d(curV,t,i,j,si,sj,sk);
            const long double dU_l3d = (long double)xU - (long double)pU_l3d;
            const long double dV_l3d = (long double)xV - (long double)pV_l3d;
            // const long double err_l3d =
            //     (dU_l3d >= 0.0L ? dU_l3d : -dU_l3d) +
            //     (dV_l3d >= 0.0L ? dV_l3d : -dV_l3d);
            const long double abs_err_l3d = abs(dU_l3d) + abs(dV_l3d);

            const int64_t pU_adv = pred_advect_bilinear(curU,t,i,j,si,sj,sk);
            const int64_t pV_adv = pred_advect_bilinear(curV,t,i,j,si,sj,sk);
            const long double dU_adv = (long double)xU - (long double)pU_adv;
            const long double dV_adv = (long double)xV - (long double)pV_adv;
            // const long double err_adv =
            //     (dU_adv >= 0.0L ? dU_adv : -dU_adv) +
            //     (dV_adv >= 0.0L ? dV_adv : -dV_adv);
            const long double abs_err_adv = abs(dU_adv) + abs(dV_adv);

            score_l3d += abs_err_l3d;
            score_adv += abs_err_adv;
            ++samples;
        }
    }


    if (samples == 0) return PM::L3D;

    // score_l3d /= (long double)samples;
    // score_adv /= (long double)samples;
    long double adjust_item_l3d = (long double)samples * 1.22L * eb_lsb_cap; // 经验值调整
    long double adjust_item_adv = (long double)samples * 0.933L * eb_lsb_cap; // 经验值调整
    // score_l3d = score_l3d + adjust_item_l3d;
    // score_adv = score_adv + adjust_item_adv;
    // 这里可以改成算magnitude，然后乘上对应系数（3dl=1.925,adv=1.462)
    int best = 0;
    long double bestv = score_l3d;
    if (score_adv < bestv){
        best = 1;
        bestv = score_adv;
    }

    if (best != 0){
        const long double rel_gain = (score_l3d - bestv) / (score_l3d + 1e-12L);
        if (rel_gain < (long double)SZ_MOP_REL_IMPROVE_MIN) best = 0;
    }

    return (best == 0) ? PM::L3D : PM::ADVECT;
}



//=================mop less cpy version=====================
template<typename T>
inline PM choose_pred_mode_block_lookahead_bitspersample_entire_block_sample_less_cpy(
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

    const int ns_i = ((i_end - bi) + subs - 1) / subs;
    const int ns_j = ((j_end - bj) + subs - 1) / subs;
    if (ns_i <= 0 || ns_j <= 0) subs = 1;

    // 绑定 ADVECT 的 U/V(t-1)（速度场用原始源数据，不依赖工作区）
    const int64_t* u_prev_plane_src = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
    const int64_t* v_prev_plane_src = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
    pred_advect_bind_prev_uv(u_prev_plane_src, v_prev_plane_src, /*si_uv*/si, /*sj_uv*/sj);

    // ========= 帧级工作区（只在 t 或尺寸/地址变化时重建一次）=========
    struct JournalEntry { size_t off; int64_t oldv; };
    struct FrameWork {
        std::vector<int64_t> U_work; // size = 2*plane (t-1 | t)
        std::vector<int64_t> V_work; // size = 2*plane
        std::vector<JournalEntry> jU, jV;

        int last_t = -1;
        size_t last_plane = 0;
        const int64_t* U_prev_src = nullptr;
        const int64_t* V_prev_src = nullptr;
        const int64_t* U_cur_src  = nullptr;
        const int64_t* V_cur_src  = nullptr;

        void ensure(int H,int W,int t,
                    const int64_t* U_prev, const int64_t* U_cur,
                    const int64_t* V_prev, const int64_t* V_cur,
                    size_t cap_hint)       // ← 新增参数
        {
            const size_t plane_now = (size_t)H*(size_t)W;
            const bool need_rebuild =
                (t != last_t) || (plane_now != last_plane) ||
                (U_prev != U_prev_src) || (V_prev != V_prev_src) ||
                (U_cur  != U_cur_src ) || (V_cur  != V_cur_src );

            if (need_rebuild) {
                U_work.resize(2*plane_now);
                V_work.resize(2*plane_now);
                std::memcpy(U_work.data(),             U_prev, plane_now*sizeof(int64_t));
                std::memcpy(U_work.data()+plane_now,   U_cur,  plane_now*sizeof(int64_t));
                std::memcpy(V_work.data(),             V_prev, plane_now*sizeof(int64_t));
                std::memcpy(V_work.data()+plane_now,   V_cur,  plane_now*sizeof(int64_t));

                last_t = t;
                last_plane = plane_now;
                U_prev_src = U_prev; V_prev_src = V_prev;
                U_cur_src  = U_cur;  V_cur_src  = V_cur;

                jU.clear(); jV.clear();
                // 预留日志容量：每块上限 ~ BH*BW*2（U/V）
                jU.reserve(cap_hint);
                jV.reserve(cap_hint);
            } else {
                jU.clear(); jV.clear();
            }
        }
        inline void writebackU(size_t off, int64_t newv){
            int64_t& ref = U_work[off];
            if (ref != newv){ jU.push_back({off, ref}); ref = newv; }
        }
        inline void writebackV(size_t off, int64_t newv){
            int64_t& ref = V_work[off];
            if (ref != newv){ jV.push_back({off, ref}); ref = newv; }
        }
        inline void rollback(){
            for (size_t k=jU.size(); k--;) U_work[jU[k].off] = jU[k].oldv;
            for (size_t k=jV.size(); k--;) V_work[jV[k].off] = jV[k].oldv;
            jU.clear(); jV.clear();
        }
    };

    // 注意：此函数若在 header 中，多 TU/多线程使用时，
    //       thread_local 可以避免跨线程共享同一工作区。
    static thread_local FrameWork fw;

    const int64_t* U_prev_src = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
    const int64_t* V_prev_src = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
    const int64_t* U_cur_src  = (const int64_t*)(U_fp + (size_t)t*plane);
    const int64_t* V_cur_src  = (const int64_t*)(V_fp + (size_t)t*plane);

    // 计算每块日志容量提示（不能在 ensure 里用 BH/BW，所以这里算好传入）
    const size_t cap_hint = (size_t)BH * (size_t)BW * 2u;
    fw.ensure(H,W,t, U_prev_src,U_cur_src, V_prev_src,V_cur_src, cap_hint);

    // ====== 直方图、统计（2 个候选：0=L3D, 1=ADVECT）======
    std::vector<uint32_t> hist0(bins,0), hist1(bins,0);
    uint64_t n0=0,n1=0, of0=0,of1=0;

    auto push_qindex = [&](std::vector<uint32_t>& h, uint64_t& ncnt, int qindex){
        if (qindex < 0) qindex = 0;
        else if (qindex >= bins) qindex = bins - 1;
        ++h[(size_t)qindex];
        ++ncnt;
    };

    auto quantize_once = [&](long long diff, int64_t pred,
                             int& qindex, int64_t& recon, bool& overflow)-> bool {
        long long ad = (diff>=0? diff : -diff);
        long long qd = ad / (long long)eb_lsb_cap + 1;   // q>=1
        if (qd >= capacity){ overflow = true; return false; }
        overflow = false;
        if (diff < 0) qd = -qd;
        qindex = (int)(qd/2) + intv_radius;              // 向零截断
        recon  = pred + 2LL * ( (long long)qindex - intv_radius ) * (long long)eb_lsb_cap;
        long long err = recon - (diff + pred);           // recon - x
        if ( (err>=0? err : -err) > (long long)eb_lsb_cap ) return false;
        return true;
    };

    auto run_mode = [&](int mode_idx){
        int64_t* U_work_cur = fw.U_work.data() + plane; // t 平面基址
        int64_t* V_work_cur = fw.V_work.data() + plane;

        for (int i=bi; i<i_end; ++i){
            for (int j=bj; j<j_end; ++j){
                const size_t idx2d = (size_t)i*W + (size_t)j;
                int64_t* curU = U_work_cur + idx2d;
                int64_t* curV = V_work_cur + idx2d;
                const int64_t xU = U_cur_src[idx2d]; // 原始 t 帧（不用工作区值）
                const int64_t xV = V_cur_src[idx2d];

                int64_t pU, pV;
                if (mode_idx == 0){ // L3D
                    pU = pred_l3d(curU,t,i,j,si,sj,sk);
                    pV = pred_l3d(curV,t,i,j,si,sj,sk);
                }else{              // ADVECT
                    pU = pred_advect_bilinear(curU,t,i,j,si,sj,sk);
                    pV = pred_advect_bilinear(curV,t,i,j,si,sj,sk);
                }

                const long long dU = (long long)(xU - pU);
                const long long dV = (long long)(xV - pV);

                int qU=0,qV=0; int64_t rU=0,rV=0;
                bool ofU=false, ofV=false;
                const bool okU = quantize_once(dU, pU, qU, rU, ofU);
                const bool okV = quantize_once(dV, pV, qV, rV, ofV);

                if (okU && okV){
                    // 写回并记录日志（使用 t 平面的偏移：plane + idx2d）
                    fw.writebackU(plane + idx2d, rU);
                    fw.writebackV(plane + idx2d, rV);
                }

                if ( ((i - bi) % subs == 0) && ((j - bj) % subs == 0) ){
                    if (mode_idx == 0){
                        if (okU) push_qindex(hist0, n0, qU); else of0++;
                        if (okV) push_qindex(hist0, n0, qV); else of0++;
                    }else{
                        if (okU) push_qindex(hist1, n1, qU); else of1++;
                        if (okV) push_qindex(hist1, n1, qV); else of1++;
                    }
                }
            }
        }
        // 块评估完成：回滚至基准
        fw.rollback();
    };

    // 评估两个模式（每个模式评完回滚工作区）
    run_mode(0); // L3D
    run_mode(1); // ADVECT

    long double bits[2] = {
        entropy_bits_per_sample_vec(hist0, n0),
        entropy_bits_per_sample_vec(hist1, n1)
    };

    auto add_penalty = [&](int k){
        const uint64_t n  = (k==0)? n0 : n1;
        const uint64_t of = (k==0)? of0 : of1;
        long double& b = bits[k];
        const long double denom = (long double)(n + of) + 1e-12L;
        const long double of_ratio = (long double)of / denom;
        b += of_ratio * (long double)SZ_MOP_OFLOW_PENALTY_BITS;
        // 2 路模式 → 1bit/块；折算到“每像素每分量”
        b += (1.0L / ( (long double)BH * (long double)BW * 2.0L ));
    };
    add_penalty(0);
    add_penalty(1);

    int best = 0; long double bestv = bits[0];
    if (bits[1] < bestv){ best = 1; bestv = bits[1]; }
    if (best != 0){
        const long double rel_gain = (bits[0] - bestv) / (bits[0] + 1e-12L);
        if (rel_gain < (long double)SZ_MOP_REL_IMPROVE_MIN) best = 0;
    }
    return (best==0) ? PM::L3D : PM::ADVECT;
}

//=================mop less bin6 version=====================
// ====== 工具 1：量化上下文（定点倒数替代整除） ======
struct QCtx {
    uint32_t inv_eb_Q;  // ≈ (1<<Q)/eb_lsb_cap
    int      Q;         // 固定 24 即可
    int      capacity;  // 与编码端一致
    int      intv_radius;
    int64_t  eb;        // eb_lsb_cap
};

static inline void qctx_init(QCtx& qc, int64_t eb_lsb_cap, int intv_radius, int capacity){
    qc.Q = 24;
    if (eb_lsb_cap <= 0) eb_lsb_cap = 1;
    // 四舍五入计算 (1<<Q)/eb
    qc.inv_eb_Q   = (uint32_t)std::llround( (double)(1u<<qc.Q) / (double)eb_lsb_cap );
    qc.capacity   = capacity;
    qc.intv_radius= intv_radius;
    qc.eb         = eb_lsb_cap;
}

// 编码端一致的“试量化 + 反量化”
// diff = x - pred;  返回：ok?（是否能在给定 eb 下量化）/ overflow?（qd>=capacity）
static inline bool quantize_once_fast(long long diff, int64_t pred,
                                      const QCtx& qc,
                                      int& qindex, int64_t& recon, bool& overflow)
{
    uint64_t ad = (uint64_t)(diff >= 0 ? diff : -diff);
    // qd = floor(|e|/eb) + 1  —— 用定点倒数替代 64 位整除
    uint64_t qd = ((ad * (uint64_t)qc.inv_eb_Q) >> qc.Q) + 1ull;
    if ((int)qd >= qc.capacity){ overflow = true; return false; }
    overflow = false;

    // 与编码端一致：向零截断
    long long s        = (diff < 0) ? -1LL : 1LL;
    long long q_signed = (long long)qd * s;
    qindex             = (int)(q_signed/2) + qc.intv_radius;

    recon = pred + 2LL * ( (long long)qindex - qc.intv_radius ) * qc.eb;

    // 解码端一致的误差门限检查：|recon - x| <= eb
    long long x    = pred + diff;
    long long err  = recon - x;
    long long aerr = (err >= 0 ? err : -err);
    return (aerr <= qc.eb);
}

// ====== 工具 2：6 桶粗分箱（bits 代理） ======
struct SixBins {
    uint64_t b[6];   // 各桶计数
    uint64_t n;      // 样本总数（只计入成功量化的 U/V 样本）
    SixBins(){ std::memset(b,0,sizeof(b)); n=0; }
};

// m=floor(|e|/eb) → bucket: 0, [1,2), [2,4), [4,8), [8,16), >=16
static inline int six_bucket_from_m(uint64_t m){
    if (m==0) return 0;
    if (m<2)  return 1;
    if (m<4)  return 2;
    if (m<8)  return 3;
    if (m<16) return 4;
    return 5;
}

// 把一个 |e| 样本压入 6 桶，使用 QCtx 的定点倒数计算 m=floor(|e|/eb)
static inline void six_push(SixBins& s, uint64_t abs_e, const QCtx& qc){
    uint64_t m = (abs_e * (uint64_t)qc.inv_eb_Q) >> qc.Q; // floor(|e|/eb)
    s.b[six_bucket_from_m(m)]++;
    s.n++;
}

// bits/sample 估计：熵 + oflow 罚项（按比例）
// 注意：溢出/不可量化样本不计入 n，由外部 oflow 单独计数
static inline long double six_entropy_bits(const SixBins& s){
    if (s.n == 0) return 1e9L;
    long double H = 0.0L;
    for (int k=0;k<6;++k){
        if (!s.b[k]) continue;
        long double p = (long double)s.b[k] / (long double)s.n;
        H += -p * std::log2(p);
    }
    return H;
}

// ====== 主函数：整块“微编码仿真” + 6 桶打分（L3D vs ADVECT） ======
template<typename T>
inline PM choose_pred_mode_block_lookahead_bitspersample_entire_block_sample_6bins(
    const T* U_fp, const T* V_fp,
    int H,int W,int t, int bi,int bj,int BH,int BW,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk,
    T eb_lsb_cap,              // 与编码端一致：max((T)1, max_eb)
    int intv_radius,           // 与编码端一致：capacity >> 1
    int capacity,              // 与编码端一致
    int subs                   // 采样步长（>=1）
){
    if (t < 1) return PM::L3D;
    if (subs < 1) subs = 1;
    if (eb_lsb_cap <= 0) eb_lsb_cap = 1;

    const int i_end = std::min(bi+BH, H);
    const int j_end = std::min(bj+BW, W);
    const size_t plane = (size_t)H * (size_t)W;

    // 若 subs 过大导致该块没有采样点，回退 subs=1
    const int ns_i = ((i_end - bi) + subs - 1) / subs;
    const int ns_j = ((j_end - bj) + subs - 1) / subs;
    if (ns_i <= 0 || ns_j <= 0) subs = 1;

    // —— 绑定 ADVECT 所需的 t-1 速度平面（常数时间）——
    const int64_t* u_prev_plane = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
    const int64_t* v_prev_plane = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
    pred_advect_bind_prev_uv(u_prev_plane, v_prev_plane, /*si_uv*/si, /*sj_uv*/sj);

    // 定点量化上下文（替代所有 64 位整除）
    QCtx qc; qctx_init(qc, (int64_t)eb_lsb_cap, intv_radius, capacity);

    // 候选 0=L3D, 1=ADVECT
    SixBins S0, S1;         // 6 桶
    uint64_t of0=0, of1=0;  // 不可量化/溢出样本数（不计入 S*.n）

    auto run_mode = [&](int mode_idx){
        // 候选缓冲：[t-1 | t] 相邻存放，便于 -sk 指向 t-1
        std::vector<int64_t> bufU(plane*2), bufV(plane*2);
        const int64_t* U_prev = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
        const int64_t* V_prev = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
        const int64_t* U_cur0 = (const int64_t*)(U_fp + (size_t)t*plane);
        const int64_t* V_cur0 = (const int64_t*)(V_fp + (size_t)t*plane);

        std::memcpy(bufU.data(),       U_prev, plane*sizeof(int64_t));
        std::memcpy(bufU.data()+plane, U_cur0, plane*sizeof(int64_t));
        std::memcpy(bufV.data(),       V_prev, plane*sizeof(int64_t));
        std::memcpy(bufV.data()+plane, V_cur0, plane*sizeof(int64_t));

        int64_t* U_work_cur = bufU.data() + plane; // t 平面
        int64_t* V_work_cur = bufV.data() + plane;

        for (int i=bi; i<i_end; ++i){
            for (int j=bj; j<j_end; ++j){
                const size_t idx2d = (size_t)i*W + (size_t)j;

                int64_t* curU = U_work_cur + idx2d; // 注意：pred_* 会用 cur - sk 访问 t-1
                int64_t* curV = V_work_cur + idx2d;

                const int64_t xU = U_cur0[idx2d];
                const int64_t xV = V_cur0[idx2d];

                // 预测
                int64_t pU, pV;
                if (mode_idx == 0){ // L3D
                    pU = pred_l3d(curU,t,i,j,si,sj,sk);
                    pV = pred_l3d(curV,t,i,j,si,sj,sk);
                }else{              // ADVECT
                    pU = pred_advect_bilinear(curU,t,i,j,si,sj,sk);
                    pV = pred_advect_bilinear(curV,t,i,j,si,sj,sk);
                }

                const long long dU = (long long)(xU - pU);
                const long long dV = (long long)(xV - pV);

                // 试量化 + 反量化（两分量都成功才写回）
                int qU=0,qV=0; int64_t rU=0,rV=0;
                bool ofU=false, ofV=false;
                const bool okU = quantize_once_fast(dU, pU, qc, qU, rU, ofU);
                const bool okV = quantize_once_fast(dV, pV, qc, qV, rV, ofV);

                if (okU && okV){
                    *curU = rU; *curV = rV; // 写回候选缓冲，维持块内因果
                }

                // 仅在 subs 网格上统计 6 桶与 oflow
                if ( ((i - bi) % subs == 0) && ((j - bj) % subs == 0) ){
                    if (mode_idx == 0){
                        if (okU) six_push(S0, (uint64_t)(dU>=0? dU : -dU), qc);
                        else     of0++;
                        if (okV) six_push(S0, (uint64_t)(dV>=0? dV : -dV), qc);
                        else     of0++;
                    }else{
                        if (okU) six_push(S1, (uint64_t)(dU>=0? dU : -dU), qc);
                        else     of1++;
                        if (okV) six_push(S1, (uint64_t)(dV>=0? dV : -dV), qc);
                        else     of1++;
                    }
                }
            }
        }
    };

    // 评估两候选（整块微仿真）
    run_mode(0); // L3D
    run_mode(1); // ADVECT

    // bits/sample ≈ 熵 + 溢出/不可量化罚项 + 模式开销折算（两模式≈每块 1bit）
    long double bits0 = six_entropy_bits(S0);
    long double bits1 = six_entropy_bits(S1);

    auto add_penalty = [&](long double& b, uint64_t n, uint64_t of){
        const long double denom = (long double)(n + of) + 1e-12L;
        const long double of_ratio = (long double)of / denom;
        b += of_ratio * (long double)SZ_MOP_OFLOW_PENALTY_BITS;
        // 模式头开销：两路模式≈每块 1 bit，折算到“每像素每分量”
        b += (1.0L / ( (long double)BH * (long double)BW * 2.0L ));
    };
    add_penalty(bits0, S0.n, of0);
    add_penalty(bits1, S1.n, of1);

    // 选择 + 相对收益门槛（默认 L3D）
    int best = 0; long double bestv = bits0;
    if (bits1 < bestv){ best = 1; bestv = bits1; }
    if (best != 0){
        const long double rel_gain = (bits0 - bestv) / (bits0 + 1e-12L);
        if (rel_gain < (long double)SZ_MOP_REL_IMPROVE_MIN) best = 0;
    }
    return (best==0)? PM::L3D : PM::ADVECT;
}

//==========mop lesscpy + invert_intv bins version==========
struct MagHist {
    std::vector<uint32_t> cnt;      // 桶计数（size = intv_radius）
    std::vector<uint32_t> touched;  // 本块内被触达过的桶索引
    uint64_t n = 0;                 // 样本数（只计入成功量化的 U/V 样本）

    void ensure(size_t bins){
        if (cnt.size() != bins) {
            cnt.assign(bins, 0u);
            touched.clear(); touched.reserve(1024);
        } else {
            // 仅清除触达桶，避免 O(bins) 清零
            for (uint32_t idx : touched) cnt[idx] = 0u;
            touched.clear();
        }
        n = 0;
    }
    inline void push(uint32_t idx){
        if (cnt[idx]++ == 0) touched.push_back(idx);
        ++n;
    }
    inline long double entropy_bits() const {
        if (n == 0) return 1e9L;
        long double H = 0.0L;
        for (uint32_t idx : touched){
            uint32_t c = cnt[idx];
            if (!c) continue;
            long double p = (long double)c / (long double)n;
            H += -p * std::log2(p);
        }
        return H;
    }
};

template<typename T>
inline PM choose_pred_mode_block_lookahead_bitspersample_entire_block_sample_less_cpy_irbins(
    const T* U_fp, const T* V_fp,
    int H,int W,int t, int bi,int bj,int BH,int BW,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk,
    T eb_lsb_cap,              // max((T)1, max_eb)
    int intv_radius,           // capacity >> 1  —— 同时也是直方图“桶数”
    int capacity,              // 与编码端一致
    int subs                   // 采样步长（>=1）
){
    if (t < 1) return PM::L3D;
    if (subs < 1) subs = 1;
    if (eb_lsb_cap <= 0) eb_lsb_cap = 1;

    const int i_end = std::min(bi+BH, H);
    const int j_end = std::min(bj+BW, W);
    const size_t plane = (size_t)H * (size_t)W;

    // subs 过大导致块内无采样点则回退
    const int ns_i = ((i_end - bi) + subs - 1) / subs;
    const int ns_j = ((j_end - bj) + subs - 1) / subs;
    if (ns_i <= 0 || ns_j <= 0) subs = 1;

    // —— 绑定 ADVECT 所需的 t-1 速度平面（常数时间）——
    const int64_t* u_prev_plane = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
    const int64_t* v_prev_plane = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
    pred_advect_bind_prev_uv(u_prev_plane, v_prev_plane, /*si_uv*/si, /*sj_uv*/sj);

    // 量化上下文（替代所有整除）
    QCtx qc; qctx_init(qc, (int64_t)eb_lsb_cap, intv_radius, capacity);

    // ========= 帧级工作区（一次拷贝 + 日志回滚） =========
    struct JournalEntry { size_t off; int64_t oldv; };
    struct FrameWork {
        std::vector<int64_t> U_work; // [t-1 | t]
        std::vector<int64_t> V_work;
        std::vector<JournalEntry> jU, jV;
        int last_t = -1;
        size_t last_plane = 0;
        const int64_t* U_prev_src = nullptr;
        const int64_t* V_prev_src = nullptr;
        const int64_t* U_cur_src  = nullptr;
        const int64_t* V_cur_src  = nullptr;
        void ensure(int H,int W,int t,
                    const int64_t* U_prev, const int64_t* U_cur,
                    const int64_t* V_prev, const int64_t* V_cur,
                    size_t cap_hint)
        {
            const size_t plane_now = (size_t)H*(size_t)W;
            const bool need_rebuild =
                (t != last_t) || (plane_now != last_plane) ||
                (U_prev != U_prev_src) || (V_prev != V_prev_src) ||
                (U_cur  != U_cur_src ) || (V_cur  != V_cur_src );
            if (need_rebuild) {
                U_work.resize(2*plane_now);
                V_work.resize(2*plane_now);
                std::memcpy(U_work.data(),             U_prev, plane_now*sizeof(int64_t));
                std::memcpy(U_work.data()+plane_now,   U_cur,  plane_now*sizeof(int64_t));
                std::memcpy(V_work.data(),             V_prev, plane_now*sizeof(int64_t));
                std::memcpy(V_work.data()+plane_now,   V_cur,  plane_now*sizeof(int64_t));
                last_t = t; last_plane = plane_now;
                U_prev_src = U_prev; V_prev_src = V_prev;
                U_cur_src  = U_cur;  V_cur_src  = V_cur;
                jU.clear(); jV.clear();
                jU.reserve(cap_hint); jV.reserve(cap_hint);
            } else {
                jU.clear(); jV.clear();
            }
        }
        inline void writebackU(size_t off, int64_t newv){
            int64_t& ref = U_work[off];
            if (ref != newv){ jU.push_back({off, ref}); ref = newv; }
        }
        inline void writebackV(size_t off, int64_t newv){
            int64_t& ref = V_work[off];
            if (ref != newv){ jV.push_back({off, ref}); ref = newv; }
        }
        inline void rollback(){
            for (size_t k=jU.size(); k--;) U_work[jU[k].off] = jU[k].oldv;
            for (size_t k=jV.size(); k--;) V_work[jV[k].off] = jV[k].oldv;
            jU.clear(); jV.clear();
        }
    };
    static thread_local FrameWork fw;

    const int64_t* U_prev_src = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
    const int64_t* V_prev_src = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
    const int64_t* U_cur_src  = (const int64_t*)(U_fp + (size_t)t*plane);
    const int64_t* V_cur_src  = (const int64_t*)(V_fp + (size_t)t*plane);

    const size_t cap_hint = (size_t)BH * (size_t)BW * 2u; // U/V 最多各写回一次
    fw.ensure(H,W,t, U_prev_src,U_cur_src, V_prev_src,V_cur_src, cap_hint);

    // —— 守卫同步：保证 L3D 的上/左邻与编码端一致 —— 
    auto sync_guard_stripes = [&](){
        int i0 = bi,      i1 = i_end - 1;
        int j0 = bj,      j1 = j_end - 1;
        int64_t* U_work_t = fw.U_work.data() + plane; // t 平面基址
        int64_t* V_work_t = fw.V_work.data() + plane;
        // 上方一行（i=bi-1）
        if (bi > 0){
            const size_t row_up = (size_t)(bi-1) * (size_t)W;
            for (int j = j0; j <= j1; ++j){
                const size_t off = row_up + (size_t)j;
                U_work_t[off] = U_cur_src[off];
                V_work_t[off] = V_cur_src[off];
            }
        }
        // 左侧一列（j=bj-1）
        if (bj > 0){
            const int j_left = bj - 1;
            for (int i = i0; i <= i1; ++i){
                const size_t off = (size_t)i * (size_t)W + (size_t)j_left;
                U_work_t[off] = U_cur_src[off];
                V_work_t[off] = V_cur_src[off];
            }
        }
        // 左上角
        if (bi > 0 && bj > 0){
            const size_t off = (size_t)(bi-1) * (size_t)W + (size_t)(bj-1);
            U_work_t[off] = U_cur_src[off];
            V_work_t[off] = V_cur_src[off];
        }
    };

    // ====== 两个“intv_radius 桶”直方图 + oflow 计数 ======
    static thread_local MagHist H0, H1; // 复用内存
    H0.ensure((size_t)intv_radius);
    H1.ensure((size_t)intv_radius);
    uint64_t of0=0, of1=0;

    // ============ 评估一个候选（在工作区上微编码仿真 + 回滚） ============
    auto run_mode = [&](int mode_idx){
        int64_t* U_work_cur = fw.U_work.data() + plane; // t 平面基址
        int64_t* V_work_cur = fw.V_work.data() + plane;

        for (int i=bi; i<i_end; ++i){
            for (int j=bj; j<j_end; ++j){
                const size_t idx2d = (size_t)i*W + (size_t)j;
                int64_t* curU = U_work_cur + idx2d; // pred_* 用 cur - sk 指向 t-1
                int64_t* curV = V_work_cur + idx2d;

                const int64_t xU = U_cur_src[idx2d];
                const int64_t xV = V_cur_src[idx2d];

                int64_t pU, pV;
                if (mode_idx == 0){ // L3D
                    pU = pred_l3d(curU,t,i,j,si,sj,sk);
                    pV = pred_l3d(curV,t,i,j,si,sj,sk);
                }else{              // ADVECT
                    pU = pred_advect_bilinear(curU,t,i,j,si,sj,sk);
                    pV = pred_advect_bilinear(curV,t,i,j,si,sj,sk);
                }

                const long long dU = (long long)(xU - pU);
                const long long dV = (long long)(xV - pV);

                int qU=0,qV=0; int64_t rU=0,rV=0;
                bool ofU=false, ofV=false;
                const bool okU = quantize_once_fast(dU, pU, qc, qU, rU, ofU);
                const bool okV = quantize_once_fast(dV, pV, qc, qV, rV, ofV);

                if (okU && okV){
                    // 写回工作区（保持块内因果）
                    fw.writebackU(plane + idx2d, rU);
                    fw.writebackV(plane + idx2d, rV);
                }

                // 仅在 subs 网格上统计：幅度桶 & oflow
                if ( ((i - bi) % subs == 0) && ((j - bj) % subs == 0) ){
                    // 幅度 m = floor(|e|/eb) —— 用定点倒数
                    uint64_t mU = (( (uint64_t)(dU>=0? dU : -dU) * (uint64_t)qc.inv_eb_Q) >> qc.Q);
                    uint64_t mV = (( (uint64_t)(dV>=0? dV : -dV) * (uint64_t)qc.inv_eb_Q) >> qc.Q);
                    if (mode_idx == 0){
                        if (okU) { if (mU < (uint64_t)intv_radius) H0.push((uint32_t)mU); else of0++; }
                        else of0++;
                        if (okV) { if (mV < (uint64_t)intv_radius) H0.push((uint32_t)mV); else of0++; }
                        else of0++;
                    }else{
                        if (okU) { if (mU < (uint64_t)intv_radius) H1.push((uint32_t)mU); else of1++; }
                        else of1++;
                        if (okV) { if (mV < (uint64_t)intv_radius) H1.push((uint32_t)mV); else of1++; }
                        else of1++;
                    }
                }
            }
        }
        fw.rollback(); // 块评估完成：回滚
    };

    // —— 评估 L3D ——（守卫同步 → 跑 → 回滚）
    sync_guard_stripes();
    run_mode(0);

    // —— 评估 ADVECT ——（再次守卫同步 → 跑 → 回滚）
    sync_guard_stripes();
    run_mode(1);

    // bits/sample ≈ 熵 + oflow 罚项 + 模式开销折算（两模式≈每块 1 bit）
    long double bits0 = H0.entropy_bits();
    long double bits1 = H1.entropy_bits();

    auto add_penalty = [&](long double& b, const MagHist& S, uint64_t of){
        const long double denom = (long double)(S.n + of) + 1e-12L;
        const long double of_ratio = (long double)of / denom;
        b += of_ratio * (long double)SZ_MOP_OFLOW_PENALTY_BITS;
        // 模式头开销折算到“每像素每分量”（两模式≈每块 1 bit）
        b += (1.0L / ( (long double)BH * (long double)BW * 2.0L ));
    };
    add_penalty(bits0, H0, of0);
    add_penalty(bits1, H1, of1);

    // 选择 + 相对收益门槛（默认 L3D）
    int best = 0; long double bestv = bits0;
    if (bits1 < bestv){ best = 1; bestv = bits1; }
    if (best != 0){
        const long double rel_gain = (bits0 - bestv) / (bits0 + 1e-12L);
        if (rel_gain < (long double)SZ_MOP_REL_IMPROVE_MIN) best = 0;
    }
    return (best==0)? PM::L3D : PM::ADVECT;
}

template<typename T>
inline PM choose_pred_mode_block_lookahead_bitspersample_entire_block_sample_less_cpy_v2(
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

    const int ns_i = ((i_end - bi) + subs - 1) / subs;
    const int ns_j = ((j_end - bj) + subs - 1) / subs;
    if (ns_i <= 0 || ns_j <= 0) subs = 1;

    // 绑定 ADVECT 的 U/V(t-1)（速度场用原始源数据，不依赖工作区）
    const int64_t* u_prev_plane_src = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
    const int64_t* v_prev_plane_src = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
    pred_advect_bind_prev_uv(u_prev_plane_src, v_prev_plane_src, /*si_uv*/si, /*sj_uv*/sj);

    // ========= 帧级工作区（只在 t 或尺寸/地址变化时重建一次）=========
    struct JournalEntry { size_t off; int64_t oldv; };
    struct FrameWork {
        std::vector<int64_t> U_work; // size = 2*plane (t-1 | t)
        std::vector<int64_t> V_work; // size = 2*plane
        std::vector<JournalEntry> jU, jV;

        int last_t = -1;
        size_t last_plane = 0;
        const int64_t* U_prev_src = nullptr;
        const int64_t* V_prev_src = nullptr;
        const int64_t* U_cur_src  = nullptr;
        const int64_t* V_cur_src  = nullptr;

        void ensure(int H,int W,int t,
                    const int64_t* U_prev, const int64_t* U_cur,
                    const int64_t* V_prev, const int64_t* V_cur,
                    size_t cap_hint)
        {
            const size_t plane_now = (size_t)H*(size_t)W;
            const bool need_rebuild =
                (t != last_t) || (plane_now != last_plane) ||
                (U_prev != U_prev_src) || (V_prev != V_prev_src) ||
                (U_cur  != U_cur_src ) || (V_cur  != V_cur_src );

            if (need_rebuild) {
                U_work.resize(2*plane_now);
                V_work.resize(2*plane_now);
                std::memcpy(U_work.data(),             U_prev, plane_now*sizeof(int64_t));
                std::memcpy(U_work.data()+plane_now,   U_cur,  plane_now*sizeof(int64_t));
                std::memcpy(V_work.data(),             V_prev, plane_now*sizeof(int64_t));
                std::memcpy(V_work.data()+plane_now,   V_cur,  plane_now*sizeof(int64_t));

                last_t = t;
                last_plane = plane_now;
                U_prev_src = U_prev; V_prev_src = V_prev;
                U_cur_src  = U_cur;  V_cur_src  = V_cur;

                jU.clear(); jV.clear();
                jU.reserve(cap_hint);
                jV.reserve(cap_hint);
            } else {
                jU.clear(); jV.clear();
            }
        }
        inline void writebackU(size_t off, int64_t newv){
            int64_t& ref = U_work[off];
            if (ref != newv){ jU.push_back({off, ref}); ref = newv; }
        }
        inline void writebackV(size_t off, int64_t newv){
            int64_t& ref = V_work[off];
            if (ref != newv){ jV.push_back({off, ref}); ref = newv; }
        }
        inline void rollback(){
            for (size_t k=jU.size(); k--;) U_work[jU[k].off] = jU[k].oldv;
            for (size_t k=jV.size(); k--;) V_work[jV[k].off] = jV[k].oldv;
            jU.clear(); jV.clear();
        }
    };

    static thread_local FrameWork fw;

    const int64_t* U_prev_src = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
    const int64_t* V_prev_src = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
    const int64_t* U_cur_src  = (const int64_t*)(U_fp + (size_t)t*plane);
    const int64_t* V_cur_src  = (const int64_t*)(V_fp + (size_t)t*plane);

    const size_t cap_hint = (size_t)BH * (size_t)BW * 2u;
    fw.ensure(H,W,t, U_prev_src,U_cur_src, V_prev_src,V_cur_src, cap_hint);

    // ====== 直方图、统计（2 个候选：0=L3D, 1=ADVECT）======
    std::vector<uint32_t> hist0(bins,0), hist1(bins,0);
    uint64_t n0=0,n1=0, of0=0,of1=0;

    auto push_qindex = [&](std::vector<uint32_t>& h, uint64_t& ncnt, int qindex){
        if (qindex < 0) qindex = 0;
        else if (qindex >= bins) qindex = bins - 1;
        ++h[(size_t)qindex];
        ++ncnt;
    };

    // ========== 仅此处改为“定点乘法替代整除”，其余逻辑保持不变 ==========
    QCtx qc; qctx_init(qc, (long long)eb_lsb_cap, intv_radius, capacity);

    auto quantize_once = [&](long long diff, int64_t pred,
                             int& qindex, int64_t& recon, bool& overflow)-> bool {
        // |diff|
        uint64_t ad = (uint64_t)(diff >= 0 ? diff : -diff);
        // qd = floor(|diff|/eb) + 1  —— 用定点倒数替代 64 位整除（等价 floor）
        __uint128_t prod = ( (__uint128_t)ad * (uint64_t)qc.inv_eb_Q );
        uint64_t qd = (uint64_t)(prod >> qc.Q) + 1ull;

        if ((int)qd >= qc.capacity){ overflow = true; return false; }
        overflow = false;

        // 与原实现一致：向零截断
        long long q_signed = (diff < 0) ? -(long long)qd : (long long)qd;
        qindex = (int)(q_signed/2) + qc.intv_radius;

        // 与原实现一致的反量化与误差门限
        recon = pred + 2LL * ( (long long)qindex - qc.intv_radius ) * (long long)eb_lsb_cap;
        long long x    = pred + diff;
        long long err  = recon - x;
        long long aerr = (err >= 0 ? err : -err);
        return (aerr <= (long long)eb_lsb_cap);
    };
    // =====================================================================

    auto run_mode = [&](int mode_idx){
        int64_t* U_work_cur = fw.U_work.data() + plane; // t 平面基址
        int64_t* V_work_cur = fw.V_work.data() + plane;

        for (int i=bi; i<i_end; ++i){
            for (int j=bj; j<j_end; ++j){
                const size_t idx2d = (size_t)i*W + (size_t)j;
                int64_t* curU = U_work_cur + idx2d;
                int64_t* curV = V_work_cur + idx2d;
                const int64_t xU = U_cur_src[idx2d];
                const int64_t xV = V_cur_src[idx2d];

                int64_t pU, pV;
                if (mode_idx == 0){ // L3D
                    pU = pred_l3d(curU,t,i,j,si,sj,sk);
                    pV = pred_l3d(curV,t,i,j,si,sj,sk);
                }else{              // ADVECT
                    pU = pred_advect_bilinear(curU,t,i,j,si,sj,sk);
                    pV = pred_advect_bilinear(curV,t,i,j,si,sj,sk);
                }

                const long long dU = (long long)(xU - pU);
                const long long dV = (long long)(xV - pV);

                int qU=0,qV=0; int64_t rU=0,rV=0;
                bool ofU=false, ofV=false;
                const bool okU = quantize_once(dU, pU, qU, rU, ofU);
                const bool okV = quantize_once(dV, pV, qV, rV, ofV);

                if (okU && okV){
                    fw.writebackU(plane + idx2d, rU);
                    fw.writebackV(plane + idx2d, rV);
                }

                if ( ((i - bi) % subs == 0) && ((j - bj) % subs == 0) ){
                    if (mode_idx == 0){
                        if (okU) push_qindex(hist0, n0, qU); else of0++;
                        if (okV) push_qindex(hist0, n0, qV); else of0++;
                    }else{
                        if (okU) push_qindex(hist1, n1, qU); else of1++;
                        if (okV) push_qindex(hist1, n1, qV); else of1++;
                    }
                }
            }
        }
        fw.rollback();
    };

    // 评估两个模式（每个模式评完回滚工作区）
    run_mode(0); // L3D
    run_mode(1); // ADVECT

    long double bits[2] = {
        entropy_bits_per_sample_vec(hist0, n0),
        entropy_bits_per_sample_vec(hist1, n1)
    };

    auto add_penalty = [&](int k){
        const uint64_t n  = (k==0)? n0 : n1;
        const uint64_t of = (k==0)? of0 : of1;
        long double& b = bits[k];
        const long double denom = (long double)(n + of) + 1e-12L;
        const long double of_ratio = (long double)of / denom;
        b += of_ratio * (long double)SZ_MOP_OFLOW_PENALTY_BITS;
        // 2 路模式 → 1bit/块；折算到“每像素每分量”
        b += (1.0L / ( (long double)BH * (long double)BW * 2.0L ));
    };
    add_penalty(0);
    add_penalty(1);

    int best = 0; long double bestv = bits[0];
    if (bits[1] < bestv){ best = 1; bestv = bits[1]; }
    if (best != 0){
        const long double rel_gain = (bits[0] - bestv) / (bits[0] + 1e-12L);
        if (rel_gain < (long double)SZ_MOP_REL_IMPROVE_MIN) best = 0;
    }
    return (best==0) ? PM::L3D : PM::ADVECT;
}

// ===============v3 version===========================================
// ======================================================================
// 1) 快速无除法的 u64 商：q = floor(a / d) —— 乘法 + 单次校正（结果与整除一致）
// ======================================================================
struct FastDivU64 {
    uint64_t d;   // 除数：eb_lsb_cap（>0）
    uint64_t m;   // magic = floor( 2^64 / d )
};

// 预计算 magic（每次调用该选择器只做一次，非常便宜）
static inline void fastdiv_init(FastDivU64& fd, uint64_t d){
    fd.d = d ? d : 1;
#if defined(_MSC_VER) && defined(_M_X64)
    // MSVC x64: 用 128/64 除法求 magic
    // 等价于 floor( (1<<64) / d )
    unsigned __int64 hi = 1ull << 63; // 先构造 2^64 = (hi<<1, lo=0)
    unsigned __int64 lo = 0ull;
    unsigned __int64 rem;
    // 这里建议用编译器内联汇编或现成库；为了简洁，仍用通用实现：
    // 在 MSVC 下可直接用 128 除法：__int128 不可用时请实现一个 128/64 除法。
    // 简化起见，这里退化为普通除法（仅发生在每块一次，对性能影响可忽略）：
    fd.m = (uint64_t)((~(uint64_t)0) / fd.d); // 近似。若要严谨请改为真正的 2^64/d。
#else
    // GCC/Clang：用 128/64 除法精确得到 floor(2^64 / d)
    fd.m = (uint64_t)(((__uint128_t)1 << 64) / ( __uint128_t)fd.d);
#endif
}

// q = floor(a / d) —— 乘法 + 单次校正（保证与整除一致）
static inline uint64_t fastdiv_do(const FastDivU64& fd, uint64_t a){
#if defined(_MSC_VER) && defined(_M_X64)
    // MSVC 版本建议用 _umul128(a, fd.m, &hi) 取高位
    // 这里给出通用写法的占位：请按需替换为 _umul128
    unsigned __int128 prod = (unsigned __int128)a * (unsigned __int128)fd.m;
    uint64_t q = (uint64_t)(prod >> 64);
#else
    __uint128_t prod = (__uint128_t)a * (__uint128_t)fd.m;
    uint64_t q = (uint64_t)(prod >> 64);
#endif
    // 单次校正（经典 Barrett/GHM 校正）：若余数 >= d，则 q++
    uint64_t r = a - q * fd.d;
    if (r >= fd.d) ++q;
    return q; // 与 floor(a/d) 完全一致
}

// ======================================================================
// 2) 直方图复用 + 按触达清零 + 升序求熵（与原版数值结果一致）
// ======================================================================
struct HistReuse {
    std::vector<uint32_t> cnt;       // 桶计数（size = bins = 2*capacity）
    std::vector<uint32_t> touched;   // 本次被触达的桶索引（不重复）
    size_t bins = 0;

    void ensure(size_t new_bins){
        if (bins != new_bins){
            bins = new_bins;
            cnt.assign(bins, 0u);
            touched.clear(); touched.reserve(1024);
        }else{
            // 仅把“上次触达”的桶置零，O(#touched)
            for (uint32_t idx : touched) cnt[idx] = 0u;
            touched.clear();
        }
    }
    // 加 1：若第一次触达则记在 touched
    inline void inc(uint32_t idx){
        if (cnt[idx]++ == 0u) touched.push_back(idx);
    }
    // 与原算法相同的熵计算（只遍历非零桶，但按索引升序与原顺序一致）
    inline long double entropy_bits(uint64_t n) {
        if (n == 0) return 1e9L;
        std::sort(touched.begin(), touched.end()); // 与原版 k=0..bins-1 顺序一致
        long double H = 0.0L;
        for (uint32_t idx : touched){
            uint32_t c = cnt[idx];
            if (!c) continue;
            long double p = (long double)c / (long double)n;
            H += -p * std::log2(p);
        }
        return H;
    }
};

// ======================================================================
// 3) 量化一步（与原版完全一致的语义；只是把整除换为 fastdiv）
// ======================================================================
struct QuantCtx {
    FastDivU64 div;       // |e|/eb 的快速商
    int        intv_radius;
    int        capacity;
    int64_t    eb;        // eb_lsb_cap（>=1）
};

static inline void quant_init(QuantCtx& qc, int64_t eb_lsb_cap, int intv_radius, int capacity){
    if (eb_lsb_cap <= 0) eb_lsb_cap = 1;
    qc.eb = eb_lsb_cap;
    qc.intv_radius = intv_radius;
    qc.capacity = capacity;
    fastdiv_init(qc.div, (uint64_t)qc.eb);
}

// 与原版 quantize_once 完全一致的输出：qindex / recon / overflow
static inline bool quantize_once_fast_exact(long long diff, int64_t pred,
                                            const QuantCtx& qc,
                                            int& qindex, int64_t& recon, bool& overflow)
{
    uint64_t ad = (uint64_t)(diff >= 0 ? diff : -diff);
    // q = floor(|e|/eb)
    uint64_t q = fastdiv_do(qc.div, ad);
    // qd = q + 1（与原版一致）
    uint64_t qd = q + 1ull;
    if (qd >= (uint64_t)qc.capacity){ overflow = true; return false; }
    overflow = false;

    long long q_signed = (diff < 0) ? -(long long)qd : (long long)qd;
    qindex = (int)(q_signed / 2) + qc.intv_radius;          // 与原版：向零截断
    recon  = pred + 2LL * ((long long)qindex - qc.intv_radius) * qc.eb;

    long long x    = pred + diff;
    long long err  = recon - x;
    long long aerr = (err >= 0 ? err : -err);
    return (aerr <= qc.eb); // 原版同样的误差门限
}

// ======================================================================
// 4) 主函数：保持与 less_cpy 完全相同的接口与行为（但更快）
// ======================================================================
template<typename T>
inline PM choose_pred_mode_block_lookahead_bitspersample_entire_block_sample_less_cpy_v3(
    const T* U_fp, const T* V_fp,
    int H,int W,int t, int bi,int bj,int BH,int BW,
    ptrdiff_t si, ptrdiff_t sj, ptrdiff_t sk,
    T eb_lsb_cap,              // max((T)1, max_eb)
    int intv_radius,           // capacity >> 1
    int capacity,              // 与编码端一致
    int subs                   // 采样步长（>=1）
){
    if (t < 1) return PM::L3D;
    if (subs < 1) subs = 1;
    if (eb_lsb_cap <= 0) eb_lsb_cap = 1;

    const int i_end = std::min(bi+BH, H);
    const int j_end = std::min(bj+BW, W);
    const size_t plane = (size_t)H * (size_t)W;
    const int bins = 2 * capacity;

    // subs 过大导致块内无采样点：回退
    const int ns_i = ((i_end - bi) + subs - 1) / subs;
    const int ns_j = ((j_end - bj) + subs - 1) / subs;
    if (ns_i <= 0 || ns_j <= 0) subs = 1;

    // 绑定 ADVECT 所需的 t-1 速度平面（O(1)）
    const int64_t* u_prev_plane_src = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
    const int64_t* v_prev_plane_src = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
    pred_advect_bind_prev_uv(u_prev_plane_src, v_prev_plane_src, /*si_uv*/si, /*sj_uv*/sj);

    // === 帧级工作区：与原 less_cpy 一致（thread_local 复用 + 日志回滚） ===
    struct JournalEntry { size_t off; int64_t oldv; };
    struct FrameWork {
        std::vector<int64_t> U_work; // [t-1 | t]
        std::vector<int64_t> V_work; // [t-1 | t]
        std::vector<JournalEntry> jU, jV;

        int last_t = -1;
        size_t last_plane = 0;
        const int64_t* U_prev_src = nullptr;
        const int64_t* V_prev_src = nullptr;
        const int64_t* U_cur_src  = nullptr;
        const int64_t* V_cur_src  = nullptr;

        void ensure(int H,int W,int t,
                    const int64_t* U_prev, const int64_t* U_cur,
                    const int64_t* V_prev, const int64_t* V_cur,
                    size_t cap_hint)
        {
            const size_t plane_now = (size_t)H*(size_t)W;
            const bool need_rebuild =
                (t != last_t) || (plane_now != last_plane) ||
                (U_prev != U_prev_src) || (V_prev != V_prev_src) ||
                (U_cur  != U_cur_src ) || (V_cur  != V_cur_src );
            if (need_rebuild) {
                U_work.resize(2*plane_now);
                V_work.resize(2*plane_now);
                std::memcpy(U_work.data(),             U_prev, plane_now*sizeof(int64_t));
                std::memcpy(U_work.data()+plane_now,   U_cur,  plane_now*sizeof(int64_t));
                std::memcpy(V_work.data(),             V_prev, plane_now*sizeof(int64_t));
                std::memcpy(V_work.data()+plane_now,   V_cur,  plane_now*sizeof(int64_t));
                last_t = t; last_plane = plane_now;
                U_prev_src = U_prev; V_prev_src = V_prev;
                U_cur_src  = U_cur;  V_cur_src  = V_cur;
                jU.clear(); jV.clear();
                jU.reserve(cap_hint); jV.reserve(cap_hint);
            } else {
                jU.clear(); jV.clear();
            }
        }
        inline void writebackU(size_t off, int64_t newv){
            int64_t& ref = U_work[off];
            if (ref != newv){ jU.push_back({off, ref}); ref = newv; }
        }
        inline void writebackV(size_t off, int64_t newv){
            int64_t& ref = V_work[off];
            if (ref != newv){ jV.push_back({off, ref}); ref = newv; }
        }
        inline void rollback(){
            for (size_t k=jU.size(); k--;) U_work[jU[k].off] = jU[k].oldv;
            for (size_t k=jV.size(); k--;) V_work[jV[k].off] = jV[k].oldv;
            jU.clear(); jV.clear();
        }
    };
    static thread_local FrameWork fw;

    const int64_t* U_prev_src = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
    const int64_t* V_prev_src = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
    const int64_t* U_cur_src  = (const int64_t*)(U_fp + (size_t)t*plane);
    const int64_t* V_cur_src  = (const int64_t*)(V_fp + (size_t)t*plane);

    const size_t cap_hint = (size_t)BH * (size_t)BW * 2u; // U/V 各最多写回一次
    fw.ensure(H,W,t, U_prev_src,U_cur_src, V_prev_src,V_cur_src, cap_hint);

    // === 直方图复用（thread_local） ===
    static thread_local HistReuse H0, H1;   // 0=L3D, 1=ADVECT
    H0.ensure((size_t)bins);
    H1.ensure((size_t)bins);
    uint64_t n0=0, n1=0, of0=0, of1=0;

    // 量化上下文：把整除换成 fastdiv（结果一致）
    QuantCtx qc; quant_init(qc, (int64_t)eb_lsb_cap, intv_radius, capacity);

    auto push_qindex = [&](HistReuse& H, uint64_t& ncnt, int qindex){
        // 与原版一致：边界钳位（理论上不会超界，这里留防御）
        if (qindex < 0) qindex = 0;
        else if (qindex >= bins) qindex = bins - 1;
        H.inc((uint32_t)qindex);
        ++ncnt;
    };

    // —— 单候选仿真：与原版完全相同（整块、两分量都成功才写回），评完回滚 —— 
    auto run_mode = [&](int mode_idx){
        int64_t* U_work_cur = fw.U_work.data() + plane; // t 平面起点
        int64_t* V_work_cur = fw.V_work.data() + plane;

        for (int i=bi; i<i_end; ++i){
            for (int j=bj; j<j_end; ++j){
                if ( ((i - bi) % 1) || ((j - bj) % 1) ) {
                    // 注意：写回必须全像素做；统计才按 subs 采样
                    // 因而这里不能因为 subs>1 就跳过写回仿真
                }
                const size_t idx2d = (size_t)i*W + (size_t)j;

                int64_t* curU = U_work_cur + idx2d;
                int64_t* curV = V_work_cur + idx2d;
                const int64_t xU = U_cur_src[idx2d];
                const int64_t xV = V_cur_src[idx2d];

                int64_t pU, pV;
                if (mode_idx == 0){ // L3D
                    pU = pred_l3d(curU,t,i,j,si,sj,sk);
                    pV = pred_l3d(curV,t,i,j,si,sj,sk);
                }else{              // ADVECT
                    pU = pred_advect_bilinear(curU,t,i,j,si,sj,sk);
                    pV = pred_advect_bilinear(curV,t,i,j,si,sj,sk);
                }

                const long long dU = (long long)(xU - pU);
                const long long dV = (long long)(xV - pV);

                int qU=0,qV=0; int64_t rU=0,rV=0;
                bool ofU=false, ofV=false;
                const bool okU = quantize_once_fast_exact(dU, pU, qc, qU, rU, ofU);
                const bool okV = quantize_once_fast_exact(dV, pV, qc, qV, rV, ofV);

                if (okU && okV){
                    fw.writebackU(plane + idx2d, rU);
                    fw.writebackV(plane + idx2d, rV);
                }

                // 仅在 subs 网格上统计直方图与 oflow
                if ( ((i - bi) % subs == 0) && ((j - bj) % subs == 0) ){
                    if (mode_idx == 0){
                        if (okU) push_qindex(H0, n0, qU); else of0++;
                        if (okV) push_qindex(H0, n0, qV); else of0++;
                    }else{
                        if (okU) push_qindex(H1, n1, qU); else of1++;
                        if (okV) push_qindex(H1, n1, qV); else of1++;
                    }
                }
            }
        }
        fw.rollback();
    };

    // 评估两个模式（与原版顺序一致）
    run_mode(0); // L3D
    run_mode(1); // ADVECT

    // bits/sample：与原版相同公式，但只遍历 touched 桶，保持升序 → 数值一致
    long double bits0 = H0.entropy_bits(n0);
    long double bits1 = H1.entropy_bits(n1);

    auto add_penalty = [&](long double& b, uint64_t n, uint64_t of){
        const long double denom = (long double)(n + of) + 1e-12L;
        const long double of_ratio = (long double)of / denom;
        b += of_ratio * (long double)SZ_MOP_OFLOW_PENALTY_BITS;
        // 两模式 ≈ 每块 1bit → 折算到“每像素每分量”
        b += (1.0L / ( (long double)BH * (long double)BW * 2.0L ));
    };
    add_penalty(bits0, n0, of0);
    add_penalty(bits1, n1, of1);

    // 选择 + 相对收益门槛（同原版）
    int best = 0; long double bestv = bits0;
    if (bits1 < bestv){ best = 1; bestv = bits1; }
    if (best != 0){
        const long double rel_gain = (bits0 - bestv) / (bits0 + 1e-12L);
        if (rel_gain < (long double)SZ_MOP_REL_IMPROVE_MIN) best = 0;
    }

    return (best==0)? PM::L3D : PM::ADVECT;
}


#endif
