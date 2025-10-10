#pragma once

#include "sz_cp_preserve_sos_2p5d_common.hpp"
#include "sz_compression_utils.hpp"
#include "sz_def.hpp"
#include "sz_prediction.hpp"
#include "sz_lossless.hpp"
#include "utils.hpp"

#ifndef ZSTD_DETAIL
#define ZSTD_DETAIL 1
#endif



template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb,                  // 全局绝对误差上限（浮点域）：max_eb(fp)=max_pwr_eb
    EbMode mode 
){
  using T = int64_t;
  const Size3 sz{ (int)r1,(int)r2,(int)r3 };
  const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

  // 1) 定点化（先乘后 llround）
  T *U_fp=(T*)std::malloc(N*sizeof(T));
  T *V_fp=(T*)std::malloc(N*sizeof(T));
  if(!U_fp || !V_fp){
    if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp);
    compressed_size=0; return nullptr;
  }
  T range=0;
  T scale = convert_to_fixed_point<T_data,T>(U, V, N, U_fp, V_fp, range);

  // 2) 预计算：全局 CP 面集合（一次性）
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();


  #if CP_DEBUG_VISIT
  // 1.1 记录：编码端枚举/检查过的三角面
  std::unordered_set<FaceKeySZ, FaceKeySZHash> visited_faces;

  // 1.2 小工具：规范化存一下 (a,b,c)
  auto MARK_FACE = [&](size_t a, size_t b, size_t c){
      visited_faces.emplace(a,b,c); // FaceKeySZ 构造里已做排序/规范化
  };

  // 1.3 帮助打印：把 vid -> (t,i,j)
  auto decode_tij = [&](size_t v){
      int t = (int)(v / (H*W));
      size_t rem = v - (size_t)t * H * W;
      int i = (int)(rem / W);
      int j = (int)(rem % W);
      return std::tuple<int,int,int>(t,i,j);
  };

  // 1.4（可选）分类一下是哪类面
  auto classify_face = [&](size_t a, size_t b, size_t c){
      auto [ta,ia,ja] = decode_tij(a);
      auto [tb,ib,jb] = decode_tij(b);
      auto [tc,ic,jc] = decode_tij(c);
      int st = (ta==tb) + (ta==tc) + (tb==tc);
      if (ta==tb && tb==tc) {
          // 同一层
          bool has_diag = ((ia!=ib)||(ja!=jb)) && ((ia!=ic)||(ja!=jc)) && ((ib!=ic)||(jb!=jc));
          return has_diag ? "layer(tri)" : "layer(?)";
      } else {
          // 跨层
          int cnt_t0 = (ta==tb) + (ta==tc) + (ta==ta); // 粗略判断：有2点在同一层
          (void)cnt_t0;
          return "slab";
      }
  };
  #endif

  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // 3) 量化/编码缓冲
  int* eb_quant_index  = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index= (int*)std::malloc(2*N*sizeof(int)); // U/V 交错
  double enc_max_abs_eb_fp   = 0.0; //编码端自检（浮点）
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred; unpred.reserve((N / 4) * 2);

  // 参数与量化槽
  const int base = 2;                             // 仍写头，方便向后兼容
  const int capacity = 65536; //65536
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);

  // 把浮点域绝对误差上限转到定点域（LSB）
  // const T max_eb = (T) std::llround( (long double)max_pwr_eb * (long double)scale );

  T max_eb = 0;
  if(mode == EbMode::Relative){

    max_eb = max_pwr_eb * range; // 相对误差转绝对误差
    printf("Compression Using Relative Eb Mode!\n");
  }
  else if (mode == EbMode::Absolute){
    printf("Compression Using Absolute Eb Mode!\n");
    max_eb = max_pwr_eb * scale; // 浮点→定点
  }
  else{
    std::cerr << "Error: Unsupported EbMode!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  // T max_eb = range * max_pwr_eb;


  // 定点 LSB 的阈值（退化门控）
  const T threshold = 1;                          // 幂指数量化阈值（LSB）


  // 4) 逐顶点：枚举与该顶点相关的三角面 → 最小 eb
  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W; // 层间位移

  // 6 个平面邻接方向（左右、上下、主对角）
  const int di[6] = { 0,  0,  1, -1,  1, -1 }; //y-axis
  const int dj[6] = { 1, -1,  0,  0,  1, -1 }; //x-axis

  for (int t=0; t<(int)Tt; ++t){
    if(t % 100 == 0){
      printf("processing slice %d / %d\n", t, (int)Tt);
    }
    for (int i=0; i<(int)H; ++i){
      for (int j=0; j<(int)W; ++j){
        const size_t v = vid(t,i,j,sz);
        // 缓存原始定点值（预测/回写前）
        T *curU = U_fp + v;
        T *curV = V_fp + v;
        const T curU_val = *curU;
        const T curV_val = *curV;

        // —— 收集最小 eb——
        T required_eb = max_eb;

        // (A) 层内 t：影响 (i-1..i, j-1..j) 的 4 个 cell，每 cell 两三角
        for (int ci=i-1; ci<=i; ++ci){
          if (!in_range(ci, (int)H-1)) continue;
          for (int cj=j-1; cj<=j; ++cj){
            if (!in_range(cj, (int)W-1)) continue;

            size_t v00 = vid(t,ci,  cj,  sz);
            size_t v10 = vid(t,ci,  cj+1,sz); //v10=> x=1,y=0
            size_t v01 = vid(t,ci+1,cj,  sz);
            size_t v11 = vid(t,ci+1,cj+1,sz);

            // Upper: (v00,v01,v11)
            if (v==v00 || v==v01 || v==v11){ 
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v01,v11);
              #endif
              if (has_cp(cp_faces, v00,v01,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v01],U_fp[v00],
                                                    V_fp[v11],V_fp[v01],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
            // Lower: (v00,v10,v11)
            if (v==v00 || v==v10 || v==v11){
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v10,v11);
              #endif
              if (has_cp(cp_faces, v00,v10,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v10],U_fp[v00],
                                                    V_fp[v11],V_fp[v10],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }
        
        // (B1) 侧面 [t, t+1]
        if (t < (int)Tt-1){
          for (int k=0; k<6; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a  = vid(t, i, j, sz);
            size_t b  = vid(t, ni,nj, sz);
            size_t ap = a + dv, bp = b + dv;
            if (k == 0 || k==3 || k==5){
              // (a,b,bp) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (a,bp,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,bp,ap);
              #endif
              if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[bp],U_fp[a],
                                                        V_fp[ap],V_fp[bp],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,ap) for k = 1,2,4
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (b,ap,bp) for k = 1,2,4 //好像与点无关
              // if (has_cp(cp_faces, b,ap,bp)) { required_eb = 0; goto after_min_eb; }
              //   {
              //     T eb = derive_cp_abs_eb_sos_online<T>(U_fp[b],U_fp[ap],U_fp[bp],
              //                                           V_fp[b],V_fp[ap],V_fp[bp]);
              //     if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
              //     if (eb < required_eb) required_eb = eb;
              //   }
            }

          }
        }

        // (B2) 侧面 [t-1, t]
        if (t > 0){
          for (int k=0; k<6 ; ++k){
            int ni = i + di[k], nj = j + dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a = vid(t,i,j,sz); //v
            size_t b = vid(t,ni,nj,sz);
            size_t ap = a - dv, bp = b - dv; //ap,bp为上一层
            if (k == 0 || k==3 || k==5){
              // (a,b,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,bp) for k = 1,2,4 //侧面上三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
                
              // (a,ap,bp) for k = 1,2,4 //侧面下三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,ap,bp);
              #endif
              if (has_cp(cp_faces, a,ap,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[ap],U_fp[a],
                                                        V_fp[bp],V_fp[ap],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }

          }
        }

        // (C) 内部剖分面：两片 ts ∈ {t, t-1}；每相邻 cell 的 Upper/Lower 各 2 面
        // (C1) [t, t+1] 内部剖分面
        if (t < (int)Tt-1){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|
      
          //triange 1 [x,y,t] = [(0,0,0),(1,0,0),(0,-1,0)]: has 2 faces
          size_t f1a = vid(t,  i,  j,  sz);
          size_t f1b = vid(t,  i,j+1,  sz);
          size_t f1c = vid(t,  i-1,j,  sz);
          size_t f1ap = f1a + dv, f1bp = f1b + dv, f1cp = f1c + dv;
          if (in_range(i-1,(int)H) && in_range(j+1,(int)W)){
            // (f1a,f1cp,f1b)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1b);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1b)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1b],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1b],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f1a,f1cp,f1bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1bp);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1bp],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1bp],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          //triange 2 [x,y,t] = [(0,0,0),(0,-1,0),(-1,-1,0)]: has 2 faces
          size_t f2a = vid(t,  i,  j,  sz);
          size_t f2b = vid(t,  i-1,j,  sz);
          size_t f2c = vid(t,  i-1,j-1,sz);
          size_t f2ap = f2a + dv, f2bp = f2b + dv, f2cp = f2c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f2a,f2c,f2bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2c,f2bp);
            #endif
            if (has_cp(cp_faces, f2a,f2c,f2bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2bp],U_fp[f2c],U_fp[f2a],
                                                    V_fp[f2bp],V_fp[f2c],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f2a,f2bp,f2cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2bp,f2cp);
            #endif
            if (has_cp(cp_faces, f2a,f2bp,f2cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2cp],U_fp[f2bp],U_fp[f2a],
                                                    V_fp[f2cp],V_fp[f2bp],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          // triange 3 [x,y,t] = [(0,0,0),(-1,-1,0),(-1,0,0)]: has 1 faces
          size_t f3a = vid(t,  i,  j,  sz);
          size_t f3b = vid(t,  i-1,j-1,sz);
          size_t f3c = vid(t,  i,j-1,  sz);
          size_t f3ap = f3a + dv, f3bp = f3b + dv, f3cp = f3c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3bp,f3c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3bp,f3c);
            #endif
            if (has_cp(cp_faces, f3a,f3bp,f3c)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3c],U_fp[f3bp],U_fp[f3a],
                                                    V_fp[f3c],V_fp[f3bp],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // triange 4 0 faces
          // triange 5 0 faces
          // triange 6 [x,y,t] = [(0,0,0),(1,0,0),(1,1,0)]: has 1 faces
          size_t f6a = vid(t,  i,  j,  sz);
          size_t f6b = vid(t,  i,j+1,  sz);
          size_t f6c = vid(t,  i+1,j+1,sz);
          size_t f6ap = f6a + dv, f6bp = f6b + dv, f6cp = f6c + dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6bp,f6c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6bp,f6c);
            #endif
            if (has_cp(cp_faces, f6a,f6bp,f6c)) {
              required_eb=0; 
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6c],U_fp[f6bp],U_fp[f6a],
                                                    V_fp[f6c],V_fp[f6bp],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }


        // (C2) [t-1, t] 内部剖分面 并不与C1对称
        if (t > 0){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|

          //triange 1 0 face
          //triange 2 0 face
          //triange 3  has 1 faces
          size_t f3a = vid(t, i,  j,  sz);   
          size_t f3b = vid(t, i-1,j-1,sz);
          size_t f3c = vid(t, i,j-1,  sz);
          size_t f3ap = f3a - dv, f3bp = f3b - dv, f3cp = f3c - dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3b,f3cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3b,f3cp);
            #endif
            if (has_cp(cp_faces, f3a,f3b,f3cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3cp],U_fp[f3b],U_fp[f3a],
                                                    V_fp[f3cp],V_fp[f3b],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 4 has 2 faces
          size_t f4a = vid(t, i,  j,  sz);
          size_t f4b = vid(t, i,j-1,  sz);
          size_t f4c = vid(t,i+1,j,  sz);
          size_t f4ap = f4a - dv, f4bp = f4b - dv, f4cp = f4c - dv;
          if (in_range(i+1,(int)H) && in_range(j-1,(int)W)){
            // (f4a,f4bp,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4bp,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4bp,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4bp],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4bp],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f4a,f4b,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4b,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4b,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4b],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4b],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 5 has 2 faces
          size_t f5a = vid(t, i,  j,  sz);
          size_t f5b = vid(t, i+1,j,  sz);
          size_t f5c = vid(t, i+1,j+1,sz);
          size_t f5ap = f5a - dv, f5bp = f5b - dv, f5cp = f5c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f5a,f5bp,f5cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5bp,f5cp);
            #endif
            if (has_cp(cp_faces, f5a,f5bp,f5cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5cp],U_fp[f5bp],U_fp[f5a],
                                                    V_fp[f5cp],V_fp[f5bp],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f5a,f5c,f5bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5c,f5bp);
            #endif
            if (has_cp(cp_faces, f5a,f5c,f5bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5bp],U_fp[f5c],U_fp[f5a],
                                                    V_fp[f5bp],V_fp[f5c],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 6 has 1 faces
          size_t f6a = vid(t, i,  j,  sz);
          size_t f6b = vid(t, i,j+1,  sz);
          size_t f6c = vid(t, i+1,j+1,sz);
          size_t f6ap = f6a - dv, f6bp = f6b - dv, f6cp = f6c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6b,f6cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6b,f6cp);
            #endif
            if (has_cp(cp_faces, f6a,f6b,f6cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6cp],U_fp[f6b],U_fp[f6a],
                                                    V_fp[f6cp],V_fp[f6b],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }
        // {
        //   // 保证 |dec-orig| ≤ abs_eb < |orig| ⇒ 不会跨 0 翻符号
        //   T sign_guard = std::min(std::llabs(curU_val), std::llabs(curV_val));
        //   if (sign_guard > 0 && required_eb >= sign_guard){
        //       T max_eb_to_preserve_sign = sign_guard - 1; // 留 1 LSB 余量
        //       if (max_eb_to_preserve_sign < required_eb)
        //           required_eb = max_eb_to_preserve_sign;
        //   }
        // }
        T abs_eb = required_eb;
        int id = eb_exponential_quantize(abs_eb,base,log_of_base,threshold);
        
        if (abs_eb == 0 )
        {
          *(eb_pos++) = 0;            // ebid=0 → 无损点
          unpred.push_back(U[v]);     // 原始浮点（解码端直接回填）
          unpred.push_back(V[v]);
          continue;
        }

        // ===== 量化 eb（配套的新幂指数量化）并编码该顶点 =====
        {
          // T abs_eb = required_eb;
          // // int id = eb_exponential_quantize_new(abs_eb, threshold); // id>=1，abs_eb 替换为代表值
          
          // // *eb_pos = id;
          // int id = eb_exponential_quantize(abs_eb,base,log_of_base,threshold);
          *eb_pos = id;

          bool unpred_flag=false;
          T dec[2];
          T abs_err_fp_q[2] = {0,0};

          for (int p=0; p<2; ++p){
            T *cur  = (p==0)? curU : curV;
            T  curv = (p==0) ? curU_val : curV_val;

            // 3D 一阶 Lorenzo（时间最慢）
            T d0 = (t&&i&&j)? cur[-sk - si - sj] : 0;
            T d1 = (t&&i)   ? cur[-sk - si]      : 0;
            T d2 = (t&&j)   ? cur[-sk - sj]      : 0;
            T d3 = (t)      ? cur[-sk]           : 0;
            T d4 = (i&&j)   ? cur[-si - sj]      : 0;
            T d5 = (i)      ? cur[-si]           : 0;
            T d6 = (j)      ? cur[-sj]           : 0;
            T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

            T diff = curv - pred;
            T qd = (std::llabs(diff)/abs_eb) + 1;
            if (qd < capacity){
              qd = (diff > 0) ? qd : -qd;
              int qindex = (int)(qd/2) + intv_radius;
              dq_pos[p] = qindex;
              dec[p] = pred + 2*(qindex - intv_radius)*abs_eb;

              // 守门：必须用 *代表值* abs_eb 校验
              if (std::llabs(dec[p] - curv) > abs_eb){
              // if (std::llabs(dec[p] - curv) > required_eb){
                unpred_flag = true; break;
              }
              abs_err_fp_q[p] = std::llabs(dec[p] - curv);
            }else{
              unpred_flag = true; break;
            }
          }

          if (unpred_flag){
            *(eb_pos++) = 0;                  // 改回无损
            unpred.push_back(U[v]);
            unpred.push_back(V[v]);
          }else{
            ++eb_pos;
            dq_pos += 2;
            *curU = dec[0];
            *curV = dec[1];

            // 编码端自检（浮点域）
            double abs_eb_fp = (double)abs_eb / (double)scale;
            double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
            double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
            enc_max_abs_eb_fp   = std::max(enc_max_abs_eb_fp, abs_eb_fp);
            enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
          }
        }
      }
    }
  }

  // std::cerr << "[ENC] max abs_eb(fp) = " << enc_max_abs_eb_fp
  //           << ", max actual |err|(fp) = " << enc_max_real_err_fp << "\n";

  
  #if CP_DEBUG_VISIT
  {
      size_t miss = 0, shown = 0;
      for (const auto &fk : cp_faces){
          if (!visited_faces.count(fk)){
              ++miss;
              if (shown < 50){ // 只演示前 50 个，避免刷屏
                  size_t a = fk.v[0], b = fk.v[1], c = fk.v[2];
                  auto [ta,ia,ja] = decode_tij(a);
                  auto [tb,ib,jb] = decode_tij(b);
                  auto [tc,ic,jc] = decode_tij(c);
                  fprintf(stderr,
                    "[MISSING] face (%zu,%zu,%zu)  "
                    "A(t=%d,i=%d,j=%d)  B(t=%d,i=%d,j=%d)  C(t=%d,i=%d,j=%d)  type=%s\n",
                    a,b,c, ta,ia,ja, tb,ib,jb, tc,ic,jc,
                    classify_face(a,b,c));
                  ++shown;
              }
          }
      }
      fprintf(stderr, "[COVERAGE] encoder touched faces = %zu, cp_faces = %zu, missing = %zu\n",
              visited_faces.size(), cp_faces.size(), miss);
  }
  #endif
  // 5) 打包码流
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);
  std::cout << "write intv_radius = " << intv_radius << "\n";

  size_t unpred_cnt = unpred.size();
  std::cout << "write unpred cnt = " << unpred_cnt << ",ratio=" << (double)unpred_cnt/(2*N) << "\n";
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);
  // 打印求和
  {
      double unpred_sum = 0.0;
      for (size_t i = 0; i < unpred_cnt; ++i) unpred_sum += unpred[i];
      printf("unpred sum = %.6f\n", unpred_sum);
  }
  #if ZSTD_DETAIL
  {
    // use zstd, calculate size after zstd
    unsigned char * unpred_after_zstd = NULL;
    unsigned long unpred_bytes = unpred_cnt * sizeof(unpred[0]);
    size_t unpred_zstd_size = sz_lossless_compress(ZSTD_COMPRESSOR, 3, reinterpret_cast<unsigned char*>(unpred.data()),unpred_bytes,&unpred_after_zstd);
    double ratio = static_cast<double>(unpred_bytes) / unpred_zstd_size;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "ZSTD(unpred): " << unpred_zstd_size
              << " bytes (orig " << unpred_bytes
              << ", ratio = " << ratio << "x)\n";
  }
  #endif


  unsigned char *pos_before_ebq = pos;
  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::cout << "Huffman eb size = " << (size_t)(pos - pos_before_ebq) << "\n";
  #if ZSTD_DETAIL
  {
    unsigned char* ebq_huf_ptr = pos_before_ebq;
    size_t ebq_huf_size = (size_t)(pos - pos_before_ebq);

    if (ebq_huf_size > 0) {
        unsigned char* ebq_zstd = nullptr;
        unsigned long ebq_zstd_size = sz_lossless_compress(
            ZSTD_COMPRESSOR, 3,
            ebq_huf_ptr,
            (unsigned long)ebq_huf_size,
            &ebq_zstd
        );
        if (ZSTD_isError(ebq_zstd_size) || ebq_zstd_size == 0) {
            std::cerr << "ZSTD compress ebq(Huffman) error: "
                      << ZSTD_getErrorName(ebq_zstd_size) << "\n";
        } else {
            double ratio = (double)ebq_huf_size / (double)ebq_zstd_size;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "ZSTD(ebq after Huffman): " << ebq_zstd_size
                      << " bytes (orig " << ebq_huf_size
                      << ", ratio = " << ratio << "x)\n";
        }
        std::free(ebq_zstd);
    }
  }
  #endif
  std::free(eb_quant_index);

  unsigned char *pos_before_dq = pos;
  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  printf("write dq num = %zu\n", data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  //print size of huffman
  std::cout << "Huffman dq size = " << (size_t)(pos - pos_before_dq) << "\n";
  #if ZSTD_DETAIL
  // —— 对 Huffman 后的 dq 字节块做 ZSTD ——
  {
    unsigned char* dq_huf_ptr = pos_before_dq;
    size_t dq_huf_size = (size_t)(pos - pos_before_dq);

    if (dq_huf_size > 0) {
        unsigned char* dq_zstd = nullptr;
        unsigned long dq_zstd_size = sz_lossless_compress(
            ZSTD_COMPRESSOR, 3,
            dq_huf_ptr,
            (unsigned long)dq_huf_size,
            &dq_zstd
        );
        if (ZSTD_isError(dq_zstd_size) || dq_zstd_size == 0) {
            std::cerr << "ZSTD compress dq(Huffman) error: "
                      << ZSTD_getErrorName(dq_zstd_size) << "\n";
        } else {
            double ratio = (double)dq_huf_size / (double)dq_zstd_size;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "ZSTD(dq after Huffman): " << dq_zstd_size
                      << " bytes (orig " << dq_huf_size
                      << ", ratio = " << ratio << "x)\n";
        }
        std::free(dq_zstd);

        //再测试一下直接dq用0-rle的前后压缩比
        size_t original_byte = data_quant_num*sizeof(int);
        size_t encoded_length = 0;
        int* encoded = zero_rle_encode(data_quant_index, data_quant_num, encoded_length);
        //int* encoded = tri_rle_encode(data_quant_index, data_quant_num, encoded_length);
        size_t compressed_byte = encoded_length*sizeof(int);
        printf("zero-rle ratio: %f\n",original_byte * 1.0 / compressed_byte);
        free(encoded);
        //把huffman前后的文件写出来
        writefile("/project/xli281_uksr/mxia/tmp_output/data_quant_index_default.raw", (unsigned char*)data_quant_index, data_quant_num*sizeof(int));
        writefile("/project/xli281_uksr/mxia/tmp_output/data_quant_index_default.huf", pos_before_dq, (size_t)(pos - pos_before_dq));
        printf("write data_quant_index_default.raw & data_quant_index_default.huf done....\n");
    }
  }
  #endif
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  std::free(U_fp); std::free(V_fp);
  return compressed;
}


template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  using T = int64_t;
  const Size3 sz{(int)r1, (int)r2, (int)r3};
  const size_t H = r1, W = r2, Tt = r3;
  const size_t N = H * W * Tt;
  const size_t dv = H * W;

  compressed_size = 0;
  if (!U || !V || N == 0) {
    return nullptr;
  }

  T* U_fp = (T*)std::malloc(N * sizeof(T));
  T* V_fp = (T*)std::malloc(N * sizeof(T));
  if (!U_fp || !V_fp) {
    if (U_fp) std::free(U_fp);
    if (V_fp) std::free(V_fp);
    return nullptr;
  }

  T range = 0;
  T scale = convert_to_fixed_point<T_data, T>(U, V, N, U_fp, V_fp, range);

  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  std::vector<int> eb_id(N, 0);
  std::vector<uint8_t> pred_mask(N, 0);
  std::vector<int> dq0(N, 0);
  std::vector<int> dq1(N, 0);

  const int base = 2;
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);
  const T threshold = 1;

  T max_eb = 0;
  if (mode == EbMode::Relative) {
    max_eb = max_pwr_eb * range;
    printf("Compression Using Relative Eb Mode!\n");
  } else if (mode == EbMode::Absolute) {
    max_eb = max_pwr_eb * scale;
    printf("Compression Using Absolute Eb Mode!\n");
  } else {
    std::cerr << "Error: Unsupported EbMode!\n";
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }
  
  const int BlockW = 50;//32
  const int BlockH = 50;//32
  const int BlockT = 200;//16
  const int blocks_t = (int)((Tt + BlockT - 1) / BlockT);
  const int blocks_i = (int)((H + BlockH - 1) / BlockH);
  const int blocks_j = (int)((W + BlockW - 1) / BlockW);

  double enc_max_abs_eb_fp = 0.0;
  double enc_max_real_err_fp = 0.0;

#ifdef _OPENMP
// #  pragma omp parallel for collapse(3) schedule(dynamic) reduction(max:enc_max_abs_eb_fp) reduction(max:enc_max_real_err_fp)
#  pragma omp parallel for collapse(3) schedule(dynamic)
#endif
  for (int bt_idx = 0; bt_idx < blocks_t; ++bt_idx) {
    for (int bi_idx = 0; bi_idx < blocks_i; ++bi_idx) {
      for (int bj_idx = 0; bj_idx < blocks_j; ++bj_idx) {
        const int t0 = bt_idx * BlockT;
        const int t1 = std::min(t0 + BlockT, (int)Tt);
        const int i0 = bi_idx * BlockH;
        const int i1 = std::min(i0 + BlockH, (int)H);
        const int j0 = bj_idx * BlockW;
        const int j1 = std::min(j0 + BlockW, (int)W);
        if (t0 >= t1 || i0 >= i1 || j0 >= j1) continue;

        const int ht0 = std::max(0, t0 - 1);
        const int ht1 = std::min((int)Tt, t1 + 1);
        const int hi0 = std::max(0, i0 - 1);
        const int hi1 = std::min((int)H, i1 + 1);
        const int hj0 = std::max(0, j0 - 1);
        const int hj1 = std::min((int)W, j1 + 1);

        const int lTdim = ht1 - ht0;
        const int lHdim = hi1 - hi0;
        const int lWdim = hj1 - hj0;
        if (lTdim <= 0 || lHdim <= 0 || lWdim <= 0) continue;

        const size_t local_size = (size_t)lTdim * (size_t)lHdim * (size_t)lWdim;
        std::vector<T> U_blk(local_size);
        std::vector<T> V_blk(local_size);

        auto local_index = [&](int t, int i, int j) -> size_t {
          return ((size_t)(t - ht0) * (size_t)lHdim + (size_t)(i - hi0)) * (size_t)lWdim + (size_t)(j - hj0);
        };

        auto local_index_from_vid = [&](size_t v) -> size_t {
          int lt, li, lj;
          inv_vid(v, (int)H, (int)W, lt, li, lj);
          return local_index(lt, li, lj);
        };

        auto readU = [&](size_t v) -> T {
          return U_blk[local_index_from_vid(v)];
        };
        auto readV = [&](size_t v) -> T {
          return V_blk[local_index_from_vid(v)];
        };

        for (int gt = ht0; gt < ht1; ++gt) {
          for (int gi = hi0; gi < hi1; ++gi) {
            for (int gj = hj0; gj < hj1; ++gj) {
              size_t gvid = vid(gt, gi, gj, sz);
              size_t lvid = local_index(gt, gi, gj);
              U_blk[lvid] = U_fp[gvid];
              V_blk[lvid] = V_fp[gvid];
            }
          }
        }

        const ptrdiff_t lsi = (ptrdiff_t)lWdim;
        const ptrdiff_t lsj = 1;
        const ptrdiff_t lsk = (ptrdiff_t)lHdim * (ptrdiff_t)lWdim;

        for (int gt = t0; gt < t1; ++gt) {
          for (int gi = i0; gi < i1; ++gi) {
            for (int gj = j0; gj < j1; ++gj) {
              const size_t gvid = vid(gt, gi, gj, sz);
              const size_t loff = local_index(gt, gi, gj);
              T* curU = U_blk.data() + loff;
              T* curV = V_blk.data() + loff;

              const bool is_boundary = (gt == t0) || (gt == t1 - 1) ||
                                       (gi == i0) || (gi == i1 - 1) ||
                                       (gj == j0) || (gj == j1 - 1);
              if (is_boundary) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              const T curU_val = *curU;
              const T curV_val = *curV;
              T required_eb = max_eb;

              for (int ci = gi - 1; ci <= gi; ++ci) {
                if (!in_range(ci, (int)H - 1)) continue;
                for (int cj = gj - 1; cj <= gj; ++cj) {
                  if (!in_range(cj, (int)W - 1)) continue;

                  size_t v00 = vid(gt, ci, cj, sz);
                  size_t v10 = vid(gt, ci, cj + 1, sz);
                  size_t v01 = vid(gt, ci + 1, cj, sz);
                  size_t v11 = vid(gt, ci + 1, cj + 1, sz);

                  if (gvid == v00 || gvid == v01 || gvid == v11) {
                    if (has_cp(cp_faces, v00, v01, v11)) {
                      required_eb = 0;
                    }
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(v11), readU(v01), readU(v00),
                        readV(v11), readV(v01), readV(v00));
                    if (eb < required_eb) required_eb = eb;
                  }

                  if (gvid == v00 || gvid == v10 || gvid == v11) {
                    if (has_cp(cp_faces, v00, v10, v11)) {
                      required_eb = 0;
                    }
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(v11), readU(v10), readU(v00),
                        readV(v11), readV(v10), readV(v00));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              const int di[6] = {0, 0, 1, -1, 1, -1};
              const int dj[6] = {1, -1, 0, 0, 1, -1};

              if (gt < (int)Tt - 1) {
                for (int k = 0; k < 6; ++k) {
                  int ni = gi + di[k];
                  int nj = gj + dj[k];
                  if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
                  size_t a = gvid;
                  size_t b = vid(gt, ni, nj, sz);
                  size_t ap = a + dv;
                  size_t bp = b + dv;
                  if (k == 0 || k == 3 || k == 5) {
                    if (has_cp(cp_faces, a, b, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(b), readU(a),
                          readV(bp), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                    if (has_cp(cp_faces, a, bp, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(bp), readU(a),
                          readV(ap), readV(bp), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  } else {
                    if (has_cp(cp_faces, a, b, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(b), readU(a),
                          readV(ap), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  }
                }
              }

              if (gt > 0) {
                for (int k = 0; k < 6; ++k) {
                  int ni = gi + di[k];
                  int nj = gj + dj[k];
                  if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
                  size_t a = gvid;
                  size_t b = vid(gt, ni, nj, sz);
                  size_t ap = a - dv;
                  size_t bp = b - dv;
                  if (k == 0 || k == 3 || k == 5) {
                    if (has_cp(cp_faces, a, b, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(b), readU(a),
                          readV(ap), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  } else {
                    if (has_cp(cp_faces, a, b, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(b), readU(a),
                          readV(bp), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                    if (has_cp(cp_faces, a, ap, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(ap), readU(a),
                          readV(bp), readV(ap), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  }
                }
              }

              if (gt < (int)Tt - 1) {
                if (in_range(gi - 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f1a = gvid;
                  size_t f1b = vid(gt, gi, gj + 1, sz);
                  size_t f1c = vid(gt, gi - 1, gj, sz);
                  size_t f1bp = f1b + dv;
                  size_t f1cp = f1c + dv;
                  if (has_cp(cp_faces, f1a, f1b, f1cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f1b), readU(f1cp), readU(f1a),
                        readV(f1b), readV(f1cp), readV(f1a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f1a, f1bp, f1cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f1bp), readU(f1cp), readU(f1a),
                        readV(f1bp), readV(f1cp), readV(f1a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f2a = gvid;
                  size_t f2b = vid(gt, gi - 1, gj, sz);
                  size_t f2c = vid(gt, gi - 1, gj - 1, sz);
                  size_t f2bp = f2b + dv;
                  size_t f2cp = f2c + dv;
                  if (has_cp(cp_faces, f2a, f2bp, f2cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f2bp), readU(f2c), readU(f2a),
                        readV(f2bp), readV(f2c), readV(f2a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f2a, f2cp, f2bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f2cp), readU(f2bp), readU(f2a),
                        readV(f2cp), readV(f2bp), readV(f2a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f3a = gvid;
                  size_t f3b = vid(gt, gi - 1, gj - 1, sz);
                  size_t f3c = vid(gt, gi, gj - 1, sz);
                  size_t f3bp = f3b + dv;
                  size_t f3cp = f3c + dv;
                  if (has_cp(cp_faces, f3a, f3b, f3bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f3c), readU(f3bp), readU(f3a),
                        readV(f3c), readV(f3bp), readV(f3a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f6a = gvid;
                  size_t f6b = vid(gt, gi, gj + 1, sz);
                  size_t f6c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f6bp = f6b + dv;
                  if (has_cp(cp_faces, f6a, f6bp, f6c)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f6c), readU(f6bp), readU(f6a),
                        readV(f6c), readV(f6bp), readV(f6a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              if (gt > 0) {
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f3a = gvid;
                  size_t f3b = vid(gt, gi - 1, gj - 1, sz);
                  size_t f3c = vid(gt, gi, gj - 1, sz);
                  size_t f3bp = f3b - dv;
                  size_t f3cp = f3c - dv;
                  if (has_cp(cp_faces, f3a, f3b, f3cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f3cp), readU(f3b), readU(f3a),
                        readV(f3cp), readV(f3b), readV(f3a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f4a = gvid;
                  size_t f4b = vid(gt, gi, gj - 1, sz);
                  size_t f4c = vid(gt, gi + 1, gj, sz);
                  size_t f4bp = f4b - dv;
                  size_t f4cp = f4c - dv;
                  if (has_cp(cp_faces, f4a, f4bp, f4cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f4cp), readU(f4bp), readU(f4a),
                        readV(f4cp), readV(f4bp), readV(f4a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f4a, f4b, f4cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f4cp), readU(f4b), readU(f4a),
                        readV(f4cp), readV(f4b), readV(f4a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f5a = gvid;
                  size_t f5b = vid(gt, gi + 1, gj, sz);
                  size_t f5c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f5bp = f5b - dv;
                  size_t f5cp = f5c - dv;
                  if (has_cp(cp_faces, f5a, f5bp, f5cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f5cp), readU(f5bp), readU(f5a),
                        readV(f5cp), readV(f5bp), readV(f5a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f5a, f5c, f5bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f5bp), readU(f5c), readU(f5a),
                        readV(f5bp), readV(f5c), readV(f5a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f6a = gvid;
                  size_t f6b = vid(gt, gi, gj + 1, sz);
                  size_t f6c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f6cp = f6c - dv;
                  if (has_cp(cp_faces, f6a, f6b, f6cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f6cp), readU(f6b), readU(f6a),
                        readV(f6cp), readV(f6b), readV(f6a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              T abs_eb = required_eb;
              int id = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);

              if (abs_eb == 0) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              bool unpred_flag = false;
              T dec[2] = {0, 0};
              T abs_err_fp_q[2] = {0, 0};
              for (int p = 0; p < 2; ++p) {
                T* cur = (p == 0) ? curU : curV;
                T curv = (p == 0) ? curU_val : curV_val;

                T d0 = (gt && gi && gj) ? cur[-lsk - lsi - lsj] : 0;
                T d1 = (gt && gi) ? cur[-lsk - lsi] : 0;
                T d2 = (gt && gj) ? cur[-lsk - lsj] : 0;
                T d3 = (gt) ? cur[-lsk] : 0;
                T d4 = (gi && gj) ? cur[-lsi - lsj] : 0;
                T d5 = (gi) ? cur[-lsi] : 0;
                T d6 = (gj) ? cur[-lsj] : 0;
                T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

                T diff = curv - pred;
                T qd = (std::llabs(diff) / abs_eb) + 1;
                if (qd < capacity) {
                  qd = (diff > 0) ? qd : -qd;
                  int qindex = (int)(qd / 2) + intv_radius;
                  if (p == 0) dq0[gvid] = qindex; else dq1[gvid] = qindex;
                  dec[p] = pred + 2 * (qindex - intv_radius) * abs_eb;
                  if (std::llabs(dec[p] - curv) > abs_eb) {
                    unpred_flag = true;
                    break;
                  }
                  abs_err_fp_q[p] = std::llabs(dec[p] - curv);
                } else {
                  unpred_flag = true;
                  break;
                }
              }

              if (unpred_flag) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              eb_id[gvid] = id;
              pred_mask[gvid] = 1;
              *curU = dec[0];
              *curV = dec[1];

              double abs_eb_fp = (double)abs_eb / (double)scale;
              double err_u_fp = (double)abs_err_fp_q[0] / (double)scale;
              double err_v_fp = (double)abs_err_fp_q[1] / (double)scale;
              // enc_max_abs_eb_fp = std::max(enc_max_abs_eb_fp, abs_eb_fp);
              // enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
            }
          }
        }
      }
    }
  }

  std::vector<T_data> unpred;
  unpred.reserve((N / 4) * 2);
  std::vector<int> dq_stream;
  dq_stream.reserve(2 * N);

  for (size_t v = 0; v < N; ++v) {
    if (pred_mask[v]) {
      dq_stream.push_back(dq0[v]);
      dq_stream.push_back(dq1[v]);
    } else {
      unpred.push_back(U[v]);
      unpred.push_back(V[v]);
    }
  }

  unsigned char* compressed = (unsigned char*)std::malloc((size_t)(2 * N * sizeof(T)));
  if (!compressed) {
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }
  unsigned char* pos = compressed;

  write_variable_to_dst(pos, scale);
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);
  std::cout << "write intv_radius = " << intv_radius << "\n";

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) {
    write_array_to_dst(pos, unpred.data(), unpred_cnt);
  }

  size_t eb_quant_num = N;
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * 1024, eb_id.data(), eb_quant_num, pos);

  size_t data_quant_num = dq_stream.size();
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * capacity, dq_stream.data(), data_quant_num, pos);

  compressed_size = (size_t)(pos - compressed);

  std::free(U_fp);
  std::free(V_fp);
  return compressed;
}

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v2(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  using T = int64_t;
  const Size3 sz{(int)r1, (int)r2, (int)r3};
  const size_t H = r1, W = r2, Tt = r3;
  const size_t N = H * W * Tt;
  const size_t dv = H * W;

  compressed_size = 0;
  if (!U || !V || N == 0) {
    return nullptr;
  }

  T* U_fp = (T*)std::malloc(N * sizeof(T));
  T* V_fp = (T*)std::malloc(N * sizeof(T));
  if (!U_fp || !V_fp) {
    if (U_fp) std::free(U_fp);
    if (V_fp) std::free(V_fp);
    return nullptr;
  }

  T range = 0;
  T scale = convert_to_fixed_point<T_data, T>(U, V, N, U_fp, V_fp, range);

  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  std::vector<int> eb_id(N, 0);
  std::vector<uint8_t> pred_mask(N, 0);
  std::vector<int> dq0(N, 0);
  std::vector<int> dq1(N, 0);

  const int base = 2;
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);
  const T threshold = 1;

  T max_eb = 0;
  if (mode == EbMode::Relative) {
    max_eb = max_pwr_eb * range;
    printf("Compression Using Relative Eb Mode!\n");
  } else if (mode == EbMode::Absolute) {
    max_eb = max_pwr_eb * scale;
    printf("Compression Using Absolute Eb Mode!\n");
  } else {
    std::cerr << "Error: Unsupported EbMode!\n";
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }
  
  const int BlockW = 50;//32
  const int BlockH = 50;//32
  const int BlockT = 200;//16
  const int blocks_t = (int)((Tt + BlockT - 1) / BlockT);
  const int blocks_i = (int)((H + BlockH - 1) / BlockH);
  const int blocks_j = (int)((W + BlockW - 1) / BlockW);

  double enc_max_abs_eb_fp = 0.0;
  double enc_max_real_err_fp = 0.0;

#ifdef _OPENMP
// #  pragma omp parallel for collapse(3) schedule(dynamic) reduction(max:enc_max_abs_eb_fp) reduction(max:enc_max_real_err_fp)
#  pragma omp parallel for collapse(3) schedule(dynamic)
#endif
  for (int bt_idx = 0; bt_idx < blocks_t; ++bt_idx) {
    for (int bi_idx = 0; bi_idx < blocks_i; ++bi_idx) {
      for (int bj_idx = 0; bj_idx < blocks_j; ++bj_idx) {
        const int t0 = bt_idx * BlockT;
        const int t1 = std::min(t0 + BlockT, (int)Tt);
        const int i0 = bi_idx * BlockH;
        const int i1 = std::min(i0 + BlockH, (int)H);
        const int j0 = bj_idx * BlockW;
        const int j1 = std::min(j0 + BlockW, (int)W);
        if (t0 >= t1 || i0 >= i1 || j0 >= j1) continue;

        const int ht0 = std::max(0, t0 - 1);
        const int ht1 = std::min((int)Tt, t1 + 1);
        const int hi0 = std::max(0, i0 - 1);
        const int hi1 = std::min((int)H, i1 + 1);
        const int hj0 = std::max(0, j0 - 1);
        const int hj1 = std::min((int)W, j1 + 1);

        const int lTdim = ht1 - ht0;
        const int lHdim = hi1 - hi0;
        const int lWdim = hj1 - hj0;
        if (lTdim <= 0 || lHdim <= 0 || lWdim <= 0) continue;

        const size_t local_size = (size_t)lTdim * (size_t)lHdim * (size_t)lWdim;
        std::vector<T> U_blk(local_size);
        std::vector<T> V_blk(local_size);

        auto local_index = [&](int t, int i, int j) -> size_t {
          return ((size_t)(t - ht0) * (size_t)lHdim + (size_t)(i - hi0)) * (size_t)lWdim + (size_t)(j - hj0);
        };

        auto local_index_from_vid = [&](size_t v) -> size_t {
          int lt, li, lj;
          inv_vid(v, (int)H, (int)W, lt, li, lj);
          return local_index(lt, li, lj);
        };

        auto readU = [&](size_t v) -> T {
          return U_blk[local_index_from_vid(v)];
        };
        auto readV = [&](size_t v) -> T {
          return V_blk[local_index_from_vid(v)];
        };

        for (int gt = ht0; gt < ht1; ++gt) {
          for (int gi = hi0; gi < hi1; ++gi) {
            for (int gj = hj0; gj < hj1; ++gj) {
              size_t gvid = vid(gt, gi, gj, sz);
              size_t lvid = local_index(gt, gi, gj);
              U_blk[lvid] = U_fp[gvid];
              V_blk[lvid] = V_fp[gvid];
            }
          }
        }

        const ptrdiff_t lsi = (ptrdiff_t)lWdim;
        const ptrdiff_t lsj = 1;
        const ptrdiff_t lsk = (ptrdiff_t)lHdim * (ptrdiff_t)lWdim;

        const int core_height = i1 - i0;
        const int core_width  = j1 - j0;

        for (int gt = t0; gt < t1; ++gt) {
          for (int diag = 0; diag <= (core_height - 1) + (core_width - 1); ++diag) {
            for (int gi = i0; gi < i1; ++gi) {
              int gj = diag - (gi - i0) + j0;
              if (gj < j0 || gj >= j1) {
                continue;
              }

              const size_t gvid = vid(gt, gi, gj, sz);
              const bool is_boundary = (gt == t0) || (gt == t1 - 1) ||
                                       (gi == i0) || (gi == i1 - 1) ||
                                       (gj == j0) || (gj == j1 - 1);
              if (is_boundary) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              const size_t loff = local_index(gt, gi, gj);
              T* curU = U_blk.data() + loff;
              T* curV = V_blk.data() + loff;
              const T curU_val = *curU;
              const T curV_val = *curV;
              T required_eb = max_eb;

              for (int ci = gi - 1; ci <= gi; ++ci) {
                if (!in_range(ci, (int)H - 1)) continue;
                for (int cj = gj - 1; cj <= gj; ++cj) {
                  if (!in_range(cj, (int)W - 1)) continue;

                  size_t v00 = vid(gt, ci, cj, sz);
                  size_t v10 = vid(gt, ci, cj + 1, sz);
                  size_t v01 = vid(gt, ci + 1, cj, sz);
                  size_t v11 = vid(gt, ci + 1, cj + 1, sz);

                  if (gvid == v00 || gvid == v01 || gvid == v11) {
                    if (has_cp(cp_faces, v00, v01, v11)) {
                      required_eb = 0;
                    }
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(v11), readU(v01), readU(v00),
                        readV(v11), readV(v01), readV(v00));
                    if (eb < required_eb) required_eb = eb;
                  }

                  if (gvid == v00 || gvid == v10 || gvid == v11) {
                    if (has_cp(cp_faces, v00, v10, v11)) {
                      required_eb = 0;
                    }
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(v11), readU(v10), readU(v00),
                        readV(v11), readV(v10), readV(v00));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              const int di[6] = {0, 0, 1, -1, 1, -1};
              const int dj[6] = {1, -1, 0, 0, 1, -1};

              if (gt < (int)Tt - 1) {
                for (int k = 0; k < 6; ++k) {
                  int ni = gi + di[k];
                  int nj = gj + dj[k];
                  if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
                  size_t a = gvid;
                  size_t b = vid(gt, ni, nj, sz);
                  size_t ap = a + dv;
                  size_t bp = b + dv;
                  if (k == 0 || k == 3 || k == 5) {
                    if (has_cp(cp_faces, a, b, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(b), readU(a),
                          readV(bp), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                    if (has_cp(cp_faces, a, bp, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(bp), readU(a),
                          readV(ap), readV(bp), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  } else {
                    if (has_cp(cp_faces, a, b, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(b), readU(a),
                          readV(ap), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  }
                }
              }

              if (gt > 0) {
                for (int k = 0; k < 6; ++k) {
                  int ni = gi + di[k];
                  int nj = gj + dj[k];
                  if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
                  size_t a = gvid;
                  size_t b = vid(gt, ni, nj, sz);
                  size_t ap = a - dv;
                  size_t bp = b - dv;
                  if (k == 0 || k == 3 || k == 5) {
                    if (has_cp(cp_faces, a, b, ap)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(ap), readU(b), readU(a),
                          readV(ap), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  } else {
                    if (has_cp(cp_faces, a, b, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(b), readU(a),
                          readV(bp), readV(b), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                    if (has_cp(cp_faces, a, ap, bp)) {
                      required_eb = 0;
                    }
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(
                          readU(bp), readU(ap), readU(a),
                          readV(bp), readV(ap), readV(a));
                      if (eb < required_eb) required_eb = eb;
                    }
                  }
                }
              }

              if (gt < (int)Tt - 1) {
                if (in_range(gi - 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f1a = gvid;
                  size_t f1b = vid(gt, gi, gj + 1, sz);
                  size_t f1c = vid(gt, gi - 1, gj, sz);
                  size_t f1bp = f1b + dv;
                  size_t f1cp = f1c + dv;
                  if (has_cp(cp_faces, f1a, f1b, f1cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f1b), readU(f1cp), readU(f1a),
                        readV(f1b), readV(f1cp), readV(f1a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f1a, f1bp, f1cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f1bp), readU(f1cp), readU(f1a),
                        readV(f1bp), readV(f1cp), readV(f1a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f2a = gvid;
                  size_t f2b = vid(gt, gi - 1, gj, sz);
                  size_t f2c = vid(gt, gi - 1, gj - 1, sz);
                  size_t f2bp = f2b + dv;
                  size_t f2cp = f2c + dv;
                  if (has_cp(cp_faces, f2a, f2bp, f2cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f2bp), readU(f2c), readU(f2a),
                        readV(f2bp), readV(f2c), readV(f2a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f2a, f2cp, f2bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f2cp), readU(f2bp), readU(f2a),
                        readV(f2cp), readV(f2bp), readV(f2a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f3a = gvid;
                  size_t f3b = vid(gt, gi - 1, gj - 1, sz);
                  size_t f3c = vid(gt, gi, gj - 1, sz);
                  size_t f3bp = f3b + dv;
                  size_t f3cp = f3c + dv;
                  if (has_cp(cp_faces, f3a, f3b, f3bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f3c), readU(f3bp), readU(f3a),
                        readV(f3c), readV(f3bp), readV(f3a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f6a = gvid;
                  size_t f6b = vid(gt, gi, gj + 1, sz);
                  size_t f6c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f6bp = f6b + dv;
                  if (has_cp(cp_faces, f6a, f6bp, f6c)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f6c), readU(f6bp), readU(f6a),
                        readV(f6c), readV(f6bp), readV(f6a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              if (gt > 0) {
                if (in_range(gi - 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f3a = gvid;
                  size_t f3b = vid(gt, gi - 1, gj - 1, sz);
                  size_t f3c = vid(gt, gi, gj - 1, sz);
                  size_t f3bp = f3b - dv;
                  size_t f3cp = f3c - dv;
                  if (has_cp(cp_faces, f3a, f3b, f3cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f3cp), readU(f3b), readU(f3a),
                        readV(f3cp), readV(f3b), readV(f3a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj - 1, (int)W)) {
                  size_t f4a = gvid;
                  size_t f4b = vid(gt, gi, gj - 1, sz);
                  size_t f4c = vid(gt, gi + 1, gj, sz);
                  size_t f4bp = f4b - dv;
                  size_t f4cp = f4c - dv;
                  if (has_cp(cp_faces, f4a, f4bp, f4cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f4cp), readU(f4bp), readU(f4a),
                        readV(f4cp), readV(f4bp), readV(f4a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f4a, f4b, f4cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f4cp), readU(f4b), readU(f4a),
                        readV(f4cp), readV(f4b), readV(f4a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f5a = gvid;
                  size_t f5b = vid(gt, gi + 1, gj, sz);
                  size_t f5c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f5bp = f5b - dv;
                  size_t f5cp = f5c - dv;
                  if (has_cp(cp_faces, f5a, f5bp, f5cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f5cp), readU(f5bp), readU(f5a),
                        readV(f5cp), readV(f5bp), readV(f5a));
                    if (eb < required_eb) required_eb = eb;
                  }
                  if (has_cp(cp_faces, f5a, f5c, f5bp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f5bp), readU(f5c), readU(f5a),
                        readV(f5bp), readV(f5c), readV(f5a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
                if (in_range(gi + 1, (int)H) && in_range(gj + 1, (int)W)) {
                  size_t f6a = gvid;
                  size_t f6b = vid(gt, gi, gj + 1, sz);
                  size_t f6c = vid(gt, gi + 1, gj + 1, sz);
                  size_t f6cp = f6c - dv;
                  if (has_cp(cp_faces, f6a, f6b, f6cp)) {
                    required_eb = 0;
                  }
                  {
                    T eb = derive_cp_abs_eb_sos_online<T>(
                        readU(f6cp), readU(f6b), readU(f6a),
                        readV(f6cp), readV(f6b), readV(f6a));
                    if (eb < required_eb) required_eb = eb;
                  }
                }
              }

              T abs_eb = required_eb;
              int id = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);

              if (abs_eb == 0) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              bool unpred_flag = false;
              T dec[2] = {0, 0};
              T abs_err_fp_q[2] = {0, 0};
              for (int p = 0; p < 2; ++p) {
                T* cur = (p == 0) ? curU : curV;
                T curv = (p == 0) ? curU_val : curV_val;

                T d0 = (gt && gi && gj) ? cur[-lsk - lsi - lsj] : 0;
                T d1 = (gt && gi) ? cur[-lsk - lsi] : 0;
                T d2 = (gt && gj) ? cur[-lsk - lsj] : 0;
                T d3 = (gt) ? cur[-lsk] : 0;
                T d4 = (gi && gj) ? cur[-lsi - lsj] : 0;
                T d5 = (gi) ? cur[-lsi] : 0;
                T d6 = (gj) ? cur[-lsj] : 0;
                T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

                T diff = curv - pred;
                T qd = (std::llabs(diff) / abs_eb) + 1;
                if (qd < capacity) {
                  qd = (diff > 0) ? qd : -qd;
                  int qindex = (int)(qd / 2) + intv_radius;
                  if (p == 0) dq0[gvid] = qindex; else dq1[gvid] = qindex;
                  dec[p] = pred + 2 * (qindex - intv_radius) * abs_eb;
                  if (std::llabs(dec[p] - curv) > abs_eb) {
                    unpred_flag = true;
                    break;
                  }
                  abs_err_fp_q[p] = std::llabs(dec[p] - curv);
                } else {
                  unpred_flag = true;
                  break;
                }
              }

              if (unpred_flag) {
                eb_id[gvid] = 0;
                pred_mask[gvid] = 0;
                dq0[gvid] = 0;
                dq1[gvid] = 0;
                continue;
              }

              eb_id[gvid] = id;
              pred_mask[gvid] = 1;
              *curU = dec[0];
              *curV = dec[1];

              double abs_eb_fp = (double)abs_eb / (double)scale;
              double err_u_fp = (double)abs_err_fp_q[0] / (double)scale;
              double err_v_fp = (double)abs_err_fp_q[1] / (double)scale;
              // enc_max_abs_eb_fp = std::max(enc_max_abs_eb_fp, abs_eb_fp);
              // enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
            }
          }
        }
      }
    }
  }

  std::vector<T_data> unpred;
  unpred.reserve((N / 4) * 2);
  std::vector<int> dq_stream;
  dq_stream.reserve(2 * N);

  for (size_t v = 0; v < N; ++v) {
    if (pred_mask[v]) {
      dq_stream.push_back(dq0[v]);
      dq_stream.push_back(dq1[v]);
    } else {
      unpred.push_back(U[v]);
      unpred.push_back(V[v]);
    }
  }

  unsigned char* compressed = (unsigned char*)std::malloc((size_t)(2 * N * sizeof(T)));
  if (!compressed) {
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }
  unsigned char* pos = compressed;

  write_variable_to_dst(pos, scale);
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);
  std::cout << "write intv_radius = " << intv_radius << "\n";

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) {
    write_array_to_dst(pos, unpred.data(), unpred_cnt);
  }

  size_t eb_quant_num = N;
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * 1024, eb_id.data(), eb_quant_num, pos);

  size_t data_quant_num = dq_stream.size();
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * capacity, dq_stream.data(), data_quant_num, pos);

  compressed_size = (size_t)(pos - compressed);

  std::free(U_fp);
  std::free(V_fp);
  return compressed;
}

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_parallel_v3(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  using T = int64_t;
  const Size3 sz{(int)r1, (int)r2, (int)r3};
  const size_t H = r1, W = r2, Tt = r3;
  const size_t N = H * W * Tt;
  const size_t dv = H * W;

  compressed_size = 0;
  if (!U || !V || N == 0) {
    return nullptr;
  }

  T* U_fp = (T*)std::malloc(N * sizeof(T));
  T* V_fp = (T*)std::malloc(N * sizeof(T));
  if (!U_fp || !V_fp) {
    if (U_fp) std::free(U_fp);
    if (V_fp) std::free(V_fp);
    return nullptr;
  }

  T range = 0;
  T scale = convert_to_fixed_point<T_data, T>(U, V, N, U_fp, V_fp, range);

  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  std::vector<int> eb_id(N, 0);
  std::vector<uint8_t> pred_mask(N, 0);
  std::vector<int> dq0(N, 0);
  std::vector<int> dq1(N, 0);

  const int base = 2;
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);
  const T threshold = 1;

  T max_eb = 0;
  if (mode == EbMode::Relative) {
    max_eb = max_pwr_eb * range;
    printf("Compression Using Relative Eb Mode!\n");
  } else if (mode == EbMode::Absolute) {
    max_eb = max_pwr_eb * scale;
    printf("Compression Using Absolute Eb Mode!\n");
  } else {
    std::cerr << "Error: Unsupported EbMode!\n";
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }

  const int BlockH = 96;
  const int BlockW = 128;
  const size_t nbH = (H + BlockH - 1) / BlockH;
  const size_t nbW = (W + BlockW - 1) / BlockW;

  const ptrdiff_t si = (ptrdiff_t)W;
  const ptrdiff_t sj = (ptrdiff_t)1;
  const ptrdiff_t sk = (ptrdiff_t)(H * W);
  const int di[6] = { 0,  0,  1, -1,  1, -1 };
  const int dj[6] = { 1, -1,  0,  0,  1, -1 };

  auto readU = [&](size_t idx) -> T { return U_fp[idx]; };
  auto readV = [&](size_t idx) -> T { return V_fp[idx]; };

  auto process_tile = [&](int t, int bi, int bj) {
    const int i0 = bi * BlockH;
    const int i1 = std::min(i0 + BlockH, (int)H);
    const int j0 = bj * BlockW;
    const int j1 = std::min(j0 + BlockW, (int)W);
    const int Hi = i1 - i0;
    const int Wi = j1 - j0;

    for (int s = 0; s <= Hi + Wi - 2; ++s) {
      int j_lo = std::max(0, s - (Hi - 1));
      int j_hi = std::min(Wi - 1, s);
      for (int joff = j_lo; joff <= j_hi; ++joff) {
        int ioff = s - joff;
        int i = i0 + ioff;
        int j = j0 + joff;
        const size_t v = vid(t, i, j, sz);

        const bool is_boundary = (i == i0) || (i == i1 - 1) ||
                                 (j == j0) || (j == j1 - 1);
        if (is_boundary) {
          eb_id[v] = 0;
          pred_mask[v] = 0;
          dq0[v] = 0;
          dq1[v] = 0;
          continue;
        }

        T* curU = U_fp + v;
        T* curV = V_fp + v;
        const T curU_val = *curU;
        const T curV_val = *curV;
        T required_eb = max_eb;

        for (int ci = i - 1; ci <= i; ++ci) {
          if (!in_range(ci, (int)H - 1)) continue;
          for (int cj = j - 1; cj <= j; ++cj) {
            if (!in_range(cj, (int)W - 1)) continue;

            size_t v00 = vid(t, ci, cj, sz);
            size_t v10 = vid(t, ci, cj + 1, sz);
            size_t v01 = vid(t, ci + 1, cj, sz);
            size_t v11 = vid(t, ci + 1, cj + 1, sz);

            if (v == v00 || v == v01 || v == v11) {
              if (has_cp(cp_faces, v00, v01, v11)) {
                required_eb = 0;
              }
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(v11), readU(v01), readU(v00),
                  readV(v11), readV(v01), readV(v00));
              if (eb < required_eb) required_eb = eb;
            }

            if (v == v00 || v == v10 || v == v11) {
              if (has_cp(cp_faces, v00, v10, v11)) {
                required_eb = 0;
              }
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(v11), readU(v10), readU(v00),
                  readV(v11), readV(v10), readV(v00));
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        if (t < (int)Tt - 1) {
          for (int k = 0; k < 6; ++k) {
            int ni = i + di[k];
            int nj = j + dj[k];
            if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
            size_t a = v;
            size_t b = vid(t, ni, nj, sz);
            size_t ap = a + dv;
            size_t bp = b + dv;
            if (k == 0 || k == 3 || k == 5) {
              if (has_cp(cp_faces, a, b, bp)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(bp), readU(b), readU(a),
                    readV(bp), readV(b), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, a, bp, ap)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(ap), readU(bp), readU(a),
                    readV(ap), readV(bp), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
            } else {
              if (has_cp(cp_faces, a, b, ap)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(ap), readU(b), readU(a),
                    readV(ap), readV(b), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }

        if (t > 0) {
          for (int k = 0; k < 6; ++k) {
            int ni = i + di[k];
            int nj = j + dj[k];
            if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
            size_t a = v;
            size_t b = vid(t, ni, nj, sz);
            size_t ap = a - dv;
            size_t bp = b - dv;
            if (k == 0 || k == 3 || k == 5) {
              if (has_cp(cp_faces, a, b, ap)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(ap), readU(b), readU(a),
                    readV(ap), readV(b), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
            } else {
              if (has_cp(cp_faces, a, b, bp)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(bp), readU(b), readU(a),
                    readV(bp), readV(b), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, a, ap, bp)) {
                required_eb = 0;
              }
              {
                T eb = derive_cp_abs_eb_sos_online<T>(
                    readU(bp), readU(ap), readU(a),
                    readV(bp), readV(ap), readV(a));
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }

        // (C1) and (C2) internal faces
        if (t < (int)Tt - 1) {
          if (in_range(i - 1, (int)H) && in_range(j + 1, (int)W)) {
            size_t f1a = v;
            size_t f1b = vid(t, i, j + 1, sz);
            size_t f1c = vid(t, i - 1, j, sz);
            size_t f1bp = f1b + dv;
            size_t f1cp = f1c + dv;
            if (has_cp(cp_faces, f1a, f1b, f1cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f1b), readU(f1cp), readU(f1a),
                  readV(f1b), readV(f1cp), readV(f1a));
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f1a, f1bp, f1cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f1bp), readU(f1cp), readU(f1a),
                  readV(f1bp), readV(f1cp), readV(f1a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i - 1, (int)H) && in_range(j - 1, (int)W)) {
            size_t f2a = v;
            size_t f2b = vid(t, i - 1, j, sz);
            size_t f2c = vid(t, i - 1, j - 1, sz);
            size_t f2bp = f2b + dv;
            size_t f2cp = f2c + dv;
            if (has_cp(cp_faces, f2a, f2bp, f2cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f2bp), readU(f2c), readU(f2a),
                  readV(f2bp), readV(f2c), readV(f2a));
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f2a, f2cp, f2bp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f2cp), readU(f2bp), readU(f2a),
                  readV(f2cp), readV(f2bp), readV(f2a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i - 1, (int)H) && in_range(j - 1, (int)W)) {
            size_t f3a = v;
            size_t f3b = vid(t, i - 1, j - 1, sz);
            size_t f3c = vid(t, i, j - 1, sz);
            size_t f3bp = f3b + dv;
            size_t f3cp = f3c + dv;
            if (has_cp(cp_faces, f3a, f3b, f3bp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f3c), readU(f3bp), readU(f3a),
                  readV(f3c), readV(f3bp), readV(f3a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i + 1, (int)H) && in_range(j + 1, (int)W)) {
            size_t f6a = v;
            size_t f6b = vid(t, i, j + 1, sz);
            size_t f6c = vid(t, i + 1, j + 1, sz);
            size_t f6bp = f6b + dv;
            if (has_cp(cp_faces, f6a, f6bp, f6c)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f6c), readU(f6bp), readU(f6a),
                  readV(f6c), readV(f6bp), readV(f6a));
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        if (t > 0) {
          if (in_range(i - 1, (int)H) && in_range(j - 1, (int)W)) {
            size_t f3a = v;
            size_t f3b = vid(t, i - 1, j - 1, sz);
            size_t f3c = vid(t, i, j - 1, sz);
            size_t f3bp = f3b - dv;
            size_t f3cp = f3c - dv;
            if (has_cp(cp_faces, f3a, f3b, f3cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f3cp), readU(f3b), readU(f3a),
                  readV(f3cp), readV(f3b), readV(f3a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i + 1, (int)H) && in_range(j - 1, (int)W)) {
            size_t f4a = v;
            size_t f4b = vid(t, i, j - 1, sz);
            size_t f4c = vid(t, i + 1, j, sz);
            size_t f4bp = f4b - dv;
            size_t f4cp = f4c - dv;
            if (has_cp(cp_faces, f4a, f4bp, f4cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f4cp), readU(f4bp), readU(f4a),
                  readV(f4cp), readV(f4bp), readV(f4a));
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f4a, f4b, f4cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f4cp), readU(f4b), readU(f4a),
                  readV(f4cp), readV(f4b), readV(f4a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i + 1, (int)H) && in_range(j + 1, (int)W)) {
            size_t f5a = v;
            size_t f5b = vid(t, i + 1, j, sz);
            size_t f5c = vid(t, i + 1, j + 1, sz);
            size_t f5bp = f5b - dv;
            size_t f5cp = f5c - dv;
            if (has_cp(cp_faces, f5a, f5bp, f5cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f5cp), readU(f5bp), readU(f5a),
                  readV(f5cp), readV(f5bp), readV(f5a));
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f5a, f5c, f5bp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f5bp), readU(f5c), readU(f5a),
                  readV(f5bp), readV(f5c), readV(f5a));
              if (eb < required_eb) required_eb = eb;
            }
          }

          if (in_range(i + 1, (int)H) && in_range(j + 1, (int)W)) {
            size_t f6a = v;
            size_t f6b = vid(t, i, j + 1, sz);
            size_t f6c = vid(t, i + 1, j + 1, sz);
            size_t f6cp = f6c - dv;
            if (has_cp(cp_faces, f6a, f6b, f6cp)) {
              required_eb = 0;
            }
            {
              T eb = derive_cp_abs_eb_sos_online<T>(
                  readU(f6cp), readU(f6b), readU(f6a),
                  readV(f6cp), readV(f6b), readV(f6a));
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        T abs_eb = required_eb;
        int id = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);

        if (abs_eb == 0) {
          eb_id[v] = 0;
          pred_mask[v] = 0;
          dq0[v] = 0;
          dq1[v] = 0;
          continue;
        }

        bool unpred_flag = false;
        T dec[2] = {0, 0};
        T abs_err_fp_q[2] = {0, 0};

        for (int p = 0; p < 2; ++p) {
          T* cur = (p == 0) ? curU : curV;
          T curv = (p == 0) ? curU_val : curV_val;

          T d0 = (t && i && j) ? cur[-sk - si - sj] : 0;
          T d1 = (t && i)      ? cur[-sk - si]      : 0;
          T d2 = (t && j)      ? cur[-sk - sj]      : 0;
          T d3 = (t)           ? cur[-sk]           : 0;
          T d4 = (i && j)      ? cur[-si - sj]      : 0;
          T d5 = (i)           ? cur[-si]           : 0;
          T d6 = (j)           ? cur[-sj]           : 0;
          T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

          T diff = curv - pred;
          T qd = (std::llabs(diff) / abs_eb) + 1;
          if (qd < capacity) {
            qd = (diff > 0) ? qd : -qd;
            int qindex = (int)(qd / 2) + intv_radius;
            if (p == 0) {
              dq0[v] = qindex;
            } else {
              dq1[v] = qindex;
            }
            dec[p] = pred + 2 * (qindex - intv_radius) * abs_eb;
            if (std::llabs(dec[p] - curv) > abs_eb) {
              unpred_flag = true;
              break;
            }
            abs_err_fp_q[p] = std::llabs(dec[p] - curv);
          } else {
            unpred_flag = true;
            break;
          }
        }

        if (unpred_flag) {
          eb_id[v] = 0;
          pred_mask[v] = 0;
          dq0[v] = 0;
          dq1[v] = 0;
        } else {
          eb_id[v] = id;
          pred_mask[v] = 1;
          *curU = dec[0];
          *curV = dec[1];
        }
      }
    }
  };

  for (int t = 0; t < (int)Tt; ++t) {
    int Smax = (int)(nbH + nbW - 2);
    for (int s = 0; s <= Smax; ++s) {
      int bi_lo = std::max(0, s - (int)nbW + 1);
      int bi_hi = std::min((int)nbH - 1, s);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int bi = bi_lo; bi <= bi_hi; ++bi) {
        int bj = s - bi;
        process_tile(t, bi, bj);
      }
    }
  }

  std::vector<T_data> unpred;
  unpred.reserve((N / 4) * 2);
  std::vector<int> data_quant_index;
  data_quant_index.reserve(2 * N);

  size_t unpred_cnt = 0;
  for (size_t v = 0; v < N; ++v) {
    if (!pred_mask[v]) unpred_cnt += 2;
  }

  unsigned char* compressed = (unsigned char*)std::malloc((size_t)(2 * N * sizeof(T)));
  if (!compressed) {
    std::free(U_fp);
    std::free(V_fp);
    return nullptr;
  }
  unsigned char* pos = compressed;

  write_variable_to_dst(pos, scale);
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);
  std::cout << "write intv_radius = " << intv_radius << "\n";

  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) {
    for (size_t v = 0; v < N; ++v) {
      if (!pred_mask[v]) {
        unpred.push_back(U[v]);
        unpred.push_back(V[v]);
      }
    }
    write_array_to_dst(pos, unpred.data(), unpred.size());
  }

  size_t eb_quant_num = N;
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * 1024, eb_id.data(), eb_quant_num, pos);

  for (size_t v = 0; v < N; ++v) {
    if (pred_mask[v]) {
      data_quant_index.push_back(dq0[v]);
      data_quant_index.push_back(dq1[v]);
    }
  }
  size_t data_quant_num = data_quant_index.size();
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2 * capacity, data_quant_index.data(), data_quant_num, pos);

  compressed_size = (size_t)(pos - compressed);

  std::free(U_fp);
  std::free(V_fp);
  return compressed;
}


template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_warped_lorenzo(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb,                  // 全局绝对误差上限（浮点域）：max_eb(fp)=max_pwr_eb
    EbMode mode 
){
  // Warped-Lorenzo 运行参数（编解码两端需一致；也可写入码流头部）
  const double WARP_ALPHA_X = 0.77;//1.49;   // 像素/帧 = alpha * u,v  （若你的u,v单位即“像素/帧”，取1.0）
  const double WARP_ALPHA_Y = 0.73;//1.49;   // 像素/帧 = alpha * u,v  （若你的u,v单位即“像素/帧”，取1.0）
  const double WARP_DMAX  = 6.0;   // 位移裁剪（像素）
  const double WARP_GATE  = 0.5;  // 小位移门槛（像素）；<0.30像素时退化为0，省去无谓的插值抖动
  using T = int64_t;
  const Size3 sz{ (int)r1,(int)r2,(int)r3 };
  const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

  // 1) 定点化（先乘后 llround）
  T *U_fp=(T*)std::malloc(N*sizeof(T));
  T *V_fp=(T*)std::malloc(N*sizeof(T));
  if(!U_fp || !V_fp){
    if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp);
    compressed_size=0; return nullptr;
  }
  T range=0;
  T scale = convert_to_fixed_point<T_data,T>(U, V, N, U_fp, V_fp, range);

  // 2) 预计算：全局 CP 面集合（一次性）
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();


  #if CP_DEBUG_VISIT
  // 1.1 记录：编码端枚举/检查过的三角面
  std::unordered_set<FaceKeySZ, FaceKeySZHash> visited_faces;

  // 1.2 小工具：规范化存一下 (a,b,c)
  auto MARK_FACE = [&](size_t a, size_t b, size_t c){
      visited_faces.emplace(a,b,c); // FaceKeySZ 构造里已做排序/规范化
  };

  // 1.3 帮助打印：把 vid -> (t,i,j)
  auto decode_tij = [&](size_t v){
      int t = (int)(v / (H*W));
      size_t rem = v - (size_t)t * H * W;
      int i = (int)(rem / W);
      int j = (int)(rem % W);
      return std::tuple<int,int,int>(t,i,j);
  };

  // 1.4（可选）分类一下是哪类面
  auto classify_face = [&](size_t a, size_t b, size_t c){
      auto [ta,ia,ja] = decode_tij(a);
      auto [tb,ib,jb] = decode_tij(b);
      auto [tc,ic,jc] = decode_tij(c);
      int st = (ta==tb) + (ta==tc) + (tb==tc);
      if (ta==tb && tb==tc) {
          // 同一层
          bool has_diag = ((ia!=ib)||(ja!=jb)) && ((ia!=ic)||(ja!=jc)) && ((ib!=ic)||(jb!=jc));
          return has_diag ? "layer(tri)" : "layer(?)";
      } else {
          // 跨层
          int cnt_t0 = (ta==tb) + (ta==tc) + (ta==ta); // 粗略判断：有2点在同一层
          (void)cnt_t0;
          return "slab";
      }
  };
  #endif

  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // 3) 量化/编码缓冲
  int* eb_quant_index  = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index= (int*)std::malloc(2*N*sizeof(int)); // U/V 交错
  double enc_max_abs_eb_fp   = 0.0; //编码端自检（浮点）
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred; unpred.reserve((N / 4) * 2);

  // 参数与量化槽
  const int base = 2;                             // 仍写头，方便向后兼容
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);

  // 把浮点域绝对误差上限转到定点域（LSB）
  // const T max_eb = (T) std::llround( (long double)max_pwr_eb * (long double)scale );

  T max_eb = 0;
  if(mode == EbMode::Relative){

    max_eb = max_pwr_eb * range; // 相对误差转绝对误差
    printf("Compression Using Relative Eb Mode!\n");
  }
  else if (mode == EbMode::Absolute){
    printf("Compression Using Absolute Eb Mode!\n");
    max_eb = max_pwr_eb * scale; // 浮点→定点
  }
  else{
    std::cerr << "Error: Unsupported EbMode!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  // T max_eb = range * max_pwr_eb;


  // 定点 LSB 的阈值（退化门控）
  const T threshold = 1;                          // 幂指数量化阈值（LSB）


  // 4) 逐顶点：枚举与该顶点相关的三角面 → 最小 eb
  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W; // 层间位移

  // 6 个平面邻接方向（左右、上下、主对角）
  const int di[6] = { 0,  0,  1, -1,  1, -1 }; //y-axis
  const int dj[6] = { 1, -1,  0,  0,  1, -1 }; //x-axis

  for (int t=0; t<(int)Tt; ++t){
    if(t % 100 == 0){
      printf("processing slice %d / %d\n", t, (int)Tt);
    }
    for (int i=0; i<(int)H; ++i){
      for (int j=0; j<(int)W; ++j){
        const size_t v = vid(t,i,j,sz);
        // 缓存原始定点值（预测/回写前）
        T *curU = U_fp + v;
        T *curV = V_fp + v;
        const T curU_val = *curU;
        const T curV_val = *curV;

        // —— 收集最小 eb——
        T required_eb = max_eb;

        // (A) 层内 t：影响 (i-1..i, j-1..j) 的 4 个 cell，每 cell 两三角
        for (int ci=i-1; ci<=i; ++ci){
          if (!in_range(ci, (int)H-1)) continue;
          for (int cj=j-1; cj<=j; ++cj){
            if (!in_range(cj, (int)W-1)) continue;

            size_t v00 = vid(t,ci,  cj,  sz);
            size_t v10 = vid(t,ci,  cj+1,sz); //v10=> x=1,y=0
            size_t v01 = vid(t,ci+1,cj,  sz);
            size_t v11 = vid(t,ci+1,cj+1,sz);

            // Upper: (v00,v01,v11)
            if (v==v00 || v==v01 || v==v11){ 
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v01,v11);
              #endif
              if (has_cp(cp_faces, v00,v01,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v01],U_fp[v00],
                                                    V_fp[v11],V_fp[v01],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
            // Lower: (v00,v10,v11)
            if (v==v00 || v==v10 || v==v11){
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v10,v11);
              #endif
              if (has_cp(cp_faces, v00,v10,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v10],U_fp[v00],
                                                    V_fp[v11],V_fp[v10],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }
        
        // (B1) 侧面 [t, t+1]
        if (t < (int)Tt-1){
          for (int k=0; k<6; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a  = vid(t, i, j, sz);
            size_t b  = vid(t, ni,nj, sz);
            size_t ap = a + dv, bp = b + dv;
            if (k == 0 || k==3 || k==5){
              // (a,b,bp) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (a,bp,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,bp,ap);
              #endif
              if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[bp],U_fp[a],
                                                        V_fp[ap],V_fp[bp],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,ap) for k = 1,2,4
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (b,ap,bp) for k = 1,2,4 //好像与点无关
              // if (has_cp(cp_faces, b,ap,bp)) { required_eb = 0; goto after_min_eb; }
              //   {
              //     T eb = derive_cp_abs_eb_sos_online<T>(U_fp[b],U_fp[ap],U_fp[bp],
              //                                           V_fp[b],V_fp[ap],V_fp[bp]);
              //     if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
              //     if (eb < required_eb) required_eb = eb;
              //   }
            }

          }
        }

        // (B2) 侧面 [t-1, t]
        if (t > 0){
          for (int k=0; k<6 ; ++k){
            int ni = i + di[k], nj = j + dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a = vid(t,i,j,sz); //v
            size_t b = vid(t,ni,nj,sz);
            size_t ap = a - dv, bp = b - dv; //ap,bp为上一层
            if (k == 0 || k==3 || k==5){
              // (a,b,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,bp) for k = 1,2,4 //侧面上三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
                
              // (a,ap,bp) for k = 1,2,4 //侧面下三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,ap,bp);
              #endif
              if (has_cp(cp_faces, a,ap,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[ap],U_fp[a],
                                                        V_fp[bp],V_fp[ap],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }

          }
        }

        // (C) 内部剖分面：两片 ts ∈ {t, t-1}；每相邻 cell 的 Upper/Lower 各 2 面
        // (C1) [t, t+1] 内部剖分面
        if (t < (int)Tt-1){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|
      
          //triange 1 [x,y,t] = [(0,0,0),(1,0,0),(0,-1,0)]: has 2 faces
          size_t f1a = vid(t,  i,  j,  sz);
          size_t f1b = vid(t,  i,j+1,  sz);
          size_t f1c = vid(t,  i-1,j,  sz);
          size_t f1ap = f1a + dv, f1bp = f1b + dv, f1cp = f1c + dv;
          if (in_range(i-1,(int)H) && in_range(j+1,(int)W)){
            // (f1a,f1cp,f1b)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1b);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1b)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1b],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1b],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f1a,f1cp,f1bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1bp);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1bp],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1bp],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          //triange 2 [x,y,t] = [(0,0,0),(0,-1,0),(-1,-1,0)]: has 2 faces
          size_t f2a = vid(t,  i,  j,  sz);
          size_t f2b = vid(t,  i-1,j,  sz);
          size_t f2c = vid(t,  i-1,j-1,sz);
          size_t f2ap = f2a + dv, f2bp = f2b + dv, f2cp = f2c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f2a,f2c,f2bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2c,f2bp);
            #endif
            if (has_cp(cp_faces, f2a,f2c,f2bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2bp],U_fp[f2c],U_fp[f2a],
                                                    V_fp[f2bp],V_fp[f2c],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f2a,f2bp,f2cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2bp,f2cp);
            #endif
            if (has_cp(cp_faces, f2a,f2bp,f2cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2cp],U_fp[f2bp],U_fp[f2a],
                                                    V_fp[f2cp],V_fp[f2bp],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          // triange 3 [x,y,t] = [(0,0,0),(-1,-1,0),(-1,0,0)]: has 1 faces
          size_t f3a = vid(t,  i,  j,  sz);
          size_t f3b = vid(t,  i-1,j-1,sz);
          size_t f3c = vid(t,  i,j-1,  sz);
          size_t f3ap = f3a + dv, f3bp = f3b + dv, f3cp = f3c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3bp,f3c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3bp,f3c);
            #endif
            if (has_cp(cp_faces, f3a,f3bp,f3c)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3c],U_fp[f3bp],U_fp[f3a],
                                                    V_fp[f3c],V_fp[f3bp],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // triange 4 0 faces
          // triange 5 0 faces
          // triange 6 [x,y,t] = [(0,0,0),(1,0,0),(1,1,0)]: has 1 faces
          size_t f6a = vid(t,  i,  j,  sz);
          size_t f6b = vid(t,  i,j+1,  sz);
          size_t f6c = vid(t,  i+1,j+1,sz);
          size_t f6ap = f6a + dv, f6bp = f6b + dv, f6cp = f6c + dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6bp,f6c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6bp,f6c);
            #endif
            if (has_cp(cp_faces, f6a,f6bp,f6c)) {
              required_eb=0; 
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6c],U_fp[f6bp],U_fp[f6a],
                                                    V_fp[f6c],V_fp[f6bp],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }


        // (C2) [t-1, t] 内部剖分面 并不与C1对称
        if (t > 0){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|

          //triange 1 0 face
          //triange 2 0 face
          //triange 3  has 1 faces
          size_t f3a = vid(t, i,  j,  sz);   
          size_t f3b = vid(t, i-1,j-1,sz);
          size_t f3c = vid(t, i,j-1,  sz);
          size_t f3ap = f3a - dv, f3bp = f3b - dv, f3cp = f3c - dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3b,f3cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3b,f3cp);
            #endif
            if (has_cp(cp_faces, f3a,f3b,f3cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3cp],U_fp[f3b],U_fp[f3a],
                                                    V_fp[f3cp],V_fp[f3b],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 4 has 2 faces
          size_t f4a = vid(t, i,  j,  sz);
          size_t f4b = vid(t, i,j-1,  sz);
          size_t f4c = vid(t,i+1,j,  sz);
          size_t f4ap = f4a - dv, f4bp = f4b - dv, f4cp = f4c - dv;
          if (in_range(i+1,(int)H) && in_range(j-1,(int)W)){
            // (f4a,f4bp,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4bp,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4bp,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4bp],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4bp],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f4a,f4b,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4b,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4b,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4b],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4b],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 5 has 2 faces
          size_t f5a = vid(t, i,  j,  sz);
          size_t f5b = vid(t, i+1,j,  sz);
          size_t f5c = vid(t, i+1,j+1,sz);
          size_t f5ap = f5a - dv, f5bp = f5b - dv, f5cp = f5c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f5a,f5bp,f5cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5bp,f5cp);
            #endif
            if (has_cp(cp_faces, f5a,f5bp,f5cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5cp],U_fp[f5bp],U_fp[f5a],
                                                    V_fp[f5cp],V_fp[f5bp],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f5a,f5c,f5bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5c,f5bp);
            #endif
            if (has_cp(cp_faces, f5a,f5c,f5bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5bp],U_fp[f5c],U_fp[f5a],
                                                    V_fp[f5bp],V_fp[f5c],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 6 has 1 faces
          size_t f6a = vid(t, i,  j,  sz);
          size_t f6b = vid(t, i,j+1,  sz);
          size_t f6c = vid(t, i+1,j+1,sz);
          size_t f6ap = f6a - dv, f6bp = f6b - dv, f6cp = f6c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6b,f6cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6b,f6cp);
            #endif
            if (has_cp(cp_faces, f6a,f6b,f6cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6cp],U_fp[f6b],U_fp[f6a],
                                                    V_fp[f6cp],V_fp[f6b],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }
        // {
        //   // 保证 |dec-orig| ≤ abs_eb < |orig| ⇒ 不会跨 0 翻符号
        //   T sign_guard = std::min(std::llabs(curU_val), std::llabs(curV_val));
        //   if (sign_guard > 0 && required_eb >= sign_guard){
        //       T max_eb_to_preserve_sign = sign_guard - 1; // 留 1 LSB 余量
        //       if (max_eb_to_preserve_sign < required_eb)
        //           required_eb = max_eb_to_preserve_sign;
        //   }
        // }
        T abs_eb = required_eb;
        int id = eb_exponential_quantize(abs_eb,base,log_of_base,threshold);
        
        if (abs_eb == 0 )
        {
          *(eb_pos++) = 0;            // ebid=0 → 无损点
          unpred.push_back(U[v]);     // 原始浮点（解码端直接回填）
          unpred.push_back(V[v]);
          continue;
        }

        // ===== 量化 eb（配套的新幂指数量化）并编码该顶点 =====
        *eb_pos = id;   // 先写代表值 id（abs_eb 已被代表值替换）
        {
          bool unpred_flag=false;
          T dec[2];
          T abs_err_fp_q[2] = {0,0};

          for (int p=0; p<2; ++p){
            T *cur  = (p==0)? curU : curV;
            T  curv = (p==0) ? curU_val : curV_val;

            // ---- Warped‑Lorenzo 预测 ----
            T pred = 0;
            const T* recon_t_plane =
                (p==0) ? (U_fp + (size_t)t*H*W) : (V_fp + (size_t)t*H*W);

            if (t == 0){
              // t=0：退化为 2D Lorenzo
              if (i>0 && j>0) {
                pred = recon_t_plane[(size_t)(i-1)*W + j]
                     + recon_t_plane[(size_t)i*W + (j-1)]
                     - recon_t_plane[(size_t)(i-1)*W + (j-1)];
              } else if (i>0) {
                pred = recon_t_plane[(size_t)(i-1)*W + j];
              } else if (j>0) {
                pred = recon_t_plane[(size_t)i*W + (j-1)];
              } else {
                pred = 0;
              }
            } else {
              const T* recon_tm1_plane =
                  (p==0) ? (U_fp + (size_t)(t-1)*H*W) : (V_fp + (size_t)(t-1)*H*W);
              const T* U_tm1_plane = U_fp + (size_t)(t-1)*H*W; // 位移来源：上一帧速度
              const T* V_tm1_plane = V_fp + (size_t)(t-1)*H*W;

              pred = warp_pred::predict<T>(
                  /*recon_t   =*/recon_t_plane,
                  /*recon_tm1 =*/recon_tm1_plane,
                  /*U_tm1     =*/U_tm1_plane,
                  /*V_tm1     =*/V_tm1_plane,
                  (int)H, (int)W, i, j,
                  /*scale     =*/scale,
                  /*alphaX   =*/WARP_ALPHA_X,
                  /*alphaY   =*/WARP_ALPHA_Y,
                  /*dmax      =*/WARP_DMAX,
                  /*gate_pix  =*/WARP_GATE
              );
            }

            // 量化当前分量
            T diff = curv - pred;
            T qd = (std::llabs(diff)/abs_eb) + 1;
            if (qd < capacity){
              qd = (diff > 0) ? qd : -qd;
              int qindex = (int)(qd/2) + intv_radius;
              dq_pos[p] = qindex;
              dec[p] = pred + 2*(qindex - intv_radius)*abs_eb;

              if (std::llabs(dec[p] - curv) > abs_eb){
                unpred_flag = true; break;
              }
              abs_err_fp_q[p] = std::llabs(dec[p] - curv);
            } else {
              unpred_flag = true; break;
            }
          }

          if (unpred_flag){
            *(eb_pos++) = 0;                  // 改回无损
            unpred.push_back(U[v]);
            unpred.push_back(V[v]);
          }else{
            ++eb_pos;
            dq_pos += 2;
            *curU = dec[0];
            *curV = dec[1];

            // 编码端自检（浮点域）
            double abs_eb_fp = (double)abs_eb / (double)scale;
            double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
            double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
            enc_max_abs_eb_fp   = std::max(enc_max_abs_eb_fp, abs_eb_fp);
            enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
          }
        }
      }
    }
  }

  std::cerr << "[ENC] max abs_eb(fp) = " << enc_max_abs_eb_fp
            << ", max actual |err|(fp) = " << enc_max_real_err_fp << "\n";

  
  #if CP_DEBUG_VISIT
  {
      size_t miss = 0, shown = 0;
      for (const auto &fk : cp_faces){
          if (!visited_faces.count(fk)){
              ++miss;
              if (shown < 50){ // 只演示前 50 个，避免刷屏
                  size_t a = fk.v[0], b = fk.v[1], c = fk.v[2];
                  auto [ta,ia,ja] = decode_tij(a);
                  auto [tb,ib,jb] = decode_tij(b);
                  auto [tc,ic,jc] = decode_tij(c);
                  fprintf(stderr,
                    "[MISSING] face (%zu,%zu,%zu)  "
                    "A(t=%d,i=%d,j=%d)  B(t=%d,i=%d,j=%d)  C(t=%d,i=%d,j=%d)  type=%s\n",
                    a,b,c, ta,ia,ja, tb,ib,jb, tc,ic,jc,
                    classify_face(a,b,c));
                  ++shown;
              }
          }
      }
      fprintf(stderr, "[COVERAGE] encoder touched faces = %zu, cp_faces = %zu, missing = %zu\n",
              visited_faces.size(), cp_faces.size(), miss);
  }
  #endif
  // 5) 打包码流
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);
  std::cout << "write intv_radius = " << intv_radius << "\n";

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);

  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::free(eb_quant_index);

  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  std::free(U_fp); std::free(V_fp);
  return compressed;
}

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_AR2_2D_LORENZO(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb,                 // 全局绝对误差上限（浮点域）
    EbMode mode
){
  using T = int64_t;
  const Size3 sz{ (int)r1,(int)r2,(int)r3 };
  const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

  // 1) 定点化（先乘后 llround）
  T *U_fp=(T*)std::malloc(N*sizeof(T));
  T *V_fp=(T*)std::malloc(N*sizeof(T));
  if(!U_fp || !V_fp){
    if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp);
    compressed_size=0; return nullptr;
  }
  T range=0;
  T scale = convert_to_fixed_point<T_data,T>(U, V, N, U_fp, V_fp, range);
  const T threshold = 1;  

  // 2) 预计算：全局 CP 面集合（一次性）
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // 3) 量化/编码缓冲
  int* eb_quant_index  = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index= (int*)std::malloc(2*N*sizeof(int)); // U/V 交错
  double enc_max_abs_eb_fp   = 0.0; // 编码端自检（浮点）
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred; unpred.reserve((N / 4) * 2);

  // 4) 参数与量化槽
  const int base = 2;
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);

  // 5) 把浮点域绝对误差上限转到定点域（LSB）
  T max_eb = 0;
  if(mode == EbMode::Relative){
    max_eb = (T)std::llround((long double)max_pwr_eb * (long double)range); // 相对 → 绝对
    printf("Compression Using Relative Eb Mode!\n");
  } else if (mode == EbMode::Absolute){
    max_eb = (T)std::llround((long double)max_pwr_eb * (long double)scale); // 浮点 → 定点
    printf("Compression Using Absolute Eb Mode!\n");
  } else{
    std::cerr << "Error: Unsupported EbMode!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  if (max_eb <= 0){
    std::cerr << "Error: max_eb <= 0 after scaling!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }

  // 6) 下标步长 & 邻接
  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W; // 层间位移
  const int di[6] = { 0,  0,  1, -1,  1, -1 }; //y-axis
  const int dj[6] = { 1, -1,  0,  0,  1, -1 }; //x-axis

  // 7) —— A 方案预测器 ——（新增）
  // 时间 AR(2) 预测，输入为“已重构”数据
  auto time_pred_at = [&](const T* base, size_t idx, int t)->T{
    if (t <= 0) return (T)0;            // t==0，用 0 冷启动
    if (t == 1) return base[idx - sk];  // t==1，用 AR(1): x_{t-1}
    return (T)(2*base[idx - sk] - base[idx - 2*sk]); // t>=2, AR(2)
  };
  // 残差的 2D Lorenzo：e_pred = e(i-1,j) + e(i,j-1) - e(i-1,j-1)
  auto e2d_pred_at = [&](const T* base, size_t idx, int t, int i, int j)->T{
    T s = 0;
    if (i > 0){
      size_t up = idx - si;
      s += (base[up] - time_pred_at(base, up, t));
    }
    if (j > 0){
      size_t lf = idx - sj;
      s += (base[lf] - time_pred_at(base, lf, t));
    }
    if (i > 0 && j > 0){
      size_t ul = idx - si - sj;
      s -= (base[ul] - time_pred_at(base, ul, t));
    }
    return s;
  };

  // 8) 主循环
  for (int t=0; t<(int)Tt; ++t){
    if(t % 100 == 0){
      printf("processing slice %d / %d\n", t, (int)Tt);
    }
    for (int i=0; i<(int)H; ++i){
      for (int j=0; j<(int)W; ++j){

        const size_t v = vid(t,i,j,sz);
        // 缓存原始定点值（预测/回写前）
        T *curU = U_fp + v;
        T *curV = V_fp + v;
        const T curU_val = *curU;
        const T curV_val = *curV;

        // —— 收集该点的“最小可行 eb”（逐面枚举）——
        T required_eb = max_eb;
        
        // (A) 层内 t：影响 (i-1..i, j-1..j) 的 4 个 cell，每 cell 两三角
        for (int ci=i-1; ci<=i; ++ci){
          if (!in_range(ci, (int)H-1)) continue;
          for (int cj=j-1; cj<=j; ++cj){
            if (!in_range(cj, (int)W-1)) continue;

            size_t v00 = vid(t,ci,  cj,  sz);
            size_t v10 = vid(t,ci,  cj+1,sz);
            size_t v01 = vid(t,ci+1,cj,  sz);
            size_t v11 = vid(t,ci+1,cj+1,sz);

            // Upper: (v00,v01,v11)
            if (v==v00 || v==v01 || v==v11){
              if (has_cp(cp_faces, v00,v01,v11)) { required_eb = 0; }
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v01],U_fp[v00],
                                                    V_fp[v11],V_fp[v01],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
            // Lower: (v00,v10,v11)
            if (v==v00 || v==v10 || v==v11){
              if (has_cp(cp_faces, v00,v10,v11)) { required_eb = 0; }
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v10],U_fp[v00],
                                                    V_fp[v11],V_fp[v10],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        // (B1) 侧面 [t, t+1]
        if (t < (int)Tt-1){
          for (int k=0; k<6; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a  = vid(t, i, j, sz);
            size_t b  = vid(t, ni,nj, sz);
            size_t ap = a + dv, bp = b + dv;
            if (k == 0 || k==3 || k==5){
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                      V_fp[bp],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[bp],U_fp[a],
                                                      V_fp[ap],V_fp[bp],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
            } else{
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                      V_fp[ap],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }

        // (B2) 侧面 [t-1, t]
        if (t > 0){
          for (int k=0; k<6 ; ++k){
            int ni = i + di[k], nj = j + dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a = vid(t,i,j,sz); //v
            size_t b = vid(t,ni,nj,sz);
            size_t ap = a - dv, bp = b - dv; //上一层
            if (k == 0 || k==3 || k==5){
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                      V_fp[ap],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
            } else{
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                      V_fp[bp],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
              if (has_cp(cp_faces, a,ap,bp)) { required_eb = 0; }
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[ap],U_fp[a],
                                                      V_fp[bp],V_fp[ap],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }

        // (C1) [t, t+1] 内部剖分面（与原实现一致）
        if (t < (int)Tt-1){
          // tri1
          size_t f1a = vid(t,  i,  j,  sz);
          size_t f1b = vid(t,  i,j+1,  sz);
          size_t f1c = vid(t,  i-1,j,  sz);
          size_t f1ap = f1a + dv, f1bp = f1b + dv, f1cp = f1c + dv;
          if (in_range(i-1,(int)H) && in_range(j+1,(int)W)){
            if (has_cp(cp_faces, f1a,f1cp,f1b)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1b],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1b],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f1a,f1cp,f1bp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1bp],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1bp],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri2
          size_t f2a = vid(t,  i,  j,  sz);
          size_t f2b = vid(t,  i-1,j,  sz);
          size_t f2c = vid(t,  i-1,j-1,sz);
          size_t f2ap = f2a + dv, f2bp = f2b + dv, f2cp = f2c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            if (has_cp(cp_faces, f2a,f2c,f2bp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2bp],U_fp[f2c],U_fp[f2a],
                                                    V_fp[f2bp],V_fp[f2c],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f2a,f2bp,f2cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2cp],U_fp[f2bp],U_fp[f2a],
                                                    V_fp[f2cp],V_fp[f2bp],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri3
          size_t f3a = vid(t,  i,  j,  sz);
          size_t f3b = vid(t,  i-1,j-1,sz);
          size_t f3c = vid(t,  i,j-1,  sz);
          size_t f3ap = f3a + dv, f3bp = f3b + dv, f3cp = f3c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            if (has_cp(cp_faces, f3a,f3bp,f3c)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3c],U_fp[f3bp],U_fp[f3a],
                                                    V_fp[f3c],V_fp[f3bp],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri6
          size_t f6a = vid(t,  i,  j,  sz);
          size_t f6b = vid(t,  i,j+1,  sz);
          size_t f6c = vid(t,  i+1,j+1,sz);
          size_t f6ap = f6a + dv, f6bp = f6b + dv, f6cp = f6c + dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            if (has_cp(cp_faces, f6a,f6bp,f6c)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6c],U_fp[f6bp],U_fp[f6a],
                                                    V_fp[f6c],V_fp[f6bp],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        // (C2) [t-1, t] 内部剖分面
        if (t > 0){
          // tri3
          size_t f3a = vid(t, i,  j,  sz);
          size_t f3b = vid(t, i-1,j-1,sz);
          size_t f3c = vid(t, i,  j-1,sz);
          size_t f3ap = f3a - dv, f3bp = f3b - dv, f3cp = f3c - dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            if (has_cp(cp_faces, f3a,f3b,f3cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3cp],U_fp[f3b],U_fp[f3a],
                                                    V_fp[f3cp],V_fp[f3b],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri4
          size_t f4a = vid(t, i,  j,  sz);
          size_t f4b = vid(t, i,  j-1,sz);
          size_t f4c = vid(t, i+1,j,  sz);
          size_t f4ap = f4a - dv, f4bp = f4b - dv, f4cp = f4c - dv;
          if (in_range(i+1,(int)H) && in_range(j-1,(int)W)){
            if (has_cp(cp_faces, f4a,f4bp,f4cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4bp],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4bp],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f4a,f4b,f4cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4b],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4b],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri5
          size_t f5a = vid(t, i,  j,  sz);
          size_t f5b = vid(t, i+1,j,  sz);
          size_t f5c = vid(t, i+1,j+1,sz);
          size_t f5ap = f5a - dv, f5bp = f5b - dv, f5cp = f5c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            if (has_cp(cp_faces, f5a,f5bp,f5cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5cp],U_fp[f5bp],U_fp[f5a],
                                                    V_fp[f5cp],V_fp[f5bp],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
            if (has_cp(cp_faces, f5a,f5c,f5bp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5bp],U_fp[f5c],U_fp[f5a],
                                                    V_fp[f5bp],V_fp[f5c],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // tri6
          size_t f6a = vid(t, i,  j,  sz);
          size_t f6b = vid(t, i,  j+1,sz);
          size_t f6c = vid(t, i+1,j+1,sz);
          size_t f6ap = f6a - dv, f6bp = f6b - dv, f6cp = f6c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            if (has_cp(cp_faces, f6a,f6b,f6cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6cp],U_fp[f6b],U_fp[f6a],
                                                    V_fp[f6cp],V_fp[f6b],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        // === 逐点 eb 完成 ===
        T abs_eb = required_eb;                         // 用枚举得到的点级别 eb
        int id = eb_exponential_quantize(abs_eb,        // 注意：与解码端保持一致
                                         base, log_of_base, threshold);
        // 如果 abs_eb==0 → 该点必须无损直存
        if (abs_eb == 0){
          *(eb_pos++) = 0;            // eb-id = 0 → 无损点
          unpred.push_back(U[v]);     // 原始浮点（解码端直接回填）
          unpred.push_back(V[v]);
          continue;
        }
        *eb_pos = id; // 写入该点的 eb-id（Huffman 之后很短）

        // ===== 量化该点 U/V（预测器：AR(2) + 残差 2D Lorenzo）=====
        bool unpred_flag=false;
        T dec[2];
        T abs_err_fp_q[2] = {0,0};

        for (int p=0; p<2; ++p){
          T *base = (p==0)? U_fp : V_fp;
          T  curv = (p==0) ? curU_val : curV_val;

          // 时间预测 p_t(i,j)
          T p_tij = time_pred_at(base, v, t);
          // 残差的 2D Lorenzo 预测
          T epred = e2d_pred_at(base, v, t, i, j);

          // 对“残差的残差”做区间量化
          T diff = (curv - p_tij) - epred;
          T qd = (std::llabs(diff)/abs_eb) + 1;
          if (qd < capacity){
            qd = (diff > 0) ? qd : -qd;
            int qindex = (int)(qd/2) + intv_radius;
            dq_pos[p] = qindex;

            // 解码侧对应：ehat = epred + 2*(qindex-intv_radius)*abs_eb; xhat = p_tij + ehat
            T ehat = epred + (T)2*(qindex - intv_radius)*abs_eb;
            T xhat = p_tij + ehat;

            // 守门：必须满足 |xhat - x| <= abs_eb
            if (std::llabs(xhat - curv) > abs_eb){
              unpred_flag = true; break;
            }
            dec[p] = xhat;
            abs_err_fp_q[p] = std::llabs(xhat - curv);
          }else{
            unpred_flag = true; break;
          }
        }

        if (unpred_flag){
          *(eb_pos++) = 0;                  // 覆盖为无损
          unpred.push_back(U[v]);
          unpred.push_back(V[v]);
        }else{
          ++eb_pos;
          dq_pos += 2;
          *curU = dec[0];                   // 回写“已重构值”，供后续 i,j,t 的预测使用
          *curV = dec[1];

          // 编码端自检（浮点域）
          double abs_eb_fp = (double)abs_eb / (double)scale;
          double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
          double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
          enc_max_abs_eb_fp   = std::max(enc_max_abs_eb_fp, abs_eb_fp);
          enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
        }
      }
    }
  }

  // 9) 打包码流（与原版一致）
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);
  std::cout << "write intv_radius = " << intv_radius << "\n";

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  std::cout << "write unpred cnt = " << unpred_cnt << ",ratio=" << (double)unpred_cnt/(2*N) << "\n";
  if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);

  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::free(eb_quant_index);

  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  std::free(U_fp); std::free(V_fp);
  return compressed;
}

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_3DL_AR2(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb,                  // 全局绝对误差上限（浮点域）：max_eb(fp)=max_pwr_eb
    EbMode mode 
){
  using T = int64_t;
  const Size3 sz{ (int)r1,(int)r2,(int)r3 };
  const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

  // 1) 定点化（先乘后 llround）
  T *U_fp=(T*)std::malloc(N*sizeof(T));
  T *V_fp=(T*)std::malloc(N*sizeof(T));
  if(!U_fp || !V_fp){
    if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp);
    compressed_size=0; return nullptr;
  }
  T range=0;
  T scale = convert_to_fixed_point<T_data,T>(U, V, N, U_fp, V_fp, range);

  // 2) 预计算：全局 CP 面集合（一次性）
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();


  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // 3) 量化/编码缓冲
  int* eb_quant_index  = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index= (int*)std::malloc(2*N*sizeof(int)); // U/V 交错
  double enc_max_abs_eb_fp   = 0.0; //编码端自检（浮点）
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred; unpred.reserve((N/10)*2);
  double alpha;
  double beta;
    {
      // 用前 K 帧做估计（3~200 帧），按你数据量设置；K>=3 才能估计到 gamma_2
      const int K = (int)std::min<size_t>(200,Tt);
      if (K >= 3) {
          // 为避免与主循环中的 si/sj/sk 重名，这里用 e 开头
          const ptrdiff_t esi=(ptrdiff_t)W, esj=(ptrdiff_t)1, esk=(ptrdiff_t)(H*W);

          // 纯 3D Lorenzo 预测（基于“原始定点场”），不回写
          auto pred3D_src = [&](const T* cur, int t, int i, int j)->T {
              T d0 = (t&&i&&j)? cur[-esk - esi - esj] : 0;
              T d1 = (t&&i)   ? cur[-esk - esi]      : 0;
              T d2 = (t&&j)   ? cur[-esk - esj]      : 0;
              T d3 = (t)      ? cur[-esk]            : 0;
              T d4 = (i&&j)   ? cur[-esi - esj]      : 0;
              T d5 = (i)      ? cur[-esi]            : 0;
              T d6 = (j)      ? cur[-esj]            : 0;
              return d0 + d3 + d5 + d6 - d1 - d2 - d4;
          };

          // 只需缓存两片历史残差（U/V 各两片）
          std::vector<T> r1U(H*W,(T)0), r2U(H*W,(T)0), rcurU(H*W,(T)0);
          std::vector<T> r1V(H*W,(T)0), r2V(H*W,(T)0), rcurV(H*W,(T)0);

          long double g0 = 0.0L, g1 = 0.0L, g2 = 0.0L; // Σ r_t^2, Σ r_t r_{t-1}, Σ r_t r_{t-2}

          for (int t=0; t<K; ++t) {
              for (int i=0; i<(int)H; ++i){
                  for (int j=0; j<(int)W; ++j){
                      const size_t v = vid(t,i,j,sz);
                      // 残差 = 原始值 - 纯 L3D 预测（均在定点域）
                      const T rU = U_fp[v] - pred3D_src(U_fp + v, t,i,j);
                      const T rV = V_fp[v] - pred3D_src(V_fp + v, t,i,j);
                      const size_t idx2D = (size_t)i*W + (size_t)j;
                      rcurU[idx2D] = rU;
                      rcurV[idx2D] = rV;

                      if (t >= 2){
                          const T r1u = r1U[idx2D], r2u = r2U[idx2D];
                          const T r1v = r1V[idx2D], r2v = r2V[idx2D];

                          // U 分量计入自/互协方差
                          g0 += (long double)rU * (long double)rU;
                          g1 += (long double)rU * (long double)r1u;
                          g2 += (long double)rU * (long double)r2u;
                          // V 分量同样计入（合到一起更稳）
                          g0 += (long double)rV * (long double)rV;
                          g1 += (long double)rV * (long double)r1v;
                          g2 += (long double)rV * (long double)r2v;
                      }
                  }
              }
              // 滚动（t→t-1→t-2）
              std::swap(r2U, r1U); std::swap(r1U, rcurU);
              std::swap(r2V, r1V); std::swap(r1V, rcurV);
          }

          // Yule–Walker 2x2： [g0 g1; g1 g0] [α;β] = [g1; g2]
          double alpha = 0.0, beta = 0.0;
          const long double lam = 1e-9L * (fabsl(g0) + 1.0L); // 轻微正则
          const long double a11 = g0 + lam, a22 = g0 + lam, a12 = g1, a21 = g1;
          const long double b1  = g1,      b2  = g2;
          const long double det = a11*a22 - a12*a21;

          double rho1 = (g0 != 0) ? (double)(g1 / g0) : 0.0;
          double rho2 = (g1 != 0) ? (double)(g2 / g1) : 0.0;
          std::cout << "[AR2-DIAG] rho1="<<rho1<<" rho2="<<rho2
          << " det="<< (double)det << " g0="<<(double)g0
          << " g1="<<(double)g1 << " g2="<<(double)g2 << "\n";

          if (fabsl(det) > 1e-30L) {
              alpha = (double)((b1*a22 - b2*a12)/det);
              beta  = (double)((a11*b2 - a21*b1)/det);
          } else {
              // 退化：用 AR(1) ρ 的双重根近似到 AR(2)
              double rho = 0.0;
              if (g0 > 0) rho = (double)(g1 / g0);
              rho = std::max(-0.99, std::min(0.99, rho));
              alpha = 2.0*rho;
              beta  = -rho*rho;
          }

          // 稳定性投影（AR(2) 的稳定三角： |β|<1, α∈(β-1, 1-β) ）
          const double eps = 1e-6;
          beta = std::max(-1.0 + eps, std::min(1.0 - eps, beta));
          const double amin = beta - 1.0 + eps;
          const double amax = 1.0 - beta - eps;
          if (alpha < amin) alpha = amin;
          if (alpha > amax) alpha = amax;
          std::cout << "[AR2-EST] K="<<K
                    << " alpha="<<alpha<<" beta="<<beta<<"\n";
      }
  }
  // === PATCH [AR2-EST] end ===

  // 参数与量化槽
  const int base = 2;                             // 仍写头，方便向后兼容
  const int capacity = 65536;
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);


  std::vector<T> r1U(H*W, (T)0), r2U(H*W, (T)0), rcurU(H*W, (T)0);
  std::vector<T> r1V(H*W, (T)0), r2V(H*W, (T)0), rcurV(H*W, (T)0);

  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W; // 层间位移（你已有）

  // 3D 一阶 Lorenzo 主预测（时间最慢）
  auto pred3D = [&](const T* cur, int t, int i, int j)->T {
      T d0 = (t&&i&&j)? cur[-sk - si - sj] : 0;
      T d1 = (t&&i)   ? cur[-sk - si]      : 0;
      T d2 = (t&&j)   ? cur[-sk - sj]      : 0;
      T d3 = (t)      ? cur[-sk]           : 0;
      T d4 = (i&&j)   ? cur[-si - sj]      : 0;
      T d5 = (i)      ? cur[-si]           : 0;
      T d6 = (j)      ? cur[-sj]           : 0;
      return d0 + d3 + d5 + d6 - d1 - d2 - d4;
  };

  T max_eb = 0;
  if(mode == EbMode::Relative){

    max_eb = max_pwr_eb * range; // 相对误差转绝对误差
    printf("Compression Using Relative Eb Mode!\n");
  }
  else if (mode == EbMode::Absolute){
    printf("Compression Using Absolute Eb Mode!\n");
    max_eb = max_pwr_eb * scale; // 浮点→定点
  }
  else{
    std::cerr << "Error: Unsupported EbMode!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  // T max_eb = range * max_pwr_eb;


  // 定点 LSB 的阈值（退化门控）
  const T threshold = 1;                          // 幂指数量化阈值（LSB）


  // 4) 逐顶点：枚举与该顶点相关的三角面 → 最小 eb
  // 6 个平面邻接方向（左右、上下、主对角）
  const int di[6] = { 0,  0,  1, -1,  1, -1 }; //y-axis
  const int dj[6] = { 1, -1,  0,  0,  1, -1 }; //x-axis

  for (int t=0; t<(int)Tt; ++t){
    if(t % 100 == 0){
      printf("processing slice %d / %d\n", t, (int)Tt);
    }
    for (int i=0; i<(int)H; ++i){
      for (int j=0; j<(int)W; ++j){
        const size_t v = vid(t,i,j,sz);
        // 缓存原始定点值（预测/回写前）
        T *curU = U_fp + v;
        T *curV = V_fp + v;
        const T curU_val = *curU;
        const T curV_val = *curV;

        // —— 收集最小 eb——
        T required_eb = max_eb;

        // (A) 层内 t：影响 (i-1..i, j-1..j) 的 4 个 cell，每 cell 两三角
        for (int ci=i-1; ci<=i; ++ci){
          if (!in_range(ci, (int)H-1)) continue;
          for (int cj=j-1; cj<=j; ++cj){
            if (!in_range(cj, (int)W-1)) continue;

            size_t v00 = vid(t,ci,  cj,  sz);
            size_t v10 = vid(t,ci,  cj+1,sz); //v10=> x=1,y=0
            size_t v01 = vid(t,ci+1,cj,  sz);
            size_t v11 = vid(t,ci+1,cj+1,sz);

            // Upper: (v00,v01,v11)
            if (v==v00 || v==v01 || v==v11){ 
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v01,v11);
              #endif
              if (has_cp(cp_faces, v00,v01,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v01],U_fp[v00],
                                                    V_fp[v11],V_fp[v01],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
            // Lower: (v00,v10,v11)
            if (v==v00 || v==v10 || v==v11){
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v10,v11);
              #endif
              if (has_cp(cp_faces, v00,v10,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v10],U_fp[v00],
                                                    V_fp[v11],V_fp[v10],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }
        
        // (B1) 侧面 [t, t+1]
        if (t < (int)Tt-1){
          for (int k=0; k<6; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a  = vid(t, i, j, sz);
            size_t b  = vid(t, ni,nj, sz);
            size_t ap = a + dv, bp = b + dv;
            if (k == 0 || k==3 || k==5){
              // (a,b,bp) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (a,bp,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,bp,ap);
              #endif
              if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[bp],U_fp[a],
                                                        V_fp[ap],V_fp[bp],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,ap) for k = 1,2,4
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (b,ap,bp) for k = 1,2,4 //好像与点无关
              // if (has_cp(cp_faces, b,ap,bp)) { required_eb = 0; goto after_min_eb; }
              //   {
              //     T eb = derive_cp_abs_eb_sos_online<T>(U_fp[b],U_fp[ap],U_fp[bp],
              //                                           V_fp[b],V_fp[ap],V_fp[bp]);
              //     if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
              //     if (eb < required_eb) required_eb = eb;
              //   }
            }

          }
        }

        // (B2) 侧面 [t-1, t]
        if (t > 0){
          for (int k=0; k<6 ; ++k){
            int ni = i + di[k], nj = j + dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a = vid(t,i,j,sz); //v
            size_t b = vid(t,ni,nj,sz);
            size_t ap = a - dv, bp = b - dv; //ap,bp为上一层
            if (k == 0 || k==3 || k==5){
              // (a,b,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,bp) for k = 1,2,4 //侧面上三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
                
              // (a,ap,bp) for k = 1,2,4 //侧面下三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,ap,bp);
              #endif
              if (has_cp(cp_faces, a,ap,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[ap],U_fp[a],
                                                        V_fp[bp],V_fp[ap],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }

          }
        }

        // (C) 内部剖分面：两片 ts ∈ {t, t-1}；每相邻 cell 的 Upper/Lower 各 2 面
        // (C1) [t, t+1] 内部剖分面
        if (t < (int)Tt-1){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|
      
          //triange 1 [x,y,t] = [(0,0,0),(1,0,0),(0,-1,0)]: has 2 faces
          size_t f1a = vid(t,  i,  j,  sz);
          size_t f1b = vid(t,  i,j+1,  sz);
          size_t f1c = vid(t,  i-1,j,  sz);
          size_t f1ap = f1a + dv, f1bp = f1b + dv, f1cp = f1c + dv;
          if (in_range(i-1,(int)H) && in_range(j+1,(int)W)){
            // (f1a,f1cp,f1b)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1b);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1b)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1b],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1b],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f1a,f1cp,f1bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1bp);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1bp],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1bp],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          //triange 2 [x,y,t] = [(0,0,0),(0,-1,0),(-1,-1,0)]: has 2 faces
          size_t f2a = vid(t,  i,  j,  sz);
          size_t f2b = vid(t,  i-1,j,  sz);
          size_t f2c = vid(t,  i-1,j-1,sz);
          size_t f2ap = f2a + dv, f2bp = f2b + dv, f2cp = f2c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f2a,f2c,f2bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2c,f2bp);
            #endif
            if (has_cp(cp_faces, f2a,f2c,f2bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2bp],U_fp[f2c],U_fp[f2a],
                                                    V_fp[f2bp],V_fp[f2c],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f2a,f2bp,f2cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2bp,f2cp);
            #endif
            if (has_cp(cp_faces, f2a,f2bp,f2cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2cp],U_fp[f2bp],U_fp[f2a],
                                                    V_fp[f2cp],V_fp[f2bp],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          // triange 3 [x,y,t] = [(0,0,0),(-1,-1,0),(-1,0,0)]: has 1 faces
          size_t f3a = vid(t,  i,  j,  sz);
          size_t f3b = vid(t,  i-1,j-1,sz);
          size_t f3c = vid(t,  i,j-1,  sz);
          size_t f3ap = f3a + dv, f3bp = f3b + dv, f3cp = f3c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3bp,f3c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3bp,f3c);
            #endif
            if (has_cp(cp_faces, f3a,f3bp,f3c)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3c],U_fp[f3bp],U_fp[f3a],
                                                    V_fp[f3c],V_fp[f3bp],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // triange 4 0 faces
          // triange 5 0 faces
          // triange 6 [x,y,t] = [(0,0,0),(1,0,0),(1,1,0)]: has 1 faces
          size_t f6a = vid(t,  i,  j,  sz);
          size_t f6b = vid(t,  i,j+1,  sz);
          size_t f6c = vid(t,  i+1,j+1,sz);
          size_t f6ap = f6a + dv, f6bp = f6b + dv, f6cp = f6c + dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6bp,f6c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6bp,f6c);
            #endif
            if (has_cp(cp_faces, f6a,f6bp,f6c)) {
              required_eb=0; 
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6c],U_fp[f6bp],U_fp[f6a],
                                                    V_fp[f6c],V_fp[f6bp],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }


        // (C2) [t-1, t] 内部剖分面 并不与C1对称
        if (t > 0){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|

          //triange 1 0 face
          //triange 2 0 face
          //triange 3  has 1 faces
          size_t f3a = vid(t, i,  j,  sz);   
          size_t f3b = vid(t, i-1,j-1,sz);
          size_t f3c = vid(t, i,j-1,  sz);
          size_t f3ap = f3a - dv, f3bp = f3b - dv, f3cp = f3c - dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3b,f3cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3b,f3cp);
            #endif
            if (has_cp(cp_faces, f3a,f3b,f3cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3cp],U_fp[f3b],U_fp[f3a],
                                                    V_fp[f3cp],V_fp[f3b],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 4 has 2 faces
          size_t f4a = vid(t, i,  j,  sz);
          size_t f4b = vid(t, i,j-1,  sz);
          size_t f4c = vid(t,i+1,j,  sz);
          size_t f4ap = f4a - dv, f4bp = f4b - dv, f4cp = f4c - dv;
          if (in_range(i+1,(int)H) && in_range(j-1,(int)W)){
            // (f4a,f4bp,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4bp,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4bp,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4bp],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4bp],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f4a,f4b,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4b,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4b,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4b],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4b],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 5 has 2 faces
          size_t f5a = vid(t, i,  j,  sz);
          size_t f5b = vid(t, i+1,j,  sz);
          size_t f5c = vid(t, i+1,j+1,sz);
          size_t f5ap = f5a - dv, f5bp = f5b - dv, f5cp = f5c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f5a,f5bp,f5cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5bp,f5cp);
            #endif
            if (has_cp(cp_faces, f5a,f5bp,f5cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5cp],U_fp[f5bp],U_fp[f5a],
                                                    V_fp[f5cp],V_fp[f5bp],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f5a,f5c,f5bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5c,f5bp);
            #endif
            if (has_cp(cp_faces, f5a,f5c,f5bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5bp],U_fp[f5c],U_fp[f5a],
                                                    V_fp[f5bp],V_fp[f5c],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 6 has 1 faces
          size_t f6a = vid(t, i,  j,  sz);
          size_t f6b = vid(t, i,j+1,  sz);
          size_t f6c = vid(t, i+1,j+1,sz);
          size_t f6ap = f6a - dv, f6bp = f6b - dv, f6cp = f6c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6b,f6cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6b,f6cp);
            #endif
            if (has_cp(cp_faces, f6a,f6b,f6cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6cp],U_fp[f6b],U_fp[f6a],
                                                    V_fp[f6cp],V_fp[f6b],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }
        // {
        //   // 保证 |dec-orig| ≤ abs_eb < |orig| ⇒ 不会跨 0 翻符号
        //   T sign_guard = std::min(std::llabs(curU_val), std::llabs(curV_val));
        //   if (sign_guard > 0 && required_eb >= sign_guard){
        //       T max_eb_to_preserve_sign = sign_guard - 1; // 留 1 LSB 余量
        //       if (max_eb_to_preserve_sign < required_eb)
        //           required_eb = max_eb_to_preserve_sign;
        //   }
        // }
        T abs_eb = required_eb;
        int id = eb_exponential_quantize(abs_eb,base,log_of_base,threshold);
        
        if (abs_eb == 0 )
        {
          *(eb_pos++) = 0;            // ebid=0 → 无损点
          unpred.push_back(U[v]);     // 原始浮点（解码端直接回填）
          unpred.push_back(V[v]);
          continue;
        }

        // ===== 量化 eb（配套的新幂指数量化）并编码该顶点 =====
        {
          // T abs_eb = required_eb;
          // // int id = eb_exponential_quantize_new(abs_eb, threshold); // id>=1，abs_eb 替换为代表值
          
          // // *eb_pos = id;
          // int id = eb_exponential_quantize(abs_eb,base,log_of_base,threshold);
          *eb_pos = id;

          bool unpred_flag=false;
          T dec[2];
          T abs_err_fp_q[2] = {0,0};

          // === PATCH [AR2]：L3D 主预测 + 残差 AR(2) ===
          const size_t idx2D = (size_t)i * W + (size_t)j;

          // 主预测（基于已回写邻域）
          const T predL3D_U = pred3D(U_fp + v, t, i, j);
          const T predL3D_V = pred3D(V_fp + v, t, i, j);

          // 残差的 AR(2) 预测
          auto ar_pred = [&](T r1, T r2) -> T {
              double num = alpha * static_cast<double>(r1)
                        + beta * static_cast<double>(r2);
              return static_cast<T>(llround(num)); 
          };
          const T arU = ar_pred(r1U[idx2D], r2U[idx2D]);
          const T arV = ar_pred(r1V[idx2D], r2V[idx2D]);

          const T predU_final = predL3D_U + arU;
          const T predV_final = predL3D_V + arV;

          // eb==0（必保点/退化）仍走无损，但要更新当前帧残差供下一帧使用
          if (abs_eb == 0){
            *(eb_pos++) = 0;
            unpred.push_back(U[v]);
            unpred.push_back(V[v]);
            rcurU[idx2D] = (T)curU_val - predL3D_U;   // 按 L3D 记录残差
            rcurV[idx2D] = (T)curV_val - predL3D_V;
            continue;
          }

          // U 分量量化
          {
            const T diff = (T)curU_val - predU_final;
            T qd = (std::llabs(diff)/abs_eb) + 1;
            if (qd < capacity){
              qd = (diff > 0) ? qd : -qd;
              int qindex = (int)(qd/2) + intv_radius;
              dq_pos[0] = qindex;
              dec[0] = predU_final + 2*(qindex - intv_radius)*abs_eb;
              if (std::llabs(dec[0] - (T)curU_val) > abs_eb) unpred_flag = true;
              else abs_err_fp_q[0] = std::llabs(dec[0] - (T)curU_val);
            }else{
              unpred_flag = true;
            }
          }

          // V 分量量化
          if (!unpred_flag){
            const T diff = (T)curV_val - predV_final;
            T qd = (std::llabs(diff)/abs_eb) + 1;
            if (qd < capacity){
              qd = (diff > 0) ? qd : -qd;
              int qindex = (int)(qd/2) + intv_radius;
              dq_pos[1] = qindex;
              dec[1] = predV_final + 2*(qindex - intv_radius)*abs_eb;
              if (std::llabs(dec[1] - (T)curV_val) > abs_eb) unpred_flag = true;
              else abs_err_fp_q[1] = std::llabs(dec[1] - (T)curV_val);
            }else{
              unpred_flag = true;
            }
          }

          if (unpred_flag){
            *(eb_pos++) = 0;
            unpred.push_back(U[v]);
            unpred.push_back(V[v]);

            // 更新当前帧残差（按 L3D）
            rcurU[idx2D] = (T)curU_val - predL3D_U;
            rcurV[idx2D] = (T)curV_val - predL3D_V;
          }else{
            ++eb_pos;
            dq_pos += 2;
            *curU = dec[0];
            *curV = dec[1];

            // 记录“已重建-主预测(L3D)”残差，供下一帧 AR(2)
            rcurU[idx2D] = dec[0] - predL3D_U;
            rcurV[idx2D] = dec[1] - predL3D_V;

            // 编码端自检（保持不变）
            double abs_eb_fp = (double)abs_eb / (double)scale;
            double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
            double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
            enc_max_abs_eb_fp   = std::max(enc_max_abs_eb_fp, abs_eb_fp);
            enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
          }
        }
      }
    }
    // === PATCH [AR2] 帧末滚动：rcur -> r1 -> r2
    std::swap(r2U, r1U); std::swap(r1U, rcurU);
    std::swap(r2V, r1V); std::swap(r1V, rcurV);
    std::fill(rcurU.begin(), rcurU.end(), (T)0);
    std::fill(rcurV.begin(), rcurV.end(), (T)0);
  }

  // std::cerr << "[ENC] max abs_eb(fp) = " << enc_max_abs_eb_fp
  //           << ", max actual |err|(fp) = " << enc_max_real_err_fp << "\n";

  
  #if CP_DEBUG_VISIT
  {
      size_t miss = 0, shown = 0;
      for (const auto &fk : cp_faces){
          if (!visited_faces.count(fk)){
              ++miss;
              if (shown < 50){ // 只演示前 50 个，避免刷屏
                  size_t a = fk.v[0], b = fk.v[1], c = fk.v[2];
                  auto [ta,ia,ja] = decode_tij(a);
                  auto [tb,ib,jb] = decode_tij(b);
                  auto [tc,ic,jc] = decode_tij(c);
                  fprintf(stderr,
                    "[MISSING] face (%zu,%zu,%zu)  "
                    "A(t=%d,i=%d,j=%d)  B(t=%d,i=%d,j=%d)  C(t=%d,i=%d,j=%d)  type=%s\n",
                    a,b,c, ta,ia,ja, tb,ib,jb, tc,ic,jc,
                    classify_face(a,b,c));
                  ++shown;
              }
          }
      }
      fprintf(stderr, "[COVERAGE] encoder touched faces = %zu, cp_faces = %zu, missing = %zu\n",
              visited_faces.size(), cp_faces.size(), miss);
  }
  #endif
  // 5) 打包码流
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);
  std::cout << "write intv_radius = " << intv_radius << "\n";

  size_t unpred_cnt = unpred.size();
  std::cout << "write unpred cnt = " << unpred_cnt << ",ratio=" << (double)unpred_cnt/(2*N) << "\n";
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);

  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::free(eb_quant_index);

  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  std::free(U_fp); std::free(V_fp);
  return compressed;
}


template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_mop(   //Mixture of Predictors
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb,                  // 全局绝对误差上限（浮点域）：max_eb(fp)=max_pwr_eb
    EbMode mode 
){
  using T = int64_t;
  const Size3 sz{ (int)r1,(int)r2,(int)r3 };
  const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

  // 1) 定点化（先乘后 llround）
  T *U_fp=(T*)std::malloc(N*sizeof(T));
  T *V_fp=(T*)std::malloc(N*sizeof(T));
  if(!U_fp || !V_fp){
    if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp);
    compressed_size=0; return nullptr;
  }
  T range=0;
  T scale = convert_to_fixed_point<T_data,T>(U, V, N, U_fp, V_fp, range);

  // 2) 预计算：全局 CP 面集合（一次性）
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();



  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // 3) 量化/编码缓冲
  int* eb_quant_index  = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index= (int*)std::malloc(2*N*sizeof(int)); // U/V 交错
  double enc_max_abs_eb_fp   = 0.0; //编码端自检（浮点）
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred; unpred.reserve((N / 4) * 2);

  // 参数与量化槽
  const int base = 2;                             // 仍写头，方便向后兼容
  const int capacity = 65536; //65536
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);

  // 把浮点域绝对误差上限转到定点域（LSB）
  // const T max_eb = (T) std::llround( (long double)max_pwr_eb * (long double)scale );

  T max_eb = 0;
  if(mode == EbMode::Relative){

    max_eb = max_pwr_eb * range; // 相对误差转绝对误差
    printf("Compression Using Relative Eb Mode!\n");
  }
  else if (mode == EbMode::Absolute){
    printf("Compression Using Absolute Eb Mode!\n");
    max_eb = max_pwr_eb * scale; // 浮点→定点
  }
  else{
    std::cerr << "Error: Unsupported EbMode!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  // T max_eb = range * max_pwr_eb;


  // 定点 LSB 的阈值（退化门控）
  const T threshold = 1;                          // 幂指数量化阈值（LSB）


  // 4) 逐顶点：枚举与该顶点相关的三角面 → 最小 eb
  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W; // 层间位移
  const int BH = SZ_MOP_BH, BW = SZ_MOP_BW;
  // const int BH = H, BW = W; // 全图为块
  const int blocks_i = (int)((H + BH - 1) / BH);
  const int blocks_j = (int)((W + BW - 1) / BW);
  const size_t blocks_per_t = (size_t)blocks_i * (size_t)blocks_j;
  // ========== 1) 准备块模式流（每块 2bit）==========
  std::vector<uint8_t> block_modes; block_modes.resize(Tt * blocks_per_t, (uint8_t)PM::L3D);
  
  //记录数据metadata，计算出delta
  // double t_range_start = 0.0;
  // double t_range_end = 15.0;
  // double delta_t = (t_range_end - t_range_start) / (Tt - 1);
  // double x_range_start = -0.5;
  // double x_range_end = 7.5;
  // double y_range_start = -0.5;
  // double y_range_end = 0.5;
  // double delta_x = (x_range_end - x_range_start) / (W);
  // double delta_y = (y_range_end - y_range_start) / (H);
  // double dt_dx = delta_t / delta_x;
  // double dt_dy = delta_t / delta_y;
  // 临时写法
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

  pred_advect_set_params(/*H*/H, /*W*/W, /*dt/dx*/dt_dx, /*dt/dy*/dt_dy, /*scale*/scale, /*max_disp*/-1.0);
  const size_t plane = (size_t)H * (size_t)W;
  // 6 个平面邻接方向（左右、上下、主对角）
  const int di[6] = { 0,  0,  1, -1,  1, -1 }; //y-axis
  const int dj[6] = { 1, -1,  0,  0,  1, -1 }; //x-axis

  for (int t=0; t<(int)Tt; ++t){
    if(t % 100 == 0){
      printf("processing slice %d / %d\n", t, (int)Tt);
    }

    // === 每帧一次：绑定 t-1 的 U/V 速度平面（供 ADVECT 使用）===
    if (t >= 1) {
        const int64_t* u_prev_plane = (const int64_t*)(U_fp + (size_t)(t-1)*plane);
        const int64_t* v_prev_plane = (const int64_t*)(V_fp + (size_t)(t-1)*plane);
        pred_advect_bind_prev_uv(u_prev_plane, v_prev_plane, /*si_uv=*/si, /*sj_uv=*/sj);
    } else {
        // 可选：显式“解绑”以防残留
        pred_advect_bind_prev_uv(nullptr, nullptr, 0, 0); // 实现里把 ready_uv=0
    }

    for (int bi=0; bi<(int)H; bi+=BH){
      for (int bj=0; bj<(int)W; bj+=BW){

        // ========== 2) 前视抽样，确定该块预测器 ==========
        // PM pm = choose_pred_mode_block_lookahead<int64_t>(
        //             U_fp, V_fp, (int)H, (int)W, t, bi, bj, BH, BW, si, sj, sk);

        // PM pm = choose_pred_mode_block_lookahead_qaware<int64_t>(
        //     U_fp, V_fp, cp_faces, (int)H, (int)W, (int)Tt, t,
        //     bi, bj, BH, BW, si, sj, sk,
        //     /*max_eb_global=*/max_eb, /*capacity=*/capacity,
        //     /*subs=*/SZ_MOP_EVAL_SUBS,
        //     /*use_log_cost=*/true,
        //     /*switch_thresh=*/0.99L);
        // PM pm = choose_pred_mode_block_lookahead_q(
        //     U_fp, V_fp, H,W, t, bi,bj, SZ_MOP_BH, SZ_MOP_BW,
        //     si,sj,sk, /*eb_lsb_cap=*/ std::max<T>(1, max_eb),
        //     /*subs=*/ SZ_MOP_EVAL_SUBS);
        // PM pm = choose_pred_mode_block_lookahead_bitspersample(
        //   U_fp, V_fp,
        //   (int)H,(int)W, t,
        //   bi,bj, BH,BW,
        //   si,sj,sk,
        //   /*eb_lsb_cap=*/ std::max<T>(1, max_eb),   // 与编码端一致
        //   /*intv_radius=*/ intv_radius,             // 与编码端一致
        //   /*capacity=*/ capacity,
        //   /*subs=*/ SZ_MOP_EVAL_SUBS);
        PM pm = choose_pred_mode_block_lookahead_bitspersample_entire_block_sample(
          U_fp, V_fp,
          (int)H,(int)W, t,
          bi,bj, BH,BW,
          si,sj,sk,
          /*eb_lsb_cap=*/ std::max<T>(1, max_eb),   // 与编码端一致
          /*intv_radius=*/ intv_radius,             // 与编码端一致
          /*capacity=*/ capacity,
          /*subs=*/ SZ_MOP_EVAL_SUBS);
        const int b_i = bi / BH, b_j = bj / BW;
        block_modes[(size_t)t*blocks_per_t + (size_t)b_i*blocks_j + (size_t)b_j] = (uint8_t)pm;

        const int i_end = std::min(bi+BH, (int)H);
        const int j_end = std::min(bj+BW, (int)W);

        for (int i=bi; i<i_end; ++i){
          for (int j=bj; j<j_end; ++j){

            const size_t v = (size_t)t*H*W + (size_t)i*W + (size_t)j;
            T *curU = U_fp + v;
            T *curV = V_fp + v;
            const T curU_val = *curU;
            const T curV_val = *curV;

            // —— 你原来的 CP/SoS 面枚举，得到 required_eb —— （不变）
            T required_eb = max_eb;
            // (A) 层内 t：影响 (i-1..i, j-1..j) 的 4 个 cell，每 cell 两三角
            for (int ci=i-1; ci<=i; ++ci){
              if (!in_range(ci, (int)H-1)) continue;
              for (int cj=j-1; cj<=j; ++cj){
                if (!in_range(cj, (int)W-1)) continue;

                size_t v00 = vid(t,ci,  cj,  sz);
                size_t v10 = vid(t,ci,  cj+1,sz); //v10=> x=1,y=0
                size_t v01 = vid(t,ci+1,cj,  sz);
                size_t v11 = vid(t,ci+1,cj+1,sz);

                // Upper: (v00,v01,v11)
                if (v==v00 || v==v01 || v==v11){ 
                  #if CP_DEBUG_VISIT
                  MARK_FACE(v00,v01,v11);
                  #endif
                  if (has_cp(cp_faces, v00,v01,v11)) { required_eb = 0;}
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v01],U_fp[v00],
                                                        V_fp[v11],V_fp[v01],V_fp[v00]);
                  if (eb < required_eb) required_eb = eb;
                }
                // Lower: (v00,v10,v11)
                if (v==v00 || v==v10 || v==v11){
                  #if CP_DEBUG_VISIT
                  MARK_FACE(v00,v10,v11);
                  #endif
                  if (has_cp(cp_faces, v00,v10,v11)) { required_eb = 0;}
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v10],U_fp[v00],
                                                        V_fp[v11],V_fp[v10],V_fp[v00]);
                  if (eb < required_eb) required_eb = eb;
                }
              }
            }
            
            // (B1) 侧面 [t, t+1]
            if (t < (int)Tt-1){
              for (int k=0; k<6; ++k){
                int ni=i+di[k], nj=j+dj[k];
                if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
                size_t a  = vid(t, i, j, sz);
                size_t b  = vid(t, ni,nj, sz);
                size_t ap = a + dv, bp = b + dv;
                if (k == 0 || k==3 || k==5){
                  // (a,b,bp) for k = 0,3,5
                  #if CP_DEBUG_VISIT
                  MARK_FACE(a,b,bp);
                  #endif
                  if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                            V_fp[bp],V_fp[b],V_fp[a]);
                      if (eb < required_eb) required_eb = eb;
                    }
                  // (a,bp,ap) for k = 0,3,5
                  #if CP_DEBUG_VISIT
                  MARK_FACE(a,bp,ap);
                  #endif
                  if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0;}
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[bp],U_fp[a],
                                                            V_fp[ap],V_fp[bp],V_fp[a]);
                      if (eb < required_eb) required_eb = eb;
                    }
                }
                else{
                  // (a,b,ap) for k = 1,2,4
                  #if CP_DEBUG_VISIT
                  MARK_FACE(a,b,ap);
                  #endif
                  if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                            V_fp[ap],V_fp[b],V_fp[a]);
                      if (eb < required_eb) required_eb = eb;
                    }
                }

              }
            }

            // (B2) 侧面 [t-1, t]
            if (t > 0){
              for (int k=0; k<6 ; ++k){
                int ni = i + di[k], nj = j + dj[k];
                if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
                size_t a = vid(t,i,j,sz); //v
                size_t b = vid(t,ni,nj,sz);
                size_t ap = a - dv, bp = b - dv; //ap,bp为上一层
                if (k == 0 || k==3 || k==5){
                  // (a,b,ap) for k = 0,3,5
                  #if CP_DEBUG_VISIT
                  MARK_FACE(a,b,ap);
                  #endif
                  if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                            V_fp[ap],V_fp[b],V_fp[a]);
                      if (eb < required_eb) required_eb = eb;
                    }
                }
                else{
                  // (a,b,bp) for k = 1,2,4 //侧面上三角
                  #if CP_DEBUG_VISIT
                  MARK_FACE(a,b,bp);
                  #endif
                  if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                            V_fp[bp],V_fp[b],V_fp[a]);
                      if (eb < required_eb) required_eb = eb;
                    }
                    
                  // (a,ap,bp) for k = 1,2,4 //侧面下三角
                  #if CP_DEBUG_VISIT
                  MARK_FACE(a,ap,bp);
                  #endif
                  if (has_cp(cp_faces, a,ap,bp)) { required_eb = 0;}
                    {
                      T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[ap],U_fp[a],
                                                            V_fp[bp],V_fp[ap],V_fp[a]);
                      if (eb < required_eb) required_eb = eb;
                    }
                }

              }
            }

            // (C) 内部剖分面：两片 ts ∈ {t, t-1}；每相邻 cell 的 Upper/Lower 各 2 面
            // (C1) [t, t+1] 内部剖分面
            if (t < (int)Tt-1){
              //     ---------
              //     |  /| T5/|
              //     | / |  / |
              //     |/T4| /T6|
              //     ----X----|
              //     |T3/|T1/ |
              //     | / | /  |
              //     |/T2|/   |
              //     |---|----|
          
              //triange 1 [x,y,t] = [(0,0,0),(1,0,0),(0,-1,0)]: has 2 faces
              size_t f1a = vid(t,  i,  j,  sz);
              size_t f1b = vid(t,  i,j+1,  sz);
              size_t f1c = vid(t,  i-1,j,  sz);
              size_t f1ap = f1a + dv, f1bp = f1b + dv, f1cp = f1c + dv;
              if (in_range(i-1,(int)H) && in_range(j+1,(int)W)){
                // (f1a,f1cp,f1b)
                #if CP_DEBUG_VISIT
                MARK_FACE(f1a,f1cp,f1b);
                #endif
                if (has_cp(cp_faces, f1a,f1cp,f1b)) { required_eb=0;}
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1b],U_fp[f1cp],U_fp[f1a],
                                                        V_fp[f1b],V_fp[f1cp],V_fp[f1a]);
                  if (eb < required_eb) required_eb = eb;
                }
                // (f1a,f1cp,f1bp)
                #if CP_DEBUG_VISIT
                MARK_FACE(f1a,f1cp,f1bp);
                #endif
                if (has_cp(cp_faces, f1a,f1cp,f1bp)) { required_eb=0;}
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1bp],U_fp[f1cp],U_fp[f1a],
                                                        V_fp[f1bp],V_fp[f1cp],V_fp[f1a]);
                  if (eb < required_eb) required_eb = eb;
                }
              }

              //triange 2 [x,y,t] = [(0,0,0),(0,-1,0),(-1,-1,0)]: has 2 faces
              size_t f2a = vid(t,  i,  j,  sz);
              size_t f2b = vid(t,  i-1,j,  sz);
              size_t f2c = vid(t,  i-1,j-1,sz);
              size_t f2ap = f2a + dv, f2bp = f2b + dv, f2cp = f2c + dv;
              if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
                // (f2a,f2c,f2bp)
                #if CP_DEBUG_VISIT
                MARK_FACE(f2a,f2c,f2bp);
                #endif
                if (has_cp(cp_faces, f2a,f2c,f2bp)) { required_eb=0;}
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2bp],U_fp[f2c],U_fp[f2a],
                                                        V_fp[f2bp],V_fp[f2c],V_fp[f2a]);
                  if (eb < required_eb) required_eb = eb;
                }
                // (f2a,f2bp,f2cp)
                #if CP_DEBUG_VISIT
                MARK_FACE(f2a,f2bp,f2cp);
                #endif
                if (has_cp(cp_faces, f2a,f2bp,f2cp)) { required_eb=0;}
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2cp],U_fp[f2bp],U_fp[f2a],
                                                        V_fp[f2cp],V_fp[f2bp],V_fp[f2a]);
                  if (eb < required_eb) required_eb = eb;
                }
              }

              // triange 3 [x,y,t] = [(0,0,0),(-1,-1,0),(-1,0,0)]: has 1 faces
              size_t f3a = vid(t,  i,  j,  sz);
              size_t f3b = vid(t,  i-1,j-1,sz);
              size_t f3c = vid(t,  i,j-1,  sz);
              size_t f3ap = f3a + dv, f3bp = f3b + dv, f3cp = f3c + dv;
              if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
                // (f3a,f3bp,f3c)
                #if CP_DEBUG_VISIT
                MARK_FACE(f3a,f3bp,f3c);
                #endif
                if (has_cp(cp_faces, f3a,f3bp,f3c)) {
                  required_eb=0;
                }
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3c],U_fp[f3bp],U_fp[f3a],
                                                        V_fp[f3c],V_fp[f3bp],V_fp[f3a]);
                  if (eb < required_eb) required_eb = eb;
                }
              }
              // triange 4 0 faces
              // triange 5 0 faces
              // triange 6 [x,y,t] = [(0,0,0),(1,0,0),(1,1,0)]: has 1 faces
              size_t f6a = vid(t,  i,  j,  sz);
              size_t f6b = vid(t,  i,j+1,  sz);
              size_t f6c = vid(t,  i+1,j+1,sz);
              size_t f6ap = f6a + dv, f6bp = f6b + dv, f6cp = f6c + dv;
              if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
                // (f6a,f6bp,f6c)
                #if CP_DEBUG_VISIT
                MARK_FACE(f6a,f6bp,f6c);
                #endif
                if (has_cp(cp_faces, f6a,f6bp,f6c)) {
                  required_eb=0; 
                }
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6c],U_fp[f6bp],U_fp[f6a],
                                                        V_fp[f6c],V_fp[f6bp],V_fp[f6a]);
                  if (eb < required_eb) required_eb = eb;
                }
              }
            }


            // (C2) [t-1, t] 内部剖分面 并不与C1对称
            if (t > 0){
              //     ---------
              //     |  /| T5/|
              //     | / |  / |
              //     |/T4| /T6|
              //     ----X----|
              //     |T3/|T1/ |
              //     | / | /  |
              //     |/T2|/   |
              //     |---|----|

              //triange 1 0 face
              //triange 2 0 face
              //triange 3  has 1 faces
              size_t f3a = vid(t, i,  j,  sz);   
              size_t f3b = vid(t, i-1,j-1,sz);
              size_t f3c = vid(t, i,j-1,  sz);
              size_t f3ap = f3a - dv, f3bp = f3b - dv, f3cp = f3c - dv;
              if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
                // (f3a,f3b,f3cp)
                #if CP_DEBUG_VISIT
                MARK_FACE(f3a,f3b,f3cp);
                #endif
                if (has_cp(cp_faces, f3a,f3b,f3cp)) {
                  required_eb=0;
                }
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3cp],U_fp[f3b],U_fp[f3a],
                                                        V_fp[f3cp],V_fp[f3b],V_fp[f3a]);
                  if (eb < required_eb) required_eb = eb;
                }
              }
              //triange 4 has 2 faces
              size_t f4a = vid(t, i,  j,  sz);
              size_t f4b = vid(t, i,j-1,  sz);
              size_t f4c = vid(t,i+1,j,  sz);
              size_t f4ap = f4a - dv, f4bp = f4b - dv, f4cp = f4c - dv;
              if (in_range(i+1,(int)H) && in_range(j-1,(int)W)){
                // (f4a,f4bp,f4cp)
                #if CP_DEBUG_VISIT
                MARK_FACE(f4a,f4bp,f4cp);
                #endif
                if (has_cp(cp_faces, f4a,f4bp,f4cp)) { required_eb=0;}
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4bp],U_fp[f4a],
                                                        V_fp[f4cp],V_fp[f4bp],V_fp[f4a]);
                  if (eb < required_eb) required_eb = eb;
                }
                // (f4a,f4b,f4cp)
                #if CP_DEBUG_VISIT
                MARK_FACE(f4a,f4b,f4cp);
                #endif
                if (has_cp(cp_faces, f4a,f4b,f4cp)) { required_eb=0;}
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4b],U_fp[f4a],
                                                        V_fp[f4cp],V_fp[f4b],V_fp[f4a]);
                  if (eb < required_eb) required_eb = eb;
                }
              }
              //triange 5 has 2 faces
              size_t f5a = vid(t, i,  j,  sz);
              size_t f5b = vid(t, i+1,j,  sz);
              size_t f5c = vid(t, i+1,j+1,sz);
              size_t f5ap = f5a - dv, f5bp = f5b - dv, f5cp = f5c - dv;
              if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
                // (f5a,f5bp,f5cp)
                #if CP_DEBUG_VISIT
                MARK_FACE(f5a,f5bp,f5cp);
                #endif
                if (has_cp(cp_faces, f5a,f5bp,f5cp)) { required_eb=0; }
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5cp],U_fp[f5bp],U_fp[f5a],
                                                        V_fp[f5cp],V_fp[f5bp],V_fp[f5a]);
                  if (eb < required_eb) required_eb = eb;
                }
                // (f5a,f5c,f5bp)
                #if CP_DEBUG_VISIT
                MARK_FACE(f5a,f5c,f5bp);
                #endif
                if (has_cp(cp_faces, f5a,f5c,f5bp)) { required_eb=0;}
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5bp],U_fp[f5c],U_fp[f5a],
                                                        V_fp[f5bp],V_fp[f5c],V_fp[f5a]);
                  if (eb < required_eb) required_eb = eb;
                }
              }
              //triange 6 has 1 faces
              size_t f6a = vid(t, i,  j,  sz);
              size_t f6b = vid(t, i,j+1,  sz);
              size_t f6c = vid(t, i+1,j+1,sz);
              size_t f6ap = f6a - dv, f6bp = f6b - dv, f6cp = f6c - dv;
              if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
                // (f6a,f6b,f6cp)
                #if CP_DEBUG_VISIT
                MARK_FACE(f6a,f6b,f6cp);
                #endif
                if (has_cp(cp_faces, f6a,f6b,f6cp)) {
                  required_eb=0;
                }
                { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6cp],U_fp[f6b],U_fp[f6a],
                                                        V_fp[f6cp],V_fp[f6b],V_fp[f6a]);
                  if (eb < required_eb) required_eb = eb;
                }
              }
            }

            T abs_eb = required_eb;
            int id = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);

            if (abs_eb == 0){
              *(eb_pos++) = 0;
              unpred.push_back(U[v]);
              unpred.push_back(V[v]);
              continue;
            }

            {
              *eb_pos = id;

              bool unpred_flag=false;
              T dec[2];
              T abs_err_fp_q[2] = {0,0};

              for (int p=0; p<2; ++p){
                T *cur  = (p==0)? curU : curV;
                T  curv = (p==0) ? curU_val : curV_val;

                // ========== 3) 用该块的 pm 做主预测 ==========
                T pred = predict_dispatch(pm, cur, t, i, j, si, sj, sk);

                T diff = curv - pred;
                T qd = (std::llabs(diff)/abs_eb) + 1;
                if (qd < capacity){
                  qd = (diff > 0) ? qd : -qd;
                  int qindex = (int)(qd/2) + intv_radius;
                  dq_pos[p] = qindex;
                  dec[p] = pred + 2*(qindex - intv_radius)*abs_eb;

                  if (std::llabs(dec[p] - curv) > abs_eb){ unpred_flag = true; break; }
                  //abs_err_fp_q[p] = std::llabs(dec[p] - curv);
                }else{
                  unpred_flag = true; break;
                }
              }

              if (unpred_flag){
                *(eb_pos++) = 0;
                unpred.push_back(U[v]); unpred.push_back(V[v]);
              }else{
                ++eb_pos; dq_pos += 2;
                *curU = dec[0]; *curV = dec[1];
                // （可选）编码端误差统计 enc_max_abs_eb_fp/enc_max_real_err_fp 保持不变
              }
            }
          } // for j
        } // for i

      } // for bj
    } // for bi
  } // for t

  // 5) 打包码流
  std::vector<uint8_t> pm_bytes;
  pack_modes_2bit(block_modes, pm_bytes);
  //print mode percentage
  print_mode_counts(block_modes);
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);
  std::cout << "write intv_radius = " << intv_radius << "\n";

  // —— 写入一个简单的扩展头，以区分 MOP‑SI 版本 —— 
  const char MOP2_MAGIC[4] = {'M','O','P','2'};
  write_array_to_dst(pos, MOP2_MAGIC, 4);                  // 4 bytes magic
  printf("sub = %d\n", SZ_MOP_EVAL_SUBS);
  uint32_t bh=BH, bw=BW, bi_cnt=blocks_i, bj_cnt=blocks_j; // 便于解码端复现划分
  write_variable_to_dst(pos, bh);
  std::cout << "write bh = " << bh << "\n";
  write_variable_to_dst(pos, bw);
  std::cout << "write bw = " << bw << "\n";
  write_variable_to_dst(pos, bi_cnt);
  std::cout << "write blocks_i = " << bi_cnt << "\n";
  write_variable_to_dst(pos, bj_cnt);
  std::cout << "write blocks_j = " << bj_cnt << "\n";
  uint64_t pm_len = (uint64_t)pm_bytes.size();
  write_variable_to_dst(pos, pm_len);
  std::cout << "write pm len = " << (long long)pm_len << "\n";
  if (pm_len) write_array_to_dst(pos, pm_bytes.data(), (size_t)pm_len);

  size_t unpred_cnt = unpred.size();
  std::cout << "write unpred cnt = " << unpred_cnt << ",ratio=" << (double)unpred_cnt/(2*N) << "\n";
  write_variable_to_dst(pos, unpred_cnt);

  #if ZSTD_DETAIL
  {
    // use zstd, calculate size after zstd
    unsigned char * unpred_after_zstd = NULL;
    unsigned long unpred_bytes = unpred_cnt * sizeof(unpred[0]);
    size_t unpred_zstd_size = sz_lossless_compress(ZSTD_COMPRESSOR, 3, reinterpret_cast<unsigned char*>(unpred.data()),unpred_bytes,&unpred_after_zstd);
    double ratio = static_cast<double>(unpred_bytes) / unpred_zstd_size;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "ZSTD(unpred): " << unpred_zstd_size
              << " bytes (orig " << unpred_bytes
              << ", ratio = " << ratio << "x)\n";
  }
  #endif


  if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);
    //print sum of unpred.data()
  {
    double unpred_sum = 0.0;
    for (size_t i=0; i<unpred_cnt; ++i){
      unpred_sum += unpred[i];
    }
    printf("unpred sum = %.6f\n", unpred_sum);
  }

  unsigned char * pos_before_ebq = pos;
  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::cout << "Huffman ebq size = " << (size_t)(pos - pos_before_ebq) << "\n";

  #if ZSTD_DETAIL
  {
      unsigned char* ebq_huf_ptr = pos_before_ebq;
      size_t ebq_huf_size = (size_t)(pos - pos_before_ebq);

      if (ebq_huf_size > 0) {
          unsigned char* ebq_zstd = nullptr;
          unsigned long ebq_zstd_size = sz_lossless_compress(
              ZSTD_COMPRESSOR, 3,
              ebq_huf_ptr,
              (unsigned long)ebq_huf_size,
              &ebq_zstd
          );
          if (ZSTD_isError(ebq_zstd_size) || ebq_zstd_size == 0) {
              std::cerr << "ZSTD compress ebq(Huffman) error: "
                        << ZSTD_getErrorName(ebq_zstd_size) << "\n";
          } else {
              double ratio = (double)ebq_huf_size / (double)ebq_zstd_size;
              std::cout << std::fixed << std::setprecision(2);
              std::cout << "ZSTD(ebq after Huffman): " << ebq_zstd_size
                        << " bytes (orig " << ebq_huf_size
                        << ", ratio = " << ratio << "x)\n";
          }
          std::free(ebq_zstd);
      }
  }
  #endif
  std::free(eb_quant_index);

  unsigned char * pos_before_dq = pos;
  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  printf("write dq num = %zu\n", data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  std::cout << "Huffman dq size = " << (size_t)(pos - pos_before_dq) << "\n";
  #if ZSTD_DETAIL
  // —— 对 Huffman 后的 dq 字节块做 ZSTD ——
  {
      unsigned char* dq_huf_ptr = pos_before_dq;
      size_t dq_huf_size = (size_t)(pos - pos_before_dq);

      if (dq_huf_size > 0) {
          unsigned char* dq_zstd = nullptr;
          unsigned long dq_zstd_size = sz_lossless_compress(
              ZSTD_COMPRESSOR, 3,
              dq_huf_ptr,
              (unsigned long)dq_huf_size,
              &dq_zstd
          );
          if (ZSTD_isError(dq_zstd_size) || dq_zstd_size == 0) {
              std::cerr << "ZSTD compress dq(Huffman) error: "
                        << ZSTD_getErrorName(dq_zstd_size) << "\n";
          } else {
              double ratio = (double)dq_huf_size / (double)dq_zstd_size;
              std::cout << std::fixed << std::setprecision(2);
              std::cout << "ZSTD(dq after Huffman): " << dq_zstd_size
                        << " bytes (orig " << dq_huf_size
                        << ", ratio = " << ratio << "x)\n";
          }
          std::free(dq_zstd);
      }
      //再测试一下直接dq用0-rle的前后压缩比
      size_t original_byte = data_quant_num*sizeof(int);
      size_t encoded_length = 0;
      // int* encoded = zero_rle_encode(data_quant_index, 2 * N, encoded_length);
      int* encoded = zero_rle_encode(data_quant_index, data_quant_num, encoded_length);
      size_t compressed_byte = encoded_length*sizeof(int);
      printf("zero-rle ratio: %f\n",original_byte * 1.0 / compressed_byte);
      free(encoded);
      //把huffman前后的文件写出来
      writefile("/project/xli281_uksr/mxia/tmp_output/data_quant_index_mop.raw", (unsigned char*)data_quant_index, data_quant_num*sizeof(int));
      writefile("/project/xli281_uksr/mxia/tmp_output/data_quant_index_mop.huf", pos_before_dq, (size_t)(pos - pos_before_dq));
      printf("write data_quant_index_mop.raw & data_quant_index_mop.huf done....\n");

  }
  #endif
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  std::free(U_fp); std::free(V_fp);
  return compressed;
}


template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_mixed_order(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb,                  // 全局绝对误差上限（浮点域）：max_eb(fp)=max_pwr_eb
    EbMode mode 
){
  using T = int64_t;
  const Size3 sz{ (int)r1,(int)r2,(int)r3 };
  const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

  // 1) 定点化（先乘后 llround）
  T *U_fp=(T*)std::malloc(N*sizeof(T));
  T *V_fp=(T*)std::malloc(N*sizeof(T));
  if(!U_fp || !V_fp){
    if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp);
    compressed_size=0; return nullptr;
  }
  T range=0;
  T scale = convert_to_fixed_point<T_data,T>(U, V, N, U_fp, V_fp, range);

  // 2) 预计算：全局 CP 面集合（一次性）
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();


  #if CP_DEBUG_VISIT
  // 1.1 记录：编码端枚举/检查过的三角面
  std::unordered_set<FaceKeySZ, FaceKeySZHash> visited_faces;

  // 1.2 小工具：规范化存一下 (a,b,c)
  auto MARK_FACE = [&](size_t a, size_t b, size_t c){
      visited_faces.emplace(a,b,c); // FaceKeySZ 构造里已做排序/规范化
  };

  // 1.3 帮助打印：把 vid -> (t,i,j)
  auto decode_tij = [&](size_t v){
      int t = (int)(v / (H*W));
      size_t rem = v - (size_t)t * H * W;
      int i = (int)(rem / W);
      int j = (int)(rem % W);
      return std::tuple<int,int,int>(t,i,j);
  };

  // 1.4（可选）分类一下是哪类面
  auto classify_face = [&](size_t a, size_t b, size_t c){
      auto [ta,ia,ja] = decode_tij(a);
      auto [tb,ib,jb] = decode_tij(b);
      auto [tc,ic,jc] = decode_tij(c);
      int st = (ta==tb) + (ta==tc) + (tb==tc);
      if (ta==tb && tb==tc) {
          // 同一层
          bool has_diag = ((ia!=ib)||(ja!=jb)) && ((ia!=ic)||(ja!=jc)) && ((ib!=ic)||(jb!=jc));
          return has_diag ? "layer(tri)" : "layer(?)";
      } else {
          // 跨层
          int cnt_t0 = (ta==tb) + (ta==tc) + (ta==ta); // 粗略判断：有2点在同一层
          (void)cnt_t0;
          return "slab";
      }
  };
  #endif

  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // 3) 量化/编码缓冲
  int* eb_quant_index  = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index= (int*)std::malloc(2*N*sizeof(int)); // U/V 交错
  double enc_max_abs_eb_fp   = 0.0; //编码端自检（浮点）
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred; unpred.reserve((N / 4) * 2);

  // 参数与量化槽
  const int base = 2;                             // 仍写头，方便向后兼容
  const int capacity = 65536; //65536
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);

  // 把浮点域绝对误差上限转到定点域（LSB）
  // const T max_eb = (T) std::llround( (long double)max_pwr_eb * (long double)scale );

  T max_eb = 0;
  if(mode == EbMode::Relative){

    max_eb = max_pwr_eb * range; // 相对误差转绝对误差
    printf("Compression Using Relative Eb Mode!\n");
  }
  else if (mode == EbMode::Absolute){
    printf("Compression Using Absolute Eb Mode!\n");
    max_eb = max_pwr_eb * scale; // 浮点→定点
  }
  else{
    std::cerr << "Error: Unsupported EbMode!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  // T max_eb = range * max_pwr_eb;


  // 定点 LSB 的阈值（退化门控）
  const T threshold = 1;                          // 幂指数量化阈值（LSB）


  // 4) 逐顶点：枚举与该顶点相关的三角面 → 最小 eb
  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W; // 层间位移

  // 6 个平面邻接方向（左右、上下、主对角）
  const int di[6] = { 0,  0,  1, -1,  1, -1 }; //y-axis
  const int dj[6] = { 1, -1,  0,  0,  1, -1 }; //x-axis

  for (int t=0; t<(int)Tt; ++t){
    if(t % 100 == 0){
      printf("processing slice %d / %d\n", t, (int)Tt);
    }
    for (int i=0; i<(int)H; ++i){
      for (int j=0; j<(int)W; ++j){
        const size_t v = vid(t,i,j,sz);
        // 缓存原始定点值（预测/回写前）
        T *curU = U_fp + v;
        T *curV = V_fp + v;
        const T curU_val = *curU;
        const T curV_val = *curV;

        // —— 收集最小 eb——
        T required_eb = max_eb;

        // (A) 层内 t：影响 (i-1..i, j-1..j) 的 4 个 cell，每 cell 两三角
        for (int ci=i-1; ci<=i; ++ci){
          if (!in_range(ci, (int)H-1)) continue;
          for (int cj=j-1; cj<=j; ++cj){
            if (!in_range(cj, (int)W-1)) continue;

            size_t v00 = vid(t,ci,  cj,  sz);
            size_t v10 = vid(t,ci,  cj+1,sz); //v10=> x=1,y=0
            size_t v01 = vid(t,ci+1,cj,  sz);
            size_t v11 = vid(t,ci+1,cj+1,sz);

            // Upper: (v00,v01,v11)
            if (v==v00 || v==v01 || v==v11){ 
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v01,v11);
              #endif
              if (has_cp(cp_faces, v00,v01,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v01],U_fp[v00],
                                                    V_fp[v11],V_fp[v01],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
            // Lower: (v00,v10,v11)
            if (v==v00 || v==v10 || v==v11){
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v10,v11);
              #endif
              if (has_cp(cp_faces, v00,v10,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v10],U_fp[v00],
                                                    V_fp[v11],V_fp[v10],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }
        
        // (B1) 侧面 [t, t+1]
        if (t < (int)Tt-1){
          for (int k=0; k<6; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a  = vid(t, i, j, sz);
            size_t b  = vid(t, ni,nj, sz);
            size_t ap = a + dv, bp = b + dv;
            if (k == 0 || k==3 || k==5){
              // (a,b,bp) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (a,bp,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,bp,ap);
              #endif
              if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[bp],U_fp[a],
                                                        V_fp[ap],V_fp[bp],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,ap) for k = 1,2,4
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (b,ap,bp) for k = 1,2,4 //好像与点无关
              // if (has_cp(cp_faces, b,ap,bp)) { required_eb = 0; goto after_min_eb; }
              //   {
              //     T eb = derive_cp_abs_eb_sos_online<T>(U_fp[b],U_fp[ap],U_fp[bp],
              //                                           V_fp[b],V_fp[ap],V_fp[bp]);
              //     if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
              //     if (eb < required_eb) required_eb = eb;
              //   }
            }

          }
        }

        // (B2) 侧面 [t-1, t]
        if (t > 0){
          for (int k=0; k<6 ; ++k){
            int ni = i + di[k], nj = j + dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a = vid(t,i,j,sz); //v
            size_t b = vid(t,ni,nj,sz);
            size_t ap = a - dv, bp = b - dv; //ap,bp为上一层
            if (k == 0 || k==3 || k==5){
              // (a,b,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,bp) for k = 1,2,4 //侧面上三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
                
              // (a,ap,bp) for k = 1,2,4 //侧面下三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,ap,bp);
              #endif
              if (has_cp(cp_faces, a,ap,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[ap],U_fp[a],
                                                        V_fp[bp],V_fp[ap],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }

          }
        }

        // (C) 内部剖分面：两片 ts ∈ {t, t-1}；每相邻 cell 的 Upper/Lower 各 2 面
        // (C1) [t, t+1] 内部剖分面
        if (t < (int)Tt-1){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|
      
          //triange 1 [x,y,t] = [(0,0,0),(1,0,0),(0,-1,0)]: has 2 faces
          size_t f1a = vid(t,  i,  j,  sz);
          size_t f1b = vid(t,  i,j+1,  sz);
          size_t f1c = vid(t,  i-1,j,  sz);
          size_t f1ap = f1a + dv, f1bp = f1b + dv, f1cp = f1c + dv;
          if (in_range(i-1,(int)H) && in_range(j+1,(int)W)){
            // (f1a,f1cp,f1b)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1b);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1b)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1b],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1b],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f1a,f1cp,f1bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1bp);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1bp],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1bp],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          //triange 2 [x,y,t] = [(0,0,0),(0,-1,0),(-1,-1,0)]: has 2 faces
          size_t f2a = vid(t,  i,  j,  sz);
          size_t f2b = vid(t,  i-1,j,  sz);
          size_t f2c = vid(t,  i-1,j-1,sz);
          size_t f2ap = f2a + dv, f2bp = f2b + dv, f2cp = f2c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f2a,f2c,f2bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2c,f2bp);
            #endif
            if (has_cp(cp_faces, f2a,f2c,f2bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2bp],U_fp[f2c],U_fp[f2a],
                                                    V_fp[f2bp],V_fp[f2c],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f2a,f2bp,f2cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2bp,f2cp);
            #endif
            if (has_cp(cp_faces, f2a,f2bp,f2cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2cp],U_fp[f2bp],U_fp[f2a],
                                                    V_fp[f2cp],V_fp[f2bp],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          // triange 3 [x,y,t] = [(0,0,0),(-1,-1,0),(-1,0,0)]: has 1 faces
          size_t f3a = vid(t,  i,  j,  sz);
          size_t f3b = vid(t,  i-1,j-1,sz);
          size_t f3c = vid(t,  i,j-1,  sz);
          size_t f3ap = f3a + dv, f3bp = f3b + dv, f3cp = f3c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3bp,f3c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3bp,f3c);
            #endif
            if (has_cp(cp_faces, f3a,f3bp,f3c)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3c],U_fp[f3bp],U_fp[f3a],
                                                    V_fp[f3c],V_fp[f3bp],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // triange 4 0 faces
          // triange 5 0 faces
          // triange 6 [x,y,t] = [(0,0,0),(1,0,0),(1,1,0)]: has 1 faces
          size_t f6a = vid(t,  i,  j,  sz);
          size_t f6b = vid(t,  i,j+1,  sz);
          size_t f6c = vid(t,  i+1,j+1,sz);
          size_t f6ap = f6a + dv, f6bp = f6b + dv, f6cp = f6c + dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6bp,f6c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6bp,f6c);
            #endif
            if (has_cp(cp_faces, f6a,f6bp,f6c)) {
              required_eb=0; 
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6c],U_fp[f6bp],U_fp[f6a],
                                                    V_fp[f6c],V_fp[f6bp],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }


        // (C2) [t-1, t] 内部剖分面 并不与C1对称
        if (t > 0){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|

          //triange 1 0 face
          //triange 2 0 face
          //triange 3  has 1 faces
          size_t f3a = vid(t, i,  j,  sz);   
          size_t f3b = vid(t, i-1,j-1,sz);
          size_t f3c = vid(t, i,j-1,  sz);
          size_t f3ap = f3a - dv, f3bp = f3b - dv, f3cp = f3c - dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3b,f3cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3b,f3cp);
            #endif
            if (has_cp(cp_faces, f3a,f3b,f3cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3cp],U_fp[f3b],U_fp[f3a],
                                                    V_fp[f3cp],V_fp[f3b],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 4 has 2 faces
          size_t f4a = vid(t, i,  j,  sz);
          size_t f4b = vid(t, i,j-1,  sz);
          size_t f4c = vid(t,i+1,j,  sz);
          size_t f4ap = f4a - dv, f4bp = f4b - dv, f4cp = f4c - dv;
          if (in_range(i+1,(int)H) && in_range(j-1,(int)W)){
            // (f4a,f4bp,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4bp,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4bp,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4bp],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4bp],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f4a,f4b,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4b,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4b,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4b],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4b],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 5 has 2 faces
          size_t f5a = vid(t, i,  j,  sz);
          size_t f5b = vid(t, i+1,j,  sz);
          size_t f5c = vid(t, i+1,j+1,sz);
          size_t f5ap = f5a - dv, f5bp = f5b - dv, f5cp = f5c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f5a,f5bp,f5cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5bp,f5cp);
            #endif
            if (has_cp(cp_faces, f5a,f5bp,f5cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5cp],U_fp[f5bp],U_fp[f5a],
                                                    V_fp[f5cp],V_fp[f5bp],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f5a,f5c,f5bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5c,f5bp);
            #endif
            if (has_cp(cp_faces, f5a,f5c,f5bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5bp],U_fp[f5c],U_fp[f5a],
                                                    V_fp[f5bp],V_fp[f5c],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 6 has 1 faces
          size_t f6a = vid(t, i,  j,  sz);
          size_t f6b = vid(t, i,j+1,  sz);
          size_t f6c = vid(t, i+1,j+1,sz);
          size_t f6ap = f6a - dv, f6bp = f6b - dv, f6cp = f6c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6b,f6cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6b,f6cp);
            #endif
            if (has_cp(cp_faces, f6a,f6b,f6cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6cp],U_fp[f6b],U_fp[f6a],
                                                    V_fp[f6cp],V_fp[f6b],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }
        // {
        //   // 保证 |dec-orig| ≤ abs_eb < |orig| ⇒ 不会跨 0 翻符号
        //   T sign_guard = std::min(std::llabs(curU_val), std::llabs(curV_val));
        //   if (sign_guard > 0 && required_eb >= sign_guard){
        //       T max_eb_to_preserve_sign = sign_guard - 1; // 留 1 LSB 余量
        //       if (max_eb_to_preserve_sign < required_eb)
        //           required_eb = max_eb_to_preserve_sign;
        //   }
        // }
        T abs_eb = required_eb;
        int id = eb_exponential_quantize(abs_eb,base,log_of_base,threshold);
        
        if (abs_eb == 0 )
        {
          *(eb_pos++) = 0;            // ebid=0 → 无损点
          unpred.push_back(U[v]);     // 原始浮点（解码端直接回填）
          unpred.push_back(V[v]);
          continue;
        }

        // ===== 量化 eb（配套的新幂指数量化）并编码该顶点 =====
        {
          *eb_pos = id;

          bool unpred_flag=false;
          T dec[2];
          T abs_err_fp_q[2] = {0,0};

          for (int p=0; p<2; ++p){
            T *cur  = (p==0)? curU : curV;
            T  curv = (p==0) ? curU_val : curV_val;

            // 新：混合阶 3D Lorenzo（示例：kt=2, ki=1, kj=1）
            constexpr int KT = 2;   // 时间阶数
            constexpr int KI = 1;   // Y(i) 阶数
            constexpr int KJ = 1;   // X(j) 阶数
            T pred = lorenzo3d_predict_mixed_order<T>(cur, t,i,j, si,sj,sk, KT,KI,KJ);


            T diff = curv - pred;
            T qd = (std::llabs(diff)/abs_eb) + 1;
            if (qd < capacity){
              qd = (diff > 0) ? qd : -qd;
              int qindex = (int)(qd/2) + intv_radius;
              dq_pos[p] = qindex;
              dec[p] = pred + 2*(qindex - intv_radius)*abs_eb;

              // 守门：必须用 *代表值* abs_eb 校验
              if (std::llabs(dec[p] - curv) > abs_eb){
              // if (std::llabs(dec[p] - curv) > required_eb){
                unpred_flag = true; break;
              }
              abs_err_fp_q[p] = std::llabs(dec[p] - curv);
            }else{
              unpred_flag = true; break;
            }
          }

          if (unpred_flag){
            *(eb_pos++) = 0;                  // 改回无损
            unpred.push_back(U[v]);
            unpred.push_back(V[v]);
          }else{
            ++eb_pos;
            dq_pos += 2;
            *curU = dec[0];
            *curV = dec[1];

            // 编码端自检（浮点域）
            double abs_eb_fp = (double)abs_eb / (double)scale;
            double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
            double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
            enc_max_abs_eb_fp   = std::max(enc_max_abs_eb_fp, abs_eb_fp);
            enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
          }
        }
      }
    }
  }

  // std::cerr << "[ENC] max abs_eb(fp) = " << enc_max_abs_eb_fp
  //           << ", max actual |err|(fp) = " << enc_max_real_err_fp << "\n";

  
  // 5) 打包码流
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);
  std::cout << "write intv_radius = " << intv_radius << "\n";

  size_t unpred_cnt = unpred.size();
  std::cout << "write unpred cnt = " << unpred_cnt << ",ratio=" << (double)unpred_cnt/(2*N) << "\n";
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);

  unsigned char *pos_before_ebq = pos;
  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::cout << "Huffman eb size = " << (size_t)(pos - pos_before_ebq) << "\n";
  std::free(eb_quant_index);

  unsigned char *pos_before_dq = pos;
  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  //print size of huffman
  std::cout << "Huffman dq size = " << (size_t)(pos - pos_before_dq) << "\n";
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  std::free(U_fp); std::free(V_fp);
  return compressed;
}

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_chanrot(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb,                  // 全局绝对误差上限（浮点域）：max_eb(fp)=max_pwr_eb
    EbMode mode 
){
  using T = int64_t;
  const Size3 sz{ (int)r1,(int)r2,(int)r3 };
  const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

  // 1) 定点化（先乘后 llround）
  T *U_fp=(T*)std::malloc(N*sizeof(T));
  T *V_fp=(T*)std::malloc(N*sizeof(T));
  if(!U_fp || !V_fp){
    if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp);
    compressed_size=0; return nullptr;
  }

  // ================================
  // [CHANROT-1] 估计旋转角 + 在旋转域中定点化
  // ================================
  Rot2D R = estimate_channel_rotation(U, V, N);
  std::cout << "[CHANROT] cos=" << R.c << " sin=" << R.s << std::endl;

  // 临时旋转后的浮点通道，仅用于定点化输入
  T_data* Arot = (T_data*)std::malloc(N*sizeof(T_data));
  T_data* Brot = (T_data*)std::malloc(N*sizeof(T_data));
  if(!Arot || !Brot){
    if(Arot) std::free(Arot); if(Brot) std::free(Brot);
    std::free(U_fp); std::free(V_fp);
    compressed_size=0; return nullptr;
  }
  for (size_t idx=0; idx<N; ++idx){
    double a =  R.c * (double)U[idx] + R.s * (double)V[idx];
    double b = -R.s * (double)U[idx] + R.c * (double)V[idx];
    Arot[idx] = (T_data)a;
    Brot[idx] = (T_data)b;
  }

  T range=0;
  T scale = convert_to_fixed_point<T_data,T>(Arot, Brot, N, U_fp, V_fp, range);
  // 旋转域临时缓冲不再需要
  std::free(Arot); std::free(Brot);
  // ================================

  // 2) 预计算：全局 CP 面集合（一次性）——在旋转域定点数据上进行
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();

  #if CP_DEBUG_VISIT
  std::unordered_set<FaceKeySZ, FaceKeySZHash> visited_faces;
  auto MARK_FACE = [&](size_t a, size_t b, size_t c){
      visited_faces.emplace(a,b,c);
  };
  auto decode_tij = [&](size_t v){
      int t = (int)(v / (H*W));
      size_t rem = v - (size_t)t * H * W;
      int i = (int)(rem / W);
      int j = (int)(rem % W);
      return std::tuple<int,int,int>(t,i,j);
  };
  auto classify_face = [&](size_t a, size_t b, size_t c){
      auto [ta,ia,ja] = decode_tij(a);
      auto [tb,ib,jb] = decode_tij(b);
      auto [tc,ic,jc] = decode_tij(c);
      int st = (ta==tb) + (ta==tc) + (tb==tc);
      if (ta==tb && tb==tc) {
          bool has_diag = ((ia!=ib)||(ja!=jb)) && ((ia!=ic)||(ja!=jc)) && ((ib!=ic)||(jb!=jc));
          return has_diag ? "layer(tri)" : "layer(?)";
      } else {
          return "slab";
      }
  };
  #endif

  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // 3) 量化/编码缓冲
  int* eb_quant_index  = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index= (int*)std::malloc(2*N*sizeof(int)); // U/V 交错（旋转域的 A/B）
  double enc_max_abs_eb_fp   = 0.0; //编码端自检（浮点）
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred; unpred.reserve((N / 4) * 2);

  // 参数与量化槽
  const int base = 2;
  const int capacity = 65536; // 或你的现有配置
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);

  // 浮点域绝对误差上限 -> 定点域（旋转域）
  T max_eb = 0;
  if(mode == EbMode::Relative){
    max_eb = max_pwr_eb * range;
    printf("Compression Using Relative Eb Mode!\n");
  } else if (mode == EbMode::Absolute){
    printf("Compression Using Absolute Eb Mode!\n");
    max_eb = max_pwr_eb * scale; // 注意：此处是旋转域的 scale
    // 可选严格上界（原坐标系）：解开注释会略降 CR
    // double safety = 1.0 / (std::fabs(R.c) + std::fabs(R.s));
    // max_eb = (T)std::llround( (long double)max_eb * (long double)safety );
  } else{
    std::cerr << "Error: Unsupported EbMode!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index) std::free(data_quant_index);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }

  // 定点 LSB 的阈值（退化门控）
  const T threshold = 1;

  // 4) 逐顶点：枚举与该顶点相关的三角面 → 最小 eb（旋转域上）
  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W; // 层间位移

  const int di[6] = { 0,  0,  1, -1,  1, -1 };
  const int dj[6] = { 1, -1,  0,  0,  1, -1 };

  for (int t=0; t<(int)Tt; ++t){
    if(t % 100 == 0){
      printf("processing slice %d / %d\n", t, (int)Tt);
    }
    for (int i=0; i<(int)H; ++i){
      for (int j=0; j<(int)W; ++j){
        const size_t v = vid(t,i,j,sz);

        // 旋转域定点值（预测/回写前）
        T *curU = U_fp + v; // 对应 a 分量
        T *curV = V_fp + v; // 对应 b 分量
        const T curU_val = *curU;
        const T curV_val = *curV;

        // —— 收集最小 eb——
        T required_eb = max_eb;

        // (A) 层内 t：4 cell × 两三角
        for (int ci=i-1; ci<=i; ++ci){
          if (!in_range(ci, (int)H-1)) continue;
          for (int cj=j-1; cj<=j; ++cj){
            if (!in_range(cj, (int)W-1)) continue;

            size_t v00 = vid(t,ci,  cj,  sz);
            size_t v10 = vid(t,ci,  cj+1,sz);
            size_t v01 = vid(t,ci+1,cj,  sz);
            size_t v11 = vid(t,ci+1,cj+1,sz);

            // Upper: (v00,v01,v11)
            if (v==v00 || v==v01 || v==v11){
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v01,v11);
              #endif
              if (has_cp(cp_faces, v00,v01,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v01],U_fp[v00],
                                                    V_fp[v11],V_fp[v01],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
            // Lower: (v00,v10,v11)
            if (v==v00 || v==v10 || v==v11){
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v10,v11);
              #endif
              if (has_cp(cp_faces, v00,v10,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v10],U_fp[v00],
                                                    V_fp[v11],V_fp[v10],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        // (B1) 侧面 [t, t+1]
        if (t < (int)Tt-1){
          for (int k=0; k<6; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a  = vid(t, i, j, sz);
            size_t b  = vid(t, ni,nj, sz);
            size_t ap = a + dv, bp = b + dv;
            if (k == 0 || k==3 || k==5){
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                      V_fp[bp],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
              #if CP_DEBUG_VISIT
              MARK_FACE(a,bp,ap);
              #endif
              if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0;}
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[bp],U_fp[a],
                                                      V_fp[ap],V_fp[bp],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
            } else {
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                      V_fp[ap],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }

        // (B2) 侧面 [t-1, t]
        if (t > 0){
          for (int k=0; k<6 ; ++k){
            int ni = i + di[k], nj = j + dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a = vid(t,i,j,sz);
            size_t b = vid(t,ni,nj,sz);
            size_t ap = a - dv, bp = b - dv;
            if (k == 0 || k==3 || k==5){
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                      V_fp[ap],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
            } else {
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                      V_fp[bp],V_fp[b],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
              #if CP_DEBUG_VISIT
              MARK_FACE(a,ap,bp);
              #endif
              if (has_cp(cp_faces, a,ap,bp)) { required_eb = 0;}
              { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[ap],U_fp[a],
                                                      V_fp[bp],V_fp[ap],V_fp[a]);
                if (eb < required_eb) required_eb = eb;
              }
            }
          }
        }

        // (C) 内部分割面，略（与你原逻辑一致，均在 U_fp/V_fp 旋转域上计算）
        // --- 下面原样保留你的 C1、C2 代码 ---
        // C1 ...
        if (t < (int)Tt-1){
          size_t f1a = vid(t,  i,  j,  sz);
          size_t f1b = vid(t,  i,j+1,  sz);
          size_t f1c = vid(t,  i-1,j,  sz);
          size_t f1ap = f1a + dv, f1bp = f1b + dv, f1cp = f1c + dv;
          if (in_range(i-1,(int)H) && in_range(j+1,(int)W)){
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1b);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1b)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1b],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1b],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1bp);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1bp],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1bp],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          size_t f2a = vid(t,  i,  j,  sz);
          size_t f2b = vid(t,  i-1,j,  sz);
          size_t f2c = vid(t,  i-1,j-1,sz);
          size_t f2ap = f2a + dv, f2bp = f2b + dv, f2cp = f2c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2c,f2bp);
            #endif
            if (has_cp(cp_faces, f2a,f2c,f2bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2bp],U_fp[f2c],U_fp[f2a],
                                                    V_fp[f2bp],V_fp[f2c],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2bp,f2cp);
            #endif
            if (has_cp(cp_faces, f2a,f2bp,f2cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2cp],U_fp[f2bp],U_fp[f2a],
                                                    V_fp[f2cp],V_fp[f2bp],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          size_t f3a = vid(t,  i,  j,  sz);
          size_t f3b = vid(t,  i-1,j-1,sz);
          size_t f3c = vid(t,  i,j-1,  sz);
          size_t f3ap = f3a + dv, f3bp = f3b + dv, f3cp = f3c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3bp,f3c);
            #endif
            if (has_cp(cp_faces, f3a,f3bp,f3c)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3c],U_fp[f3bp],U_fp[f3a],
                                                    V_fp[f3c],V_fp[f3bp],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          size_t f6a = vid(t,  i,  j,  sz);
          size_t f6b = vid(t,  i,j+1,  sz);
          size_t f6c = vid(t,  i+1,j+1,sz);
          size_t f6ap = f6a + dv, f6bp = f6b + dv, f6cp = f6c + dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6bp,f6c);
            #endif
            if (has_cp(cp_faces, f6a,f6bp,f6c)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6c],U_fp[f6bp],U_fp[f6a],
                                                    V_fp[f6c],V_fp[f6bp],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        // C2 ...
        if (t > 0){
          size_t f3a = vid(t, i,  j,  sz);
          size_t f3b = vid(t, i-1,j-1,sz);
          size_t f3c = vid(t, i,j-1,  sz);
          size_t f3ap = f3a - dv, f3bp = f3b - dv, f3cp = f3c - dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3b,f3cp);
            #endif
            if (has_cp(cp_faces, f3a,f3b,f3cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3cp],U_fp[f3b],U_fp[f3a],
                                                    V_fp[f3cp],V_fp[f3b],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          size_t f4a = vid(t, i,  j,  sz);
          size_t f4b = vid(t, i,j-1,  sz);
          size_t f4c = vid(t,i+1,j,  sz);
          size_t f4ap = f4a - dv, f4bp = f4b - dv, f4cp = f4c - dv;
          if (in_range(i+1,(int)H) && in_range(j-1,(int)W)){
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4bp,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4bp,f4cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4bp],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4bp],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4b,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4b,f4cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4b],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4b],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          size_t f5a = vid(t, i,  j,  sz);
          size_t f5b = vid(t, i+1,j,  sz);
          size_t f5c = vid(t, i+1,j+1,sz);
          size_t f5ap = f5a - dv, f5bp = f5b - dv, f5cp = f5c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5bp,f5cp);
            #endif
            if (has_cp(cp_faces, f5a,f5bp,f5cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5cp],U_fp[f5bp],U_fp[f5a],
                                                    V_fp[f5cp],V_fp[f5bp],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5c,f5bp);
            #endif
            if (has_cp(cp_faces, f5a,f5c,f5bp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5bp],U_fp[f5c],U_fp[f5a],
                                                    V_fp[f5bp],V_fp[f5c],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          size_t f6a = vid(t, i,  j,  sz);
          size_t f6b = vid(t, i,j+1,  sz);
          size_t f6c = vid(t, i+1,j+1,sz);
          size_t f6ap = f6a - dv, f6bp = f6b - dv, f6cp = f6c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6b,f6cp);
            #endif
            if (has_cp(cp_faces, f6a,f6b,f6cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6cp],U_fp[f6b],U_fp[f6a],
                                                    V_fp[f6cp],V_fp[f6b],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }

        // ===== 量化 eb 并编码该顶点 =====
        T abs_eb = required_eb;
        int id = eb_exponential_quantize(abs_eb,base,log_of_base,threshold);

        if (abs_eb == 0 ){
          // ================================
          // [CHANROT-2] 无损点写入：写旋转域 (a,b)
          // ================================
          *(eb_pos++) = 0;
          T_data a_f, b_f;
          apply_rot_forward_point(U[v], V[v], R, a_f, b_f);
          unpred.push_back(a_f);
          unpred.push_back(b_f);
          continue;
        }

        {
          *eb_pos = id;

          bool unpred_flag=false;
          T dec[2];
          T abs_err_fp_q[2] = {0,0};

          for (int p=0; p<2; ++p){
            T *cur  = (p==0)? curU : curV;
            T  curv = (p==0) ? curU_val : curV_val;

            // 3D 一阶 Lorenzo（时间最慢）
            T d0 = (t&&i&&j)? cur[-sk - si - sj] : 0;
            T d1 = (t&&i)   ? cur[-sk - si]      : 0;
            T d2 = (t&&j)   ? cur[-sk - sj]      : 0;
            T d3 = (t)      ? cur[-sk]           : 0;
            T d4 = (i&&j)   ? cur[-si - sj]      : 0;
            T d5 = (i)      ? cur[-si]           : 0;
            T d6 = (j)      ? cur[-sj]           : 0;
            T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

            T diff = curv - pred;
            T qd = (std::llabs(diff)/abs_eb) + 1;
            if (qd < capacity){
              qd = (diff > 0) ? qd : -qd;
              int qindex = (int)(qd/2) + intv_radius;
              dq_pos[p] = qindex;
              dec[p] = pred + 2*(qindex - intv_radius)*abs_eb;

              if (std::llabs(dec[p] - curv) > abs_eb){
                unpred_flag = true; break;
              }
              abs_err_fp_q[p] = std::llabs(dec[p] - curv);
            }else{
              unpred_flag = true; break;
            }
          }

          if (unpred_flag){
            // ================================
            // [CHANROT-3] 退化为无损：同样写旋转域 (a,b)
            // ================================
            *(eb_pos++) = 0;
            T_data a_f, b_f;
            apply_rot_forward_point(U[v], V[v], R, a_f, b_f);
            unpred.push_back(a_f);
            unpred.push_back(b_f);
          }else{
            ++eb_pos;
            dq_pos += 2;
            *curU = dec[0];
            *curV = dec[1];

            double abs_eb_fp = (double)abs_eb / (double)scale;
            double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
            double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
            enc_max_abs_eb_fp   = std::max(enc_max_abs_eb_fp, abs_eb_fp);
            enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
          }
        }
      }
    }
  }

  // 5) 打包码流
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);
  std::cout << "write intv_radius = " << intv_radius << "\n";

  // ===== 写入旋转系数（double）=====
  write_variable_to_dst(pos, R.c);
  write_variable_to_dst(pos, R.s);
  std::cout << "write chanrot cos,sin = " << R.c << ", " << R.s << "\n";

  size_t unpred_cnt = unpred.size();
  std::cout << "write unpred cnt = " << unpred_cnt << ",ratio=" << (double)unpred_cnt/(2*N) << "\n";
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);

  unsigned char *pos_before_ebq = pos;
  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::cout << "Huffman eb size = " << (size_t)(pos - pos_before_ebq) << "\n";
  std::free(eb_quant_index);

  unsigned char *pos_before_dq = pos;
  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  std::cout << "Huffman dq size = " << (size_t)(pos - pos_before_dq) << "\n";
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  std::free(U_fp); std::free(V_fp);
  return compressed;
}

//分流编码 U/V, 差分qindex
template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_vertexwise_cpmap_simple(
    const T_data* U, const T_data* V,
    size_t r1, size_t r2, size_t r3,   // r1=H, r2=W, r3=T (时间最慢)
    size_t& compressed_size,
    double max_pwr_eb,                  // 全局绝对误差上限（浮点域）：max_eb(fp)=max_pwr_eb
    EbMode mode 
){
  using T = int64_t;
  const Size3 sz{ (int)r1,(int)r2,(int)r3 };
  const size_t H=r1, W=r2, Tt=r3, N=H*W*Tt;

  // 1) 定点化（先乘后 llround）
  T *U_fp=(T*)std::malloc(N*sizeof(T));
  T *V_fp=(T*)std::malloc(N*sizeof(T));
  if(!U_fp || !V_fp){
    if(U_fp) std::free(U_fp); if(V_fp) std::free(V_fp);
    compressed_size=0; return nullptr;
  }
  T range=0;
  T scale = convert_to_fixed_point<T_data,T>(U, V, N, U_fp, V_fp, range);

  // 2) 预计算：全局 CP 面集合（一次性）
  auto pre_compute_time = std::chrono::high_resolution_clock::now();
  auto cp_faces = compute_cp_2p5d_faces<T>(U_fp, V_fp, (int)H, (int)W, (int)Tt);
  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();


  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // 3) 量化/编码缓冲
  int* eb_quant_index     = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index_u = (int*)std::malloc(N*sizeof(int));
  int* data_quant_index_v = (int*)std::malloc(N*sizeof(int));
  double enc_max_abs_eb_fp   = 0.0; //编码端自检（浮点）
  double enc_max_real_err_fp = 0.0;
  if(!eb_quant_index || !data_quant_index_u || !data_quant_index_v){
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index_u) std::free(data_quant_index_u);
    if(data_quant_index_v) std::free(data_quant_index_v);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos_u = data_quant_index_u;
  int* dq_pos_v = data_quant_index_v;
  std::vector<T_data> unpred; unpred.reserve((N / 4) * 2);

  // 参数与量化槽
  const int base = 2;                             // 仍写头，方便向后兼容
  const int capacity = 65536; //65536
  const double log_of_base = log2(base);
  const int intv_radius = (capacity >> 1);

  // 把浮点域绝对误差上限转到定点域（LSB）
  // const T max_eb = (T) std::llround( (long double)max_pwr_eb * (long double)scale );

  T max_eb = 0;
  if(mode == EbMode::Relative){

    max_eb = max_pwr_eb * range; // 相对误差转绝对误差
    printf("Compression Using Relative Eb Mode!\n");
  }
  else if (mode == EbMode::Absolute){
    printf("Compression Using Absolute Eb Mode!\n");
    max_eb = max_pwr_eb * scale; // 浮点→定点
  }
  else{
    std::cerr << "Error: Unsupported EbMode!\n";
    if(eb_quant_index) std::free(eb_quant_index);
    if(data_quant_index_u) std::free(data_quant_index_u);
    if(data_quant_index_v) std::free(data_quant_index_v);
    std::free(U_fp); std::free(V_fp); compressed_size=0; return nullptr;
  }
  // T max_eb = range * max_pwr_eb;


  // 定点 LSB 的阈值（退化门控）
  const T threshold = 1;                          // 幂指数量化阈值（LSB）


  // 4) 逐顶点：枚举与该顶点相关的三角面 → 最小 eb
  const ptrdiff_t si=(ptrdiff_t)W, sj=(ptrdiff_t)1, sk=(ptrdiff_t)(H*W);
  const size_t dv = (size_t)H*(size_t)W; // 层间位移

  // 6 个平面邻接方向（左右、上下、主对角）
  const int di[6] = { 0,  0,  1, -1,  1, -1 }; //y-axis
  const int dj[6] = { 1, -1,  0,  0,  1, -1 }; //x-axis

  for (int t=0; t<(int)Tt; ++t){
    if(t % 100 == 0){
      printf("processing slice %d / %d\n", t, (int)Tt);
    }
    for (int i=0; i<(int)H; ++i){
      for (int j=0; j<(int)W; ++j){
        const size_t v = vid(t,i,j,sz);
        // 缓存原始定点值（预测/回写前）
        T *curU = U_fp + v;
        T *curV = V_fp + v;
        const T curU_val = *curU;
        const T curV_val = *curV;

        // —— 收集最小 eb——
        T required_eb = max_eb;

        // (A) 层内 t：影响 (i-1..i, j-1..j) 的 4 个 cell，每 cell 两三角
        for (int ci=i-1; ci<=i; ++ci){
          if (!in_range(ci, (int)H-1)) continue;
          for (int cj=j-1; cj<=j; ++cj){
            if (!in_range(cj, (int)W-1)) continue;

            size_t v00 = vid(t,ci,  cj,  sz);
            size_t v10 = vid(t,ci,  cj+1,sz); //v10=> x=1,y=0
            size_t v01 = vid(t,ci+1,cj,  sz);
            size_t v11 = vid(t,ci+1,cj+1,sz);

            // Upper: (v00,v01,v11)
            if (v==v00 || v==v01 || v==v11){ 
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v01,v11);
              #endif
              if (has_cp(cp_faces, v00,v01,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v01],U_fp[v00],
                                                    V_fp[v11],V_fp[v01],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
            // Lower: (v00,v10,v11)
            if (v==v00 || v==v10 || v==v11){
              #if CP_DEBUG_VISIT
              MARK_FACE(v00,v10,v11);
              #endif
              if (has_cp(cp_faces, v00,v10,v11)) { required_eb = 0;}
              T eb = derive_cp_abs_eb_sos_online<T>(U_fp[v11],U_fp[v10],U_fp[v00],
                                                    V_fp[v11],V_fp[v10],V_fp[v00]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }
        
        // (B1) 侧面 [t, t+1]
        if (t < (int)Tt-1){
          for (int k=0; k<6; ++k){
            int ni=i+di[k], nj=j+dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a  = vid(t, i, j, sz);
            size_t b  = vid(t, ni,nj, sz);
            size_t ap = a + dv, bp = b + dv;
            if (k == 0 || k==3 || k==5){
              // (a,b,bp) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (a,bp,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,bp,ap);
              #endif
              if (has_cp(cp_faces, a,bp,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[bp],U_fp[a],
                                                        V_fp[ap],V_fp[bp],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,ap) for k = 1,2,4
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
              // (b,ap,bp) for k = 1,2,4 //好像与点无关
              // if (has_cp(cp_faces, b,ap,bp)) { required_eb = 0; goto after_min_eb; }
              //   {
              //     T eb = derive_cp_abs_eb_sos_online<T>(U_fp[b],U_fp[ap],U_fp[bp],
              //                                           V_fp[b],V_fp[ap],V_fp[bp]);
              //     if (eb <= degenerate_lsb){ degenerate_face = true; goto after_min_eb; }
              //     if (eb < required_eb) required_eb = eb;
              //   }
            }

          }
        }

        // (B2) 侧面 [t-1, t]
        if (t > 0){
          for (int k=0; k<6 ; ++k){
            int ni = i + di[k], nj = j + dj[k];
            if (!in_range(ni,(int)H) || !in_range(nj,(int)W)) continue;
            size_t a = vid(t,i,j,sz); //v
            size_t b = vid(t,ni,nj,sz);
            size_t ap = a - dv, bp = b - dv; //ap,bp为上一层
            if (k == 0 || k==3 || k==5){
              // (a,b,ap) for k = 0,3,5
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,ap);
              #endif
              if (has_cp(cp_faces, a,b,ap)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[ap],U_fp[b],U_fp[a],
                                                        V_fp[ap],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }
            else{
              // (a,b,bp) for k = 1,2,4 //侧面上三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,b,bp);
              #endif
              if (has_cp(cp_faces, a,b,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[b],U_fp[a],
                                                        V_fp[bp],V_fp[b],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
                
              // (a,ap,bp) for k = 1,2,4 //侧面下三角
              #if CP_DEBUG_VISIT
              MARK_FACE(a,ap,bp);
              #endif
              if (has_cp(cp_faces, a,ap,bp)) { required_eb = 0;}
                {
                  T eb = derive_cp_abs_eb_sos_online<T>(U_fp[bp],U_fp[ap],U_fp[a],
                                                        V_fp[bp],V_fp[ap],V_fp[a]);
                  if (eb < required_eb) required_eb = eb;
                }
            }

          }
        }

        // (C) 内部剖分面：两片 ts ∈ {t, t-1}；每相邻 cell 的 Upper/Lower 各 2 面
        // (C1) [t, t+1] 内部剖分面
        if (t < (int)Tt-1){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|
      
          //triange 1 [x,y,t] = [(0,0,0),(1,0,0),(0,-1,0)]: has 2 faces
          size_t f1a = vid(t,  i,  j,  sz);
          size_t f1b = vid(t,  i,j+1,  sz);
          size_t f1c = vid(t,  i-1,j,  sz);
          size_t f1ap = f1a + dv, f1bp = f1b + dv, f1cp = f1c + dv;
          if (in_range(i-1,(int)H) && in_range(j+1,(int)W)){
            // (f1a,f1cp,f1b)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1b);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1b)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1b],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1b],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f1a,f1cp,f1bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f1a,f1cp,f1bp);
            #endif
            if (has_cp(cp_faces, f1a,f1cp,f1bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f1bp],U_fp[f1cp],U_fp[f1a],
                                                    V_fp[f1bp],V_fp[f1cp],V_fp[f1a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          //triange 2 [x,y,t] = [(0,0,0),(0,-1,0),(-1,-1,0)]: has 2 faces
          size_t f2a = vid(t,  i,  j,  sz);
          size_t f2b = vid(t,  i-1,j,  sz);
          size_t f2c = vid(t,  i-1,j-1,sz);
          size_t f2ap = f2a + dv, f2bp = f2b + dv, f2cp = f2c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f2a,f2c,f2bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2c,f2bp);
            #endif
            if (has_cp(cp_faces, f2a,f2c,f2bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2bp],U_fp[f2c],U_fp[f2a],
                                                    V_fp[f2bp],V_fp[f2c],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f2a,f2bp,f2cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f2a,f2bp,f2cp);
            #endif
            if (has_cp(cp_faces, f2a,f2bp,f2cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f2cp],U_fp[f2bp],U_fp[f2a],
                                                    V_fp[f2cp],V_fp[f2bp],V_fp[f2a]);
              if (eb < required_eb) required_eb = eb;
            }
          }

          // triange 3 [x,y,t] = [(0,0,0),(-1,-1,0),(-1,0,0)]: has 1 faces
          size_t f3a = vid(t,  i,  j,  sz);
          size_t f3b = vid(t,  i-1,j-1,sz);
          size_t f3c = vid(t,  i,j-1,  sz);
          size_t f3ap = f3a + dv, f3bp = f3b + dv, f3cp = f3c + dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3bp,f3c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3bp,f3c);
            #endif
            if (has_cp(cp_faces, f3a,f3bp,f3c)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3c],U_fp[f3bp],U_fp[f3a],
                                                    V_fp[f3c],V_fp[f3bp],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          // triange 4 0 faces
          // triange 5 0 faces
          // triange 6 [x,y,t] = [(0,0,0),(1,0,0),(1,1,0)]: has 1 faces
          size_t f6a = vid(t,  i,  j,  sz);
          size_t f6b = vid(t,  i,j+1,  sz);
          size_t f6c = vid(t,  i+1,j+1,sz);
          size_t f6ap = f6a + dv, f6bp = f6b + dv, f6cp = f6c + dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6bp,f6c)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6bp,f6c);
            #endif
            if (has_cp(cp_faces, f6a,f6bp,f6c)) {
              required_eb=0; 
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6c],U_fp[f6bp],U_fp[f6a],
                                                    V_fp[f6c],V_fp[f6bp],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }


        // (C2) [t-1, t] 内部剖分面 并不与C1对称
        if (t > 0){
          //     ---------
          //     |  /| T5/|
          //     | / |  / |
          //     |/T4| /T6|
          //     ----X----|
          //     |T3/|T1/ |
          //     | / | /  |
          //     |/T2|/   |
          //     |---|----|

          //triange 1 0 face
          //triange 2 0 face
          //triange 3  has 1 faces
          size_t f3a = vid(t, i,  j,  sz);   
          size_t f3b = vid(t, i-1,j-1,sz);
          size_t f3c = vid(t, i,j-1,  sz);
          size_t f3ap = f3a - dv, f3bp = f3b - dv, f3cp = f3c - dv;
          if (in_range(i-1,(int)H) && in_range(j-1,(int)W)){
            // (f3a,f3b,f3cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f3a,f3b,f3cp);
            #endif
            if (has_cp(cp_faces, f3a,f3b,f3cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f3cp],U_fp[f3b],U_fp[f3a],
                                                    V_fp[f3cp],V_fp[f3b],V_fp[f3a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 4 has 2 faces
          size_t f4a = vid(t, i,  j,  sz);
          size_t f4b = vid(t, i,j-1,  sz);
          size_t f4c = vid(t,i+1,j,  sz);
          size_t f4ap = f4a - dv, f4bp = f4b - dv, f4cp = f4c - dv;
          if (in_range(i+1,(int)H) && in_range(j-1,(int)W)){
            // (f4a,f4bp,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4bp,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4bp,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4bp],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4bp],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f4a,f4b,f4cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f4a,f4b,f4cp);
            #endif
            if (has_cp(cp_faces, f4a,f4b,f4cp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f4cp],U_fp[f4b],U_fp[f4a],
                                                    V_fp[f4cp],V_fp[f4b],V_fp[f4a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 5 has 2 faces
          size_t f5a = vid(t, i,  j,  sz);
          size_t f5b = vid(t, i+1,j,  sz);
          size_t f5c = vid(t, i+1,j+1,sz);
          size_t f5ap = f5a - dv, f5bp = f5b - dv, f5cp = f5c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f5a,f5bp,f5cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5bp,f5cp);
            #endif
            if (has_cp(cp_faces, f5a,f5bp,f5cp)) { required_eb=0; }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5cp],U_fp[f5bp],U_fp[f5a],
                                                    V_fp[f5cp],V_fp[f5bp],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
            // (f5a,f5c,f5bp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f5a,f5c,f5bp);
            #endif
            if (has_cp(cp_faces, f5a,f5c,f5bp)) { required_eb=0;}
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f5bp],U_fp[f5c],U_fp[f5a],
                                                    V_fp[f5bp],V_fp[f5c],V_fp[f5a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
          //triange 6 has 1 faces
          size_t f6a = vid(t, i,  j,  sz);
          size_t f6b = vid(t, i,j+1,  sz);
          size_t f6c = vid(t, i+1,j+1,sz);
          size_t f6ap = f6a - dv, f6bp = f6b - dv, f6cp = f6c - dv;
          if (in_range(i+1,(int)H) && in_range(j+1,(int)W)){
            // (f6a,f6b,f6cp)
            #if CP_DEBUG_VISIT
            MARK_FACE(f6a,f6b,f6cp);
            #endif
            if (has_cp(cp_faces, f6a,f6b,f6cp)) {
              required_eb=0;
            }
            { T eb = derive_cp_abs_eb_sos_online<T>(U_fp[f6cp],U_fp[f6b],U_fp[f6a],
                                                    V_fp[f6cp],V_fp[f6b],V_fp[f6a]);
              if (eb < required_eb) required_eb = eb;
            }
          }
        }
        // {
        //   // 保证 |dec-orig| ≤ abs_eb < |orig| ⇒ 不会跨 0 翻符号
        //   T sign_guard = std::min(std::llabs(curU_val), std::llabs(curV_val));
        //   if (sign_guard > 0 && required_eb >= sign_guard){
        //       T max_eb_to_preserve_sign = sign_guard - 1; // 留 1 LSB 余量
        //       if (max_eb_to_preserve_sign < required_eb)
        //           required_eb = max_eb_to_preserve_sign;
        //   }
        // }
        T abs_eb = required_eb;
        int id = eb_exponential_quantize(abs_eb,base,log_of_base,threshold);
        
        if (abs_eb == 0 )
        {
          *(eb_pos++) = 0;            // ebid=0 → 无损点
          unpred.push_back(U[v]);     // 原始浮点（解码端直接回填）
          unpred.push_back(V[v]);
          continue;
        }

        // ===== 量化 eb（配套的新幂指数量化）并编码该顶点 =====
        {
          // T abs_eb = required_eb;
          // // int id = eb_exponential_quantize_new(abs_eb, threshold); // id>=1，abs_eb 替换为代表值
          
          // // *eb_pos = id;
          // int id = eb_exponential_quantize(abs_eb,base,log_of_base,threshold);
          *eb_pos = id;

          bool unpred_flag=false;
          T dec[2];
          T abs_err_fp_q[2] = {0,0};
          int qindex_cache[2] = {0, 0};

          for (int p=0; p<2; ++p){
            T *cur  = (p==0)? curU : curV;
            T  curv = (p==0) ? curU_val : curV_val;

            // 3D 一阶 Lorenzo（时间最慢）
            T d0 = (t&&i&&j)? cur[-sk - si - sj] : 0;
            T d1 = (t&&i)   ? cur[-sk - si]      : 0;
            T d2 = (t&&j)   ? cur[-sk - sj]      : 0;
            T d3 = (t)      ? cur[-sk]           : 0;
            T d4 = (i&&j)   ? cur[-si - sj]      : 0;
            T d5 = (i)      ? cur[-si]           : 0;
            T d6 = (j)      ? cur[-sj]           : 0;
            T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

            T diff = curv - pred;
            T qd = (std::llabs(diff)/abs_eb) + 1;
            if (qd < capacity){
              qd = (diff > 0) ? qd : -qd;
              int qindex = (int)(qd/2) + intv_radius;
              qindex_cache[p] = qindex;
              dec[p] = pred + 2*(qindex - intv_radius)*abs_eb;

              // 守门：必须用 *代表值* abs_eb 校验
              if (std::llabs(dec[p] - curv) > abs_eb){
              // if (std::llabs(dec[p] - curv) > required_eb){
                unpred_flag = true; break;
              }
              abs_err_fp_q[p] = std::llabs(dec[p] - curv);
            }else{
              unpred_flag = true; break;
            }
          }

          if (unpred_flag){
            *(eb_pos++) = 0;                  // 改回无损
            unpred.push_back(U[v]);
            unpred.push_back(V[v]);
          }else{
            ++eb_pos;
            *(dq_pos_u++) = qindex_cache[0];
            *(dq_pos_v++) = qindex_cache[1];
            *curU = dec[0];
            *curV = dec[1];

            // 编码端自检（浮点域）
            double abs_eb_fp = (double)abs_eb / (double)scale;
            double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
            double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
            enc_max_abs_eb_fp   = std::max(enc_max_abs_eb_fp, abs_eb_fp);
            enc_max_real_err_fp = std::max(enc_max_real_err_fp, std::max(err_u_fp, err_v_fp));
          }
        }
      }
    }
  }

  // std::cerr << "[ENC] max abs_eb(fp) = " << enc_max_abs_eb_fp
  //           << ", max actual |err|(fp) = " << enc_max_real_err_fp << "\n";

  
  #if CP_DEBUG_VISIT
  {
      size_t miss = 0, shown = 0;
      for (const auto &fk : cp_faces){
          if (!visited_faces.count(fk)){
              ++miss;
              if (shown < 50){ // 只演示前 50 个，避免刷屏
                  size_t a = fk.v[0], b = fk.v[1], c = fk.v[2];
                  auto [ta,ia,ja] = decode_tij(a);
                  auto [tb,ib,jb] = decode_tij(b);
                  auto [tc,ic,jc] = decode_tij(c);
                  fprintf(stderr,
                    "[MISSING] face (%zu,%zu,%zu)  "
                    "A(t=%d,i=%d,j=%d)  B(t=%d,i=%d,j=%d)  C(t=%d,i=%d,j=%d)  type=%s\n",
                    a,b,c, ta,ia,ja, tb,ib,jb, tc,ic,jc,
                    classify_face(a,b,c));
                  ++shown;
              }
          }
      }
      fprintf(stderr, "[COVERAGE] encoder touched faces = %zu, cp_faces = %zu, missing = %zu\n",
              visited_faces.size(), cp_faces.size(), miss);
  }
  #endif
  // 5) 打包码流
  unsigned char *compressed = (unsigned char*)std::malloc( (size_t)(2*N*sizeof(T)) );
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);
  std::cout << "write intv_radius = " << intv_radius << "\n";

  size_t unpred_cnt = unpred.size();
  std::cout << "write unpred cnt = " << unpred_cnt << ",ratio=" << (double)unpred_cnt/(2*N) << "\n";
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) write_array_to_dst(pos, unpred.data(), unpred_cnt);

  unsigned char *pos_before_ebq = pos;
  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::cout << "Huffman eb size = " << (size_t)(pos - pos_before_ebq) << "\n";
  std::free(eb_quant_index);

  unsigned char *pos_before_dq = pos;
  const size_t data_quant_num_u = (size_t)(dq_pos_u - data_quant_index_u);
  const size_t data_quant_num_v = (size_t)(dq_pos_v - data_quant_index_v);

  auto apply_delta_zigzag = [](int* data, size_t len){
    if (!len) return;
    int prev = 0;
    for (size_t idx = 0; idx < len; ++idx){
      int curr = data[idx];
      int delta = curr - prev;
      prev = curr;
      data[idx] = (delta >= 0) ? (delta << 1) : ((-delta << 1) - 1);
    }
  };

  apply_delta_zigzag(data_quant_index_u, data_quant_num_u);
  apply_delta_zigzag(data_quant_index_v, data_quant_num_v);

  write_variable_to_dst(pos, data_quant_num_u);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index_u, data_quant_num_u, pos);
  std::cout << "Huffman dq U size = " << (size_t)(pos - pos_before_dq) << "\n";
  pos_before_dq = pos;
  write_variable_to_dst(pos, data_quant_num_v);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index_v, data_quant_num_v, pos);
  std::cout << "Huffman dq V size = " << (size_t)(pos - pos_before_dq) << "\n";
  std::free(data_quant_index_u);
  std::free(data_quant_index_v);

  compressed_size = (size_t)(pos - compressed);
  std::free(U_fp); std::free(V_fp);
  return compressed;
}




#if 0
template<typename T_data, typename LayerFetcher>
static unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_streaming_impl(
    LayerFetcher&& fetch_layer,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  using T = int64_t;
  const size_t H = r1, W = r2, Tt = r3;
  if (!H || !W || !Tt) {
    compressed_size = 0;
    return nullptr;
  }

  const Size3 sz{(int)H, (int)W, (int)Tt};
  const size_t layer_size = H * W;

  std::vector<T_data> tmpU(layer_size);
  std::vector<T_data> tmpV(layer_size);

  // -------- Pass 1: determine scaling factor --------
  double vector_field_resolution = 0.0;
  for (size_t t = 0; t < Tt; ++t) {
    if (!fetch_layer(t, tmpU.data(), tmpV.data())) {
      compressed_size = 0;
      return nullptr;
    }
    for (size_t idx = 0; idx < layer_size; ++idx) {
      double max_val = std::max(std::fabs((double)tmpU[idx]), std::fabs((double)tmpV[idx]));
      if (max_val > vector_field_resolution) vector_field_resolution = max_val;
    }
  }

  const int type_bits = 63;
  const int nbits = (type_bits - 3) / 2;
  int vbits = 0;
  if (vector_field_resolution > 0.0) {
    vbits = (int)std::ceil(std::log2(vector_field_resolution));
  }
  int shift_bits = nbits - vbits;
  if (shift_bits < 0) shift_bits = 0;
  T scale = (T)1 << shift_bits;

  std::cerr << "resolution=" << vector_field_resolution
            << ", factor=" << (long long)scale
            << ", nbits=" << nbits
            << ", vbits=" << vbits
            << ", shift_bits=" << shift_bits << std::endl;

  T fp_max = std::numeric_limits<T>::min();
  T fp_min = std::numeric_limits<T>::max();
  printf("max = %lld, min = %lld\n", (long long)fp_max, (long long)fp_min);

  auto load_layer_fixed = [&](size_t t,
                              std::vector<T>& outU,
                              std::vector<T>& outV,
                              std::vector<T_data>* storeU,
                              std::vector<T_data>* storeV,
                              bool update_minmax)->bool {
    if (!fetch_layer(t, tmpU.data(), tmpV.data())) {
      return false;
    }
    for (size_t idx = 0; idx < layer_size; ++idx) {
      T u_fp = static_cast<T>(tmpU[idx] * (double)scale);
      T v_fp = static_cast<T>(tmpV[idx] * (double)scale);
      outU[idx] = u_fp;
      outV[idx] = v_fp;
      if (storeU) (*storeU)[idx] = tmpU[idx];
      if (storeV) (*storeV)[idx] = tmpV[idx];
      if (update_minmax) {
        if (u_fp > fp_max) fp_max = u_fp;
        if (v_fp > fp_max) fp_max = v_fp;
        if (u_fp < fp_min) fp_min = u_fp;
        if (v_fp < fp_min) fp_min = v_fp;
      }
    }
    return true;
  };

  auto pre_compute_time = std::chrono::high_resolution_clock::now();

  std::vector<T> u_prev(layer_size, 0), v_prev(layer_size, 0);
  std::vector<T> u_curr(layer_size, 0), v_curr(layer_size, 0);
  std::vector<T> u_next(layer_size, 0), v_next(layer_size, 0);

  if (!load_layer_fixed(0, u_curr, v_curr, nullptr, nullptr, true)) {
    compressed_size = 0;
    return nullptr;
  }
  if (Tt > 1) {
    if (!load_layer_fixed(1, u_next, v_next, nullptr, nullptr, true)) {
      compressed_size = 0;
      return nullptr;
    }
  }

  std::unordered_set<FaceKeySZ, FaceKeySZHash> cp_faces;
  cp_faces.reserve((size_t)(H*(size_t)W*(size_t)Tt / 8));

  auto eval_face = [&](size_t t_base,
                       size_t a,size_t b,size_t c)->bool {
    int idxs[3] = { (int)a, (int)b, (int)c };
    int64_t vf[3][2];
    auto fetch_value = [&](size_t vidx, int pos){
      int vt, vi, vj;
      inv_vid(vidx, (int)H, (int)W, vt, vi, vj);
      size_t off = (size_t)vi * W + (size_t)vj;
      const std::vector<T>* su = nullptr;
      const std::vector<T>* sv = nullptr;
      if (vt == (int)t_base) {
        su = &u_curr; sv = &v_curr;
      } else if (vt == (int)t_base + 1) {
        su = &u_next; sv = &v_next;
      } else if (vt == (int)t_base - 1) {
        su = &u_prev; sv = &v_prev;
      } else {
        su = &u_curr; sv = &v_curr;
      }
      vf[pos][0] = (*su)[off];
      vf[pos][1] = (*sv)[off];
    };
    fetch_value(a, 0);
    fetch_value(b, 1);
    fetch_value(c, 2);
    return ftk::robust_critical_point_in_simplex2(vf, idxs);
  };

  const size_t dv = H * W;

  #ifdef _OPENMP
  const size_t block_rows_cp = 32;
  const size_t block_cols_cp = 32;
  const int cp_threads = omp_get_max_threads();
  std::vector<std::vector<FaceKeySZ>> cp_thread_faces(cp_threads);
  auto flush_thread_faces = [&](std::vector<std::vector<FaceKeySZ>>& buffers){
    for (auto &vec : buffers){
      for (const auto &f : vec){
        cp_faces.emplace(f);
      }
      vec.clear();
    }
  };
  #endif

  for (size_t t = 0; t < Tt; ++t) {
    if (t % 1000 == 0) {
      printf("pre-compute cp lower layer %zu / %zu\n", t, Tt);
    }

    #ifdef _OPENMP
    const size_t layer_i_limit = (H > 0) ? (H - 1) : 0;
    const size_t layer_j_limit = (W > 0) ? (W - 1) : 0;
    const size_t blocks_i_layer = (layer_i_limit + block_rows_cp - 1) / block_rows_cp;
    const size_t blocks_j_layer = (layer_j_limit + block_cols_cp - 1) / block_cols_cp;
    const size_t blocks_i_full  = (H + block_rows_cp - 1) / block_rows_cp;
    const size_t blocks_j_fullm1 = (W > 0) ? ((W - 1 + block_cols_cp - 1) / block_cols_cp) : 0;
    const size_t blocks_i_m1 = (H > 0) ? ((H - 1 + block_rows_cp - 1) / block_rows_cp) : 0;
    const size_t blocks_j_full = (W + block_cols_cp - 1) / block_cols_cp;

    #pragma omp parallel
    {
      auto& local_faces = cp_thread_faces[omp_get_thread_num()];
      local_faces.clear();

      if (layer_i_limit && layer_j_limit) {
        const size_t total_blocks = blocks_i_layer * blocks_j_layer;
        #pragma omp for schedule(static)
        for (size_t block = 0; block < total_blocks; ++block) {
          size_t bi = block / blocks_j_layer;
          size_t bj = block % blocks_j_layer;
          size_t i_begin = bi * block_rows_cp;
          size_t i_end = std::min(i_begin + block_rows_cp, layer_i_limit);
          size_t j_begin = bj * block_cols_cp;
          size_t j_end = std::min(j_begin + block_cols_cp, layer_j_limit);
          for (size_t i = i_begin; i < i_end; ++i) {
            for (size_t j = j_begin; j < j_end; ++j) {
              size_t v00 = vid((int)t,(int)i,(int)j,sz);
              size_t v10 = vid((int)t,(int)i,(int)(j+1),sz);
              size_t v01 = vid((int)t,(int)(i+1),(int)j,sz);
              size_t v11 = vid((int)t,(int)(i+1),(int)(j+1),sz);
              if (eval_face(t, v00, v01, v11)) local_faces.emplace_back(v00, v01, v11);
              if (eval_face(t, v00, v10, v11)) local_faces.emplace_back(v00, v10, v11);
            }
          }
        }
      }

      if (t + 1 < Tt) {
        if (t % 1000 == 0) {
          #pragma omp single
          printf("pre-compute cp side_hor layer %zu / %zu\n", t, Tt);
        }
        if (blocks_i_full && (W > 1)) {
          const size_t total_blocks_h = blocks_i_full * blocks_j_layer;
          #pragma omp for schedule(static)
          for (size_t block = 0; block < total_blocks_h; ++block) {
            size_t bi = block / blocks_j_layer;
            size_t bj = block % blocks_j_layer;
            size_t i_begin = bi * block_rows_cp;
            size_t i_end = std::min(i_begin + block_rows_cp, H);
            size_t j_begin = bj * block_cols_cp;
            size_t j_end = std::min(j_begin + block_cols_cp, W - 1);
            for (size_t i = i_begin; i < i_end; ++i) {
              for (size_t j = j_begin; j < j_end; ++j) {
                size_t a = vid((int)t,(int)i,(int)j,sz);
                size_t b = vid((int)t,(int)i,(int)(j+1),sz);
                size_t ap = a + dv;
                size_t bp = b + dv;
                if (eval_face(t, a, b, bp)) local_faces.emplace_back(a, b, bp);
                if (eval_face(t, a, bp, ap)) local_faces.emplace_back(a, bp, ap);
              }
            }
          }
        }

        if (t % 1000 == 0) {
          #pragma omp single
          printf("pre-compute cp side_ver layer %zu / %zu\n", t, Tt);
        }
        if (blocks_i_m1 && blocks_j_full) {
          const size_t total_blocks_v = blocks_i_m1 * blocks_j_full;
          #pragma omp for schedule(static)
          for (size_t block = 0; block < total_blocks_v; ++block) {
            size_t bi = block / blocks_j_full;
            size_t bj = block % blocks_j_full;
            size_t i_begin = bi * block_rows_cp;
            size_t i_end = std::min(i_begin + block_rows_cp, (H > 0) ? (H - 1) : 0);
            size_t j_begin = bj * block_cols_cp;
            size_t j_end = std::min(j_begin + block_cols_cp, W);
            for (size_t i = i_begin; i < i_end; ++i) {
              for (size_t j = j_begin; j < j_end; ++j) {
                size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
                size_t ax0y1 = vid((int)t,(int)(i+1),(int)j,sz);
                size_t bx0y0 = ax0y0 + dv;
                size_t bx0y1 = ax0y1 + dv;
                if (eval_face(t, ax0y0, ax0y1, bx0y0)) local_faces.emplace_back(ax0y0, ax0y1, bx0y0);
                if (eval_face(t, ax0y1, bx0y0, bx0y1)) local_faces.emplace_back(ax0y1, bx0y0, bx0y1);
              }
            }
          }
        }

        if (t % 1000 == 0) {
          #pragma omp single
          printf("pre-compute cp diag layer %zu / %zu\n", t, Tt);
        }
        if (layer_i_limit && layer_j_limit) {
          const size_t total_blocks_d = blocks_i_layer * blocks_j_layer;
          #pragma omp for schedule(static)
          for (size_t block = 0; block < total_blocks_d; ++block) {
            size_t bi = block / blocks_j_layer;
            size_t bj = block % blocks_j_layer;
            size_t i_begin = bi * block_rows_cp;
            size_t i_end = std::min(i_begin + block_rows_cp, layer_i_limit);
            size_t j_begin = bj * block_cols_cp;
            size_t j_end = std::min(j_begin + block_cols_cp, layer_j_limit);
            for (size_t i = i_begin; i < i_end; ++i) {
              for (size_t j = j_begin; j < j_end; ++j) {
                size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
                size_t ax1y1 = vid((int)t,(int)(i+1),(int)(j+1),sz);
                size_t bx0y0 = ax0y0 + dv;
                size_t bx1y1 = ax1y1 + dv;
                if (eval_face(t, ax0y0, ax1y1, bx0y0)) local_faces.emplace_back(ax0y0, ax1y1, bx0y0);
                if (eval_face(t, ax1y1, bx0y0, bx1y1)) local_faces.emplace_back(ax1y1, bx0y0, bx1y1);
              }
            }
          }
        }

        if (t % 1000 == 0) {
          #pragma omp single
          printf("pre-compute cp inside layer %zu / %zu\n", t, Tt);
        }
        if (layer_i_limit && layer_j_limit) {
          const size_t total_blocks_in = blocks_i_layer * blocks_j_layer;
          #pragma omp for schedule(static)
          for (size_t block = 0; block < total_blocks_in; ++block) {
            size_t bi = block / blocks_j_layer;
            size_t bj = block % blocks_j_layer;
            size_t i_begin = bi * block_rows_cp;
            size_t i_end = std::min(i_begin + block_rows_cp, layer_i_limit);
            size_t j_begin = bj * block_cols_cp;
            size_t j_end = std::min(j_begin + block_cols_cp, layer_j_limit);
            for (size_t i = i_begin; i < i_end; ++i) {
              for (size_t j = j_begin; j < j_end; ++j) {
                size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
                size_t ax0y1 = vid((int)t,(int)(i+1),(int)j,sz);
                size_t ax1y1 = vid((int)t,(int)(i+1),(int)(j+1),sz);
                size_t ax1y0 = vid((int)t,(int)i,(int)(j+1),sz);
                size_t bx0y0 = ax0y0 + dv;
                size_t bx0y1 = ax0y1 + dv;
                size_t bx1y1 = ax1y1 + dv;
                size_t bx1y0 = ax1y0 + dv;
                if (eval_face(t, ax0y1, bx0y0, ax1y1)) local_faces.emplace_back(ax0y1, bx0y0, ax1y1);
                if (eval_face(t, ax0y1, bx0y0, bx1y1)) local_faces.emplace_back(ax0y1, bx0y0, bx1y1);
                if (eval_face(t, ax0y0, ax1y1, bx1y0)) local_faces.emplace_back(ax0y0, ax1y1, bx1y0);
                if (eval_face(t, ax1y1, bx0y0, bx1y0)) local_faces.emplace_back(ax1y1, bx0y0, bx1y0);
              }
            }
          }
        }
      }
    }

    flush_thread_faces(cp_thread_faces);
    #else
    if (t % 1000 == 0) {
      printf("pre-compute cp lower layer %zu / %zu\n", t, Tt);
    }
    for (size_t i = 0; i + 1 < H; ++i) {
      for (size_t j = 0; j + 1 < W; ++j) {
        size_t v00 = vid((int)t,(int)i,(int)j,sz);
        size_t v10 = vid((int)t,(int)i,(int)(j+1),sz);
        size_t v01 = vid((int)t,(int)(i+1),(int)j,sz);
        size_t v11 = vid((int)t,(int)(i+1),(int)(j+1),sz);
        if (eval_face(t, v00, v01, v11)) cp_faces.emplace(v00, v01, v11);
        if (eval_face(t, v00, v10, v11)) cp_faces.emplace(v00, v10, v11);
      }
    }
    if (t + 1 < Tt) {
      if (t % 1000 == 0) {
        printf("pre-compute cp side_hor layer %zu / %zu\n", t, Tt);
      }
      for (size_t i = 0; i < H; ++i) {
        for (size_t j = 0; j + 1 < W; ++j) {
          size_t a = vid((int)t,(int)i,(int)j,sz);
          size_t b = vid((int)t,(int)i,(int)(j+1),sz);
          size_t ap = a + dv;
          size_t bp = b + dv;
          if (eval_face(t, a, b, bp)) cp_faces.emplace(a, b, bp);
          if (eval_face(t, a, bp, ap)) cp_faces.emplace(a, bp, ap);
        }
      }
      if (t % 1000 == 0) {
        printf("pre-compute cp side_ver layer %zu / %zu\n", t, Tt);
      }
      for (size_t i = 0; i + 1 < H; ++i) {
        for (size_t j = 0; j < W; ++j) {
          size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
          size_t ax0y1 = vid((int)t,(int)(i+1),(int)j,sz);
          size_t bx0y0 = ax0y0 + dv;
          size_t bx0y1 = ax0y1 + dv;
          if (eval_face(t, ax0y0, ax0y1, bx0y0)) cp_faces.emplace(ax0y0, ax0y1, bx0y0);
          if (eval_face(t, ax0y1, bx0y0, bx0y1)) cp_faces.emplace(ax0y1, bx0y0, bx0y1);
        }
      }
      if (t % 1000 == 0) {
        printf("pre-compute cp diag layer %zu / %zu\n", t, Tt);
      }
      for (size_t i = 0; i + 1 < H; ++i) {
        for (size_t j = 0; j + 1 < W; ++j) {
          size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
          size_t ax1y1 = vid((int)t,(int)(i+1),(int)(j+1),sz);
          size_t bx0y0 = ax0y0 + dv;
          size_t bx1y1 = ax1y1 + dv;
          if (eval_face(t, ax0y0, ax1y1, bx0y0)) cp_faces.emplace(ax0y0, ax1y1, bx0y0);
          if (eval_face(t, ax1y1, bx0y0, bx1y1)) cp_faces.emplace(ax1y1, bx0y0, bx1y1);
        }
      }
      if (t % 1000 == 0) {
        printf("pre-compute cp inside layer %zu / %zu\n", t, Tt);
      }
      for (size_t i = 0; i + 1 < H; ++i) {
        for (size_t j = 0; j + 1 < W; ++j) {
          size_t ax0y0 = vid((int)t,(int)i,(int)j,sz);
          size_t ax0y1 = vid((int)t,(int)(i+1),(int)j,sz);
          size_t ax1y1 = vid((int)t,(int)(i+1),(int)(j+1),sz);
          size_t ax1y0 = vid((int)t,(int)i,(int)(j+1),sz);
          size_t bx0y0 = ax0y0 + dv;
          size_t bx0y1 = ax0y1 + dv;
          size_t bx1y1 = ax1y1 + dv;
          size_t bx1y0 = ax1y0 + dv;
          if (eval_face(t, ax0y1, bx0y0, ax1y1)) cp_faces.emplace(ax0y1, bx0y0, ax1y1);
          if (eval_face(t, ax0y1, bx0y0, bx1y1)) cp_faces.emplace(ax0y1, bx0y0, bx1y1);
          if (eval_face(t, ax0y0, ax1y1, bx1y0)) cp_faces.emplace(ax0y0, ax1y1, bx1y0);
          if (eval_face(t, ax1y1, bx0y0, bx1y0)) cp_faces.emplace(ax1y1, bx0y0, bx1y0);
        }
      }
    }
    #endif

    // rotate buffers
    if (t + 1 < Tt) {
      u_prev.swap(u_curr);
      v_prev.swap(v_curr);
      u_curr.swap(u_next);
      v_curr.swap(v_next);
      if (t + 2 < Tt) {
        if (!load_layer_fixed(t + 2, u_next, v_next, nullptr, nullptr, true)) {
          compressed_size = 0;
          return nullptr;
        }
      }
    }
  }

  printf("max = %lld, min = %lld\n", (long long)fp_max, (long long)fp_min);
  const T range = fp_max - fp_min;

  auto pre_compute_time_end = std::chrono::high_resolution_clock::now();
  std::cout << "pre-compute cp faces time second: "
            << std::chrono::duration<double>(pre_compute_time_end - pre_compute_time).count()
            << std::endl;
  std::cout << "Total faces with CP ori: " << cp_faces.size() << std::endl;

  // -------- Pass 3: quantization in streaming manner --------
  int* eb_quant_index  = (int*)std::malloc((size_t)(H*W*Tt) * sizeof(int));
  int* data_quant_index= (int*)std::malloc((size_t)(2*H*W*Tt) * sizeof(int));
  if (!eb_quant_index || !data_quant_index) {
    if (eb_quant_index) std::free(eb_quant_index);
    if (data_quant_index) std::free(data_quant_index);
    compressed_size = 0;
    return nullptr;
  }
  int* eb_pos = eb_quant_index;
  int* dq_pos = data_quant_index;
  std::vector<T_data> unpred;
  unpred.reserve((N / 4) * 2);

  double enc_max_abs_eb_fp   = 0.0;
  double enc_max_real_err_fp = 0.0;

  const int base = 2;
  const int capacity = 65536;
  const double log_of_base = std::log2(base);
  const int intv_radius = (capacity >> 1);

  T max_eb = 0;
  if (mode == EbMode::Relative) {
    printf("Compression Using Relative Eb Mode!\n");
    max_eb = (T)(max_pwr_eb * (double)range);
  } else if (mode == EbMode::Absolute) {
    printf("Compression Using Absolute Eb Mode!\n");
    max_eb = (T)(max_pwr_eb * (double)scale);
  } else {
    std::cerr << "Error: Unsupported EbMode!\n";
    std::free(eb_quant_index);
    std::free(data_quant_index);
    compressed_size = 0;
    return nullptr;
  }
  const T threshold = 1;

  std::vector<T> dec_prev_u(layer_size, 0), dec_prev_v(layer_size, 0);
  std::vector<T> dec_curr_u(layer_size, 0), dec_curr_v(layer_size, 0);
  std::vector<T> required_eb_layer(layer_size, max_eb);

  std::vector<T_data> float_prev_u(layer_size, (T_data)0);
  std::vector<T_data> float_prev_v(layer_size, (T_data)0);
  std::vector<T_data> float_curr_u(layer_size, (T_data)0);
  std::vector<T_data> float_curr_v(layer_size, (T_data)0);
  std::vector<T_data> float_next_u(layer_size, (T_data)0);
  std::vector<T_data> float_next_v(layer_size, (T_data)0);

  std::fill(u_prev.begin(), u_prev.end(), 0);
  std::fill(v_prev.begin(), v_prev.end(), 0);

  if (!load_layer_fixed(0, u_curr, v_curr, &float_curr_u, &float_curr_v, false)) {
    std::free(eb_quant_index);
    std::free(data_quant_index);
    compressed_size = 0;
    return nullptr;
  }
  if (Tt > 1) {
    if (!load_layer_fixed(1, u_next, v_next, &float_next_u, &float_next_v, false)) {
      std::free(eb_quant_index);
      std::free(data_quant_index);
      compressed_size = 0;
      return nullptr;
    }
  }

  const int di[6] = { 0,  0,  1, -1,  1, -1 };
  const int dj[6] = { 1, -1,  0,  0,  1, -1 };

  auto fetch_fixed_pair = [&](size_t t_base, size_t vidx)->std::pair<T,T> {
    int vt, vi, vj;
    inv_vid(vidx, (int)H, (int)W, vt, vi, vj);
    size_t off = (size_t)vi * W + (size_t)vj;
    if (vt == (int)t_base) {
      return { u_curr[off], v_curr[off] };
    } else if (vt == (int)t_base + 1) {
      return { u_next[off], v_next[off] };
    } else if (vt == (int)t_base - 1) {
      return { u_prev[off], v_prev[off] };
    }
    return { u_curr[off], v_curr[off] };
  };

  auto compute_required_eb = [&](size_t t_idx, size_t i_idx, size_t j_idx)->T {
    size_t global_idx = vid((int)t_idx,(int)i_idx,(int)j_idx,sz);
    T required_eb = max_eb;

    for (int ci = (int)i_idx - 1; ci <= (int)i_idx; ++ci) {
      if (!in_range(ci, (int)H - 1)) continue;
      for (int cj = (int)j_idx - 1; cj <= (int)j_idx; ++cj) {
        if (!in_range(cj, (int)W - 1)) continue;
        size_t v00 = vid((int)t_idx, ci,   cj,   sz);
        size_t v10 = vid((int)t_idx, ci,   cj+1, sz);
        size_t v01 = vid((int)t_idx, ci+1, cj,   sz);
        size_t v11 = vid((int)t_idx, ci+1, cj+1, sz);

        auto u01 = fetch_fixed_pair(t_idx, v01);
        auto u11 = fetch_fixed_pair(t_idx, v11);
        auto u10 = fetch_fixed_pair(t_idx, v10);
        auto u00 = fetch_fixed_pair(t_idx, v00);

        if (global_idx == v00 || global_idx == v01 || global_idx == v11) {
          if (has_cp(cp_faces, v00, v01, v11)) required_eb = 0;
          T eb = derive_cp_abs_eb_sos_online<T>(u11.first, u01.first, u00.first,
                                                u11.second, u01.second, u00.second);
          if (eb < required_eb) required_eb = eb;
        }
        if (global_idx == v00 || global_idx == v10 || global_idx == v11) {
          if (has_cp(cp_faces, v00, v10, v11)) required_eb = 0;
          T eb = derive_cp_abs_eb_sos_online<T>(u11.first, u10.first, u00.first,
                                                u11.second, u10.second, u00.second);
          if (eb < required_eb) required_eb = eb;
        }
      }
    }

    if (t_idx + 1 < Tt) {
      for (int k = 0; k < 6; ++k) {
        int ni = (int)i_idx + di[k];
        int nj = (int)j_idx + dj[k];
        if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
        size_t a = vid((int)t_idx, (int)i_idx, (int)j_idx, sz);
        size_t b = vid((int)t_idx, ni, nj, sz);
        size_t ap = a + dv;
        size_t bp = b + dv;
        auto val_a  = fetch_fixed_pair(t_idx, a);
        auto val_b  = fetch_fixed_pair(t_idx, b);
        auto val_ap = fetch_fixed_pair(t_idx, ap);
        auto val_bp = fetch_fixed_pair(t_idx, bp);

        if (k == 0 || k == 3 || k == 5) {
          if (has_cp(cp_faces, a, b, bp)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_bp.first, val_b.first, val_a.first,
                                                  val_bp.second, val_b.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
          if (has_cp(cp_faces, a, bp, ap)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_ap.first, val_bp.first, val_a.first,
                                                  val_ap.second, val_bp.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
        } else {
          if (has_cp(cp_faces, a, b, ap)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_ap.first, val_b.first, val_a.first,
                                                  val_ap.second, val_b.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
        }
      }
    }

    if (t_idx > 0) {
      for (int k = 0; k < 6; ++k) {
        int ni = (int)i_idx + di[k];
        int nj = (int)j_idx + dj[k];
        if (!in_range(ni, (int)H) || !in_range(nj, (int)W)) continue;
        size_t a = vid((int)t_idx, (int)i_idx, (int)j_idx, sz);
        size_t b = vid((int)t_idx, ni, nj, sz);
        size_t ap = a - dv;
        size_t bp = b - dv;
        auto val_a  = fetch_fixed_pair(t_idx, a);
        auto val_b  = fetch_fixed_pair(t_idx, b);
        auto val_ap = fetch_fixed_pair(t_idx, ap);
        auto val_bp = fetch_fixed_pair(t_idx, bp);

        if (k == 0 || k == 3 || k == 5) {
          if (has_cp(cp_faces, a, b, ap)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_ap.first, val_b.first, val_a.first,
                                                  val_ap.second, val_b.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
          if (has_cp(cp_faces, a, ap, bp)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_bp.first, val_ap.first, val_a.first,
                                                  val_bp.second, val_ap.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
        } else {
          if (has_cp(cp_faces, a, b, bp)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_bp.first, val_b.first, val_a.first,
                                                  val_bp.second, val_b.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
          if (has_cp(cp_faces, a, bp, ap)) required_eb = 0;
          {
            T eb = derive_cp_abs_eb_sos_online<T>(val_ap.first, val_bp.first, val_a.first,
                                                  val_ap.second, val_bp.second, val_a.second);
            if (eb < required_eb) required_eb = eb;
          }
        }
      }
    }

    auto valid = [&](int x, int limit){ return x >= 0 && x < limit; };

    if (t_idx + 1 < Tt) {
      int i_int = (int)i_idx;
      int j_int = (int)j_idx;
      if (valid(i_int - 1, (int)H) && valid(j_int + 1, (int)W)) {
        size_t f1a  = vid((int)t_idx, i_int,     j_int,     sz);
        size_t f1b  = vid((int)t_idx, i_int,     j_int + 1, sz);
        size_t f1c  = vid((int)t_idx, i_int - 1, j_int,     sz);
        size_t f1ap = f1a + dv;
        size_t f1bp = f1b + dv;
        size_t f1cp = f1c + dv;
        auto va = fetch_fixed_pair(t_idx, f1a);
        auto vb = fetch_fixed_pair(t_idx, f1b);
        auto vcp = fetch_fixed_pair(t_idx, f1cp);
        auto vbp = fetch_fixed_pair(t_idx, f1bp);
        if (has_cp(cp_faces, f1a, f1cp, f1b)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vb.first, vcp.first, va.first,
                                                vb.second, vcp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
        if (has_cp(cp_faces, f1a, f1cp, f1bp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vbp.first, vcp.first, va.first,
                                                vbp.second, vcp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int - 1, (int)H) && valid(j_int - 1, (int)W)) {
        size_t f2a  = vid((int)t_idx, i_int,     j_int,     sz);
        size_t f2b  = vid((int)t_idx, i_int - 1, j_int,     sz);
        size_t f2c  = vid((int)t_idx, i_int - 1, j_int - 1, sz);
        size_t f2ap = f2a + dv;
        size_t f2bp = f2b + dv;
        size_t f2cp = f2c + dv;
        auto va = fetch_fixed_pair(t_idx, f2a);
        auto vc = fetch_fixed_pair(t_idx, f2c);
        auto vbp = fetch_fixed_pair(t_idx, f2bp);
        auto vcp = fetch_fixed_pair(t_idx, f2cp);
        if (has_cp(cp_faces, f2a, f2c, f2bp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vbp.first, vc.first, va.first,
                                                vbp.second, vc.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
        if (has_cp(cp_faces, f2a, f2bp, f2cp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vcp.first, vbp.first, va.first,
                                                vcp.second, vbp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int - 1, (int)H) && valid(j_int - 1, (int)W)) {
        size_t f3a  = vid((int)t_idx, i_int,     j_int,     sz);
        size_t f3b  = vid((int)t_idx, i_int - 1, j_int - 1, sz);
        size_t f3c  = vid((int)t_idx, i_int,     j_int - 1, sz);
        size_t f3bp = f3b + dv;
        auto va = fetch_fixed_pair(t_idx, f3a);
        auto vb = fetch_fixed_pair(t_idx, f3b);
        auto vbp = fetch_fixed_pair(t_idx, f3bp);
        auto vc = fetch_fixed_pair(t_idx, f3c);
        if (has_cp(cp_faces, f3a, f3bp, f3c)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vc.first, vbp.first, va.first,
                                                vc.second, vbp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int + 1, (int)H) && valid(j_int + 1, (int)W)) {
        size_t f6a  = vid((int)t_idx, i_int,     j_int,     sz);
        size_t f6b  = vid((int)t_idx, i_int,     j_int + 1, sz);
        size_t f6c  = vid((int)t_idx, i_int + 1, j_int + 1, sz);
        size_t f6bp = f6b + dv;
        auto va = fetch_fixed_pair(t_idx, f6a);
        auto vb = fetch_fixed_pair(t_idx, f6b);
        auto vbp = fetch_fixed_pair(t_idx, f6bp);
        auto vc = fetch_fixed_pair(t_idx, f6c);
        if (has_cp(cp_faces, f6a, f6bp, f6c)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vc.first, vbp.first, va.first,
                                                vc.second, vbp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }
    }

    if (t_idx > 0) {
      int i_int = (int)i_idx;
      int j_int = (int)j_idx;
      if (valid(i_int - 1, (int)H) && valid(j_int - 1, (int)W)) {
        size_t f3a  = vid((int)t_idx,     i_int,     j_int,     sz);
        size_t f3b  = vid((int)t_idx - 1, i_int - 1, j_int - 1, sz);
        size_t f3c  = vid((int)t_idx,     i_int,     j_int - 1, sz);
        size_t f3cp = f3c - dv;
        auto va = fetch_fixed_pair(t_idx, f3a);
        auto vb = fetch_fixed_pair(t_idx, f3b);
        auto vc = fetch_fixed_pair(t_idx, f3c);
        auto vcp = fetch_fixed_pair(t_idx, f3cp);
        if (has_cp(cp_faces, f3a, f3c, f3cp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vcp.first, vc.first, va.first,
                                                vcp.second, vc.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int + 1, (int)H) && valid(j_int - 1, (int)W)) {
        size_t f4a  = vid((int)t_idx,     i_int,     j_int,     sz);
        size_t f4b  = vid((int)t_idx - 1, i_int,     j_int - 1, sz);
        size_t f4c  = vid((int)t_idx - 1, i_int + 1, j_int - 1, sz);
        size_t f4bp = f4b + dv;
        size_t f4cp = f4c + dv;
        auto va = fetch_fixed_pair(t_idx, f4a);
        auto vb = fetch_fixed_pair(t_idx, f4b);
        auto vc = fetch_fixed_pair(t_idx, f4c);
        auto vbp = fetch_fixed_pair(t_idx, f4bp);
        auto vcp = fetch_fixed_pair(t_idx, f4cp);
        if (has_cp(cp_faces, f4a, f4bp, f4cp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vcp.first, vbp.first, va.first,
                                                vcp.second, vbp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
        if (has_cp(cp_faces, f4a, f4cp, f4b)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vb.first, vcp.first, va.first,
                                                vb.second, vcp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int + 1, (int)H) && valid(j_int + 1, (int)W)) {
        size_t f5a  = vid((int)t_idx,     i_int,     j_int,     sz);
        size_t f5b  = vid((int)t_idx - 1, i_int + 1, j_int,     sz);
        size_t f5c  = vid((int)t_idx - 1, i_int + 1, j_int + 1, sz);
        size_t f5bp = f5b + dv;
        size_t f5cp = f5c + dv;
        auto va = fetch_fixed_pair(t_idx, f5a);
        auto vb = fetch_fixed_pair(t_idx, f5b);
        auto vc = fetch_fixed_pair(t_idx, f5c);
        auto vbp = fetch_fixed_pair(t_idx, f5bp);
        auto vcp = fetch_fixed_pair(t_idx, f5cp);
        if (has_cp(cp_faces, f5a, f5bp, f5cp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vcp.first, vbp.first, va.first,
                                                vcp.second, vbp.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
        if (has_cp(cp_faces, f5a, f5c, f5bp)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vbp.first, vc.first, va.first,
                                                vbp.second, vc.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }

      if (valid(i_int + 1, (int)H) && valid(j_int + 1, (int)W)) {
        size_t f6a  = vid((int)t_idx,     i_int,     j_int,     sz);
        size_t f6b  = vid((int)t_idx,     i_int,     j_int + 1, sz);
        size_t f6c  = vid((int)t_idx - 1, i_int + 1, j_int + 1, sz);
        auto va = fetch_fixed_pair(t_idx, f6a);
        auto vb = fetch_fixed_pair(t_idx, f6b);
        auto vc = fetch_fixed_pair(t_idx, f6c);
        if (has_cp(cp_faces, f6a, f6b, f6c)) required_eb = 0;
        {
          T eb = derive_cp_abs_eb_sos_online<T>(vc.first, vb.first, va.first,
                                                vc.second, vb.second, va.second);
          if (eb < required_eb) required_eb = eb;
        }
      }
    }

    return required_eb;
  };

  for (size_t t = 0; t < Tt; ++t) {
    if (t % 100 == 0) {
      printf("processing slice %zu / %zu\n", t, Tt);
    }

#ifdef _OPENMP
    const size_t block_rows_eb = 8;
    const size_t block_cols_eb = 8;
    const size_t blocks_i_eb = (H + block_rows_eb - 1) / block_rows_eb;
    const size_t blocks_j_eb = (W + block_cols_eb - 1) / block_cols_eb;
    #pragma omp parallel for schedule(static)
    for (size_t block = 0; block < blocks_i_eb * blocks_j_eb; ++block) {
      size_t bi = block / blocks_j_eb;
      size_t bj = block % blocks_j_eb;
      size_t i_begin = bi * block_rows_eb;
      size_t i_end = std::min(i_begin + block_rows_eb, H);
      size_t j_begin = bj * block_cols_eb;
      size_t j_end = std::min(j_begin + block_cols_eb, W);
      for (size_t i = i_begin; i < i_end; ++i) {
        for (size_t j = j_begin; j < j_end; ++j) {
          size_t off = i * W + j;
          required_eb_layer[off] = compute_required_eb(t, i, j);
        }
      }
    }
#else
    for (size_t i = 0; i < H; ++i) {
      for (size_t j = 0; j < W; ++j) {
        size_t off = i * W + j;
        required_eb_layer[off] = compute_required_eb(t, i, j);
      }
    }
#endif

    for (size_t i = 0; i < H; ++i) {
      for (size_t j = 0; j < W; ++j) {
        size_t off = i * W + j;
        const T curU_val = u_curr[off];
        const T curV_val = v_curr[off];
        T required_eb = required_eb_layer[off];

        T abs_eb = required_eb;
        int id = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
        if (abs_eb == 0) {
          *(eb_pos++) = 0;
          unpred.push_back(float_curr_u[off]);
          unpred.push_back(float_curr_v[off]);
          dec_curr_u[off] = curU_val;
          dec_curr_v[off] = curV_val;
          continue;
        }

        *eb_pos = id;
        bool unpred_flag = false;
        T dec_val[2];
        T abs_err_fp_q[2] = {0,0};

        for (int comp = 0; comp < 2; ++comp) {
          const T curv = (comp == 0) ? curU_val : curV_val;
          const std::vector<T>& prev_buf = (comp == 0) ? dec_prev_u : dec_prev_v;
          const std::vector<T>& curr_buf = (comp == 0) ? dec_curr_u : dec_curr_v;
          auto sample_prev = [&](size_t row, size_t col)->T {
            return prev_buf[row * W + col];
          };
          auto sample_curr = [&](size_t row, size_t col)->T {
            return curr_buf[row * W + col];
          };

          T d0 = (t && i && j) ? sample_prev(i - 1, j - 1) : 0;
          T d1 = (t && i)     ? sample_prev(i - 1, j)     : 0;
          T d2 = (t && j)     ? sample_prev(i,     j - 1) : 0;
          T d3 = (t)          ? sample_prev(i,     j)     : 0;
          T d4 = (i && j)     ? sample_curr(i - 1, j - 1) : 0;
          T d5 = (i)          ? sample_curr(i - 1, j)     : 0;
          T d6 = (j)          ? sample_curr(i,     j - 1) : 0;
          T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

          T diff = curv - pred;
          T qd = (std::llabs(diff)/abs_eb) + 1;
          if (qd < capacity) {
            qd = (diff > 0) ? qd : -qd;
            int qindex = (int)(qd/2) + intv_radius;
            dq_pos[comp] = qindex;
            dec_val[comp] = pred + 2 * ( (T)qindex - (T)intv_radius ) * abs_eb;
            if (std::llabs(dec_val[comp] - curv) > abs_eb) {
              unpred_flag = true;
              break;
            }
            abs_err_fp_q[comp] = std::llabs(dec_val[comp] - curv);
          } else {
            unpred_flag = true;
            break;
          }
        }

        if (unpred_flag) {
          *(eb_pos++) = 0;
          unpred.push_back(float_curr_u[off]);
          unpred.push_back(float_curr_v[off]);
          dec_curr_u[off] = curU_val;
          dec_curr_v[off] = curV_val;
        } else {
          ++eb_pos;
          dq_pos += 2;
          dec_curr_u[off] = dec_val[0];
          dec_curr_v[off] = dec_val[1];
          double abs_eb_fp = (double)abs_eb / (double)scale;
          double err_u_fp  = (double)abs_err_fp_q[0] / (double)scale;
          double err_v_fp  = (double)abs_err_fp_q[1] / (double)scale;
          if (abs_eb_fp > enc_max_abs_eb_fp) enc_max_abs_eb_fp = abs_eb_fp;
          double real_err = std::max(err_u_fp, err_v_fp);
          if (real_err > enc_max_real_err_fp) enc_max_real_err_fp = real_err;
        }
      }
    }

    // rotate buffers for next timestep
    dec_prev_u.swap(dec_curr_u);
    dec_prev_v.swap(dec_curr_v);
    std::fill(dec_curr_u.begin(), dec_curr_u.end(), 0);
    std::fill(dec_curr_v.begin(), dec_curr_v.end(), 0);

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
        if (!load_layer_fixed(t + 2, u_next, v_next, &float_next_u, &float_next_v, false)) {
          std::free(eb_quant_index);
          std::free(data_quant_index);
          compressed_size = 0;
          return nullptr;
        }
      }
    }
  }

  unsigned char *compressed = (unsigned char*)std::malloc((size_t)(2 * H * W * Tt * sizeof(T)));
  if (!compressed) {
    std::free(eb_quant_index);
    std::free(data_quant_index);
    compressed_size = 0;
    return nullptr;
  }
  unsigned char *pos = compressed;

  write_variable_to_dst(pos, scale);
  std::cout << "write scale = " << (long long)scale << "\n";
  write_variable_to_dst(pos, base);
  write_variable_to_dst(pos, threshold);
  write_variable_to_dst(pos, intv_radius);

  size_t unpred_cnt = unpred.size();
  write_variable_to_dst(pos, unpred_cnt);
  if (unpred_cnt) {
    write_array_to_dst(pos, unpred.data(), unpred_cnt);
  }

  size_t eb_quant_num = (size_t)(eb_pos - eb_quant_index);
  write_variable_to_dst(pos, eb_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*1024, eb_quant_index, eb_quant_num, pos);
  std::free(eb_quant_index);

  size_t data_quant_num = (size_t)(dq_pos - data_quant_index);
  write_variable_to_dst(pos, data_quant_num);
  Huffman_encode_tree_and_data(/*state_num=*/2*capacity, data_quant_index, data_quant_num, pos);
  std::free(data_quant_index);

  compressed_size = (size_t)(pos - compressed);
  return compressed;
}

template<typename T_data, typename LayerFetcher>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_streaming(
    LayerFetcher&& fetch_layer,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  return sz_compress_cp_preserve_sos_2p5d_online_fp_streaming_impl<T_data>(
      std::forward<LayerFetcher>(fetch_layer), r1, r2, r3,
      compressed_size, max_pwr_eb, mode);
}

template<typename T_data>
unsigned char*
sz_compress_cp_preserve_sos_2p5d_online_fp_streaming(
    const T_data* U,
    const T_data* V,
    size_t r1, size_t r2, size_t r3,
    size_t& compressed_size,
    double max_pwr_eb,
    EbMode mode)
{
  const size_t H = r1, W = r2;
  const size_t layer_size = H * W;
  auto loader = [&](size_t t, T_data* dstU, T_data* dstV) {
    const size_t offset = t * layer_size;
    std::copy_n(U + offset, layer_size, dstU);
    std::copy_n(V + offset, layer_size, dstV);
    return true;
  };
  return sz_compress_cp_preserve_sos_2p5d_online_fp_streaming_impl<T_data>(
      loader, r1, r2, r3, compressed_size, max_pwr_eb, mode);
}
#endif

// ---------------- 解压主函数 ----------------
