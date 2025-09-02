// mesh_index_stream.hpp
#include <atomic>
#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <ftk/numeric/critical_point_type.hh>
#include <ftk/numeric/critical_point_test.hh>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <ftk/numeric/clamp.hh>
#include <ftk/numeric/linear_interpolation.hh>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef BITMAP_ATOMIC
#define BITMAP_ATOMIC 0
#endif

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>


namespace m3d {

struct Size { int H, W, T; };
inline int vid(int i,int j,int t,const Size& sz){ return t*(sz.H*sz.W)+i*sz.W+j; }

enum class TriInCell : unsigned char { Upper=0, Lower=1 };

inline std::array<int,3> tri_vertices_2d(int i,int j,TriInCell w,int t,const Size& sz){
    int v00=vid(i,  j,  t,sz), v01=vid(i,  j+1,t,sz), v10=vid(i+1,j,  t,sz), v11=vid(i+1,j+1,t,sz);
    return (w==TriInCell::Upper) ? std::array<int,3>{v00,v01,v11}
                                 : std::array<int,3>{v00,v10,v11};
}

struct Tet { int v[4]; }; // 仅用于回调传参（不做全量存储）

// 固定 3-tet 剖分
inline std::array<Tet,3> prism_split_3tets(const std::array<int,3>& v_t,
                                           const std::array<int,3>& v_tp1){
    return {{
        Tet{ v_t[0],   v_t[1],   v_t[2],   v_tp1[2] },
        Tet{ v_t[0],   v_t[1],   v_tp1[1], v_tp1[2] },
        Tet{ v_t[0],   v_tp1[0], v_tp1[1], v_tp1[2] }
    }};
}

// 以 FaceKey(升序) 作为无向面键
struct FaceKey{
    std::array<int,3> v;
    FaceKey()=default;
    FaceKey(int a,int b,int c){ v={a,b,c}; std::sort(v.begin(),v.end()); }
    bool operator==(const FaceKey& o) const { return v==o.v; }
};
struct FaceKeyHash{
    size_t operator()(const FaceKey& k) const noexcept{
        // 简洁的 64bit 组合哈希
        size_t h=1469598103934665603ull;
        for(int x: k.v){ h ^= (size_t)x + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2); }
        return h;
    }
};

// 面邻接的固定 2 槽
struct FaceAdj2{
    int a=-1, b=-1;
    inline void add(int t){ if(a<0) a=t; else if(b<0) b=t; /*>2忽略或报警*/ }
    inline int size() const { return (a<0?0:(b<0?1:2)); }
};

// —— 计算片 [t, t+1] 的 face→tets（仅此片的 tet）——
using FaceMap = std::unordered_map<FaceKey, FaceAdj2, FaceKeyHash>;

inline void build_face_map_one_slab(const Size& sz, int t, FaceMap& fmap){
    const int dv = sz.H*sz.W;
    // 经验预估：每片约 3*triangles_per_layer 个 tet，唯一面 ~ 2×tets
    const size_t triangles_per_layer = size_t(2)*(sz.H-1)*(sz.W-1);
    const size_t tets_in_slab = 3*triangles_per_layer;
    fmap.clear();
    fmap.reserve(size_t(2.2*tets_in_slab)); // 留余量
    fmap.max_load_factor(0.7f);

    // 遍历本片所有棱柱并拆成3个tet；把每个tet的4个面塞进 fmap
    // 可并行（注意 unordered_map 需分线程局部再归并；此处提供单线程安全版本）
    for(int i=0;i<sz.H-1;++i){
        for(int j=0;j<sz.W-1;++j){
            for(auto w : {TriInCell::Upper, TriInCell::Lower}){
                auto v_t   = tri_vertices_2d(i,j,w,t,sz);
                std::array<int,3> v_tp1{ v_t[0]+dv, v_t[1]+dv, v_t[2]+dv };
                auto T = prism_split_3tets(v_t, v_tp1);
                for(const auto& K : T){
                    // 四个三角面（未排序→FaceKey内部排序）
                    FaceKey f0(K.v[0],K.v[1],K.v[2]);
                    FaceKey f1(K.v[0],K.v[1],K.v[3]);
                    FaceKey f2(K.v[0],K.v[2],K.v[3]);
                    FaceKey f3(K.v[1],K.v[2],K.v[3]);
                    int tid = -1; // 本示例不分配全局tid；只做面->片内tets编号可选
                    fmap[f0].add(tid); fmap[f1].add(tid);
                    fmap[f2].add(tid); fmap[f3].add(tid);
                }
            }
        }
    }
}

// —— 仅遍历（流式）：不给你存任何大容器 ——
// 回调签名：visit(i,j,t,w,k, Tet{...}, faces[4](FaceKey))
template<class Visitor>
inline void traverse_all_tets_stream(const Size& sz, Visitor&& visit){
    const int dv = sz.H*sz.W;
    for(int t=0; t<sz.T-1; ++t){
      #pragma omp parallel for collapse(2) if(sz.H*sz.W>20000)
        for(int i=0;i<sz.H-1;++i){
            for(int j=0;j<sz.W-1;++j){
                for(auto w : {TriInCell::Upper, TriInCell::Lower}){
                    auto v_t   = tri_vertices_2d(i,j,w,t,sz);
                    std::array<int,3> v_tp1{ v_t[0]+dv, v_t[1]+dv, v_t[2]+dv };
                    auto T = prism_split_3tets(v_t, v_tp1);
                    for(int k=0;k<3;++k){
                        const Tet& K = T[k];
                        std::array<FaceKey,4> faces = {
                            FaceKey(K.v[0],K.v[1],K.v[2]),
                            FaceKey(K.v[0],K.v[1],K.v[3]),
                            FaceKey(K.v[0],K.v[2],K.v[3]),
                            FaceKey(K.v[1],K.v[2],K.v[3])
                        };
                        visit(i,j,t,w,k,K,faces);
                    }
                }
            }
        }
    }
}

// —— 给定三点(面)→ 找相邻 tet（通过“片”决定查询范围）——
// 注意：此函数示意“如何选片”；真正查找需你在该片调用 build_face_map_one_slab 后用 fmap.count(key) 判断。
inline void which_slabs_for_face(const Size& sz, int a,int b,int c,
                                 std::vector<int>& slabs) {
    slabs.clear();
    auto layer = [&](int v){ return v / (sz.H*sz.W); };
    int ta=layer(a), tb=layer(b), tc=layer(c);
    int mn = std::min({ta,tb,tc});
    int mx = std::max({ta,tb,tc});
    if (mn==mx){
        // 同一时层：可能是 [t-1,t] 的顶面 或 [t,t+1] 的底面
        int t = mn;
        if (t>0)      slabs.push_back(t-1); // 查下片的顶面
        if (t<sz.T-1) slabs.push_back(t);   // 查上片的底面
    } else if (mx==mn+1){
        // 跨两层：一定属于片 [mn, mn+1]
        slabs.push_back(mn);
    } else {
        // 跳层（无效或异常）
    }
}


template<class FaceVisitor>
inline void traverse_unique_faces_across_slabs(const Size& sz, FaceVisitor&& face_visit) {
    const int dv = sz.H * sz.W;
    auto layer = [&](int v){ return v / (sz.H * sz.W); };

    // 对一个三棱柱的 3 个 tet 做局部去重，并按跨片规则筛面
    auto emit_unique_faces_of_prism = [&](int i,int j,int t,TriInCell w,
                                          const std::array<Tet,3>& T){
        // 局部 seen：存已发射过的“升序三元组”（最多 10 个）
        int seen_n = 0;
        int seen[12][3];

        auto try_emit = [&](int x,int y,int z){
            // 片内去重：按升序做无向键
            int a=x, b=y, c=z;
            if (a>b) std::swap(a,b);
            if (b>c) std::swap(b,c);
            if (a>b) std::swap(a,b);
            for (int q=0; q<seen_n; ++q)
                if (seen[q][0]==a && seen[q][1]==b && seen[q][2]==c) return; // 已处理

            // 跨片去重：同层面只在“所属片”发射
            int la = layer(x), lb = layer(y), lc = layer(z);
            bool same_layer = (la==lb && lb==lc);
            if (same_layer) {
                // 同层面只在它的“底面片”发射；最后一片再补顶面
                if      (la == t)        { /* 发射底面 */ }
                else if (la == t+1)      { if (t != sz.T-2) return; /* 仅最后一片发射顶面 */ }
                else                      return; // 异常层（不应出现）
            } else {
                // 侧面（跨 t 与 t+1），每片只出现一次，直接发射
            }

            // 记录为已发射，并回调
            seen[seen_n][0]=a; seen[seen_n][1]=b; seen[seen_n][2]=c; ++seen_n;
            face_visit(i,j,t,w, x,y,z);
        };

        for (const auto& K : T) {
            try_emit(K.v[0], K.v[1], K.v[2]);
            try_emit(K.v[0], K.v[1], K.v[3]);
            try_emit(K.v[0], K.v[2], K.v[3]);
            try_emit(K.v[1], K.v[2], K.v[3]);
        }
    };

    for (int t=0; t<sz.T-1; ++t){
      #pragma omp parallel for collapse(2) if(sz.H*sz.W>20000)
        for (int i=0; i<sz.H-1; ++i){
            for (int j=0; j<sz.W-1; ++j){
                for (auto w : {TriInCell::Upper, TriInCell::Lower}){
                    auto v_t   = tri_vertices_2d(i,j,w,t,sz);
                    std::array<int,3> v_tp1{ v_t[0]+dv, v_t[1]+dv, v_t[2]+dv };
                    auto T = prism_split_3tets(v_t, v_tp1);
                    emit_unique_faces_of_prism(i,j,t,w, T);
                }
            }
        }
    }
}

inline void print_omp_runtime_info(const char* tag = "OMP") {
#ifdef _OPENMP
  // _OPENMP 是形如 yyyymm 的整数，例如 202011 表示 OpenMP 5.0 (Nov 2020)
  std::cerr << "[" << tag << "] compiled with OpenMP: _OPENMP=" << _OPENMP << "\n";
  std::cerr << "[" << tag << "] omp_get_num_procs()   = " << omp_get_num_procs()   << " (可用逻辑处理器)\n";
  std::cerr << "[" << tag << "] omp_get_max_threads() = " << omp_get_max_threads() << " (默认最大并行线程)\n";
  if (const char* s = std::getenv("OMP_NUM_THREADS"))
    std::cerr << "[" << tag << "] OMP_NUM_THREADS      = " << s << "\n";
  if (const char* s = std::getenv("OMP_PROC_BIND"))
    std::cerr << "[" << tag << "] OMP_PROC_BIND        = " << s << "\n";
  if (const char* s = std::getenv("OMP_PLACES"))
    std::cerr << "[" << tag << "] OMP_PLACES           = " << s << "\n";

  // 启一个并行区，测实际能拿到多少线程
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      std::cerr << "[" << tag << "] parallel region threads = " << omp_get_num_threads()
                << " (inside #pragma omp parallel)\n";
    }
  }
#else
  std::cerr << "[" << tag << "] OpenMP DISABLED (未使用 -fopenmp 编译)\n";
#endif
}

} // namespace m3d


// main.cpp
// #include "mesh_index_stream.hpp"
#include <iostream>
#include <vector>
#include <array>
#include <unordered_set>
#include <fstream>
#include <chrono>
#include <cmath>

using namespace m3d;

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


// 小工具：给定一个 tet，返回其四个面（FaceKey）
static inline std::array<FaceKey,4> faces_of_tet_local(const Tet& K) {
    return {
        FaceKey(K.v[0], K.v[1], K.v[2]),
        FaceKey(K.v[0], K.v[1], K.v[3]),
        FaceKey(K.v[0], K.v[2], K.v[3]),
        FaceKey(K.v[1], K.v[2], K.v[3])
    };
}

// 功能2：给定三点(面)，查找相邻四面体（最多2个）。
// 做法：根据三点所在时间层定位需检查的 slab（最多2个），仅扫描这些 slab。
static std::vector<Tet> find_adjacent_tets_by_face(const Size& sz, int a, int b, int c) {
    FaceKey target(a,b,c);
    std::vector<int> slabs;
    which_slabs_for_face(sz, a,b,c, slabs);

    std::vector<Tet> owners; owners.reserve(2);
    const int dv = sz.H * sz.W;

    for (int t : slabs) {
        for (int i = 0; i < sz.H-1; ++i) {
            for (int j = 0; j < sz.W-1; ++j) {
                for (auto w : {TriInCell::Upper, TriInCell::Lower}) {
                    auto v_t   = tri_vertices_2d(i,j,w,t,sz);
                    std::array<int,3> v_tp1{ v_t[0]+dv, v_t[1]+dv, v_t[2]+dv };
                    auto T = prism_split_3tets(v_t, v_tp1);
                    for (const auto& K : T) {
                        auto F = faces_of_tet_local(K);
                        for (const auto& fk : F) {
                            if (fk == target) {
                                owners.push_back(K);
                                break;
                            }
                        }
                        if (owners.size() == 2) break; // 内部面最多两个tet
                    }
                    if (owners.size() == 2) break;
                }
                if (owners.size() == 2) break;
            }
            if (owners.size() == 2) break;
        }
        if (owners.size() == 2) break;
    }
    return owners; // 0（未命中）/1（边界）/2（内部）
}

// ========== 简单的二进制读取工具 ==========
static bool read_floats(const std::string& path, std::vector<float>& out, size_t expected_count) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        std::cerr << "[ERR] Cannot open file: " << path << "\n";
        return false;
    }
    fin.seekg(0, std::ios::end);
    std::streamsize bytes = fin.tellg();
    fin.seekg(0, std::ios::beg);
    if (bytes % sizeof(float) != 0) {
        std::cerr << "[ERR] File size not multiple of 4 bytes: " << path << "\n";
        return false;
    }
    size_t n = static_cast<size_t>(bytes / sizeof(float));
    if (expected_count != 0 && n != expected_count) {
        std::cerr << "[ERR] Count mismatch. File " << path << " has " << n
                  << " floats, expected " << expected_count << ".\n";
        return false;
    }
    out.resize(n);
    if (!fin.read(reinterpret_cast<char*>(out.data()), bytes)) {
        std::cerr << "[ERR] Failed to read file: " << path << "\n";
        return false;
    }
    return true;
}

// ========== 便捷索引（可根据实际文件打平顺序调整）==========
// 假设 linear id = t*(H*W) + i*W + j
static inline size_t index_of(int i,int j,int t,const Size& sz) {
    return static_cast<size_t>(t) * (sz.H * sz.W) + static_cast<size_t>(i) * sz.W + j;
}

// 返回一个四面体的 4 个三角面（未排序，保持“自然面”顶点次序）
static inline std::array<std::array<int,3>,4> raw_faces_of_tet(const Tet& K) {
    return {
        std::array<int,3>{K.v[0], K.v[1], K.v[2]},
        std::array<int,3>{K.v[0], K.v[1], K.v[3]},
        std::array<int,3>{K.v[0], K.v[2], K.v[3]},
        std::array<int,3>{K.v[1], K.v[2], K.v[3]}
    };
}



// —— 新增：顶点位图（1bit/vertex），支持并发置位 ——
// N = H*W*T 位（约 135,067,500 bit ≈ 16.1 MB）
// 顶部：可选开关。并行写入同一位图时设为 1；单线程设为 0（默认）


struct VertexBitmap {
#if BITMAP_ATOMIC
    std::vector<std::atomic<uint64_t>> words;
#else
    std::vector<uint64_t> words;
#endif
    size_t nbits = 0;

    void init(size_t nbits_) {
        nbits = nbits_;
        size_t nw = (nbits + 63) >> 6;
        words.resize(nw);
#if BITMAP_ATOMIC
        for (size_t i=0; i<nw; ++i) words[i].store(0, std::memory_order_relaxed);
#else
        std::fill(words.begin(), words.end(), 0ull);
#endif
    }

    inline void set(size_t idx) {
        size_t w = idx >> 6;
        uint64_t mask = 1ull << (idx & 63);
#if BITMAP_ATOMIC
        words[w].fetch_or(mask, std::memory_order_relaxed);
#else
        words[w] |= mask; // 非原子，单线程下最快
#endif
    }

    inline bool get(size_t idx) const {
        size_t w = idx >> 6;
        uint64_t mask = 1ull << (idx & 63);
#if BITMAP_ATOMIC
        uint64_t v = words[w].load(std::memory_order_relaxed);
#else
        uint64_t v = words[w];
#endif
        return (v & mask) != 0;
    }

    size_t count_ones() const {
        size_t c=0;
        for (size_t i=0;i<words.size();++i) {
#if BITMAP_ATOMIC
            uint64_t v = words[i].load(std::memory_order_relaxed);
#else
            uint64_t v = words[i];
#endif
#if __cpp_lib_bitops >= 201907L
            c += std::popcount(v);
#else
            c += (size_t)__builtin_popcountll(v);
#endif
        }
        return c;
    }
};

#pragma pack(push,1)
struct UVPackHeader {
    char     magic[8];     // "UVPACK\0"
    uint32_t version;      // 1
    uint32_t W, H, T;      // 网格维度
    uint64_t Ntotal;       // W*H*T
    uint64_t Mmarked;      // 被标记的顶点个数
    uint8_t  value_kind;   // 0=f32(U/V)，1=i64_fixed(U_fp/V_fp)
    uint8_t  reserved8[7]; // 对齐/扩展
    int64_t  scale;        // 定点缩放因子（value_kind==1 有效）
    int64_t  reserved64[3];// 预留
};
#pragma pack(pop)

// “内存包”——仅驻留内存，不写文件
struct UVPackMem {
    UVPackHeader hdr{};
    std::vector<uint32_t> indices;     // 升序唯一顶点ID
    // 二选一：根据 hdr.value_kind 使用
    std::vector<float>    Uvals, Vvals;      // value_kind==0
    std::vector<int64_t>  Ufixed, Vfixed;    // value_kind==1
};

// 从位图提取升序唯一索引（沿用您现有 VertexBitmap）
static std::vector<uint32_t> build_marked_indices(const VertexBitmap& bmp) {
    const size_t nbits = bmp.nbits;
    const size_t nw    = (nbits + 63) >> 6;
    std::vector<uint32_t> out;
    out.reserve(bmp.count_ones());
    for (size_t w = 0; w < nw; ++w) {
        uint64_t word =
#if defined(BITMAP_ATOMIC) && BITMAP_ATOMIC
            bmp.words[w].load(std::memory_order_relaxed);
#else
            bmp.words[w];
#endif
        while (word) {
#if defined(_MSC_VER)
            unsigned long tz; _BitScanForward64(&tz, word);
            uint32_t bit = (uint32_t)tz;
#else
            uint32_t bit = (uint32_t)__builtin_ctzll(word);
#endif
            out.push_back((uint32_t)((w << 6) + bit));
            word &= (word - 1); // 清最低位1
        }
    }
    return out;
}

// 在内存中“打包”标记顶点的U/V（float或定点）——不写盘
static UVPackMem build_uv_pack_in_memory(
        uint32_t W, uint32_t H, uint32_t T,
        const VertexBitmap& bmp,
        const std::vector<uint32_t>& marked_idx,           // 可传 build_marked_indices 的结果
        const std::vector<float>& U, const std::vector<float>& V,
        bool use_fixed_point,                              // true: 使用 U_fp/V_fp
        const std::vector<int64_t>* U_fp,                  // 若 use_fixed_point=true，必须提供
        const std::vector<int64_t>* V_fp,
        int64_t scale_if_fixed)                            // 定点缩放因子
{
    UVPackMem pack;
    std::memset(&pack.hdr, 0, sizeof(pack.hdr));
    std::memcpy(pack.hdr.magic, "UVPACK\0", 8);
    pack.hdr.version   = 1;
    pack.hdr.W         = W; pack.hdr.H = H; pack.hdr.T = T;
    pack.hdr.Ntotal    = (uint64_t)W * H * T;
    pack.hdr.Mmarked   = marked_idx.size();
    pack.hdr.value_kind= use_fixed_point ? 1 : 0;
    pack.hdr.scale     = use_fixed_point ? scale_if_fixed : 0;

    pack.indices = marked_idx; // 拷贝索引

    if (!use_fixed_point) {
        pack.Uvals.resize(marked_idx.size());
        pack.Vvals.resize(marked_idx.size());
        for (size_t k=0; k<marked_idx.size(); ++k) {
            uint32_t id = marked_idx[k];
            pack.Uvals[k] = U[id];
            pack.Vvals[k] = V[id];
        }
    } else {
        if (!U_fp || !V_fp) {
            std::cerr << "[ERR] build_uv_pack_in_memory: missing fixed-point arrays.\n";
            return pack;
        }
        pack.Ufixed.resize(marked_idx.size());
        pack.Vfixed.resize(marked_idx.size());
        for (size_t k=0; k<marked_idx.size(); ++k) {
            uint32_t id = marked_idx[k];
            pack.Ufixed[k] = (*U_fp)[id];
            pack.Vfixed[k] = (*V_fp)[id];
        }
    }
    return pack;
}

// 从“内存包”回填到 U/V 数组（用于解压后替换回去）
static bool apply_marked_uv_from_memory(
        const UVPackMem& pack,
        std::vector<float>& U, std::vector<float>& V)
{
    const uint64_t Nexpect = (uint64_t)U.size();
    if (pack.hdr.Ntotal != Nexpect || pack.hdr.Ntotal != V.size()) {
        std::cerr << "[ERR] apply_from_memory: size mismatch.\n";
        return false;
    }
    if (pack.hdr.value_kind == 0) {
        if (pack.Uvals.size() != pack.indices.size() || pack.Vvals.size() != pack.indices.size()) {
            std::cerr << "[ERR] apply_from_memory: payload size mismatch (float).\n";
            return false;
        }
        for (size_t k=0; k<pack.indices.size(); ++k) {
            uint32_t id = pack.indices[k];
            U[id] = pack.Uvals[k];
            V[id] = pack.Vvals[k];
        }
    } else if (pack.hdr.value_kind == 1) {
        if (pack.Ufixed.size() != pack.indices.size() || pack.Vfixed.size() != pack.indices.size()) {
            std::cerr << "[ERR] apply_from_memory: payload size mismatch (fixed).\n";
            return false;
        }
        const double inv = (pack.hdr.scale != 0 ? 1.0/double(pack.hdr.scale) : 1.0);
        for (size_t k=0; k<pack.indices.size(); ++k) {
            uint32_t id = pack.indices[k];
            U[id] = float(double(pack.Ufixed[k]) * inv);
            V[id] = float(double(pack.Vfixed[k]) * inv);
        }
    } else {
        std::cerr << "[ERR] apply_from_memory: unknown value_kind.\n";
        return false;
    }
    return true;
}


int main(int argc, char** argv) {
    print_omp_runtime_info("BOOT");
    // ---- 配置区 ----
    // 尺寸：W=450, H=150, T=2000
    Size sz{ .H = 150, .W = 450, .T = 2001 };

    // 文件路径：可从命令行传入；否则用默认名
    std::string u_path = (argc > 1 ? argv[1] : "u.bin");
    std::string v_path = (argc > 2 ? argv[2] : "v.bin");

    // 打印的最大面数（避免海量输出）；需要全部打印可改成 very large
    const size_t MAX_PRINT = 50;  // <- 修改这里

    // ---- 读取数据 ----
    const size_t N = static_cast<size_t>(sz.H) * sz.W * sz.T;
    std::vector<float> U, V;
    if (!read_floats(u_path, U, N)) return 1;
    if (!read_floats(v_path, V, N)) return 1;
    std::vector<int64_t> U_fp(N), V_fp(N);
    int64_t range = 0;
    int64_t vector_field_scaling_factor = convert_to_fixed_point(U.data(), V.data(), N, U_fp.data(), V_fp.data(), range);

    std::cout << "Loaded U and V: count=" << N << " floats each.\n";
    std::cout << "Grid: W="<<sz.W<<", H="<<sz.H<<", T="<<sz.T<<"\n\n";

    // —— 位图初始化 —— 
    VertexBitmap vbmp;
    vbmp.init(N);

    // ---- 遍历所有四面体 → 遍历其 4 个面，并打印每个面的三顶点的 (u,v) ----
    // 说明：这是“逐面打印”。若担心重复（共面会在两个 tet 都出现），
    // 可改为“分片构建面哈希取 unique face 再打印”。此处直接按需求逐面输出。
    size_t printed = 0, total_faces = 0, total_tets = 0, face_contain_cp = 0, total_unique_faces = 0;

    //add timing
    auto start = std::chrono::high_resolution_clock::now();

    #if 0
    //TODO: 相邻的两个四面体会有一个面重复计算了一次
    traverse_all_tets_stream(sz,
        [&](int i, int j, int t, TriInCell which, int k, const Tet& K,
            const std::array<FaceKey,4>& /*faces_sorted*/)
        {
            ++total_tets;
            auto faces_raw = raw_faces_of_tet(K);
            for (int f = 0; f < 4; ++f) {
                ++total_faces;
                // if (printed < MAX_PRINT) {
                    const auto& tri = faces_raw[f];
                    int a = tri[0], b = tri[1], c = tri[2];

                    // 取出 (i,j,t) 坐标（仅用于演示可读性；实际上直接按 id 取值即可）
                    auto fetch_uv = [&](int vid)->std::pair<float,float> {
                        // 由顶点 id 反推出 (t,i,j) 仅用于 sanity check / 打印；不需要也可省略
                        // 反解：t = vid / (H*W), r = vid % (H*W), i = r / W, j = r % W
                        // 注意：这是“反演”，仅用于断言，可注释以提速。
                        // const int layer = vid / (sz.H * sz.W);
                        // const int rem   = vid % (sz.H * sz.W);
                        // const int ii    = rem / sz.W;
                        // const int jj    = rem % sz.W;
                        // assert(index_of(ii,jj,layer,sz) == static_cast<size_t>(vid));
                        float u = U[static_cast<size_t>(vid)];
                        float v = V[static_cast<size_t>(vid)];
                        return {u,v};
                    };

                    auto fetch_uv_fp = [&](int vid)->std::pair<int64_t,int64_t> { //fix point
                        int64_t u_fp = U_fp[static_cast<size_t>(vid)];
                        int64_t v_fp = V_fp[static_cast<size_t>(vid)];
                        return {u_fp, v_fp};
                    };

                    auto [ua_fp,va_fp] = fetch_uv_fp(a);
                    auto [ub_fp,vb_fp] = fetch_uv_fp(b);
                    auto [uc_fp,vc_fp] = fetch_uv_fp(c);

                    auto [ua,va] = fetch_uv(a);
                    auto [ub,vb] = fetch_uv(b);
                    auto [uc,vc] = fetch_uv(c);

                    //move the value of ua,ub,uc and va,vb,vc to vf[3][2]
                    int64_t vf[3][2];
                    vf[0][0] = ua_fp; vf[0][1] = va_fp;
                    vf[1][0] = ub_fp; vf[1][1] = vb_fp;
                    vf[2][0] = uc_fp; vf[2][1] = vc_fp;

                    double v[3][2]; //convert float to double
                    v[0][0] = static_cast<double>(ua); v[0][1] = static_cast<double>(va);
                    v[1][0] = static_cast<double>(ub); v[1][1] = static_cast<double>(vb);
                    v[2][0] = static_cast<double>(uc); v[2][1] = static_cast<double>(vc);

                    // std::cout << "Face vertices = {"<<a<<","<<b<<","<<c<<"}\n"
                    //           << "  U = {"<<ua<<","<<ub<<","<<uc<<"}\n"
                    //           << "  V = {"<<va<<","<<vb<<","<<vc<<"}\n";
                    // std::cout << "  U_fp = {"<<ua_fp<<","<<ub_fp<<","<<uc_fp<<"}\n"
                    //           << "  V_fp = {"<<va_fp<<","<<vb_fp<<","<<vc_fp<<"}\n";
                    // ++printed;

                    //计算cp-sos
                    // robust critical point test
                    int indices[3];
                    indices[0] = a; indices[1] = b; indices[2] = c;
                    // std::cout << "  indices = {"<<indices[0]<<","<<indices[1]<<","<<indices[2]<<"}\n";
                    // std::cout << "  vf = {"<<vf[0][0]<<","<<vf[0][1]<<","<<vf[1][0]<<","<<vf[1][1]<<","<<vf[2][0]<<","<<vf[2][1]<<"}\n";
                    bool succ = ftk::robust_critical_point_in_simplex2(vf, indices);
                    if (succ) {
                        std::cout << "  robust_critical_point_in_simplex2 = true indices = {" << indices[0] << "," << indices[1] << "," << indices[2] << "}\n";
                    }
                    if (!succ) return false;

                    double mu[3]; // check intersection
                    double cond;
                    bool succ2 = ftk::inverse_lerp_s2v2(v, mu, &cond);
                    // if (!succ2) return false;
                    // if (std::isnan(mu[0]) || std::isnan(mu[1]) || std::isnan(mu[2])) return false;
                    // fprintf(stderr, "mu=%f, %f, %f\n", mu[0], mu[1], mu[2]);

                    if (!succ2){
                        ftk::clamp_barycentric<3>(mu);
                    }
                    double X[3][4], x[4]; // position
                    for (int i = 0; i < 3; ++i) {
                        // X[i][0] => x, X[i][1] => y, X[i][2] => z(zero), X[i][3] => t
                        //X[i][0] : p mod W
                        //X[i][1] : p / W mod H
                        X[i][0] = static_cast<double>(indices[i] % sz.W);
                        X[i][1] = static_cast<double>((indices[i] / sz.W) % sz.H);
                        X[i][2] = 0.0; // 2D plane, z
                        X[i][3] = static_cast<double>(indices[i] / (sz.W * sz.H));
                    }
                    //print X
                    for (int i = 0; i < 3; ++i) {
                        std::cout << "X[" << i << "] = {";
                        for (int j = 0; j < 4; ++j) {
                            std::cout << X[i][j];
                            if (j < 3) std::cout << ", ";
                        }
                        std::cout << "}\n";
                    }
                    ftk::lerp_s2v4(X, mu, x);
                    //print x coord
                    std::cout << "x[0] = " << x[0] << "\n";
                    std::cout << "x[1] = " << x[1] << "\n";
                    std::cout << "x[2] = " << x[2] << "\n";
                    std::cout << "x[t] = " << x[3] << "\n";
                    face_contain_cp++;
                // }
            }
        });
    std::cout << "\nTotal tets traversed : " << total_tets  << "\n";
    std::cout << "Total faces visited   : " << total_faces << " (4 per tet)\n";
    std::cout << "Printed faces         : " << printed     << " (capped by MAX_PRINT="<<MAX_PRINT<<")\n";
    std::cout << "Faces containing CP   : " << face_contain_cp << "\n";
    #endif

    auto fetch_uv     = [&](int vid)->std::pair<float,float>   { return { U[vid],    V[vid]    }; };
    auto fetch_uv_fp  = [&](int vid)->std::pair<int64_t,int64_t>{ return { U_fp[vid], V_fp[vid] }; };

    auto decode = [&](int p, double &X0,double &X1,double &X3){
        X0 = (double)(p % sz.W);
        X1 = (double)((p / sz.W) % sz.H);
        X3 = (double)(p / (sz.W * sz.H));
    };

    traverse_unique_faces_across_slabs(sz,[&](int i, int j, int t, TriInCell which, int a, int b, int c)
    {
        ++total_unique_faces;

        // 取 U/V（float 与定点）
        auto [ua,va]     = fetch_uv(a);
        auto [ub,vb]     = fetch_uv(b);
        auto [uc,vc]     = fetch_uv(c);
        auto [ua_fp,va_fp] = fetch_uv_fp(a);
        auto [ub_fp,vb_fp] = fetch_uv_fp(b);
        auto [uc_fp,vc_fp] = fetch_uv_fp(c);

        int64_t vf[3][2] = { {ua_fp,va_fp}, {ub_fp,vb_fp}, {uc_fp,vc_fp} };
        double  v3[3][2] = { {(double)ua,(double)va}, {(double)ub,(double)vb}, {(double)uc,(double)vc} };

        // robust-cp判定
        int indices[3] = { a,b,c };
        bool succ = ftk::robust_critical_point_in_simplex2(vf, indices);
        if (!succ) return;  // 本面不含 CP，直接跳过

        // —— 命中：位图置位（线程安全，原子 OR） ——
        vbmp.set((size_t)a);
        vbmp.set((size_t)b);
        vbmp.set((size_t)c);
        ++face_contain_cp;

        // 反插值与位置
        double mu[3], cond;
        bool succ2 = ftk::inverse_lerp_s2v2(v3, mu, &cond);
        if (!succ2) ftk::clamp_barycentric<3>(mu);

        double X[3][4], x[4];
        decode(a, X[0][0], X[0][1], X[0][3]); X[0][2]=0.0;
        decode(b, X[1][0], X[1][1], X[1][3]); X[1][2]=0.0;
        decode(c, X[2][0], X[2][1], X[2][3]); X[2][2]=0.0;
        ftk::lerp_s2v4(X, mu, x);
        std::cout << "CP on face {"<<a<<","<<b<<","<<c<<"}, x=("
        << x[0] << "," << x[1] << "," << x[2] << ", t=" << x[3] << ")\n";
        }
    );
    std::cout << "\nTotal unique faces visited : " << total_unique_faces << "\n";
    std::cout << "Faces containing CP        : " << face_contain_cp    << "\n";

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "traverse Time taken: " << duration.count() << " seconds.\n";
    //print out number of 1 in bitmap
    std::cout << "Number of 1s in bitmap: " << vbmp.count_ones() << "\n";

    // 从位图拿到所有被置位的顶点索引（升序唯一）
    auto marked_idx = build_marked_indices(vbmp);
    std::cout << "Marked vertices in bitmap = " << marked_idx.size() << "\n";

    // 构建内存包（两种模式：float 或 定点）
    bool use_fixed_point = true; // fp or float
    UVPackMem pack = build_uv_pack_in_memory(
        /*W*/ sz.W, /*H*/ sz.H, /*T*/ sz.T,
        /*bmp*/ vbmp,
        /*marked_idx*/ marked_idx,
        /*U,V*/ U, V,
        /*use_fixed_point*/ use_fixed_point,
        /*U_fp,V_fp*/ use_fixed_point ? &U_fp : nullptr,
        /*V_fp*/     use_fixed_point ? &V_fp : nullptr,
        /*scale*/    use_fixed_point ? vector_field_scaling_factor : 0
    );
    printf("number of lossless = %llu\n", use_fixed_point ? pack.Ufixed.size() : pack.Uvals.size());

    // start = std::chrono::high_resolution_clock::now();
    // // —— 准备一个示例面：取 (i=100,j=111,t=222,Upper) 的第一个tet的第0个面 —— 
    // const int dv = sz.H * sz.W;
    // auto v_t   = tri_vertices_2d(/*i=*/100, /*j=*/111, TriInCell::Upper, /*t=*/222, sz);
    // std::array<int,3> v_tp1{ v_t[0]+dv, v_t[1]+dv, v_t[2]+dv };
    // auto T = prism_split_3tets(v_t, v_tp1);
    // Tet sampleK = T[0];
    // auto sampleFaces = faces_of_tet_local(sampleK);
    // auto f0 = sampleFaces[0].v; // (a,b,c)

    // // —— 功能2：给定三点(面)→查相邻tet —— 
    // std::cout << "[2] Query adjacent tets of face("
    //           << f0[0] << "," << f0[1] << "," << f0[2] << ")...\n";
    // auto adj = find_adjacent_tets_by_face(sz, f0[0], f0[1], f0[2]);
    // if (adj.empty()) {
    //     std::cout << "  No owner tet found (face not present).\n";
    // } else {
    //     std::cout << "  Owners (" << adj.size() << "):\n";
    //     for (const auto& K : adj) {
    //         std::cout << "    {"<<K.v[0]<<","<<K.v[1]<<","<<K.v[2]<<","<<K.v[3]<<"}\n";
    //     }
    // }
    // std::cout << "\n";

    // // —— 功能3：给定四点(四面体)→列出其四个面 —— 
    // std::cout << "[3] Faces of tet {"<<sampleK.v[0]<<","<<sampleK.v[1]
    //           << ","<<sampleK.v[2]<<","<<sampleK.v[3]<<"}:\n";
    // for (const auto& fk : sampleFaces) {
    //     auto v = fk.v;
    //     std::cout << "  ("<<v[0]<<","<<v[1]<<","<<v[2]<<")\n";
    // }

    // std::cout << "\nDone.\n";
    
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    // std::cout << "Query Time taken: " << duration.count() << " seconds.\n";

    // // ---- 示例：给定一个“面”的三顶点 id，快速拿到它的 (u,v) 三元组并打印 ----
    // // 这里演示用第一个打印过的面的三顶点（若未打印则跳过）
    // if (printed > 0) {
    //     // 为了示例，我们在上面打印循环中无法直接保留 tri；这里再构造一个面来演示：
    //     // 选取 t=0, i=0, j=0, which=Upper 的第一个 tet 的第0个面
    //     const int dv = sz.H * sz.W;
    //     auto v_t   = tri_vertices_2d(0,0,TriInCell::Upper,0,sz);
    //     std::array<int,3> v_tp1{ v_t[0]+dv, v_t[1]+dv, v_t[2]+dv };
    //     auto T = prism_split_3tets(v_t, v_tp1);
    //     auto tri = raw_faces_of_tet(T[0])[0];
    //     int a=tri[0], b=tri[1], c=tri[2];

    //     auto uv = [&](int vid){ return std::pair<float,float>(U[vid], V[vid]); };
    //     auto [ua,va] = uv(a);
    //     auto [ub,vb] = uv(b);
    //     auto [uc,vc] = uv(c);

    //     std::cout << "\n[Demo] Query one face again: {"<<a<<","<<b<<","<<c<<"}\n"
    //               << "  U = {"<<ua<<","<<ub<<","<<uc<<"}\n"
    //               << "  V = {"<<va<<","<<vb<<","<<vc<<"}\n";
    // }

    return 0;
}


// int main(int argc, char** argv) {
//     // 为演示起见使用较小网格。真实大网格(如150×450×2000)请直接替换数值；
//     // 代码会以“流式分片”遍历，内存占用可控。
//     Size sz{ .H = 150, .W = 450, .T = 1000 };

//     // —— 功能1：遍历所有四面体及其四个面（流式，不存全量）——
//     std::cout << "[1] Streaming traverse all tets (printing first few)...\n";
//     std::size_t total_tets = 0, printed = 0;
//     traverse_all_tets_stream(sz,
//         [&](int i, int j, int t, TriInCell w, int k, const Tet& K, const std::array<FaceKey,4>& faces){
//             ++total_tets;
//             if (printed < 6) {
//                 std::cout << "  Tet(i="<<i<<", j="<<j<<", t="<<t
//                           << ", which="<<(w==TriInCell::Upper?"U":"L")
//                           << ", k="<<k<<") = {"
//                           << K.v[0]<<","<<K.v[1]<<","<<K.v[2]<<","<<K.v[3]<<"}\n";
//                 for (int f=0; f<4; ++f) {
//                     auto v = faces[f].v;
//                     std::cout << "    face"<<f<<"=("<<v[0]<<","<<v[1]<<","<<v[2]<<")\n";
//                 }
//                 ++printed;
//             }
//         });
//     std::cout << "  Total tets traversed = " << total_tets << "\n\n";

//     // —— 准备一个示例面：取 (i=0,j=0,t=0,Upper) 的第一个tet的第0个面 —— 
//     const int dv = sz.H * sz.W;
//     auto v_t   = tri_vertices_2d(/*i=*/0, /*j=*/0, TriInCell::Upper, /*t=*/0, sz);
//     std::array<int,3> v_tp1{ v_t[0]+dv, v_t[1]+dv, v_t[2]+dv };
//     auto T = prism_split_3tets(v_t, v_tp1);
//     Tet sampleK = T[0];
//     auto sampleFaces = faces_of_tet_local(sampleK);
//     auto f0 = sampleFaces[0].v; // (a,b,c)

//     // —— 功能2：给定三点(面)→查相邻tet —— 
//     std::cout << "[2] Query adjacent tets of face("
//               << f0[0] << "," << f0[1] << "," << f0[2] << ")...\n";
//     auto adj = find_adjacent_tets_by_face(sz, f0[0], f0[1], f0[2]);
//     if (adj.empty()) {
//         std::cout << "  No owner tet found (face not present).\n";
//     } else {
//         std::cout << "  Owners (" << adj.size() << "):\n";
//         for (const auto& K : adj) {
//             std::cout << "    {"<<K.v[0]<<","<<K.v[1]<<","<<K.v[2]<<","<<K.v[3]<<"}\n";
//         }
//     }
//     std::cout << "\n";

//     // —— 功能3：给定四点(四面体)→列出其四个面 —— 
//     std::cout << "[3] Faces of tet {"<<sampleK.v[0]<<","<<sampleK.v[1]
//               << ","<<sampleK.v[2]<<","<<sampleK.v[3]<<"}:\n";
//     for (const auto& fk : sampleFaces) {
//         auto v = fk.v;
//         std::cout << "  ("<<v[0]<<","<<v[1]<<","<<v[2]<<")\n";
//     }

//     std::cout << "\nDone.\n";
//     return 0;
// }
