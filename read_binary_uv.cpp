#include <ftk/ndarray.hh>
#include <ftk/filters/critical_point_tracker_2d_regular.hh>
#include "utils.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <u_file.bin> <v_file.bin> <W> <H> <T>" << std::endl;
        std::cerr << "Example: " << argv[0] << " u.bin v.bin 150 450 2001" << std::endl;
        return 1;
    }
    
    const char* u_file = argv[1];
    const char* v_file = argv[2];
    int W = std::atoi(argv[3]);  // 宽度
    int H = std::atoi(argv[4]);  // 高度  
    int T = std::atoi(argv[5]);  // 时间
    
    std::cout << "Reading binary files: " << u_file << " and " << v_file << std::endl;
    std::cout << "Target dimensions: (2, " << W << ", " << H << ", " << T << ")" << std::endl;
    
    // 使用您现有的readfile函数读取u和v分量
    size_t num_u = 0, num_v = 0;
    float* u_data = readfile<float>(u_file, num_u);
    float* v_data = readfile<float>(v_file, num_v);
    
    if (!u_data || !v_data) {
        std::cerr << "Error: Failed to read binary files" << std::endl;
        return 1;
    }
    
    std::cout << "Read " << num_u << " elements from u file" << std::endl;
    std::cout << "Read " << num_v << " elements from v file" << std::endl;
    
    // 验证数据大小
    size_t expected_size = static_cast<size_t>(W) * H * T;
    if (num_u != expected_size || num_v != expected_size) {
        std::cerr << "Error: Data size mismatch. Expected " << expected_size 
                  << " elements per component, got " << num_u << " and " << num_v << std::endl;
        free(u_data);
        free(v_data);
        return 1;
    }
    
    // 创建FTK ndarray，形状为(2, W, H, T)
    ftk::ndarray<float> vector_field;
    vector_field.reshape({2, static_cast<size_t>(W), static_cast<size_t>(H), static_cast<size_t>(T)});
    
    std::cout << "Created FTK ndarray with shape: ";
    vector_field.print_shape(std::cout);
    
    // 将u和v数据复制到vector_field数组
    // 假设输入数据是按照(T, H, W)的顺序存储的（numpy默认）
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                size_t input_idx = t * H * W + h * W + w;  // (T,H,W)顺序
                vector_field(0, w, h, t) = u_data[input_idx]; // u分量
                vector_field(1, w, h, t) = v_data[input_idx]; // v分量
            }
        }
    }
    
    std::cout << "Successfully populated FTK ndarray" << std::endl;
    
    // 设置时间属性
    if (T > 1) {
        vector_field.set_has_time(true);
        std::cout << "Set time dimension flag" << std::endl;
    }
    
    // 输出数据统计信息
    float min_u = u_data[0], max_u = u_data[0];
    float min_v = v_data[0], max_v = v_data[0];
    
    for (size_t i = 0; i < num_u; i++) {
        min_u = std::min(min_u, u_data[i]);
        max_u = std::max(max_u, u_data[i]);
        min_v = std::min(min_v, v_data[i]);
        max_v = std::max(max_v, v_data[i]);
    }
    
    std::cout << "\nData statistics:" << std::endl;
    std::cout << "U component: min=" << min_u << ", max=" << max_u << std::endl;
    std::cout << "V component: min=" << min_v << ", max=" << max_v << std::endl;
    
    // 输出几个示例值进行验证
    std::cout << "\nSample values:" << std::endl;
    std::cout << "U(0,0,0) = " << vector_field(0, 0, 0, 0) << std::endl;
    std::cout << "V(0,0,0) = " << vector_field(1, 0, 0, 0) << std::endl;
    if (T > 1) {
        std::cout << "U(0,0,1) = " << vector_field(0, 0, 0, 1) << std::endl;
        std::cout << "V(0,0,1) = " << vector_field(1, 0, 0, 1) << std::endl;
    }
    
    // 保存为FTK二进制格式
    vector_field.to_binary_file("vector_field_2wht.bin");
    std::cout << "\nSaved vector field to vector_field_2wht.bin" << std::endl;
    
    // 保存为VTK格式用于可视化
    vector_field.to_vtk_image_data_file("vector_field_2wht.vti");
    std::cout << "Saved vector field to vector_field_2wht.vti" << std::endl;
    
    // 示例：提取特定时间步的数据
    if (T > 1) {
        int example_time = std::min(10, T-1);
        std::cout << "\nExtracting time step " << example_time << ":" << std::endl;
        ftk::ndarray<double> time_slice = vector_field.slice_time(example_time);
        std::cout << "Time slice shape: ";
        time_slice.print_shape(std::cout);
    }
    
    // 如果需要进行critical point tracking
    if (T > 1) {
        std::cout << "\nSetting up critical point tracker..." << std::endl;
        diy::mpi::environment env;   
        diy::mpi::communicator comm;
        
        ftk::critical_point_tracker_2d_regular tracker(comm);
        tracker.set_domain(ftk::lattice({1, 1}, {W-2, H-2})); 
        tracker.set_array_domain(vector_field.get_lattice());
        tracker.set_scalar_field_source(ftk::SOURCE_NONE);
        tracker.set_jacobian_field_source(ftk::SOURCE_DERIVED);
        tracker.set_jacobian_symmetric(false);
        std::cout << "Critical point tracker initialized for domain " << W-2 << "x" << H-2 << std::endl;
        std::cout << "Ready for time series analysis" << std::endl;
    }
    
    // 清理内存
    free(u_data);
    free(v_data);
    
    std::cout << "\nProgram completed successfully!" << std::endl;
    std::cout << "FTK ndarray created with shape (2, " << W << ", " << H << ", " << T << ")" << std::endl;
    
    return 0;
}
