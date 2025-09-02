// #include <ftk/ndarray.hh>
// #include <ftk/filters/critical_point_tracker_2d_regular.hh>
  
// // This code reads Cylinder2D.am, writes the vector field data into 
// // vtkImageData (.vti) file for visualization, tracks critical points 
// // in the data, and writes the critical point trackign results into 
// // the designated vtkPolyData (.vtp) file.
// //
// // Information and the Cylinder2D.am data file may be found in the 
// // following web page: https://www.csc.kth.se/~weinkauf/notes/cylinder2d.html
// // Follow the copyright information in the web page; Restrictions may apply.  

// int main(int argc, char **argv)
// {
//   // The following section initializes MPI through DIY.  The code should 
//   // build and run without MPI.
//   diy::mpi::environment env;   
//   diy::mpi::communicator comm;

//   // Print help information
//   if (argc < 4) {
//     fprintf(stderr, "Usage: %s <path_to_cylinder_2d_am> <output.vti> <output.vtp>\n", argv[0]);
//     return 1;
//   }

//   // The entire spacetime volume (400x50x1001) will be read into the array
//   // Note that the dimension of the array is 2x400x500x1001, with the first
//   // dimension x- and y-components of the time-varying vector field
//   ftk::ndarray<float> array;
//   array.read_amira(argv[1]);
//   array.to_vtk_image_data_file(argv[2]);
 
//   // Get the width, height, and time dimensions of the domain
//   const size_t DW = array.dim(1), DH = array.dim(2), DT = array.dim(3);
//   printf("DW=%zu, DH=%zu, DT=%zu\n", DW, DH, DT);

//   // Derive the average velocity as the frame-of-reference for critical 
//   // point tracking
//   double ave_u = 0.0, ave_v = 0.0;
//   for (int k = 0; k < DT; k ++)
//     for (int j = 0; j < DH; j ++)
//       for (int i = 0; i < DW; i ++) {
//         ave_u += array(0, i, j, k);
//         ave_v += array(1, i, j, k);
//       }
//   ave_u = ave_u / (DW * DH * DT);
//   ave_v = ave_v / (DW * DH * DT);

//   printf("ave_u=%f, ave_v=%f\n", ave_u, ave_v);

//   // Initialze the tracker.  The domain leaves out one cell on the boundary 
//   // because Jacobians are evaluated by central differences
//   ftk::critical_point_tracker_2d_regular tracker( comm );
//   tracker.set_domain( ftk::lattice({1, 1}, {DW-2, DH-2}) ); 
//   tracker.set_array_domain( array.get_lattice() );
//   tracker.set_scalar_field_source( ftk::SOURCE_NONE ); // no scalar field
//   tracker.set_jacobian_field_source( ftk::SOURCE_DERIVED );
//   tracker.set_jacobian_symmetric( false ); // Jacobians are asymmetric
//   printf("before initialize\n");
//   tracker.initialize();

//   // Feed time-varying data into the tracker
//   for (int k = 0; k < DT; k ++) {
//     ftk::ndarray<double> data = array.slice_time(k);

//     // Use the average velocity as the frame-of-reference
//     for (int j = 0; j < DH; j ++)
//       for (int i = 0; i < DW; i ++) {
//         data(0, i, j) -= ave_u;
//         data(1, i, j) -= ave_v;
//       }

//     // Push the current timestep
//     tracker.push_vector_field_snapshot(data);

//     // Start tracking until two timesteps become available
//     if (k != 0) tracker.advance_timestep();
//   }
//   tracker.finalize();

//   // Write results
//   tracker.write_traced_critical_points_vtk(argv[3]);

//   return 0;
// }

#include <ftk/ndarray.hh>
#include <ftk/filters/critical_point_tracker_2d_regular.hh>
  
// This code reads Cylinder2D.am, writes the vector field data into 
// vtkImageData (.vti) file for visualization, tracks critical points 
// in the data, and writes the critical point trackign results into 
// the designated vtkPolyData (.vtp) file.
//
// Information and the Cylinder2D.am data file may be found in the 
// following web page: https://www.csc.kth.se/~weinkauf/notes/cylinder2d.html
// Follow the copyright information in the web page; Restrictions may apply.  

int main(int argc, char **argv)
{
  // The following section initializes MPI through DIY.  The code should 
  // build and run without MPI.
  diy::mpi::environment env;   
  diy::mpi::communicator comm;

  // Print help information
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <path_to_cylinder_2d_am> <output.vti> <output.vtp>\n", argv[0]);
    return 1;
  }

  // The entire spacetime volume (400x50x1001) will be read into the array
  // Note that the dimension of the array is 2x400x500x1001, with the first
  // dimension x- and y-components of the time-varying vector field
  ftk::ndarray<float> array;
  array.read_amira(argv[1]);
  array.print_shape(std::cout);
  array.set_has_time(true);
  //array.to_vtk_image_data_file(argv[2]);
  array.print_shape(std::cout);
  
  // 获取数组维度
  const size_t DW = array.dim(1), DH = array.dim(2), DT = array.dim(3);
  printf("Dimensions: %zu x %zu x %zu\n", DW, DH, DT);
  
  // 分离u和v分量，保存为单独的二进制文件
  // 创建u分量数组 (DW, DH, DT)
  ftk::ndarray<float> u_comp;
  u_comp.reshape({DW, DH, DT});
  
  // 创建v分量数组 (DW, DH, DT)  
  ftk::ndarray<float> v_comp;
  v_comp.reshape({DW, DH, DT});
  
  // 复制数据
  for (size_t t = 0; t < DT; t++) {
    for (size_t h = 0; h < DH; h++) {
      for (size_t w = 0; w < DW; w++) {
        u_comp(w, h, t) = array(0, w, h, t);  // u分量
        v_comp(w, h, t) = array(1, w, h, t);  // v分量
      }
    }
  }
  
  // 保存为二进制文件
  u_comp.to_binary_file("u_component.bin");
  v_comp.to_binary_file("v_component.bin");
  
  printf("Saved u component to u_component.bin with shape (%zu, %zu, %zu)\n", DW, DH, DT);
  printf("Saved v component to v_component.bin with shape (%zu, %zu, %zu)\n", DW, DH, DT);
  



  exit(0);

  // Derive the average velocity as the frame-of-reference for critical 
  // point tracking
  double ave_u = 0.0, ave_v = 0.0;
  for (int k = 0; k < DT; k ++)
    for (int j = 0; j < DH; j ++)
      for (int i = 0; i < DW; i ++) {
        ave_u += array(0, i, j, k);
        ave_v += array(1, i, j, k);
      }
  ave_u = ave_u / (DW * DH * DT);
  ave_v = ave_v / (DW * DH * DT);

  // Initialze the tracker.  The domain leaves out one cell on the boundary 
  // because Jacobians are evaluated by central differences
  ftk::critical_point_tracker_2d_regular tracker( comm );
  tracker.set_domain( ftk::lattice({1, 1}, {DW-2, DH-2}) ); 
  tracker.set_array_domain( array.get_lattice() );
  tracker.set_scalar_field_source( ftk::SOURCE_NONE ); // no scalar field
  tracker.set_jacobian_field_source( ftk::SOURCE_DERIVED );
  tracker.set_jacobian_symmetric( false ); // Jacobians are asymmetric
  printf("before initialize\n");
  tracker.initialize();

  // Feed time-varying data into the tracker
  for (int k = 0; k < DT; k ++) {
    ftk::ndarray<double> data = array.slice_time(k);
    // Use the average velocity as the frame-of-reference
    // for (int j = 0; j < DH; j ++)
    //   for (int i = 0; i < DW; i ++) {
    //     data(0, i, j) -= ave_u;
    //     data(1, i, j) -= ave_v;
    //   }

    // Push the current timestep
    tracker.push_vector_field_snapshot(data);

    // Start tracking until two timesteps become available
    if (k != 0) tracker.advance_timestep();
  }
  tracker.finalize();

  // Write results
  tracker.write_traced_critical_points_vtk(argv[3]);

  return 0;
}