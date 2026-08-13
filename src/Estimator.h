#ifndef Estimator_H
#define Estimator_H

#include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include "common_lib.h"
#include "parameters.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <ikd-Tree/ikd_Tree.h>
#include <pcl/io/pcd_io.h>
#include "../include/surfel_map.h"

extern PointCloudXYZI::Ptr normvec; //(new PointCloudXYZI(100000, 1));
extern std::vector<int> time_seq;
extern PointCloudXYZI::Ptr feats_down_body; //(new PointCloudXYZI());
extern PointCloudXYZI::Ptr feats_down_world; //(new PointCloudXYZI());
extern std::vector<V3D> pbody_list;
extern std::vector<PointVector> Nearest_Points;
// ekf_update_stride EXPERIMENTAL (see config/mid360.yaml): true for points whose group
// skipped this frame's measurement update (no fresh Nearest_Points -- see
// laserMapping.cpp's k-loop). map_incremental() skips these entirely rather than
// routing them through its no-dedup-info fallback path -- routing them there was tried
// first and caused runaway ikd-Tree growth (undeduplicated points compounding every
// frame, nn_search getting progressively slower, frame_e2e climbing 24ms->42ms+ over a
// 130s run instead of dropping). assign()'d (not resize()'d) to false every frame in
// laserMapping.cpp so stale true values never leak across frames at reused indices.
extern std::vector<bool> ekf_stride_skip_insert;
extern KD_TREE<PointType> ikdtree;
// EXPERIMENTAL (see config/mid360.yaml's use_surfel_map comment + include/surfel_map.h):
// nullptr unless use_surfel_map is true, constructed once in main() after parameters
// load (needs surfel_map_voxel_size, not available at static-init time). Never touched
// when use_surfel_map is false -- existing ikdtree path is byte-for-byte unchanged.
extern point_lio_experimental::SurfelMap *surfel_map;
// Set every frame in laserMapping.cpp's main loop once (lidar_beg_time - first_lidar_time)
// exceeds surfel_map_warmup_seconds -- see that config option's comment. Surfel queries in
// Estimator.cpp are gated on use_surfel_map && surfel_map_past_warmup, both.
extern bool surfel_map_past_warmup;
extern std::vector<float> pointSearchSqDis;
extern bool point_selected_surf[100000]; // = {0};
extern std::vector<M3D> crossmat_list;
extern int effct_feat_num;
extern int k;
extern int idx;
extern V3D angvel_avr, acc_avr;

extern V3D Lidar_T_wrt_IMU; //(Zero3d);
extern M3D Lidar_R_wrt_IMU; //(Eye3d);

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2;
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_input,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((vect3, gravity))
);

MTK_BUILD_MANIFOLD(state_output,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, omg))
((vect3, acc))
((vect3, gravity))
((vect3, bg))
((vect3, ba))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_input,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

MTK_BUILD_MANIFOLD(process_noise_output,
((vect3, vel))
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

extern esekfom::esekf<state_input, 24, input_ikfom> kf_input;
extern esekfom::esekf<state_output, 30, input_ikfom> kf_output;
extern state_input state_in;
extern state_output state_out;
extern input_ikfom input_in;

Eigen::Matrix<double, 24, 24> process_noise_cov_input();

Eigen::Matrix<double, 30, 30> process_noise_cov_output();

//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in);

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in);

Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in);

// Eigen::Matrix<double, 24, 12> df_dw_input(state_input &s, const input_ikfom &in);

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in);

// Eigen::Matrix<double, 30, 15> df_dw_output(state_output &s);

vect3 SO3ToEuler(const SO3 &orient);

void h_model_input(state_input &s, esekfom::dyn_share_modified<double> &ekfom_data);

void h_model_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data);

void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data);

void pointBodyToWorld(PointType const *const pi, PointType *const po);

const bool time_list(PointType &x, PointType &y); // {return (x.curvature < y.curvature);};

#endif