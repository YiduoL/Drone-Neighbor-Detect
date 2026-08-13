// #include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include "Estimator.h"
#include <prof_timing.h>

PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
std::vector<int> time_seq;
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
std::vector<V3D> pbody_list;
std::vector<PointVector> Nearest_Points;
std::vector<bool> ekf_stride_skip_insert;
KD_TREE<PointType> ikdtree;
point_lio_experimental::SurfelMap *surfel_map = nullptr;
bool surfel_map_past_warmup = false;
std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
bool point_selected_surf[100000] = {0};
std::vector<M3D> crossmat_list;
int effct_feat_num = 0;
int k;
int idx;
esekfom::esekf<state_input, 24, input_ikfom> kf_input;
esekfom::esekf<state_output, 30, input_ikfom> kf_output;
state_input state_in;
state_output state_out;
input_ikfom input_in;
V3D angvel_avr, acc_avr;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2;
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

Eigen::Matrix<double, 24, 24> process_noise_cov_input()
{
	Eigen::Matrix<double, 24, 24> cov;
	cov.setZero();
	cov.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
	cov.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
	cov.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	// MTK::get_cov<process_noise_input>::type cov = MTK::get_cov<process_noise_input>::type::Zero();
	// MTK::setDiagonal<process_noise_input, vect3, 0>(cov, &process_noise_input::ng, gyr_cov_input);// 0.03
	// MTK::setDiagonal<process_noise_input, vect3, 3>(cov, &process_noise_input::na, acc_cov_input); // *dt 0.01 0.01 * dt * dt 0.05
	// MTK::setDiagonal<process_noise_input, vect3, 6>(cov, &process_noise_input::nbg, b_gyr_cov); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	// MTK::setDiagonal<process_noise_input, vect3, 9>(cov, &process_noise_input::nba, b_acc_cov);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

Eigen::Matrix<double, 30, 30> process_noise_cov_output()
{
	Eigen::Matrix<double, 30, 30> cov;
	cov.setZero();
	cov.block<3, 3>(12, 12).diagonal() << vel_cov, vel_cov, vel_cov;
	cov.block<3, 3>(15, 15).diagonal() << gyr_cov_output, gyr_cov_output, gyr_cov_output;
	cov.block<3, 3>(18, 18).diagonal() << acc_cov_output, acc_cov_output, acc_cov_output;
	cov.block<3, 3>(24, 24).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(27, 27).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	// MTK::get_cov<process_noise_output>::type cov = MTK::get_cov<process_noise_output>::type::Zero();
	// MTK::setDiagonal<process_noise_output, vect3, 0>(cov, &process_noise_output::vel, vel_cov);// 0.03
	// MTK::setDiagonal<process_noise_output, vect3, 3>(cov, &process_noise_output::ng, gyr_cov_output); // *dt 0.01 0.01 * dt * dt 0.05
	// MTK::setDiagonal<process_noise_output, vect3, 6>(cov, &process_noise_output::na, acc_cov_output); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	// MTK::setDiagonal<process_noise_output, vect3, 9>(cov, &process_noise_output::nbg, b_gyr_cov);   //0.001 0.05 0.0001/out 0.01
	// MTK::setDiagonal<process_noise_output, vect3, 12>(cov, &process_noise_output::nba, b_acc_cov);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	vect3 a_inertial = s.rot.normalized() * (in.acc-s.ba); 
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) =  omega[i]; 
		res(i + 12) = a_inertial[i] + s.gravity[i]; 
	}
	return res;
}

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 1> res = Eigen::Matrix<double, 30, 1>::Zero();
	vect3 a_inertial = s.rot.normalized() * s.acc; 
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) = s.omg[i]; 
		res(i + 12) = a_inertial[i] + s.gravity[i]; 
	}
	return res;
}

Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	cov.template block<3, 3>(12, 3) = -s.rot.normalized().toRotationMatrix()*MTK::hat(acc_);
	cov.template block<3, 3>(12, 18) = -s.rot.normalized().toRotationMatrix();
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix; 
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity(); 
	return cov;
}

// Eigen::Matrix<double, 24, 12> df_dw_input(state_input &s, const input_ikfom &in)
// {
// 	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
// 	cov.template block<3, 3>(12, 3) = -s.rot.normalized().toRotationMatrix();
// 	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
// 	return cov;
// }

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 30> cov = Eigen::Matrix<double, 30, 30>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(12, 3) = -s.rot.normalized().toRotationMatrix()*MTK::hat(s.acc);
	cov.template block<3, 3>(12, 18) = s.rot.normalized().toRotationMatrix();
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix; 
	cov.template block<3, 3>(3, 15) = Eigen::Matrix3d::Identity(); 
	return cov;
}

// Eigen::Matrix<double, 30, 15> df_dw_output(state_output &s)
// {
// 	Eigen::Matrix<double, 30, 15> cov = Eigen::Matrix<double, 30, 15>::Zero();
// 	cov.template block<3, 3>(12, 0) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(15, 3) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(18, 6) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(24, 9) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(27, 12) = Eigen::Matrix3d::Identity();
// 	return cov;
// }

vect3 SO3ToEuler(const SO3 &orient) 
{
	Eigen::Matrix<double, 3, 1> _ang;
	Eigen::Vector4d q_data = orient.coeffs().transpose();
	//scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
	double sqw = q_data[3]*q_data[3];
	double sqx = q_data[0]*q_data[0];
	double sqy = q_data[1]*q_data[1];
	double sqz = q_data[2]*q_data[2];
	double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
	double test = q_data[3]*q_data[1] - q_data[2]*q_data[0];

	if (test > 0.49999*unit) { // singularity at north pole
	
		_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
	if (test < -0.49999*unit) { // singularity at south pole
		_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
		
	_ang <<
			std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
			std::asin (2*test/unit),
			std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	vect3 euler_ang(temp, 3);
	return euler_ang;
}

void h_model_input(state_input &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	normvec->resize(time_seq[k]);
	int effect_num_k = 0;
	// pabcd and the neighbor-distance buffer are per-iteration-local (both used to be a
	// single object reused/overwritten every iteration) -- kept that way even though
	// this loop is serial, since it's correct and clearer either way.
	//
	// THIS FUNCTION IS DEAD CODE in the current config: mapping_mid360.launch.py
	// hardcodes use_imu_as_input: false, so h_model_output (below) runs every frame,
	// not this one -- confirmed via the live launch_params temp file. A long chain of
	// OpenMP-parallelization experiments this session was run against THIS function
	// before that was discovered, so all of that history (multiple rounds, eventually
	// "validated" as a small win under taskset-isolated combined load) was actually
	// measuring nothing. Once corrected by porting the identical pragma to
	// h_model_output (the real hot path) under the same isolated-core methodology,
	// the result reversed: worse, not better (frame_e2e 33-36ms mean vs. serial's
	// ~30.75ms, p95/p99 43-46ms, one spike to 57ms) -- see h_model_output's HISTORY
	// comment for the full account. Root cause: each call has ~1 point on average
	// (group_size profiling metric), so there's essentially nothing to parallelize,
	// only thread-spawn/reduction-coordination overhead to pay -- true of this
	// function too, so do not re-attempt parallelizing this loop either without a
	// fundamentally different batching approach.
	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];
		pointBodyToWorld(&point_body_j, &point_world_j);
		V3D p_body = pbody_list[idx+j+1];
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		VF(4) pabcd;
		pabcd.setZero();

		{
			auto &points_near = Nearest_Points[idx+j+1];
			std::vector<float> point_dist;

			point_selected_surf[idx+j+1] = false;
			bool plane_ok = false;
			{
				PROF_SCOPE("nn_search");
				ikdtree.Nearest_Search(point_world_j, NUM_MATCH_POINTS, points_near, point_dist, 2.236); //1.0); //, 3.0); // 2.236;
			}
			bool have_enough_neighbors = (points_near.size() >= NUM_MATCH_POINTS) && (point_dist[NUM_MATCH_POINTS - 1] <= 5);
			if (use_surfel_map && surfel_map_past_warmup)
			{
				// EXPERIMENTAL path -- see config/mid360.yaml's use_surfel_map comment.
				// nn_search above still runs unconditionally (map_incremental() downstream
				// needs Nearest_Points[i] for its own bookkeeping), so this path only ever
				// saves esti_plane()'s per-query QR solve, not nn_search itself. On top of
				// that: an early build showed real trajectory divergence (tested on
				// lidar_1_ros2, x drifted to -7.5m in the first 25s where the platform is
				// actually stationary) when a voxel without enough accumulated points was
				// simply treated as "no plane, skip this point" -- during the first ~25s the
				// local map is still sparse almost everywhere, so most points were getting
				// dropped from the EKF update right when it's most fragile. Fixed by falling
				// back to the exact same ikd-Tree nn_search + esti_plane() the non-surfel path
				// uses whenever the surfel cache doesn't have a valid plane yet for this
				// voxel -- so correctness never regresses below the original path, and the
				// surfel shortcut only ever *adds* speed once a voxel is well-populated, never
				// substitutes fewer/worse measurements for it.
				Eigen::Vector4d surfel_plane;
				bool surfel_ok;
				{
					PROF_SCOPE("esti_plane");
					surfel_ok = surfel_map->query(p_world, surfel_plane);
				}
				if (surfel_ok)
				{
					plane_ok = true;
					pabcd(0) = surfel_plane(0); pabcd(1) = surfel_plane(1);
					pabcd(2) = surfel_plane(2); pabcd(3) = surfel_plane(3);
				}
				else if (have_enough_neighbors)
				{
					PROF_SCOPE("esti_plane");
					plane_ok = esti_plane(pabcd, points_near, plane_thr);
				}
			}
			else if (have_enough_neighbors)
			{
				PROF_SCOPE("esti_plane");
				plane_ok = esti_plane(pabcd, points_near, plane_thr); //(planeValid)
			}
			if (plane_ok)
			{
				float pd2 = pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3);

				if (p_body.norm() > match_s * pd2 * pd2)
				{
					point_selected_surf[idx+j+1] = true;
					normvec->points[j].x = pabcd(0);
					normvec->points[j].y = pabcd(1);
					normvec->points[j].z = pabcd(2);
					normvec->points[j].intensity = pabcd(3);
					effect_num_k ++;
				}
			}
		}
	}
	PROF_SAMPLE("dof_measurement", effect_num_k);
	PROF_SAMPLE("group_size", time_seq[k]);
	if (effect_num_k == 0)
	{
		ekfom_data.valid = false;
		return;
	}
	ekfom_data.M_Noise = laser_point_cov;
	// .resize(), not ::Zero(): every row [0, effect_num_k) is unconditionally
	// overwritten by the block<1,12>(m,0) << ... assignment in the loop below (m counts
	// up to exactly effect_num_k, by construction), so zero-filling first is wasted work
	// on every EKF update call.
	ekfom_data.h_x.resize(effect_num_k, 12);
	ekfom_data.z.resize(effect_num_k);
	int m = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		if(point_selected_surf[idx+j+1])
		{
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);

			if (extrinsic_est_en)
			{
				V3D p_body = pbody_list[idx+j+1];
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I.normalized() * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.conjugate().normalized() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.conjugate().normalized() * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else
			{
				M3D point_crossmat = crossmat_list[idx+j+1];
				V3D C(s.rot.conjugate().normalized() * norm_vec);
				V3D A(point_crossmat * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x -norm_vec(1) * feats_down_world->points[idx+j+1].y -norm_vec(2) * feats_down_world->points[idx+j+1].z-normvec->points[j].intensity;
			m++;
		}
	}
	effct_feat_num += effect_num_k;
}

void h_model_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	normvec->resize(time_seq[k]);
	int effect_num_k = 0;
	// THIS is the actually-active branch: mapping_mid360.launch.py hardcodes
	// use_imu_as_input: false (overriding mid360.yaml's own value -- launch parameter
	// lists are parsed in order with later entries winning), so h_model_output is what
	// runs every frame, not h_model_input -- confirmed via the live launch_params temp
	// file, not assumed (a session-long assumption to the contrary was wrong, and meant
	// every earlier "OpenMP parallelization works when core-isolated" test this session
	// was inadvertently exercising h_model_input's dead twin, not this function).
	// HISTORY: tried `#pragma omp parallel for reduction(+:effect_num_k)
	// schedule(dynamic) num_threads(4)` here (identical to what was validated -- on the
	// wrong function -- earlier), taskset-isolated from detection (Point-LIO on cores
	// 0-3, detection on 4-7) exactly as before. Result on the real hot path: worse, not
	// better -- frame_e2e mean 33-36ms vs the plain-serial baseline's ~30.75ms, p95/p99
	// 43-46ms, one spike to 57ms. Core isolation was never actually the fix for the
	// earlier oversubscription failures; parallelizing this specific loop just doesn't
	// help, whether isolated or not -- each call has ~1 point on average (group_size
	// profiling metric, confirmed independent of which function runs), so there's
	// essentially nothing to parallelize, only thread-spawn/reduction-coordination
	// overhead to pay. Reverted; do not re-attempt without a fundamentally different
	// approach to batching that doesn't just add a pragma to a ~1-iteration loop.
	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];
		pointBodyToWorld(&point_body_j, &point_world_j);
		V3D p_body = pbody_list[idx+j+1];
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		VF(4) pabcd;
		pabcd.setZero();
		{
			auto &points_near = Nearest_Points[idx+j+1];
			std::vector<float> point_dist;

			point_selected_surf[idx+j+1] = false;
			bool plane_ok = false;
			{
				PROF_SCOPE("nn_search");
				ikdtree.Nearest_Search(point_world_j, NUM_MATCH_POINTS, points_near, point_dist, 2.236);
			}
			bool have_enough_neighbors = (points_near.size() >= NUM_MATCH_POINTS) && (point_dist[NUM_MATCH_POINTS - 1] <= 5);
			if (use_surfel_map && surfel_map_past_warmup)
			{
				// EXPERIMENTAL path -- see h_model_input's identical block above (including
				// the divergence bug found + the ikd-Tree fallback that fixed it) for the
				// full rationale.
				Eigen::Vector4d surfel_plane;
				bool surfel_ok;
				{
					PROF_SCOPE("esti_plane");
					surfel_ok = surfel_map->query(p_world, surfel_plane);
				}
				if (surfel_ok)
				{
					plane_ok = true;
					pabcd(0) = surfel_plane(0); pabcd(1) = surfel_plane(1);
					pabcd(2) = surfel_plane(2); pabcd(3) = surfel_plane(3);
				}
				else if (have_enough_neighbors)
				{
					PROF_SCOPE("esti_plane");
					plane_ok = esti_plane(pabcd, points_near, plane_thr);
				}
			}
			else if (have_enough_neighbors)
			{
				PROF_SCOPE("esti_plane");
				plane_ok = esti_plane(pabcd, points_near, plane_thr); //(planeValid)
			}
			if (plane_ok)
			{
				float pd2 = pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3);

				if (p_body.norm() > match_s * pd2 * pd2)
				{
					// point_selected_surf[i] = true;
					point_selected_surf[idx+j+1] = true;
					normvec->points[j].x = pabcd(0);
					normvec->points[j].y = pabcd(1);
					normvec->points[j].z = pabcd(2);
					normvec->points[j].intensity = pabcd(3);
					effect_num_k ++;
				}
			}
		}
	}
	PROF_SAMPLE("dof_measurement", effect_num_k);
	PROF_SAMPLE("group_size", time_seq[k]);
	if (effect_num_k == 0)
	{
		ekfom_data.valid = false;
		return;
	}
	ekfom_data.M_Noise = laser_point_cov;
	// .resize(), not ::Zero(): every row [0, effect_num_k) is unconditionally
	// overwritten by the block<1,12>(m,0) << ... assignment in the loop below (m counts
	// up to exactly effect_num_k, by construction), so zero-filling first is wasted work
	// on every EKF update call.
	ekfom_data.h_x.resize(effect_num_k, 12);
	ekfom_data.z.resize(effect_num_k);
	int m = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		if(point_selected_surf[idx+j+1])
		{
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);
			
			if (extrinsic_est_en)
			{
				V3D p_body = pbody_list[idx+j+1];
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I.normalized() * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.conjugate().normalized() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.conjugate().normalized() * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else
			{   
				M3D point_crossmat = crossmat_list[idx+j+1];
				V3D C(s.rot.conjugate().normalized() * norm_vec);
				V3D A(point_crossmat * C);
				// V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x -norm_vec(1) * feats_down_world->points[idx+j+1].y -norm_vec(2) * feats_down_world->points[idx+j+1].z-normvec->points[j].intensity;
			m++;
		}
	}
	effct_feat_num += effect_num_k;
}

void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
    std::memset(ekfom_data.satu_check, false, 6);
	ekfom_data.z_IMU.block<3,1>(0, 0) = angvel_avr - s.omg - s.bg;
	ekfom_data.z_IMU.block<3,1>(3, 0) = acc_avr * G_m_s2 / acc_norm - s.acc - s.ba;
    ekfom_data.R_IMU << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;
	if(check_satu)
	{
		if(fabs(angvel_avr(0)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[0] = true; 
			ekfom_data.z_IMU(0) = 0.0;
		}
		
		if(fabs(angvel_avr(1)) >= 0.99 * satu_gyro) 
		{
			ekfom_data.satu_check[1] = true;
			ekfom_data.z_IMU(1) = 0.0;
		}
		
		if(fabs(angvel_avr(2)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[2] = true;
			ekfom_data.z_IMU(2) = 0.0;
		}
		
		if(fabs(acc_avr(0)) >= 0.99 * satu_acc)
		{
			ekfom_data.satu_check[3] = true;
			ekfom_data.z_IMU(3) = 0.0;
		}

		if(fabs(acc_avr(1)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[4] = true;
			ekfom_data.z_IMU(4) = 0.0;
		}

		if(fabs(acc_avr(2)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[5] = true;
			ekfom_data.z_IMU(5) = 0.0;
		}
	}
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{    
    V3D p_body(pi->x, pi->y, pi->z);
    
    V3D p_global;
	if (extrinsic_est_en)
	{	
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot.normalized() * (kf_output.x_.offset_R_L_I.normalized() * p_body + kf_output.x_.offset_T_L_I) + kf_output.x_.pos;
		}
		else
		{
			p_global = kf_input.x_.rot.normalized() * (kf_input.x_.offset_R_L_I.normalized() * p_body + kf_input.x_.offset_T_L_I) + kf_input.x_.pos;
		}
	}
	else
	{
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot.normalized() * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_output.x_.pos;
		}
		else
		{
			p_global = kf_input.x_.rot.normalized() * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_input.x_.pos;
		}
	}

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};