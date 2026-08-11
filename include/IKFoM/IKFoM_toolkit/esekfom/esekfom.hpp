/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP


#include <vector>
#include <cstdlib>

#include <boost/bind/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include "../mtk/types/vect.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/types/SEn.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/build_manifold.hpp"
#include "util.hpp"

namespace esekfom {

using namespace Eigen;

template<typename T>
struct dyn_share_modified
{
	bool valid;
	bool converge;
	T M_Noise;
	Eigen::Matrix<T, Eigen::Dynamic, 1> z;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
	Eigen::Matrix<T, 6, 1> z_IMU;
	Eigen::Matrix<T, 6, 1> R_IMU;
	bool satu_check[6];
};

template<typename state, int process_noise_dof, typename input = state, typename measurement=state, int measurement_noise_dof=0>
class esekf{

	typedef esekf self;
	enum{
		n = state::DOF, m = state::DIM, l = measurement::DOF
	};

public:
	
	typedef typename state::scalar scalar_type;
	typedef Matrix<scalar_type, n, n> cov;
	typedef Matrix<scalar_type, m, n> cov_;
	typedef SparseMatrix<scalar_type> spMt;
	typedef Matrix<scalar_type, n, 1> vectorized_state;
	typedef Matrix<scalar_type, m, 1> flatted_state;
	typedef flatted_state processModel(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
	typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;

	typedef void measurementModel_dyn_share_modified(state &, dyn_share_modified<scalar_type> &);
	typedef Eigen::Matrix<scalar_type ,l, n> measurementMatrix1(state &);
	typedef Eigen::Matrix<scalar_type , Eigen::Dynamic, n> measurementMatrix1_dyn(state &);
	typedef Eigen::Matrix<scalar_type ,l, measurement_noise_dof> measurementMatrix2(state &);
	typedef Eigen::Matrix<scalar_type ,Eigen::Dynamic, Eigen::Dynamic> measurementMatrix2_dyn(state &);
	typedef Eigen::Matrix<scalar_type, measurement_noise_dof, measurement_noise_dof> measurementnoisecovariance;
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementnoisecovariance_dyn;

	esekf(const state &x = state(),
		const cov  &P = cov::Identity()): x_(x), P_(P){};

	void init_dyn_share_modified(processModel f_in, processMatrix1 f_x_in, measurementModel_dyn_share_modified h_dyn_share_in)
	{
		f = f_in;
		f_x = f_x_in;
		// f_w = f_w_in;
		h_dyn_share_modified_1 = h_dyn_share_in;
		maximum_iter = 1;
		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
		x_.build_SEN_state();
	}
	
	void init_dyn_share_modified_2h(processModel f_in, processMatrix1 f_x_in, measurementModel_dyn_share_modified h_dyn_share_in1, measurementModel_dyn_share_modified h_dyn_share_in2)
	{
		f = f_in;
		f_x = f_x_in;
		// f_w = f_w_in;
		h_dyn_share_modified_1 = h_dyn_share_in1;
		h_dyn_share_modified_2 = h_dyn_share_in2;
		maximum_iter = 1;
		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
		x_.build_SEN_state();
	}

	// iterated error state EKF propogation
	void predict(double &dt, processnoisecovariance &Q, const input &i_in, bool predict_state, bool prop_cov){
		if (predict_state)
		{
			flatted_state f_ = f(x_, i_in);
			x_.oplus(f_, dt);
		}

		if (prop_cov)
		{
			flatted_state f_ = f(x_, i_in);
			// state x_before = x_;

			cov_ f_x_ = f_x(x_, i_in);
			cov f_x_final;
			F_x1 = cov::Identity();
			for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) {
				int idx = (*it).first.first;
				int dim = (*it).first.second;
				int dof = (*it).second;
				for(int i = 0; i < n; i++){
					for(int j=0; j<dof; j++)
					{f_x_final(idx+j, i) = f_x_(dim+j, i);}	
				}
			}

			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = -1 * f_(dim + i) * dt;
				}
				MTK::SO3<scalar_type> res;
				res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
				F_x1.template block<3, 3>(idx, idx) = res.normalized().toRotationMatrix();		
				res_temp_SO3 = MTK::A_matrix(seg_SO3);
				for(int i = 0; i < n; i++){
					f_x_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_. template block<3, 1>(dim, i));	
				}
			}
	
			F_x1 += f_x_final * dt;
			P_ = F_x1 * P_ * (F_x1).transpose() + Q * (dt * dt);
		}
	}

	bool update_iterated_dyn_share_modified() {
		dyn_share_modified<scalar_type> &dyn_share = dyn_share_buf_;
		state x_propagated = x_;
		int dof_Measurement;
		double m_noise;
		for(int i=0; i<maximum_iter; i++)
		{
			dyn_share.valid = true;
			h_dyn_share_modified_1(x_, dyn_share);
			if(! dyn_share.valid)
			{
				return false;
				// continue;
			}
			dof_Measurement = dyn_share.h_x.rows();
			m_noise = dyn_share.M_Noise;

			Matrix<scalar_type, n, 1> dx_;

			if (dof_Measurement == 1) {
				// Fast path: point-to-plane measurements from this fork's caller
				// (Estimator.cpp h_model_input/h_model_output) are grouped by
				// identical point timestamp (time_seq), and that group size is
				// empirically ~always 1 (measured: p50=p95=p99=1, max=2 across
				// 771k samples on a real bag) -- i.e. this is a true per-point
				// scalar residual, not a batched/stacked one. The general path
				// below is algebraically identical for dof_Measurement==1 (a 1x1
				// "matrix inverse" is just a scalar reciprocal) but pays for
				// Dynamic-sized Eigen temporaries (heap allocation every call)
				// and Eigen's general .inverse() (LU-based) machinery to get
				// there. This branch computes the same
				// K_ = P*H^T*(H*P*H^T + R)^-1, dx_ = K_*z, P_ -= K_*H*P_
				// using fixed-size types and a plain scalar division instead.
				// Kept as a fast path (not a replacement) rather than changing
				// dyn_share_modified's h_x/z types, since those are shared by
				// other measurement models (e.g. h_model_IMU_output) this
				// change must not touch.
				Matrix<scalar_type, 1, 12> h_row = dyn_share.h_x.template block<1, 12>(0, 0);
				scalar_type z0 = dyn_share.z(0);
				Matrix<scalar_type, n, 1> PHT = P_. template block<n, 12>(0, 0) * h_row.transpose();
				scalar_type HPHT = (h_row * PHT.template topRows<12>())(0, 0);
				HPHT += m_noise;
				Matrix<scalar_type, n, 1> K_ = PHT / HPHT;
				dx_ = K_ * z0;
				x_.boxplus(dx_);
				dyn_share.converge = true;
				P_ = P_ - (K_ * h_row) * P_. template block<12, n>(0, 0);
			} else {
				// General path, unchanged from upstream: dof_Measurement > 1 has not
				// been observed in practice for this fork's callers (max seen: 2), but
				// is kept exact and untouched as a correctness safety net.
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> z = dyn_share.z;
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;

				Matrix<scalar_type, n, Eigen::Dynamic> PHT;
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> HPHT;
				Matrix<scalar_type, n, Eigen::Dynamic> K_;
				{
					PHT = P_. template block<n, 12>(0, 0) * h_x.transpose();
					HPHT = h_x * PHT.topRows(12);
					for (int m = 0; m < dof_Measurement; m++)
					{
						HPHT(m, m) += m_noise;
					}
					K_= PHT*HPHT.inverse();
				}
				dx_ = K_ * z;

				x_.boxplus(dx_);
				dyn_share.converge = true;
				P_ = P_ - K_*h_x*P_. template block<12, n>(0, 0);
			}
			
			// L_ = P_;
			// Matrix<scalar_type, 3, 3> res_temp_SO3;
			// MTK::vect<3, scalar_type> seg_SO3;
			// for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
			// 	int idx = (*it).first;
			// 	for(int i = 0; i < 3; i++){
			// 		seg_SO3(i) = dx_(i + idx);
			// 	}
			// 	res_temp_SO3 = A_matrix(seg_SO3).transpose();
			// 	for(int i = 0; i < n; i++){
			// 		L_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i)); 
			// 	}
			// 	{
			// 		for(int i = 0; i < dof_Measurement; i++){
			// 			K_. template block<3, 1>(idx, i) = res_temp_SO3 * (K_. template block<3, 1>(idx, i));
			// 		}
			// 	}
			// 	for(int i = 0; i < n; i++){
			// 		L_. template block<1, 3>(i, idx) = (L_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
			// 		// P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
			// 	}
			// 	for(int i = 0; i < n; i++){
			// 		P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
			// 	}
			// }
			// Matrix<scalar_type, 2, 2> res_temp_S2;
			// MTK::vect<2, scalar_type> seg_S2;
			// for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
			// 	int idx = (*it).first;
		
			// 	for(int i = 0; i < 2; i++){
			// 		seg_S2(i) = dx_(i + idx);
			// 	}
		
			// 	Eigen::Matrix<scalar_type, 2, 3> Nx;
			// 	Eigen::Matrix<scalar_type, 3, 2> Mx;
			// 	x_.S2_Nx_yy(Nx, idx);
			// 	x_propagated.S2_Mx(Mx, seg_S2, idx);
			// 	res_temp_S2 = Nx * Mx; 
	
			// 	for(int i = 0; i < n; i++){
			// 		L_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i)); 
			// 	}
				
			// 	{
			// 		for(int i = 0; i < dof_Measurement; i++){
			// 			K_. template block<2, 1>(idx, i) = res_temp_S2 * (K_. template block<2, 1>(idx, i));
			// 		}
			// 	}
			// 	for(int i = 0; i < n; i++){
			// 		L_. template block<1, 2>(i, idx) = (L_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
			// 	}
			// 	for(int i = 0; i < n; i++){
			// 		P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
			// 	}
			// }
		}
		return true;
	}

	void update_iterated_dyn_share_IMU() {

		dyn_share_modified<scalar_type> &dyn_share = dyn_share_buf_;
		for(int i=0; i<maximum_iter; i++)
		{
			dyn_share.valid = true;
			h_dyn_share_modified_2(x_, dyn_share);

			Matrix<scalar_type, 6, 1> z = dyn_share.z_IMU;

			Matrix<double, 30, 6> PHT;
            Matrix<double, 6, 30> HP;
            Matrix<double, 6, 6> HPHT;
			PHT.setZero();
			HP.setZero();
			HPHT.setZero();
			for (int l_ = 0; l_ < 6; l_++)
			{
				if (!dyn_share.satu_check[l_])
				{
					PHT.col(l_) = P_.col(15+l_) + P_.col(24+l_);
					HP.row(l_) = P_.row(15+l_) + P_.row(24+l_);
				}
			}
			for (int l_ = 0; l_ < 6; l_++)
			{
				if (!dyn_share.satu_check[l_])
				{
					HPHT.col(l_) = HP.col(15+l_) + HP.col(24+l_);
				}
				HPHT(l_, l_) += dyn_share.R_IMU(l_); //, l);
			}
        	Eigen::Matrix<double, 30, 6> K = PHT * HPHT.inverse(); 
                                    
            Matrix<scalar_type, n, 1> dx_ = K * z; 

            P_ -= K * HP;
			x_.boxplus(dx_);
		}
		return;
	}
	
	void change_x(state &input_state)
	{
		x_ = input_state;

		if((!x_.vect_state.size())&&(!x_.SO3_state.size())&&(!x_.S2_state.size())&&(!x_.SEN_state.size()))
		{
			x_.build_S2_state();
			x_.build_SO3_state();
			x_.build_vect_state();
			x_.build_SEN_state();
		}
	}

	void change_P(cov &input_cov)
	{
		P_ = input_cov;
	}

	const state& get_x() const {
		return x_;
	}
	const cov& get_P() const {
		return P_;
	}
	state x_;
private:
	measurement m_;
	cov P_;
	spMt l_;
	spMt f_x_1;
	spMt f_x_2;
	cov F_x1 = cov::Identity();
	cov F_x2 = cov::Identity();
	cov L_ = cov::Identity();

	processModel *f;
	processMatrix1 *f_x;
	processMatrix2 *f_w;

	measurementMatrix1 *h_x;
	measurementMatrix2 *h_v;

	measurementMatrix1_dyn *h_x_dyn;
	measurementMatrix2_dyn *h_v_dyn;

	measurementModel_dyn_share_modified *h_dyn_share_modified_1;

	measurementModel_dyn_share_modified *h_dyn_share_modified_2;

	int maximum_iter = 0;
	scalar_type limit[n];

	// Persistent scratch object for update_iterated_dyn_share_modified() and
	// update_iterated_dyn_share_IMU() (both single-threaded, called sequentially, never
	// concurrently, so sharing one instance across both is safe). Was previously a fresh
	// stack-local dyn_share_modified<T> constructed on every call -- since T's h_x/z
	// members are Eigen::Dynamic-sized, that meant a fresh heap allocation on every
	// single point (dof_Measurement==1 in practice, see esekfom.hpp's fast path below,
	// but a brand-new object has 0 capacity regardless of what size gets resized into
	// next). Reusing one persistent instance means Eigen's resize() is a no-op once the
	// buffer has already grown to the steady-state size (1x12 / 1), eliminating that
	// allocation. Safe because both call sites unconditionally overwrite every field
	// they read (valid, h_x, z, M_Noise / z_IMU, R_IMU, satu_check) before reading it --
	// nothing here depends on a field retaining its value from a prior call.
	dyn_share_modified<scalar_type> dyn_share_buf_;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP
