#ifndef ESTIMATORS_SWBA_ESTIMATOR_HPP
#define ESTIMATORS_SWBA_ESTIMATOR_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <algorithm>
#include <iostream>

#include "simulation_io/preprocessed_interfaces.hpp"
#include "simulation_io/estimator_result_io.hpp"
#include "estimators/math_utils.hpp"

namespace estimators {

/**
 * Sliding Window Bundle Adjustment Estimator
 * 
 * This is a camera-model-independent SWBA estimator that uses pre-processed
 * visual measurements with pre-computed Jacobians. It preferentially uses
 * ideal/normalized coordinates when available for better numerical stability.
 * 
 * Template parameter FLOAT allows using either float or double precision.
 */
template<typename FLOAT = double>
class SWBAEstimator {
public:
    // Type aliases for Eigen matrices
    using Vector2 = Eigen::Matrix<FLOAT, 2, 1>;
    using Vector3 = Eigen::Matrix<FLOAT, 3, 1>;
    using Vector4 = Eigen::Matrix<FLOAT, 4, 1>;
    using Vector9 = Eigen::Matrix<FLOAT, 9, 1>;
    using Matrix2 = Eigen::Matrix<FLOAT, 2, 2>;
    using Matrix3 = Eigen::Matrix<FLOAT, 3, 3>;
    using Matrix9 = Eigen::Matrix<FLOAT, 9, 9>;
    using VectorX = Eigen::Matrix<FLOAT, Eigen::Dynamic, 1>;
    using MatrixX = Eigen::Matrix<FLOAT, Eigen::Dynamic, Eigen::Dynamic>;
    
    /**
     * Configuration parameters for the estimator
     */
    struct Config {
        // Sliding window parameters
        int window_size = 10;
        FLOAT min_keyframe_distance = static_cast<FLOAT>(0.5);  // meters
        FLOAT min_keyframe_angle = static_cast<FLOAT>(10.0);    // degrees
        int max_landmarks = 500;
        
        // Optimization parameters
        int max_optimization_iterations = 50;
        int max_iterations = 50;  // Alias for tests
        FLOAT optimization_convergence_threshold = static_cast<FLOAT>(1e-6);
        FLOAT convergence_threshold = static_cast<FLOAT>(1e-6);  // Alias for tests
        bool use_ideal_coordinates = true;  // Prefer ideal coordinates when available
        FLOAT ideal_coordinate_weight = static_cast<FLOAT>(10.0);  // Weight boost for ideal coords
        
        // Visual correction parameters
        FLOAT visual_correction_gain = static_cast<FLOAT>(0.0);  // Disable to match Python debug
        FLOAT chi2_outlier_threshold = static_cast<FLOAT>(5.991);  // 95% confidence for 2 DOF
        
        // Robust kernel parameters
        bool use_robust_kernel = true;
        std::string robust_kernel_type = "huber";  // "huber" or "cauchy"
        FLOAT huber_delta = static_cast<FLOAT>(1.0);
        FLOAT cauchy_c = static_cast<FLOAT>(2.3849);
        
        // Debug/logging
        bool verbose = false;
    };
    
    /**
     * Current state of the estimator
     */
    struct State {
        Vector3 position;
        Matrix3 rotation_matrix;
        Vector3 velocity;
        Vector3 bias_accel;
        Vector3 bias_gyro;
        FLOAT timestamp;
        
        State() 
            : position(Vector3::Zero()),
              rotation_matrix(Matrix3::Identity()),
              velocity(Vector3::Zero()),
              bias_accel(Vector3::Zero()),
              bias_gyro(Vector3::Zero()),
              timestamp(0) {}
    };
    
private:
    Config config_;
    State current_state_;
    Matrix9 state_covariance_;
    
    std::vector<State> keyframe_states_;
    std::vector<simulation_io::ProcessedVisualFrameT<FLOAT>> keyframe_observations_;
    std::unordered_map<int, Vector3> landmarks_;
    
    // Store all states for trajectory output
    std::vector<State> trajectory_states_;
    
    bool initialized_;
    int next_keyframe_id_;
    
public:
    /**
     * Constructor
     */
    explicit SWBAEstimator(const Config& config = Config())
        : config_(config),
          state_covariance_(Matrix9::Identity()),
          initialized_(false),
          next_keyframe_id_(0) {}
    
    /**
     * Initialize the estimator with an initial state
     */
    void initialize(const Vector3& initial_position,
                   const Matrix3& initial_rotation,
                   const Vector3& initial_velocity,
                   FLOAT timestamp) {
        current_state_.position = initial_position;
        current_state_.rotation_matrix = initial_rotation;
        current_state_.velocity = initial_velocity;
        current_state_.timestamp = timestamp;
        
        // Initialize covariance
        state_covariance_ = Matrix9::Identity();
        state_covariance_.template block<3, 3>(0, 0) *= 0.01;  // Position uncertainty
        state_covariance_.template block<3, 3>(3, 3) *= 0.1;   // Velocity uncertainty  
        state_covariance_.template block<3, 3>(6, 6) *= 0.001; // Rotation uncertainty
        
        // Add first keyframe
        keyframe_states_.clear();
        keyframe_states_.push_back(current_state_);
        
        // Initialize trajectory with first state
        trajectory_states_.clear();
        trajectory_states_.push_back(current_state_);
        
        landmarks_.clear();
        initialized_ = true;
        
        if (config_.verbose) {
            std::cout << "[SWBA] Initialized at position: " << initial_position.transpose() << std::endl;
        }
    }
    
    /**
     * Predict forward using IMU measurements
     */
    void predict(const simulation_io::PreprocessedIMUDataT<FLOAT>& imu_data, FLOAT dt) {
        if (!initialized_) {
            std::cerr << "[SWBA] Cannot predict - not initialized" << std::endl;
            return;
        }
        
        State prev_state = current_state_;
        Vector3 gravity(0, 0, -9.81);
        
        // Debug IMU data
        static int predict_count = 0;
        if (predict_count < 5 || config_.verbose) {
            std::cout << "[SWBA] Predict #" << predict_count << " dt=" << dt 
                     << "\n  delta_p: " << imu_data.delta_position.transpose()
                     << "\n  delta_v: " << imu_data.delta_velocity.transpose()
                     << "\n  prev_pos: " << prev_state.position.transpose()
                     << "\n  prev_vel: " << prev_state.velocity.transpose() << std::endl;
        }
        predict_count++;
        
        // Propagate rotation: R_j = R_i @ delta_R
        current_state_.rotation_matrix = prev_state.rotation_matrix * imu_data.delta_rotation;
        
        // Matching Python's formula from IMUPreintegrator.predict():
        // The deltas have gravity removed, so add it back
        
        // Propagate velocity: v_j = v_i + g*dt + R_i @ delta_v
        Vector3 gravity_term_v = gravity * dt;
        Vector3 rotated_delta_v = prev_state.rotation_matrix * imu_data.delta_velocity;
        current_state_.velocity = prev_state.velocity + gravity_term_v + rotated_delta_v;
        
        // Propagate position: p_j = p_i + v_i*dt + 0.5*g*dt^2 + R_i @ delta_p
        Vector3 vel_term = prev_state.velocity * dt;
        Vector3 gravity_term_p = 0.5 * gravity * dt * dt;
        Vector3 rotated_delta_p = prev_state.rotation_matrix * imu_data.delta_position;
        current_state_.position = prev_state.position + vel_term + gravity_term_p + rotated_delta_p;
        
        if (predict_count <= 5) {
            std::cout << "  vel_term: " << vel_term.transpose() << std::endl;
            std::cout << "  gravity_term_p: " << gravity_term_p.transpose() << std::endl;
            std::cout << "  rotated_delta_p: " << rotated_delta_p.transpose() << std::endl;
            std::cout << "  new_pos: " << current_state_.position.transpose() << std::endl;
            std::cout << "  new_vel: " << current_state_.velocity.transpose() << std::endl;
        }
        
        // Update timestamp
        current_state_.timestamp += dt;
        
        // Store state in trajectory
        trajectory_states_.push_back(current_state_);
        
        // Propagate covariance (simplified - just add process noise)
        state_covariance_.diagonal().template segment<3>(0) += Vector3::Constant(static_cast<FLOAT>(0.01));  // Position
        state_covariance_.diagonal().template segment<3>(3) += Vector3::Constant(static_cast<FLOAT>(0.01));  // Velocity
        state_covariance_.diagonal().template segment<3>(6) += Vector3::Constant(static_cast<FLOAT>(0.001)); // Rotation
        
        if (config_.verbose) {
            std::cout << "[SWBA] Predicted to t=" << current_state_.timestamp 
                     << ", pos: " << current_state_.position.transpose() << std::endl;
        }
    }
    
    /**
     * Update with visual measurements
     * Returns number of optimization iterations performed
     */
    int update(const simulation_io::ProcessedVisualFrameT<FLOAT>& visual_frame,
              const std::unordered_map<int, Vector3>* external_landmarks = nullptr) {
        if (!initialized_) {
            std::cerr << "[SWBA] Cannot update - not initialized" << std::endl;
            return 0;
        }
        
        // Initialize landmarks from external if provided
        if (external_landmarks) {
            for (const auto& [id, pos] : *external_landmarks) {
                if (landmarks_.find(id) == landmarks_.end()) {
                    landmarks_[id] = pos;
                }
            }
        }
        
        // Apply visual correction using ideal coordinates if available
        applyVisualCorrection(visual_frame);
        
        // Check if this should be a keyframe
        if (shouldAddKeyframe()) {
            keyframe_states_.push_back(current_state_);
            keyframe_observations_.push_back(visual_frame);
            
            // Maintain sliding window
            while (keyframe_states_.size() > static_cast<size_t>(config_.window_size)) {
                keyframe_states_.erase(keyframe_states_.begin());
                keyframe_observations_.erase(keyframe_observations_.begin());
            }
        }
        
        // Initialize new landmarks
        initializeNewLandmarks(visual_frame);
        
        // Run optimization
        return optimize(visual_frame);
    }
    
    // Overload for EstimatedLandmarkT
    int update(const simulation_io::ProcessedVisualFrameT<FLOAT>& visual_frame,
              const std::map<int, simulation_io::EstimatedLandmarkT<FLOAT>>& external_landmarks) {
        // Convert to simple format
        std::unordered_map<int, Vector3> simple_landmarks;
        for (const auto& [id, lmk] : external_landmarks) {
            simple_landmarks[id] = lmk.position;
        }
        return update(visual_frame, &simple_landmarks);
    }
    
    /**
     * Run optimization (simplified Gauss-Newton)
     */
    int optimize(const simulation_io::ProcessedVisualFrameT<FLOAT>& visual_frame) {
        // TEMPORARILY DISABLED FOR DEBUGGING
        return 0;
        
        if (keyframe_states_.size() < 2) {
            return 0;  // Need at least 2 keyframes
        }
        
        int iteration = 0;
        FLOAT prev_cost = std::numeric_limits<FLOAT>::max();
        
        for (iteration = 0; iteration < config_.max_optimization_iterations; ++iteration) {
            // Compute total cost
            FLOAT cost = 0;
            MatrixX H = MatrixX::Zero(9 * keyframe_states_.size(), 9 * keyframe_states_.size());
            VectorX b = VectorX::Zero(9 * keyframe_states_.size());
            
            // Add visual residuals
            for (size_t kf_idx = 0; kf_idx < keyframe_observations_.size(); ++kf_idx) {
                const auto& frame = keyframe_observations_[kf_idx];
                for (const auto& meas : frame.measurements) {
                    if (!meas.is_valid()) continue;
                    
                    // Use ideal coordinates if available
                    if (config_.use_ideal_coordinates && meas.has_ideal_coordinates()) {
                        cost += meas.ideal_residual.value().squaredNorm() * 
                               meas.robust_weight * config_.ideal_coordinate_weight;
                        
                        // Add to normal equations (simplified)
                        if (meas.ideal_jacobian_wrt_pose.has_value()) {
                            int state_idx = kf_idx * 9;
                            Matrix2 info = Matrix2::Identity() * meas.robust_weight * config_.ideal_coordinate_weight;
                            auto J = meas.ideal_jacobian_wrt_pose.value();
                            
                            // H += J^T * W * J  (simplified - only pose part)
                            H.template block<6, 6>(state_idx, state_idx) += J.transpose() * info * J;
                            
                            // b += J^T * W * r
                            b.template segment<6>(state_idx) += J.transpose() * info * meas.ideal_residual.value();
                        }
                    } else {
                        // Use pixel measurements
                        cost += meas.residual.squaredNorm() * meas.robust_weight;
                    }
                }
            }
            
            // Check convergence
            if (std::abs(prev_cost - cost) < config_.optimization_convergence_threshold) {
                if (config_.verbose) {
                    std::cout << "[SWBA] Converged at iteration " << iteration 
                             << " with cost " << cost << std::endl;
                }
                break;
            }
            
            // Solve H * dx = -b
            VectorX dx = H.ldlt().solve(-b);
            
            // Apply update (simplified - just to current state)
            if (dx.size() >= 9) {
                current_state_.position += dx.template segment<3>(0);
                current_state_.velocity += dx.template segment<3>(3);
                // Rotation update would need proper SO(3) handling
            }
            
            prev_cost = cost;
        }
        
        return iteration;
    }
    
    // Alternative optimize without visual frame
    int optimize() {
        if (keyframe_states_.size() < 2) {
            return 0;  // Need at least 2 keyframes to optimize
        }
        
        // Placeholder for full bundle adjustment
        int iteration;
        for (iteration = 0; iteration < config_.max_optimization_iterations; ++iteration) {
            // TODO: Implement full Gauss-Newton optimization
            // This is a placeholder
        }
        return iteration;
    }
    
    // Public accessors
    const State& getCurrentState() const {
        return current_state_;
    }
    
    size_t getNumKeyframes() const { return keyframe_states_.size(); }
    size_t getNumLandmarks() const { return landmarks_.size(); }
    
    Vector3 getCurrentPosition() const { return current_state_.position; }
    Matrix3 getCurrentRotation() const { return current_state_.rotation_matrix; }
    Vector3 getCurrentVelocity() const { return current_state_.velocity; }
    
    /**
     * Get estimated pose for output
     */
    simulation_io::EstimatedPoseT<FLOAT> getEstimatedPose() const {
        simulation_io::EstimatedPoseT<FLOAT> pose;
        pose.timestamp = current_state_.timestamp;
        pose.position = current_state_.position;
        pose.quaternion = rotation_matrix_to_quaternion(current_state_.rotation_matrix);
        pose.velocity = current_state_.velocity;
        return pose;
    }
    
    /**
     * Get all estimated landmarks
     */
    std::map<int, simulation_io::EstimatedLandmarkT<FLOAT>> getEstimatedLandmarks() const {
        std::map<int, simulation_io::EstimatedLandmarkT<FLOAT>> result;
        for (const auto& [id, pos] : landmarks_) {
            simulation_io::EstimatedLandmarkT<FLOAT> lmk;
            lmk.id = id;
            lmk.position = pos;
            result[id] = lmk;
        }
        return result;
    }
    
    /**
     * Get full estimated trajectory
     */
    std::vector<simulation_io::EstimatedPoseT<FLOAT>> getFullTrajectory() const {
        std::vector<simulation_io::EstimatedPoseT<FLOAT>> trajectory;
        for (const auto& state : trajectory_states_) {
            simulation_io::EstimatedPoseT<FLOAT> pose;
            pose.timestamp = state.timestamp;
            pose.position = state.position;
            pose.quaternion = rotation_matrix_to_quaternion(state.rotation_matrix);
            pose.velocity = state.velocity;
            trajectory.push_back(pose);
        }
        return trajectory;
    }
    
private:
    /**
     * Apply EKF-style visual correction
     */
    void applyVisualCorrection(const simulation_io::ProcessedVisualFrameT<FLOAT>& frame) {
        // Accumulate information matrix and information vector
        Matrix9 info_matrix = Matrix9::Zero();
        Vector9 info_vector = Vector9::Zero();
        
        int valid_measurements = 0;
        
        for (const auto& meas : frame.measurements) {
            if (!meas.is_valid()) continue;
            
            // Check if landmark exists
            auto it = landmarks_.find(meas.landmark_id);
            if (it == landmarks_.end()) continue;
            
            // Prefer ideal coordinates if available
            if (config_.use_ideal_coordinates && meas.has_ideal_coordinates() &&
                meas.ideal_jacobian_wrt_pose.has_value()) {
                
                // Use ideal coordinates with weight boost
                const auto& J = meas.ideal_jacobian_wrt_pose.value();
                Matrix2 measurement_info = Matrix2::Identity() * 
                    meas.robust_weight * config_.ideal_coordinate_weight;
                
                // Accumulate information (only pose part for now)
                info_matrix.template block<6, 6>(0, 0) += 
                    J.transpose() * measurement_info * J;
                info_vector.template segment<6>(0) += 
                    J.transpose() * measurement_info * meas.ideal_residual.value();
                
                valid_measurements++;
                
            } else if (meas.jacobian_wrt_pose.has_value()) {
                // Use pixel measurements
                const auto& J = meas.jacobian_wrt_pose.value();
                Matrix2 measurement_info = Matrix2::Identity() * meas.robust_weight;
                
                info_matrix.template block<6, 6>(0, 0) += 
                    J.transpose() * measurement_info * J;
                info_vector.template segment<6>(0) += 
                    J.transpose() * measurement_info * meas.residual;
                
                valid_measurements++;
            }
        }
        
        if (valid_measurements > 0) {
            // Solve for correction: delta = (H + lambda*I)^(-1) * b
            Matrix9 H = info_matrix + Matrix9::Identity() * 1e-6;  // Add small damping
            Vector9 correction = H.ldlt().solve(info_vector) * config_.visual_correction_gain;
            
            // Apply correction
            current_state_.position += correction.template segment<3>(0);
            current_state_.velocity += correction.template segment<3>(3);
            
            // Rotation correction needs proper SO(3) handling
            Vector3 rotation_correction = correction.template segment<3>(6);
            if (rotation_correction.norm() > 1e-8) {
                current_state_.rotation_matrix = current_state_.rotation_matrix * 
                    so3_exp(rotation_correction);
            }
            
            // Update covariance
            state_covariance_ = (Matrix9::Identity() - H.inverse() * info_matrix) * state_covariance_;
            
            if (config_.verbose) {
                std::cout << "[SWBA] Applied visual correction with " 
                         << valid_measurements << " measurements" << std::endl;
            }
        }
    }
    
    /**
     * Initialize new landmarks using bearing vectors
     */
    void initializeNewLandmarks(const simulation_io::ProcessedVisualFrameT<FLOAT>& frame) {
        for (const auto& meas : frame.measurements) {
            // Skip if landmark already exists
            if (landmarks_.find(meas.landmark_id) != landmarks_.end()) {
                continue;
            }
            
            // Initialize using bearing vector and estimated depth if available
            if (meas.bearing_vector.has_value() && meas.estimated_depth.has_value()) {
                Vector3 landmark_camera = meas.bearing_vector.value() * meas.estimated_depth.value();
                Vector3 landmark_world = current_state_.rotation_matrix * landmark_camera + 
                                       current_state_.position;
                
                landmarks_[meas.landmark_id] = landmark_world;
                
                if (config_.verbose) {
                    std::cout << "[SWBA] Initialized landmark " << meas.landmark_id
                             << " at " << landmark_world.transpose() << std::endl;
                }
            }
        }
    }
    
    /**
     * Check if we should add a new keyframe
     */
    bool shouldAddKeyframe() const {
        if (keyframe_states_.empty()) {
            return true;
        }
        
        const State& last_keyframe = keyframe_states_.back();
        
        // Distance criterion
        FLOAT distance = (current_state_.position - last_keyframe.position).norm();
        if (distance > config_.min_keyframe_distance) {
            return true;
        }
        
        // Rotation criterion
        Matrix3 relative_rotation = last_keyframe.rotation_matrix.transpose() * 
                                   current_state_.rotation_matrix;
        FLOAT angle = std::acos(std::min(static_cast<FLOAT>(1.0),
                                        (relative_rotation.trace() - 1) / 2));
        if (angle > config_.min_keyframe_angle * M_PI / 180) {
            return true;
        }
        
        return false;
    }
};

} // namespace estimators

#endif // ESTIMATORS_SWBA_ESTIMATOR_HPP