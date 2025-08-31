#ifndef ESTIMATORS_SIMPLE_SWBA_ESTIMATOR_HPP
#define ESTIMATORS_SIMPLE_SWBA_ESTIMATOR_HPP

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
 * Simplified Sliding Window Bundle Adjustment VIO Estimator
 * 
 * Key simplifications from original SWBA:
 * 1. predict() just stores preintegrated IMU data
 * 2. update() decides on keyframes and triggers optimization
 * 3. Optimization uses precomputed Jacobians from measurements
 * 4. Simple marginalization of oldest keyframe
 * 5. Stores ALL frame poses for complete trajectory output
 * 
 * Based on Python simple_swba_vio.py implementation.
 */
template<typename FLOAT = double>
class SimpleSWBAEstimator {
public:
    // Type aliases for Eigen matrices
    using Vector2 = Eigen::Matrix<FLOAT, 2, 1>;
    using Vector3 = Eigen::Matrix<FLOAT, 3, 1>;
    using Matrix2 = Eigen::Matrix<FLOAT, 2, 2>;
    using Matrix3 = Eigen::Matrix<FLOAT, 3, 3>;
    using VectorX = Eigen::Matrix<FLOAT, Eigen::Dynamic, 1>;
    using MatrixX = Eigen::Matrix<FLOAT, Eigen::Dynamic, Eigen::Dynamic>;
    
    /**
     * Simplified configuration parameters
     */
    struct Config {
        // Window parameters
        int window_size = 10;                    // Number of keyframes in window
        int keyframe_spacing = 5;                // Create keyframe every N frames
        
        // Optimization parameters  
        int max_iterations = 10;                 // Max optimization iterations
        FLOAT convergence_threshold = static_cast<FLOAT>(1e-4);  // Convergence threshold
        FLOAT damping_factor = static_cast<FLOAT>(0.01);         // Levenberg-Marquardt damping
        
        // Visual measurement parameters
        int min_measurements = 5;                // Minimum measurements for update
        FLOAT ideal_coord_weight = static_cast<FLOAT>(10.0);     // Weight for ideal coordinates
        FLOAT pixel_coord_weight = static_cast<FLOAT>(1.0);      // Weight for pixel coordinates
        
        // Debug/logging
        bool verbose = false;
        bool debug_enabled = true;
    };
    
    /**
     * Simplified state (no covariance tracking)
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
    
    /**
     * Keyframe data
     */
    struct Keyframe {
        State state;
        simulation_io::ProcessedVisualFrameT<FLOAT> visual_frame;
        int id;
    };
    
private:
    Config config_;
    State current_state_;
    
    // All poses for complete trajectory
    std::vector<State> all_poses_;
    
    // Keyframe data for optimization window
    std::vector<Keyframe> keyframes_;
    std::vector<int> keyframe_ids_;
    
    // IMU preintegration storage
    // Key: (from_kf_id, to_kf_id), Value: PreprocessedIMUData
    std::map<std::pair<int, int>, simulation_io::PreprocessedIMUDataT<FLOAT>> imu_constraints_;
    std::optional<simulation_io::PreprocessedIMUDataT<FLOAT>> pending_imu_;
    
    // Landmarks
    std::unordered_map<int, Vector3> landmarks_;
    
    // Counters
    int frame_count_;
    int keyframe_count_;
    int total_iterations_;
    
    bool initialized_;
    
public:
    /**
     * Constructor
     */
    explicit SimpleSWBAEstimator(const Config& config = Config())
        : config_(config),
          frame_count_(0),
          keyframe_count_(0),
          total_iterations_(0),
          initialized_(false) {}
    
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
        
        // Add initial pose to all_poses
        all_poses_.clear();
        all_poses_.push_back(current_state_);
        
        // Clear data structures
        keyframes_.clear();
        keyframe_ids_.clear();
        imu_constraints_.clear();
        pending_imu_.reset();
        landmarks_.clear();
        
        frame_count_ = 0;
        keyframe_count_ = 0;
        total_iterations_ = 0;
        initialized_ = true;
        
        if (config_.verbose) {
            std::cout << "[SimpleSWBA] Initialized at position: " 
                     << initial_position.transpose() << std::endl;
        }
    }
    
    /**
     * Prediction step - store preintegrated IMU and propagate state
     * 
     * @param imu_data Preintegrated IMU measurements
     * @param dt Time delta (already in preintegration)
     */
    void predict(const simulation_io::PreprocessedIMUDataT<FLOAT>& imu_data, FLOAT dt) {
        if (!initialized_) {
            std::cerr << "[SimpleSWBA] Not initialized" << std::endl;
            return;
        }
        
        // Store preintegrated IMU for later optimization
        pending_imu_ = imu_data;
        
        // Propagate state using preintegrated measurements
        State prev_state = current_state_;
        Vector3 gravity(0, 0, static_cast<FLOAT>(-9.81));
        
        // Get rotation matrix - handle both attribute names
        Matrix3 delta_R = imu_data.delta_rotation;
        
        // Standard VIO prediction equations
        current_state_.rotation_matrix = prev_state.rotation_matrix * delta_R;
        current_state_.velocity = prev_state.velocity + gravity * dt + 
                                  prev_state.rotation_matrix * imu_data.delta_velocity;
        current_state_.position = prev_state.position + prev_state.velocity * dt + 
                                  static_cast<FLOAT>(0.5) * gravity * dt * dt + 
                                  prev_state.rotation_matrix * imu_data.delta_position;
        
        // Update timestamp
        current_state_.timestamp = prev_state.timestamp + dt;
        
        // Store all poses for complete trajectory
        all_poses_.push_back(current_state_);
        
        if (config_.verbose) {
            std::cout << "[SimpleSWBA] Predicted to t=" << current_state_.timestamp 
                     << ", pos: " << current_state_.position.transpose() << std::endl;
        }
    }
    
    /**
     * Update step - process ALL frames and decide on keyframes for optimization
     * 
     * @param visual_frame Processed visual measurements with Jacobians
     * @param external_landmarks Optional landmark map for initialization
     * @return Number of optimization iterations performed
     */
    int update(const simulation_io::ProcessedVisualFrameT<FLOAT>& visual_frame,
              const std::unordered_map<int, Vector3>* external_landmarks = nullptr) {
        if (!initialized_) {
            std::cerr << "[SimpleSWBA] Not initialized" << std::endl;
            return 0;
        }
        
        frame_count_++;
        
        // IMPORTANT: Process every frame, not just keyframes
        // This ensures we track poses for all camera frames
        
        // Store current pose for this frame (already updated by predict())
        State current_frame_pose = current_state_;
        
        // Initialize landmarks from external if provided
        if (external_landmarks) {
            for (const auto& [id, pos] : *external_landmarks) {
                if (landmarks_.find(id) == landmarks_.end()) {
                    landmarks_[id] = pos;
                }
            }
        }
        
        // Decide if this should be a keyframe for optimization
        bool is_keyframe = (frame_count_ % config_.keyframe_spacing == 0) || 
                          keyframes_.empty();
        
        if (config_.verbose) {
            std::cout << "[SimpleSWBA] Frame " << frame_count_ 
                     << ": is_keyframe=" << is_keyframe 
                     << ", total_keyframes=" << keyframes_.size() << std::endl;
        }
        
        int iterations = 0;
        
        if (is_keyframe) {
            // Create keyframe
            Keyframe kf;
            kf.state = current_frame_pose;
            kf.visual_frame = visual_frame;
            kf.id = keyframe_count_;
            
            keyframes_.push_back(kf);
            keyframe_ids_.push_back(keyframe_count_);
            
            // Store IMU constraint if we have pending IMU data
            if (pending_imu_ && keyframe_ids_.size() > 1) {
                int prev_id = keyframe_ids_[keyframe_ids_.size() - 2];
                int curr_id = keyframe_ids_.back();
                imu_constraints_[{prev_id, curr_id}] = pending_imu_.value();
                pending_imu_.reset();
            }
            
            // Initialize new landmarks
            initializeNewLandmarks(visual_frame);
            
            // Maintain window size
            if (keyframes_.size() > static_cast<size_t>(config_.window_size)) {
                marginalizeOldest();
            }
            
            // Optimize if we have enough keyframes
            if (keyframes_.size() >= 3) {
                iterations = optimize();
            }
            
            keyframe_count_++;
            
            if (config_.verbose) {
                std::cout << "[SimpleSWBA] Created keyframe " << keyframe_count_ << std::endl;
            }
        }
        
        return iterations;
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
     * Bundle adjustment optimization over sliding window
     * 
     * Forms constraints from:
     * 1. IMU preintegration between consecutive keyframes
     * 2. Visual reprojection to ideal coordinates
     */
    int optimize() {
        if (keyframes_.size() < 2) {
            return 0;
        }
        
        size_t num_kf = keyframes_.size();
        size_t num_lm = landmarks_.size();
        
        // State vector: [kf_positions..., landmark_positions...]
        // Simplified: only optimize positions (3 DOF each)
        int state_dim = num_kf * 3 + num_lm * 3;
        VectorX state = VectorX::Zero(state_dim);
        
        // Initialize state
        for (size_t i = 0; i < num_kf; ++i) {
            state.template segment<3>(i * 3) = keyframes_[i].state.position;
        }
        
        std::vector<int> lm_ids;
        std::unordered_map<int, int> lm_idx_map;
        for (const auto& [lid, pos] : landmarks_) {
            lm_ids.push_back(lid);
            lm_idx_map[lid] = lm_ids.size() - 1;
            state.template segment<3>(num_kf * 3 + (lm_ids.size() - 1) * 3) = pos;
        }
        
        // Gauss-Newton optimization
        int iteration;
        for (iteration = 0; iteration < config_.max_iterations; ++iteration) {
            std::vector<FLOAT> residuals;
            std::vector<Eigen::Matrix<FLOAT, -1, -1, Eigen::RowMajor>> jacobian_rows;
            
            // 1. IMU constraints
            for (const auto& [kf_pair, imu_data] : imu_constraints_) {
                int from_id = kf_pair.first;
                int to_id = kf_pair.second;
                
                // Find indices in current keyframe list
                auto from_it = std::find(keyframe_ids_.begin(), keyframe_ids_.end(), from_id);
                auto to_it = std::find(keyframe_ids_.begin(), keyframe_ids_.end(), to_id);
                
                if (from_it != keyframe_ids_.end() && to_it != keyframe_ids_.end()) {
                    size_t from_idx = std::distance(keyframe_ids_.begin(), from_it);
                    size_t to_idx = std::distance(keyframe_ids_.begin(), to_it);
                    
                    if (from_idx < num_kf && to_idx < num_kf) {
                        // IMU residual: position consistency
                        Vector3 p_i = state.template segment<3>(from_idx * 3);
                        Vector3 p_j = state.template segment<3>(to_idx * 3);
                        
                        // Get rotation from stored keyframe (not optimizing rotation here)
                        Matrix3 R_i = keyframes_[from_idx].state.rotation_matrix;
                        FLOAT dt = imu_data.delta_t;
                        Vector3 gravity(0, 0, static_cast<FLOAT>(-9.81));
                        
                        // Predicted position change
                        Vector3 predicted_p_j = p_i + keyframes_[from_idx].state.velocity * dt + 
                                               static_cast<FLOAT>(0.5) * gravity * dt * dt + 
                                               R_i * imu_data.delta_position;
                        
                        // Residual
                        Vector3 r_imu = p_j - predicted_p_j;
                        FLOAT imu_weight = static_cast<FLOAT>(10.0);  // Weight IMU constraints higher
                        
                        for (int i = 0; i < 3; ++i) {
                            residuals.push_back(r_imu(i) * imu_weight);
                        }
                        
                        // Jacobian (simplified)
                        MatrixX J_row = MatrixX::Zero(3, state_dim);
                        J_row.template block<3, 3>(0, from_idx * 3) = -Matrix3::Identity() * imu_weight;
                        J_row.template block<3, 3>(0, to_idx * 3) = Matrix3::Identity() * imu_weight;
                        jacobian_rows.push_back(J_row);
                    }
                }
            }
            
            // 2. Visual constraints
            for (size_t kf_idx = 0; kf_idx < keyframes_.size(); ++kf_idx) {
                const auto& frame = keyframes_[kf_idx].visual_frame;
                
                for (const auto& meas : frame.measurements) {
                    if (!meas.is_valid() || lm_idx_map.find(meas.landmark_id) == lm_idx_map.end()) {
                        continue;
                    }
                    
                    int lm_idx = lm_idx_map[meas.landmark_id];
                    
                    // Use ideal coordinates if available
                    FLOAT weight;
                    Vector2 r;
                    Eigen::Matrix<FLOAT, 2, 6> J_pose;
                    Eigen::Matrix<FLOAT, 2, 3> J_lm;
                    
                    if (meas.has_ideal_coordinates() && meas.ideal_residual.has_value()) {
                        r = meas.ideal_residual.value();
                        J_pose = meas.ideal_jacobian_wrt_pose.value_or(Eigen::Matrix<FLOAT, 2, 6>::Zero());
                        J_lm = meas.ideal_jacobian_wrt_landmark.value_or(Eigen::Matrix<FLOAT, 2, 3>::Zero());
                        weight = config_.ideal_coord_weight;
                    } else {
                        r = meas.residual;
                        J_pose = meas.jacobian_wrt_pose.value_or(Eigen::Matrix<FLOAT, 2, 6>::Zero());
                        J_lm = meas.jacobian_wrt_landmark.value_or(Eigen::Matrix<FLOAT, 2, 3>::Zero());
                        weight = config_.pixel_coord_weight;
                    }
                    
                    // Apply robust weight from preprocessing
                    weight *= meas.robust_weight;
                    
                    residuals.push_back(r(0) * weight);
                    residuals.push_back(r(1) * weight);
                    
                    // Build Jacobian row (only position part)
                    MatrixX J_row = MatrixX::Zero(2, state_dim);
                    J_row.template block<2, 3>(0, kf_idx * 3) = J_pose.template block<2, 3>(0, 0) * weight;
                    J_row.template block<2, 3>(0, num_kf * 3 + lm_idx * 3) = J_lm * weight;
                    jacobian_rows.push_back(J_row);
                }
            }
            
            if (residuals.empty()) {
                break;
            }
            
            // Stack and solve
            VectorX r = Eigen::Map<VectorX>(residuals.data(), residuals.size());
            MatrixX J = MatrixX::Zero(r.size(), state_dim);
            
            size_t row_offset = 0;
            for (const auto& J_row : jacobian_rows) {
                J.middleRows(row_offset, J_row.rows()) = J_row;
                row_offset += J_row.rows();
            }
            
            // Levenberg-Marquardt
            MatrixX H = J.transpose() * J + config_.damping_factor * MatrixX::Identity(state_dim, state_dim);
            VectorX g = -J.transpose() * r;
            
            // Solve
            VectorX delta = H.ldlt().solve(g);
            
            // Update state
            state += delta;
            
            // Check convergence
            if (delta.norm() < config_.convergence_threshold) {
                if (config_.verbose) {
                    std::cout << "[SimpleSWBA] Converged at iteration " << iteration << std::endl;
                }
                break;
            }
        }
        
        // Update estimates
        for (size_t i = 0; i < num_kf; ++i) {
            keyframes_[i].state.position = state.template segment<3>(i * 3);
        }
        
        for (size_t idx = 0; idx < lm_ids.size(); ++idx) {
            landmarks_[lm_ids[idx]] = state.template segment<3>(num_kf * 3 + idx * 3);
        }
        
        // Update current pose to match last keyframe
        if (!keyframes_.empty()) {
            current_state_.position = keyframes_.back().state.position;
        }
        
        total_iterations_ += iteration + 1;
        return iteration + 1;
    }
    
    /**
     * Get full estimated trajectory (all frames, not just keyframes)
     */
    std::vector<simulation_io::EstimatedPoseT<FLOAT>> getFullTrajectory() const {
        std::vector<simulation_io::EstimatedPoseT<FLOAT>> trajectory;
        
        // Use all_poses for complete trajectory
        for (const auto& state : all_poses_) {
            simulation_io::EstimatedPoseT<FLOAT> pose;
            pose.timestamp = state.timestamp;
            pose.position = state.position;
            pose.rotation_matrix = state.rotation_matrix;
            pose.velocity = state.velocity;
            trajectory.push_back(pose);
        }
        
        // If no poses yet, add current state
        if (trajectory.empty() && initialized_) {
            simulation_io::EstimatedPoseT<FLOAT> pose;
            pose.timestamp = current_state_.timestamp;
            pose.position = current_state_.position;
            pose.rotation_matrix = current_state_.rotation_matrix;
            pose.velocity = current_state_.velocity;
            trajectory.push_back(pose);
        }
        
        return trajectory;
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
    
    // Accessors for compatibility
    const State& getCurrentState() const { return current_state_; }
    Vector3 getCurrentPosition() const { return current_state_.position; }
    Matrix3 getCurrentRotation() const { return current_state_.rotation_matrix; }
    Vector3 getCurrentVelocity() const { return current_state_.velocity; }
    size_t getNumKeyframes() const { return keyframes_.size(); }
    size_t getNumLandmarks() const { return landmarks_.size(); }
    
    simulation_io::EstimatedPoseT<FLOAT> getEstimatedPose() const {
        simulation_io::EstimatedPoseT<FLOAT> pose;
        pose.timestamp = current_state_.timestamp;
        pose.position = current_state_.position;
        pose.rotation_matrix = current_state_.rotation_matrix;
        pose.velocity = current_state_.velocity;
        return pose;
    }
    
private:
    /**
     * Initialize new landmarks from measurements
     */
    void initializeNewLandmarks(const simulation_io::ProcessedVisualFrameT<FLOAT>& frame) {
        FLOAT default_depth = static_cast<FLOAT>(5.0);
        
        for (const auto& meas : frame.measurements) {
            if (landmarks_.find(meas.landmark_id) != landmarks_.end()) {
                continue;
            }
            
            Vector3 p_cam;
            
            // Use bearing vector or ideal coordinates
            if (meas.bearing_vector.has_value()) {
                p_cam = meas.bearing_vector.value() * default_depth;
            } else if (meas.observed_ideal.has_value()) {
                FLOAT x = meas.observed_ideal.value()(0);
                FLOAT y = meas.observed_ideal.value()(1);
                p_cam = Vector3(x * default_depth, y * default_depth, default_depth);
            } else {
                // Fallback to pixel with assumed intrinsics
                FLOAT fx = static_cast<FLOAT>(500.0);
                FLOAT fy = static_cast<FLOAT>(500.0);
                FLOAT cx = static_cast<FLOAT>(320.0);
                FLOAT cy = static_cast<FLOAT>(240.0);
                
                FLOAT u = meas.observed_pixel(0);
                FLOAT v = meas.observed_pixel(1);
                FLOAT x = (u - cx) / fx * default_depth;
                FLOAT y = (v - cy) / fy * default_depth;
                p_cam = Vector3(x, y, default_depth);
            }
            
            // Transform to world
            Vector3 p_world = current_state_.rotation_matrix * p_cam + current_state_.position;
            landmarks_[meas.landmark_id] = p_world;
            
            if (config_.verbose) {
                std::cout << "[SimpleSWBA] Initialized landmark " << meas.landmark_id
                         << " at " << p_world.transpose() << std::endl;
            }
        }
    }
    
    /**
     * Remove oldest keyframe and associated constraints
     */
    void marginalizeOldest() {
        if (keyframes_.size() <= static_cast<size_t>(config_.window_size)) {
            return;
        }
        
        // Remove oldest
        int old_id = keyframes_.front().id;
        keyframes_.erase(keyframes_.begin());
        keyframe_ids_.erase(keyframe_ids_.begin());
        
        // Remove associated IMU constraints
        auto it = imu_constraints_.begin();
        while (it != imu_constraints_.end()) {
            if (it->first.first == old_id || it->first.second == old_id) {
                it = imu_constraints_.erase(it);
            } else {
                ++it;
            }
        }
        
        if (config_.verbose) {
            std::cout << "[SimpleSWBA] Marginalized keyframe " << old_id << std::endl;
        }
    }
};

} // namespace estimators

#endif // ESTIMATORS_SIMPLE_SWBA_ESTIMATOR_HPP