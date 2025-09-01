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
#include <random>
#include <set>

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
        bool marginalize_old_keyframes = true;   // Marginalize old keyframes
        
        // Keyframe selection
        bool use_keyframes_only = false;         // Use only frames marked as keyframes
        FLOAT keyframe_time_threshold = static_cast<FLOAT>(0.5);         // Time threshold for new keyframe
        FLOAT keyframe_translation_threshold = static_cast<FLOAT>(0.2);  // Translation threshold
        FLOAT keyframe_rotation_threshold = static_cast<FLOAT>(0.2);     // Rotation threshold
        
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
              
        State copy() const {
            State s;
            s.position = position;
            s.rotation_matrix = rotation_matrix;
            s.velocity = velocity;
            s.bias_accel = bias_accel;
            s.bias_gyro = bias_gyro;
            s.timestamp = timestamp;
            return s;
        }
    };
    
    /**
     * Keyframe data (matching Python implementation)
     */
    struct Keyframe {
        int id;
        FLOAT timestamp;
        State state;
        std::vector<simulation_io::VisualMeasurementT<FLOAT>> observations;
        std::optional<simulation_io::PreprocessedIMUDataT<FLOAT>> imu_preintegration;
        
        Keyframe() : id(-1), timestamp(0) {}
        
        Keyframe(int _id, FLOAT _timestamp, const State& _state)
            : id(_id), timestamp(_timestamp), state(_state) {}
    };
    
private:
    Config config_;
    State current_state_;
    
    // All poses for complete trajectory
    std::vector<State> all_poses_;
    
    // Keyframe data for optimization window
    std::vector<Keyframe> keyframes_;
    std::vector<Keyframe> trajectory_history_;  // Full history of all keyframes
    int next_keyframe_id_;
    
    // Landmarks
    std::unordered_map<int, Vector3> landmarks_;
    std::unordered_map<int, std::vector<std::pair<int, simulation_io::VisualMeasurementT<FLOAT>>>> landmark_observations_;
    
    // Counters
    int num_optimizations_;
    FLOAT last_optimization_cost_;
    
    bool initialized_;
    
public:
    /**
     * Constructor
     */
    explicit SimpleSWBAEstimator(const Config& config = Config())
        : config_(config),
          next_keyframe_id_(0),
          num_optimizations_(0),
          last_optimization_cost_(0),
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
        trajectory_history_.clear();
        landmarks_.clear();
        landmark_observations_.clear();
        
        next_keyframe_id_ = 0;
        num_optimizations_ = 0;
        last_optimization_cost_ = 0;
        initialized_ = true;
        
        if (config_.verbose) {
            std::cout << "[SimpleSWBA] Initialized at position: " 
                     << initial_position.transpose() << std::endl;
        }
    }
    
    /**
     * Prediction step - process preintegrated IMU data (Python lines 240-336)
     * Creates keyframes proactively and propagates state correctly.
     * 
     * @param preintegrated Preintegrated IMU measurements between keyframes
     */
    void predict(const simulation_io::PreprocessedIMUDataT<FLOAT>& preintegrated) {
        if (!initialized_) {
            std::cerr << "[SimpleSWBA] Not initialized" << std::endl;
            return;
        }
        
        // Create keyframes if they don't exist yet (Python lines 252-276)
        // This happens when we have preintegrated IMU but no camera frames
        while (next_keyframe_id_ <= preintegrated.to_frame_id) {
            FLOAT kf_timestamp;
            if (next_keyframe_id_ == 0) {
                kf_timestamp = current_state_.timestamp;
            } else {
                // Use the preintegration dt to space keyframes
                kf_timestamp = current_state_.timestamp + 
                    (next_keyframe_id_ - preintegrated.from_frame_id) * preintegrated.delta_t;
            }
            
            // Create a new keyframe at the current state
            Keyframe kf(next_keyframe_id_, kf_timestamp, current_state_.copy());
            keyframes_.push_back(kf);
            trajectory_history_.push_back(kf);  // Add to full history
            next_keyframe_id_++;
        }
        
        // Find the keyframes this preintegration corresponds to (Python lines 278-286)
        Keyframe* from_kf = nullptr;
        Keyframe* to_kf = nullptr;
        
        for (auto& kf : keyframes_) {
            if (kf.id == preintegrated.from_frame_id) {
                from_kf = &kf;
            }
            if (kf.id == preintegrated.to_frame_id) {
                to_kf = &kf;
            }
        }
        
        if (from_kf != nullptr) {
            // Store preintegration with the source keyframe (Python lines 289-299)
            from_kf->imu_preintegration = preintegrated;
            
            // Update current state and to_kf state based on preintegration (Python lines 301-333)
            if (from_kf != nullptr && to_kf != nullptr) {
                // Propagate state using preintegrated deltas
                Matrix3 R_old = from_kf->state.rotation_matrix;
                Vector3 gravity(0, 0, static_cast<FLOAT>(-9.81));
                
                // Add small initialization noise to create non-zero residuals
                std::default_random_engine generator;
                std::normal_distribution<FLOAT> distribution(0.0, 1.0);
                Vector3 position_noise(distribution(generator) * static_cast<FLOAT>(0.01),
                                      distribution(generator) * static_cast<FLOAT>(0.01),
                                      distribution(generator) * static_cast<FLOAT>(0.01));
                Vector3 velocity_noise(distribution(generator) * static_cast<FLOAT>(0.001),
                                      distribution(generator) * static_cast<FLOAT>(0.001),
                                      distribution(generator) * static_cast<FLOAT>(0.001));
                
                // CORRECT position equation (Python lines 313-317)
                to_kf->state.position = from_kf->state.position + 
                    from_kf->state.velocity * preintegrated.delta_t +
                    R_old * preintegrated.delta_position +
                    static_cast<FLOAT>(0.5) * gravity * preintegrated.delta_t * preintegrated.delta_t +
                    position_noise;
                
                // CORRECT velocity equation (Python lines 318-321)
                to_kf->state.velocity = from_kf->state.velocity +
                    R_old * preintegrated.delta_velocity +
                    gravity * preintegrated.delta_t +
                    velocity_noise;
                
                // Rotation propagation (Python line 322)
                to_kf->state.rotation_matrix = R_old * preintegrated.delta_rotation;
                to_kf->state.timestamp = from_kf->state.timestamp + preintegrated.delta_t;
                
                // Update current state to match the latest keyframe (Python lines 326-333)
                current_state_ = to_kf->state.copy();
            }
        } else {
            if (config_.verbose) {
                std::cout << "[SimpleSWBA] Warning: Could not find source keyframe " 
                         << preintegrated.from_frame_id << " for preintegration" << std::endl;
            }
        }
        
        // Store all poses for complete trajectory
        all_poses_.push_back(current_state_);
        
        if (config_.verbose) {
            std::cout << "[SimpleSWBA] Predicted with preintegration from frame " 
                     << preintegrated.from_frame_id << " to " << preintegrated.to_frame_id << std::endl;
        }
    }
    
    // Overload for backward compatibility with dt parameter
    void predict(const simulation_io::PreprocessedIMUDataT<FLOAT>& imu_data, FLOAT /*dt*/) {
        predict(imu_data);
    }
    
    /**
     * Update step - process camera measurements (Python lines 337-393)
     * Decides whether to create a new keyframe and triggers optimization if needed.
     * 
     * @param camera_frame Visual measurements (can be nullptr for simplified version)
     * @param landmarks Known landmarks (for initialization)
     */
    void update(const simulation_io::ProcessedVisualFrameT<FLOAT>* camera_frame,
               const std::unordered_map<int, Vector3>* landmarks = nullptr) {
        if (!keyframes_.empty() || current_state_.timestamp > 0 || !initialized_) {
            // OK to proceed
        } else {
            std::cerr << "[SimpleSWBA] Not initialized, skipping update" << std::endl;
            return;
        }
        
        // Handle nullptr camera_frame for simplified version (Python lines 353-356)
        if (camera_frame == nullptr) {
            // Create a minimal keyframe for tracking purposes
            createMinimalKeyframe();
            return;
        }
        
        // If we have observations, add them to the most recent keyframe
        // instead of creating a new one (to avoid timestamp conflicts) (Python lines 358-375)
        if (!camera_frame->measurements.empty() && !keyframes_.empty()) {
            // Find the keyframe with matching or closest timestamp
            Keyframe* best_kf = nullptr;
            FLOAT min_time_diff = std::numeric_limits<FLOAT>::max();
            for (auto& kf : keyframes_) {
                FLOAT time_diff = std::abs(kf.timestamp - camera_frame->timestamp);
                if (time_diff < min_time_diff) {
                    min_time_diff = time_diff;
                    best_kf = &kf;
                }
            }
            
            if (best_kf && min_time_diff < static_cast<FLOAT>(0.01)) {  // Within 10ms - same keyframe
                // Add observations to existing keyframe
                addObservationsToKeyframe(best_kf, camera_frame->measurements, landmarks);
                if (config_.verbose) {
                    std::cout << "[SimpleSWBA] Added " << camera_frame->measurements.size() 
                             << " observations to keyframe " << best_kf->id << std::endl;
                }
                return;
            }
        }
        
        // Check keyframe-only processing (Python lines 377-384)
        if (config_.use_keyframes_only) {
            // Only process frames marked as keyframes
            if (!camera_frame->is_keyframe) {
                return;
            }
            // Only create new keyframe if timestamp is different
            if (keyframes_.empty() || 
                std::abs(keyframes_.back().timestamp - camera_frame->timestamp) > static_cast<FLOAT>(0.01)) {
                createKeyframe(*camera_frame, landmarks);
            }
        } else {
            // Use internal keyframe selection logic (Python lines 386-388)
            if (shouldCreateKeyframe(camera_frame->timestamp)) {
                createKeyframe(*camera_frame, landmarks);
            }
        }
        
        // Run optimization if we have enough keyframes (Python lines 390-392)
        if (keyframes_.size() >= 2) {
            optimize();
        }
    }
    
    // Overload for backward compatibility 
    int update(const simulation_io::ProcessedVisualFrameT<FLOAT>& visual_frame,
              const std::unordered_map<int, Vector3>* external_landmarks = nullptr) {
        update(&visual_frame, external_landmarks);
        return num_optimizations_;
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
            
            // 1. IMU constraints between consecutive keyframes
            for (size_t i = 0; i < keyframes_.size() - 1; ++i) {
                const Keyframe& from_kf = keyframes_[i];
                const Keyframe& to_kf = keyframes_[i + 1];
                
                if (from_kf.imu_preintegration.has_value()) {
                    const auto& preint = from_kf.imu_preintegration.value();
                    
                    // Get current state estimates
                    Vector3 p_i = state.template segment<3>(i * 3);
                    Vector3 p_j = state.template segment<3>((i + 1) * 3);
                    Vector3 v_i = from_kf.state.velocity;
                    Vector3 v_j = to_kf.state.velocity;
                    Matrix3 R_i = from_kf.state.rotation_matrix;
                    Matrix3 R_j = to_kf.state.rotation_matrix;
                    
                    Vector3 gravity(0, 0, static_cast<FLOAT>(-9.81));
                    FLOAT dt = preint.delta_t;
                    
                    // CORRECT IMU residual computation in BODY frame (matching Python lines 780-788)
                    // Position residual
                    Vector3 r_p = R_i.transpose() * 
                        (p_j - p_i - v_i * dt - static_cast<FLOAT>(0.5) * gravity * dt * dt) - 
                        preint.delta_position;
                    
                    // Velocity residual (simplified - not optimizing velocity)
                    Vector3 r_v = R_i.transpose() * (v_j - v_i - gravity * dt) - preint.delta_velocity;
                    
                    // Rotation residual (simplified - not optimizing rotation)
                    Matrix3 R_error = preint.delta_rotation.transpose() * R_i.transpose() * R_j;
                    Vector3 r_R = so3_log(R_error);
                    
                    // Weight IMU constraints
                    FLOAT imu_weight = static_cast<FLOAT>(10.0);
                    
                    // Add position residual only (simplified optimization)
                    for (int k = 0; k < 3; ++k) {
                        residuals.push_back(r_p(k) * imu_weight);
                    }
                    
                    // Jacobian for position residual w.r.t positions
                    MatrixX J_row = MatrixX::Zero(3, state_dim);
                    J_row.template block<3, 3>(0, i * 3) = -R_i.transpose() * imu_weight;
                    J_row.template block<3, 3>(0, (i + 1) * 3) = R_i.transpose() * imu_weight;
                    jacobian_rows.push_back(J_row);
                }
            }
            
            // 2. Visual constraints
            for (size_t kf_idx = 0; kf_idx < keyframes_.size(); ++kf_idx) {
                const auto& kf = keyframes_[kf_idx];
                
                for (const auto& meas : kf.observations) {
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
        
        num_optimizations_++;
        return iteration + 1;
    }
    
    /**
     * Get full estimated trajectory (all frames, not just keyframes)
     */
    std::vector<simulation_io::EstimatedPoseT<FLOAT>> getFullTrajectory() const {
        std::vector<simulation_io::EstimatedPoseT<FLOAT>> trajectory;
        
        // Use trajectory_history for complete trajectory (Python approach)
        for (const auto& kf : trajectory_history_) {
            simulation_io::EstimatedPoseT<FLOAT> pose;
            pose.timestamp = kf.timestamp;
            pose.position = kf.state.position;
            pose.rotation_matrix = kf.state.rotation_matrix;
            pose.velocity = kf.state.velocity;
            trajectory.push_back(pose);
        }
        
        // Also include all_poses if available for inter-keyframe poses
        if (!all_poses_.empty()) {
            // Merge all_poses with trajectory_history, avoiding duplicates
            std::set<FLOAT> existing_timestamps;
            for (const auto& pose : trajectory) {
                existing_timestamps.insert(pose.timestamp);
            }
            
            for (const auto& state : all_poses_) {
                if (existing_timestamps.find(state.timestamp) == existing_timestamps.end()) {
                    simulation_io::EstimatedPoseT<FLOAT> pose;
                    pose.timestamp = state.timestamp;
                    pose.position = state.position;
                    pose.rotation_matrix = state.rotation_matrix;
                    pose.velocity = state.velocity;
                    trajectory.push_back(pose);
                }
            }
            
            // Sort by timestamp
            std::sort(trajectory.begin(), trajectory.end(), 
                     [](const auto& a, const auto& b) { return a.timestamp < b.timestamp; });
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
     * Check if a new keyframe should be created (Python lines 394-431)
     */
    bool shouldCreateKeyframe(FLOAT timestamp) {
        if (keyframes_.empty()) {
            return true;
        }
        
        const Keyframe& last_kf = keyframes_.back();
        
        // Time threshold
        FLOAT time_diff = timestamp - last_kf.timestamp;
        if (time_diff > config_.keyframe_time_threshold) {
            return true;
        }
        
        // Translation threshold
        Vector3 trans_diff = current_state_.position - last_kf.state.position;
        if (trans_diff.norm() > config_.keyframe_translation_threshold) {
            return true;
        }
        
        // Rotation threshold
        Matrix3 R_diff = last_kf.state.rotation_matrix.transpose() * current_state_.rotation_matrix;
        Vector3 angle_axis = so3_log(R_diff);
        if (angle_axis.norm() > config_.keyframe_rotation_threshold) {
            return true;
        }
        
        return false;
    }
    
    /**
     * Add observations to an existing keyframe (Python lines 433-463)
     */
    void addObservationsToKeyframe(Keyframe* keyframe,
                                   const std::vector<simulation_io::VisualMeasurementT<FLOAT>>& observations,
                                   const std::unordered_map<int, Vector3>* landmarks) {
        // Add observations to keyframe
        keyframe->observations.insert(keyframe->observations.end(), observations.begin(), observations.end());
        
        // Track landmarks from observations
        for (const auto& obs : observations) {
            // Add to landmark tracking
            if (landmark_observations_.find(obs.landmark_id) == landmark_observations_.end()) {
                landmark_observations_[obs.landmark_id] = std::vector<std::pair<int, simulation_io::VisualMeasurementT<FLOAT>>>();
            }
            landmark_observations_[obs.landmark_id].push_back({keyframe->id, obs});
            
            // Initialize landmark if not known
            if (landmarks_.find(obs.landmark_id) == landmarks_.end()) {
                if (landmarks && landmarks->find(obs.landmark_id) != landmarks->end()) {
                    landmarks_[obs.landmark_id] = landmarks->at(obs.landmark_id);
                    if (config_.verbose) {
                        std::cout << "[SimpleSWBA] Initialized landmark " << obs.landmark_id << std::endl;
                    }
                }
            }
        }
    }
    
    /**
     * Create a new keyframe (Python lines 464-513)
     */
    void createKeyframe(const simulation_io::ProcessedVisualFrameT<FLOAT>& camera_frame,
                       const std::unordered_map<int, Vector3>* landmarks) {
        if (config_.verbose) {
            std::cout << "[SimpleSWBA] Creating keyframe with " << camera_frame.measurements.size() 
                     << " observations" << std::endl;
        }
        
        // Create new keyframe
        Keyframe new_kf(next_keyframe_id_, camera_frame.timestamp, current_state_.copy());
        new_kf.observations = camera_frame.measurements;
        next_keyframe_id_++;
        
        // Add landmarks from observations
        for (const auto& obs : camera_frame.measurements) {
            // Add to landmark tracking
            if (landmark_observations_.find(obs.landmark_id) == landmark_observations_.end()) {
                landmark_observations_[obs.landmark_id] = std::vector<std::pair<int, simulation_io::VisualMeasurementT<FLOAT>>>();
            }
            landmark_observations_[obs.landmark_id].push_back({new_kf.id, obs});
            
            // Initialize landmark if not known
            if (landmarks_.find(obs.landmark_id) == landmarks_.end()) {
                if (landmarks && landmarks->find(obs.landmark_id) != landmarks->end()) {
                    landmarks_[obs.landmark_id] = landmarks->at(obs.landmark_id);
                    if (config_.verbose) {
                        std::cout << "[SimpleSWBA] Initialized landmark " << obs.landmark_id << std::endl;
                    }
                }
            }
        }
        
        // Add keyframe to window and history
        keyframes_.push_back(new_kf);
        trajectory_history_.push_back(new_kf);
        
        // Marginalize old keyframe if window is full
        if (config_.marginalize_old_keyframes && keyframes_.size() > static_cast<size_t>(config_.window_size)) {
            marginalizeOldest();
        }
        
        if (config_.verbose) {
            std::cout << "[SimpleSWBA] Created keyframe " << new_kf.id 
                     << " at time " << camera_frame.timestamp << std::endl;
        }
    }
    
    /**
     * Create a minimal keyframe for simplified version without camera measurements (Python lines 515-550)
     */
    void createMinimalKeyframe() {
        // Increment timestamp slightly to ensure chronological order
        if (!keyframes_.empty()) {
            // Ensure new timestamp is after the last keyframe
            current_state_.timestamp = std::max(
                current_state_.timestamp,
                keyframes_.back().timestamp + static_cast<FLOAT>(0.001)
            );
        }
        
        // Create new keyframe with current state
        Keyframe new_kf(next_keyframe_id_, current_state_.timestamp, current_state_.copy());
        // No observations in simplified version
        
        keyframes_.push_back(new_kf);
        trajectory_history_.push_back(new_kf);  // Add to full history
        next_keyframe_id_++;
        
        // Trigger optimization if enough keyframes (use window_size/2 as threshold)
        if (keyframes_.size() >= static_cast<size_t>(std::max(2, config_.window_size / 2))) {
            if (config_.debug_enabled) {
                std::cout << "DEBUG: Triggering optimization with " << keyframes_.size() 
                         << " keyframes" << std::endl;
            }
            optimize();
        }
        
        // Marginalize old keyframe if window is full
        if (config_.marginalize_old_keyframes && keyframes_.size() > static_cast<size_t>(config_.window_size)) {
            marginalizeOldest();
        }
        
        if (config_.verbose) {
            std::cout << "[SimpleSWBA] Created minimal keyframe " << new_kf.id 
                     << " at time " << current_state_.timestamp << std::endl;
        }
    }
    
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
        
        // Remove oldest keyframe
        int old_id = keyframes_.front().id;
        keyframes_.erase(keyframes_.begin());
        
        // Remove associated landmark observations
        for (auto& [lm_id, obs_list] : landmark_observations_) {
            auto it = obs_list.begin();
            while (it != obs_list.end()) {
                if (it->first == old_id) {
                    it = obs_list.erase(it);
                } else {
                    ++it;
                }
            }
        }
        
        // Clean up landmarks with no observations
        auto lm_it = landmark_observations_.begin();
        while (lm_it != landmark_observations_.end()) {
            if (lm_it->second.empty()) {
                landmarks_.erase(lm_it->first);
                lm_it = landmark_observations_.erase(lm_it);
            } else {
                ++lm_it;
            }
        }
        
        if (config_.verbose) {
            std::cout << "[SimpleSWBA] Marginalized keyframe " << old_id << std::endl;
        }
    }
};

} // namespace estimators

#endif // ESTIMATORS_SIMPLE_SWBA_ESTIMATOR_HPP