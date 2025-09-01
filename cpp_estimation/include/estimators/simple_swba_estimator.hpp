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
#include <set>
#include <random>

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
        FLOAT lambda_factor = static_cast<FLOAT>(10.0);          // LM damping adjustment factor
        
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
    
    // For normally distributed noise to match Python's randn
    std::default_random_engine generator_;
    std::normal_distribution<FLOAT> pos_noise_dist_{static_cast<FLOAT>(0.0), static_cast<FLOAT>(0.01)};
    std::normal_distribution<FLOAT> vel_noise_dist_{static_cast<FLOAT>(0.0), static_cast<FLOAT>(0.001)};

    bool initialized_;
    
    // SO(3) helper functions
    static Matrix3 skew_symmetric(const Vector3& v) {
        Matrix3 S;
        S << static_cast<FLOAT>(0), -v(2), v(1),
             v(2), static_cast<FLOAT>(0), -v(0),
             -v(1), v(0), static_cast<FLOAT>(0);
        return S;
    }
    
    static Vector3 so3_log(const Matrix3& R) {
        FLOAT trace = R.trace();
        if (trace >= static_cast<FLOAT>(3.0 - 1e-6)) {
            // Near identity
            return static_cast<FLOAT>(0.5) * Vector3(R(2,1) - R(1,2), 
                                                      R(0,2) - R(2,0), 
                                                      R(1,0) - R(0,1));
        }
        
        FLOAT theta = std::acos((trace - static_cast<FLOAT>(1.0)) * static_cast<FLOAT>(0.5));
        Vector3 axis = static_cast<FLOAT>(1.0 / (2.0 * std::sin(theta))) * 
                      Vector3(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
        return theta * axis;
    }
    
    static Matrix3 so3_exp(const Vector3& w) {
        FLOAT theta = w.norm();
        if (theta < static_cast<FLOAT>(1e-6)) {
            // Small angle approximation
            return Matrix3::Identity() + skew_symmetric(w);
        }
        
        Vector3 axis = w / theta;
        Matrix3 K = skew_symmetric(axis);
        return Matrix3::Identity() + std::sin(theta) * K + 
               (static_cast<FLOAT>(1.0) - std::cos(theta)) * K * K;
    }
    
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
        
        if (config_.debug_enabled) {
            std::cout << "\n=== C++ SWBA predict() ===" << std::endl;
            std::cout << "Preintegration from frame " << preintegrated.from_frame_id 
                     << " to " << preintegrated.to_frame_id << std::endl;
            std::cout << "Delta_t: " << preintegrated.delta_t << std::endl;
            std::cout << "Delta_position: " << preintegrated.delta_position.transpose() << std::endl;
            std::cout << "Delta_velocity: " << preintegrated.delta_velocity.transpose() << std::endl;
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
            
            if (config_.debug_enabled) {
                std::cout << "Created keyframe " << kf.id << " at timestamp " << kf_timestamp << std::endl;
            }
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
                
                // Add small initialization noise to match Python (lines 310-311)
                Vector3 position_noise(pos_noise_dist_(generator_), pos_noise_dist_(generator_), pos_noise_dist_(generator_));
                Vector3 velocity_noise(vel_noise_dist_(generator_), vel_noise_dist_(generator_), vel_noise_dist_(generator_));
                
                // FIX: Restore gravity terms to match the optimization model and Python version
                to_kf->state.position = from_kf->state.position + 
                    from_kf->state.velocity * preintegrated.delta_t +
                    R_old * preintegrated.delta_position +
                    static_cast<FLOAT>(0.5) * gravity * preintegrated.delta_t * preintegrated.delta_t + // Add this back
                    position_noise;
                
                // FIX: Restore gravity terms to match the optimization model and Python version
                to_kf->state.velocity = from_kf->state.velocity +
                    R_old * preintegrated.delta_velocity +
                    gravity * preintegrated.delta_t + // Add this back
                    velocity_noise;
                
                // Rotation propagation (Python line 322)
                to_kf->state.rotation_matrix = R_old * preintegrated.delta_rotation;
                to_kf->state.timestamp = from_kf->state.timestamp + preintegrated.delta_t;
                
                if (config_.debug_enabled) {
                    std::cout << "Propagated state from keyframe " << from_kf->id 
                             << " to " << to_kf->id << std::endl;
                    std::cout << "  From position: " << from_kf->state.position.transpose() << std::endl;
                    std::cout << "  To position: " << to_kf->state.position.transpose() << std::endl;
                    std::cout << "  From velocity: " << from_kf->state.velocity.transpose() << std::endl;
                    std::cout << "  To velocity: " << to_kf->state.velocity.transpose() << std::endl;
                }
                
                // Update current state to match the latest keyframe (Python lines 326-333)
                current_state_ = to_kf->state.copy();
            }
        } else {
            if (config_.verbose) {
                std::cout << "[SimpleSWBA] Warning: Could not find source keyframe " 
                         << preintegrated.from_frame_id << " for preintegration" << std::endl;
            }
        }
        
        // Poses are now stored at the beginning of update()
        
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
        
        // Update timestamp only - don't store poses here as Python doesn't
        current_state_.timestamp = camera_frame->timestamp;
        // NOTE: Python doesn't store poses in update() - trajectory comes from keyframes only
        
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
                // Current state already stored at beginning of update()
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
    
private:
    /**
     * Helper function for optimization: computes residuals and Jacobians for a given state.
     * This is refactored from the main optimize() loop to support the LM algorithm.
     * It is marked 'const' as it does not modify the estimator's state.
     */
    void compute_residuals_and_jacobian(
        const VectorX& state,
        const std::unordered_map<int, int>& lm_idx_map,
        VectorX& r,
        MatrixX& J) const
    {
        const size_t num_kf = keyframes_.size();
        const int state_dim = state.size();

        std::vector<FLOAT> residuals_list;
        std::vector<Eigen::Matrix<FLOAT, -1, -1, Eigen::RowMajor>> jacobian_rows_list;

        // 1. IMU constraints
        for (size_t i = 0; i < keyframes_.size() - 1; ++i) {
            const Keyframe& from_kf = keyframes_[i];

            if (from_kf.imu_preintegration.has_value()) {
                const auto& preint = from_kf.imu_preintegration.value();
                int idx_i = i * 15;
                int idx_j = (i + 1) * 15;

                Vector3 p_i = state.template segment<3>(idx_i);
                Vector3 v_i = state.template segment<3>(idx_i + 3);
                Vector3 theta_i = state.template segment<3>(idx_i + 6);
                Vector3 p_j = state.template segment<3>(idx_j);
                Vector3 v_j = state.template segment<3>(idx_j + 3);
                Vector3 theta_j = state.template segment<3>(idx_j + 6);

                Matrix3 R_i = so3_exp(theta_i);
                Vector3 gravity(0, 0, static_cast<FLOAT>(-9.81));
                FLOAT dt = preint.delta_t;

                Vector3 r_p = R_i.transpose() * (p_j - p_i - v_i * dt - static_cast<FLOAT>(0.5) * gravity * dt * dt) - preint.delta_position;
                Vector3 r_v = R_i.transpose() * (v_j - v_i - gravity * dt) - preint.delta_velocity;
                Vector3 r_R = so3_log(preint.delta_rotation.transpose() * R_i.transpose() * so3_exp(theta_j));

                FLOAT imu_weight = static_cast<FLOAT>(10.0);
                for (int k = 0; k < 3; ++k) residuals_list.push_back(r_p(k) * imu_weight);
                for (int k = 0; k < 3; ++k) residuals_list.push_back(r_v(k) * imu_weight);
                for (int k = 0; k < 3; ++k) residuals_list.push_back(r_R(k) * imu_weight);

                // Skip Jacobian computation if J has zero rows (just computing cost)
                if (J.rows() > 0) {
                    MatrixX J_i = MatrixX::Zero(9, 15);
                    MatrixX J_j = MatrixX::Zero(9, 15);
                    J_i.template block<3, 3>(0, 0) = -R_i.transpose();
                    J_i.template block<3, 3>(0, 3) = -R_i.transpose() * dt;
                    J_i.template block<3, 3>(0, 6) = skew_symmetric(R_i.transpose() * (p_j - p_i - v_i * dt - static_cast<FLOAT>(0.5) * gravity * dt * dt));
                    J_j.template block<3, 3>(0, 0) = R_i.transpose();
                    J_i.template block<3, 3>(3, 3) = -R_i.transpose();
                    J_i.template block<3, 3>(3, 6) = skew_symmetric(R_i.transpose() * (v_j - v_i - gravity * dt));
                    J_j.template block<3, 3>(3, 3) = R_i.transpose();
                    J_i.template block<3, 3>(6, 6) = -Matrix3::Identity();
                    J_j.template block<3, 3>(6, 6) = Matrix3::Identity();
                    J_i *= imu_weight;
                    J_j *= imu_weight;

                    MatrixX J_row = MatrixX::Zero(9, state_dim);
                    J_row.template block<9, 15>(0, i * 15) = J_i;
                    J_row.template block<9, 15>(0, (i + 1) * 15) = J_j;
                    jacobian_rows_list.push_back(J_row);
                }
            }
        }

        // 2. Visual constraints
        for (size_t kf_idx = 0; kf_idx < keyframes_.size(); ++kf_idx) {
            const auto& kf = keyframes_[kf_idx];
            for (const auto& meas : kf.observations) {
                if (!meas.is_valid() || lm_idx_map.find(meas.landmark_id) == lm_idx_map.end()) {
                    continue;
                }
                int lm_idx = lm_idx_map.at(meas.landmark_id);

                FLOAT weight;
                Vector2 r_vis;
                Eigen::Matrix<FLOAT, 2, 6> J_pose;
                Eigen::Matrix<FLOAT, 2, 3> J_lm;

                if (meas.has_ideal_coordinates() && meas.ideal_residual.has_value()) {
                    r_vis = meas.ideal_residual.value();
                    if (J.rows() > 0) {
                        J_pose = meas.ideal_jacobian_wrt_pose.value_or(Eigen::Matrix<FLOAT, 2, 6>::Zero());
                        J_lm = meas.ideal_jacobian_wrt_landmark.value_or(Eigen::Matrix<FLOAT, 2, 3>::Zero());
                    }
                    weight = config_.ideal_coord_weight;
                } else {
                    r_vis = meas.residual;
                    if (J.rows() > 0) {
                        J_pose = meas.jacobian_wrt_pose.value_or(Eigen::Matrix<FLOAT, 2, 6>::Zero());
                        J_lm = meas.jacobian_wrt_landmark.value_or(Eigen::Matrix<FLOAT, 2, 3>::Zero());
                    }
                    weight = config_.pixel_coord_weight;
                }
                weight *= meas.robust_weight;

                residuals_list.push_back(r_vis(0) * weight);
                residuals_list.push_back(r_vis(1) * weight);

                if (J.rows() > 0) {
                    MatrixX J_row = MatrixX::Zero(2, state_dim);
                    // DEBUG: Try Python's convention - J_pose has [rotation(0:3), position(3:6)]
                    // State vector: [..., position, velocity, rotation, ...]
                    // Map: position from J_pose cols 3-5, rotation from J_pose cols 0-2
                    J_row.template block<2, 3>(0, kf_idx * 15) = J_pose.template block<2, 3>(0, 3) * weight;      // position from cols 3-5
                    J_row.template block<2, 3>(0, kf_idx * 15 + 6) = J_pose.template block<2, 3>(0, 0) * weight;  // rotation from cols 0-2
                    J_row.template block<2, 3>(0, num_kf * 15 + lm_idx * 3) = J_lm * weight;
                    
                    // Debug first visual Jacobian
                    if (config_.debug_enabled && kf_idx == 0 && lm_idx == 0) {
                        std::cout << "Visual Jacobian for first measurement:" << std::endl;
                        std::cout << "  J_pose: " << J_pose << std::endl;
                        std::cout << "  J_lm: " << J_lm << std::endl;
                        std::cout << "  Residual: " << r_vis.transpose() << std::endl;
                    }
                    
                    jacobian_rows_list.push_back(J_row);
                }
            }
        }

        if (residuals_list.empty()) {
            r.resize(0);
            if (J.rows() > 0) J.resize(0, state_dim);
            return;
        }

        r = Eigen::Map<VectorX>(residuals_list.data(), residuals_list.size());
        
        if (J.rows() > 0) {
            J.resize(r.size(), state_dim);
            J.setZero();
            size_t row_offset = 0;
            for (const auto& J_row : jacobian_rows_list) {
                J.middleRows(row_offset, J_row.rows()) = J_row;
                row_offset += J_row.rows();
            }
        }
    }

public:
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
        
        if (config_.debug_enabled) {
            std::cout << "\n=== C++ SWBA optimize() ===" << std::endl;
            std::cout << "Optimizing with " << num_kf << " keyframes and " 
                     << num_lm << " landmarks" << std::endl;
        }
        
        // State vector: [kf_states..., landmark_positions...]
        int state_dim = num_kf * 15 + num_lm * 3;  // 15 DOF per keyframe like Python!
        VectorX state = VectorX::Zero(state_dim);
        
        // Initialize state with full 15 DOF per keyframe (matching Python lines 606-615)
        for (size_t i = 0; i < num_kf; ++i) {
            int kf_idx = i * 15;
            state.template segment<3>(kf_idx) = keyframes_[i].state.position;           // position
            state.template segment<3>(kf_idx + 3) = keyframes_[i].state.velocity;       // velocity
            state.template segment<3>(kf_idx + 6) = so3_log(keyframes_[i].state.rotation_matrix); // rotation
            state.template segment<3>(kf_idx + 9) = keyframes_[i].state.bias_accel;     // accel bias
            state.template segment<3>(kf_idx + 12) = keyframes_[i].state.bias_gyro;     // gyro bias
        }
        
        std::vector<int> lm_ids;
        std::unordered_map<int, int> lm_idx_map;
        for (const auto& [lid, pos] : landmarks_) {
            lm_ids.push_back(lid);
            lm_idx_map[lid] = lm_ids.size() - 1;
            state.template segment<3>(num_kf * 15 + (lm_ids.size() - 1) * 3) = pos;  // Offset by 15*num_kf now!
        }
        
        // Levenberg-Marquardt optimization
        FLOAT lambda = config_.damping_factor;
        int iteration;
        for (iteration = 0; iteration < config_.max_iterations; ++iteration) {
            VectorX r;
            MatrixX J = MatrixX::Zero(1, state_dim);  // Initialize with non-zero size
            compute_residuals_and_jacobian(state, lm_idx_map, r, J);

            if (r.size() == 0) {
                break;
            }
            
            FLOAT cost = static_cast<FLOAT>(0.5) * r.dot(r);
            if (config_.debug_enabled && iteration == 0) {
                std::cout << "Optimization iteration 0:" << std::endl;
                std::cout << "  Number of residuals: " << r.size() << std::endl;
                std::cout << "  Initial cost: " << cost << std::endl;
                std::cout << "  Residual norm: " << r.norm() << std::endl;
            }
            
            MatrixX H = J.transpose() * J + lambda * MatrixX::Identity(state_dim, state_dim);
            VectorX g = -J.transpose() * r;
            
            VectorX delta = H.ldlt().solve(g);
            
            VectorX state_new = state + delta;

            VectorX r_new;
            MatrixX J_dummy = MatrixX::Zero(0, 0); // Empty matrix to skip Jacobian computation
            compute_residuals_and_jacobian(state_new, lm_idx_map, r_new, J_dummy);
            FLOAT cost_new = static_cast<FLOAT>(0.5) * r_new.dot(r_new);

            if (cost_new < cost) {
                state = state_new;
                lambda /= config_.lambda_factor;
            } else {
                lambda *= config_.lambda_factor;
            }
            
            if (delta.norm() < config_.convergence_threshold) {
                if (config_.verbose || config_.debug_enabled) {
                    std::cout << "[SimpleSWBA] Converged at iteration " << iteration 
                             << ", delta norm: " << delta.norm() << std::endl;
                }
                break;
            }
        }
        
        // Update estimates with full 15 DOF state (Python lines 1040-1046)
        for (size_t i = 0; i < num_kf; ++i) {
            Vector3 old_pos = keyframes_[i].state.position;
            int idx = i * 15;
            
            keyframes_[i].state.position = state.template segment<3>(idx);           // position
            keyframes_[i].state.velocity = state.template segment<3>(idx + 3);       // velocity
            keyframes_[i].state.rotation_matrix = so3_exp(state.template segment<3>(idx + 6)); // rotation
            keyframes_[i].state.bias_accel = state.template segment<3>(idx + 9);     // accel bias
            keyframes_[i].state.bias_gyro = state.template segment<3>(idx + 12);     // gyro bias
            
            if (config_.debug_enabled && i < 2) {  // Debug first two keyframes
                Vector3 pos_change = keyframes_[i].state.position - old_pos;
                std::cout << "Keyframe " << keyframes_[i].id << " position change: " 
                         << pos_change.transpose() << " (norm: " << pos_change.norm() << ")" << std::endl;
            }
        }
        
        for (size_t idx = 0; idx < lm_ids.size(); ++idx) {
            landmarks_[lm_ids[idx]] = state.template segment<3>(num_kf * 15 + idx * 3);  // Now offset by 15*num_kf
        }
        
        // CRITICAL FIX: Update the full trajectory history with the optimized states
        // This mirrors the logic in the Python implementation's _update_states_from_solution
        for (const auto& kf_in_window : keyframes_) {
            for (auto& kf_in_history : trajectory_history_) {
                if (kf_in_history.id == kf_in_window.id) {
                    kf_in_history.state = kf_in_window.state.copy();
                    break; // Found the matching keyframe, move to the next one in the window
                }
            }
        }
        
        // Update current state to match last keyframe (full state like Python line 1055)
        if (!keyframes_.empty()) {
            current_state_ = keyframes_.back().state.copy();
        }
        
        num_optimizations_++;
        return iteration + 1;
    }
    
public:
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
    
private: // Back to private
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