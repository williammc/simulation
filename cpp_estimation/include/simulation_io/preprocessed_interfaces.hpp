#ifndef SIMULATION_IO_PREPROCESSED_INTERFACES_HPP
#define SIMULATION_IO_PREPROCESSED_INTERFACES_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <optional>
#include <algorithm>
#include <cmath>

namespace simulation_io {

// Template-based Eigen types
template<typename FLOAT>
using Vector2T = Eigen::Matrix<FLOAT, 2, 1>;

template<typename FLOAT>
using Vector3T = Eigen::Matrix<FLOAT, 3, 1>;

template<typename FLOAT>
using Matrix2x2T = Eigen::Matrix<FLOAT, 2, 2>;

template<typename FLOAT>
using Matrix3x3T = Eigen::Matrix<FLOAT, 3, 3>;

template<typename FLOAT>
using Matrix2x3T = Eigen::Matrix<FLOAT, 2, 3>;

template<typename FLOAT>
using Matrix2x6T = Eigen::Matrix<FLOAT, 2, 6>;

template<typename FLOAT>
using Matrix9x3T = Eigen::Matrix<FLOAT, 9, 3>;

template<typename FLOAT>
using Matrix9x9T = Eigen::Matrix<FLOAT, 9, 9>;

/**
 * Pre-processed visual measurement, camera-model independent.
 * 
 * This structure contains all information needed for visual updates
 * without requiring the estimator to perform projection operations.
 */
template<typename FLOAT>
struct VisualMeasurementT {
    // Landmark identification
    int landmark_id;
    
    // Pixel coordinates
    Vector2T<FLOAT> observed_pixel;     // Actual camera observation
    Vector2T<FLOAT> predicted_pixel;    // Predicted from current state
    Vector2T<FLOAT> residual;          // observed - predicted
    
    // Ideal/Normalized coordinates (optional, camera-model independent)
    // These are in the ideal pinhole camera plane (z=1)
    std::optional<Vector2T<FLOAT>> observed_ideal;   // Undistorted normalized coordinates
    std::optional<Vector2T<FLOAT>> predicted_ideal;  // Predicted normalized coordinates
    std::optional<Vector2T<FLOAT>> ideal_residual;   // Residual in ideal plane
    
    // Measurement uncertainty
    Matrix2x2T<FLOAT> pixel_covariance;
    std::optional<Matrix2x2T<FLOAT>> ideal_covariance;
    
    // Robust weight (1.0 for no robustification)
    FLOAT robust_weight;
    
    // Pre-computed Jacobians for optimization
    std::optional<Matrix2x6T<FLOAT>> jacobian_wrt_pose;      // w.r.t. SE(3) pose
    std::optional<Matrix2x3T<FLOAT>> jacobian_wrt_landmark;  // w.r.t. 3D landmark
    
    // Jacobians in ideal/normalized coordinates (more stable for optimization)
    std::optional<Matrix2x6T<FLOAT>> ideal_jacobian_wrt_pose;
    std::optional<Matrix2x3T<FLOAT>> ideal_jacobian_wrt_landmark;
    
    // Bearing vector for triangulation (unit vector in camera frame)
    std::optional<Vector3T<FLOAT>> bearing_vector;
    
    // Estimated depth (for triangulation)
    std::optional<FLOAT> estimated_depth;
    
    // Information matrix (inverse of covariance)
    std::optional<Matrix2x2T<FLOAT>> information_matrix;
    
    // Constructor
    VisualMeasurementT() 
        : landmark_id(-1), 
          observed_pixel(Vector2T<FLOAT>::Zero()),
          predicted_pixel(Vector2T<FLOAT>::Zero()),
          residual(Vector2T<FLOAT>::Zero()),
          pixel_covariance(Matrix2x2T<FLOAT>::Identity()),
          robust_weight(static_cast<FLOAT>(1.0)) {}
    
    // Check if measurement is valid for use in optimization
    bool is_valid() const {
        return robust_weight > 0 && 
               residual.allFinite() && 
               pixel_covariance.allFinite();
    }
    
    // Check if ideal/normalized coordinates are available
    bool has_ideal_coordinates() const {
        return observed_ideal.has_value() && 
               predicted_ideal.has_value() &&
               ideal_residual.has_value();
    }
    
    // Get weighted residual
    Vector2T<FLOAT> weighted_residual() const {
        return residual * robust_weight;
    }
    
    // Get ideal weighted residual
    Vector2T<FLOAT> ideal_weighted_residual() const {
        if (ideal_residual.has_value()) {
            return ideal_residual.value() * robust_weight;
        }
        return Vector2T<FLOAT>::Zero();
    }
};

/**
 * Container for pre-processed visual measurements from one camera frame.
 * 
 * This replaces CameraFrame for camera-model-independent estimators.
 */
template<typename FLOAT>
struct ProcessedVisualFrameT {
    FLOAT timestamp;
    int frame_id;
    bool is_keyframe;
    std::optional<int> keyframe_id;
    std::vector<VisualMeasurementT<FLOAT>> measurements;
    
    // Optional: pre-computed pose prediction for this frame
    std::optional<Vector3T<FLOAT>> predicted_position;
    std::optional<Matrix3x3T<FLOAT>> predicted_rotation;
    
    // Camera ID for multi-camera systems
    std::string camera_id;
    
    // Constructor
    ProcessedVisualFrameT() 
        : timestamp(0), 
          frame_id(-1), 
          is_keyframe(false),
          camera_id("cam0") {}
    
    // Get number of measurements
    size_t num_measurements() const {
        return measurements.size();
    }
    
    // Get only valid measurements
    std::vector<VisualMeasurementT<FLOAT>> valid_measurements() const {
        std::vector<VisualMeasurementT<FLOAT>> valid;
        std::copy_if(measurements.begin(), measurements.end(), 
                    std::back_inserter(valid),
                    [](const auto& m) { return m.is_valid(); });
        return valid;
    }
    
    // Compute total residual norm across all measurements
    FLOAT total_residual_norm() const {
        if (measurements.empty()) return static_cast<FLOAT>(0);
        
        FLOAT sum = 0;
        for (const auto& m : measurements) {
            if (m.is_valid()) {
                auto weighted = m.weighted_residual();
                sum += weighted.squaredNorm();
            }
        }
        return std::sqrt(sum);
    }
    
    // Get measurement for a specific landmark
    std::optional<VisualMeasurementT<FLOAT>> get_measurement_for_landmark(int landmark_id) const {
        auto it = std::find_if(measurements.begin(), measurements.end(),
                               [landmark_id](const auto& m) { 
                                   return m.landmark_id == landmark_id; 
                               });
        if (it != measurements.end()) {
            return *it;
        }
        return std::nullopt;
    }
    
    // Count measurements using ideal coordinates
    size_t num_ideal_measurements() const {
        return std::count_if(measurements.begin(), measurements.end(),
                            [](const auto& m) { 
                                return m.has_ideal_coordinates(); 
                            });
    }
};

/**
 * Enhanced PreintegratedIMUData with bias Jacobians
 * (Extends the existing structure in data_structures.hpp)
 */
template<typename FLOAT>
struct PreprocessedIMUDataT {
    // Frame indices
    int from_frame_id;
    int to_frame_id;
    
    // Pre-integrated changes
    Vector3T<FLOAT> delta_position;
    Vector3T<FLOAT> delta_velocity;
    Matrix3x3T<FLOAT> delta_rotation;  // Rotation matrix
    
    // Pre-computed covariance including noise model
    Matrix9x9T<FLOAT> covariance;  // [rotation, velocity, position]
    
    // Time interval
    FLOAT delta_t;
    
    // Number of integrated measurements
    int num_measurements;
    
    // Pre-computed Jacobians w.r.t. biases (optional)
    std::optional<Matrix9x3T<FLOAT>> jacobian_wrt_accel_bias;
    std::optional<Matrix9x3T<FLOAT>> jacobian_wrt_gyro_bias;
    
    // Optional bias estimates
    std::optional<Vector3T<FLOAT>> accel_bias;
    std::optional<Vector3T<FLOAT>> gyro_bias;
    
    // Constructor
    PreprocessedIMUDataT()
        : from_frame_id(-1),
          to_frame_id(-1),
          delta_position(Vector3T<FLOAT>::Zero()),
          delta_velocity(Vector3T<FLOAT>::Zero()),
          delta_rotation(Matrix3x3T<FLOAT>::Identity()),
          covariance(Matrix9x9T<FLOAT>::Identity()),
          delta_t(0),
          num_measurements(0) {}
    
    // Check if bias Jacobians are available
    bool has_bias_jacobians() const {
        return jacobian_wrt_accel_bias.has_value() && 
               jacobian_wrt_gyro_bias.has_value();
    }
    
    // Apply bias correction if Jacobians are available
    void apply_bias_correction(const Vector3T<FLOAT>& accel_bias_correction,
                               const Vector3T<FLOAT>& gyro_bias_correction) {
        if (!has_bias_jacobians()) return;
        
        // Compute correction: delta = J_a * delta_bias_a + J_g * delta_bias_g
        Eigen::Matrix<FLOAT, 9, 1> correction = 
            jacobian_wrt_accel_bias.value() * accel_bias_correction +
            jacobian_wrt_gyro_bias.value() * gyro_bias_correction;
        
        // Apply corrections
        // Note: Rotation correction needs special handling (SO3 manifold)
        // For small corrections, we can use: R' = R * exp(correction[0:3])
        // This is simplified here - full implementation would use proper SO3 operations
        
        // Extract corrections
        Vector3T<FLOAT> rotation_correction = correction.template segment<3>(0);
        Vector3T<FLOAT> velocity_correction = correction.template segment<3>(3);
        Vector3T<FLOAT> position_correction = correction.template segment<3>(6);
        
        // Apply position and velocity corrections directly
        delta_position += position_correction;
        delta_velocity += velocity_correction;
        
        // For rotation, use exponential map (simplified)
        // Full implementation would use proper SO3 exp
        if (rotation_correction.norm() > 1e-8) {
            // This is a simplified update - proper implementation needed
            // delta_rotation = delta_rotation * exp(rotation_correction)
        }
    }
};

// Type aliases for convenience (double precision by default)
using VisualMeasurement = VisualMeasurementT<double>;
using ProcessedVisualFrame = ProcessedVisualFrameT<double>;
using PreprocessedIMUData = PreprocessedIMUDataT<double>;

} // namespace simulation_io

#endif // SIMULATION_IO_PREPROCESSED_INTERFACES_HPP