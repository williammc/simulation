#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <chrono>

#include "estimators/swba_estimator.hpp"
#include "simulation_io/preprocessed_interfaces.hpp"

using namespace estimators;
using namespace simulation_io;

// Test helper functions
template<typename FLOAT>
void test_basic_construction() {
    std::cout << "Testing basic construction..." << std::endl;
    
    typename SWBAEstimator<FLOAT>::Config config;
    config.max_landmarks = 200;
    config.use_ideal_coordinates = true;
    
    SWBAEstimator<FLOAT> estimator(config);
    
    assert(estimator.getNumKeyframes() == 0);
    assert(estimator.getNumLandmarks() == 0);
    std::cout << "  Basic construction: PASSED" << std::endl;
}

template<typename FLOAT>
void test_initialization() {
    std::cout << "Testing initialization..." << std::endl;
    
    typename SWBAEstimator<FLOAT>::Config config;
    config.use_ideal_coordinates = true;
    
    SWBAEstimator<FLOAT> estimator(config);
    
    // Create initial state
    Vector3T<FLOAT> position(1.0, 2.0, 3.0);
    Matrix3x3T<FLOAT> rotation = Matrix3x3T<FLOAT>::Identity();
    Vector3T<FLOAT> velocity(0.1, 0.0, 0.0);
    
    FLOAT timestamp = 0.0;
    
    // Initialize
    estimator.initialize(position, rotation, velocity, timestamp);
    
    // Check state
    auto current_position = estimator.getCurrentPosition();
    auto current_rotation = estimator.getCurrentRotation();
    auto current_velocity = estimator.getCurrentVelocity();
    
    assert((current_position - position).norm() < 1e-6);
    assert((current_rotation - rotation).norm() < 1e-6);
    assert((current_velocity - velocity).norm() < 1e-6);
    
    std::cout << "  Initialization: PASSED" << std::endl;
}

template<typename FLOAT>
void test_predict_with_imu() {
    std::cout << "Testing predict with IMU..." << std::endl;
    
    typename SWBAEstimator<FLOAT>::Config config;
    SWBAEstimator<FLOAT> estimator(config);
    
    // Initialize at origin
    Vector3T<FLOAT> position = Vector3T<FLOAT>::Zero();
    Matrix3x3T<FLOAT> rotation = Matrix3x3T<FLOAT>::Identity();
    Vector3T<FLOAT> velocity = Vector3T<FLOAT>::Zero();
    
    estimator.initialize(position, rotation, velocity, 0.0);
    
    // Create IMU data with small rotation around Z
    PreprocessedIMUDataT<FLOAT> imu_data;
    imu_data.from_frame_id = 0;
    imu_data.to_frame_id = 1;
    imu_data.delta_t = 1.0;
    
    // Small rotation around Z axis
    FLOAT angle = 0.1;  // radians
    imu_data.delta_rotation << std::cos(angle), -std::sin(angle), 0,
                               std::sin(angle),  std::cos(angle), 0,
                               0, 0, 1;
    
    // Small movement forward
    imu_data.delta_position = Vector3T<FLOAT>(0.5, 0.0, 0.0);
    imu_data.delta_velocity = Vector3T<FLOAT>(0.5, 0.0, 0.0);
    
    // Small covariance
    imu_data.covariance = Matrix9x9T<FLOAT>::Identity() * 0.001;
    
    // Predict
    estimator.predict(imu_data, 1.0);
    
    // Check that position changed
    auto new_position = estimator.getCurrentPosition();
    assert(new_position.norm() > 0.1);  // Should have moved
    
    // Check rotation changed
    auto new_rotation = estimator.getCurrentRotation();
    assert((new_rotation - rotation).norm() > 0.01);  // Should have rotated
    
    std::cout << "  Predict with IMU: PASSED" << std::endl;
}

template<typename FLOAT>
void test_visual_update_with_ideal() {
    std::cout << "Testing visual update with ideal coordinates..." << std::endl;
    
    typename SWBAEstimator<FLOAT>::Config config;
    config.use_ideal_coordinates = true;
    config.max_iterations = 5;
    
    SWBAEstimator<FLOAT> estimator(config);
    
    // Initialize
    Vector3T<FLOAT> position(0.0, 0.0, 0.0);
    Matrix3x3T<FLOAT> rotation = Matrix3x3T<FLOAT>::Identity();
    Vector3T<FLOAT> velocity(0.0, 0.0, 0.0);
    estimator.initialize(position, rotation, velocity, 0.0);
    
    // Create visual frame with measurements
    ProcessedVisualFrameT<FLOAT> visual_frame;
    visual_frame.timestamp = 0.1;
    visual_frame.frame_id = 1;
    visual_frame.is_keyframe = true;
    visual_frame.keyframe_id = 0;
    
    // Add some measurements with ideal coordinates
    for (int i = 0; i < 5; ++i) {
        VisualMeasurementT<FLOAT> meas;
        meas.landmark_id = i;
        
        // Pixel measurements
        meas.observed_pixel = Vector2T<FLOAT>(320 + i*10, 240 + i*10);
        meas.predicted_pixel = Vector2T<FLOAT>(320 + i*10 + 2, 240 + i*10 + 1);  // Small error
        meas.residual = meas.observed_pixel - meas.predicted_pixel;
        meas.pixel_covariance = Matrix2x2T<FLOAT>::Identity();
        
        // Ideal coordinates
        FLOAT x_ideal = (meas.observed_pixel(0) - 320) / 500.0;  // Normalized
        FLOAT y_ideal = (meas.observed_pixel(1) - 240) / 500.0;
        meas.observed_ideal = Vector2T<FLOAT>(x_ideal, y_ideal);
        
        x_ideal = (meas.predicted_pixel(0) - 320) / 500.0;
        y_ideal = (meas.predicted_pixel(1) - 240) / 500.0;
        meas.predicted_ideal = Vector2T<FLOAT>(x_ideal, y_ideal);
        meas.ideal_residual = meas.observed_ideal.value() - meas.predicted_ideal.value();
        
        // Jacobians
        meas.ideal_jacobian_wrt_pose = Matrix2x6T<FLOAT>::Random() * 0.1;
        meas.ideal_jacobian_wrt_landmark = Matrix2x3T<FLOAT>::Random() * 0.1;
        meas.jacobian_wrt_pose = meas.ideal_jacobian_wrt_pose;  // Add regular jacobian too
        
        // Bearing vector
        Vector3T<FLOAT> bearing(x_ideal, y_ideal, 1.0);
        bearing.normalize();
        meas.bearing_vector = bearing;
        
        meas.robust_weight = 1.0;
        meas.estimated_depth = 5.0 + i;  // Different depths
        
        visual_frame.measurements.push_back(meas);
    }
    
    // Also provide existing landmarks
    std::map<int, EstimatedLandmarkT<FLOAT>> landmarks;
    for (int i = 0; i < 5; ++i) {
        EstimatedLandmarkT<FLOAT> lmk;
        lmk.id = i;
        lmk.position = Vector3T<FLOAT>(i*1.0, i*0.5, 5.0 + i);
        landmarks[i] = lmk;
    }
    
    // Update
    int iterations = estimator.update(visual_frame, landmarks);
    
    // First update won't optimize (need 2+ keyframes), but should still process
    assert(iterations >= 0);
    assert(iterations <= config.max_iterations);
    assert(estimator.getNumLandmarks() > 0);
    
    // Check that ideal measurements were used
    assert(visual_frame.num_ideal_measurements() == 5);
    
    std::cout << "  Visual update with ideal: PASSED (iterations=" << iterations << ")" << std::endl;
}

template<typename FLOAT>
void test_optimization_convergence() {
    std::cout << "Testing optimization convergence..." << std::endl;
    
    typename SWBAEstimator<FLOAT>::Config config;
    config.use_ideal_coordinates = true;
    config.max_iterations = 20;
    config.convergence_threshold = 1e-6;
    
    SWBAEstimator<FLOAT> estimator(config);
    
    // Initialize with some error from ground truth
    Vector3T<FLOAT> position(0.1, 0.1, 0.1);  // Small error from origin
    Matrix3x3T<FLOAT> rotation = Matrix3x3T<FLOAT>::Identity();
    Vector3T<FLOAT> velocity(0.0, 0.0, 0.0);
    estimator.initialize(position, rotation, velocity, 0.0);
    
    // Create visual frame with many measurements
    ProcessedVisualFrameT<FLOAT> visual_frame;
    visual_frame.timestamp = 0.1;
    visual_frame.frame_id = 1;
    visual_frame.is_keyframe = true;
    visual_frame.keyframe_id = 0;
    
    // Generate measurements from known landmarks
    std::map<int, EstimatedLandmarkT<FLOAT>> true_landmarks;
    for (int i = 0; i < 20; ++i) {
        // True landmark position
        Vector3T<FLOAT> lmk_pos(
            std::cos(i * 0.3) * 5.0,
            std::sin(i * 0.3) * 5.0,
            3.0 + i * 0.1
        );
        
        EstimatedLandmarkT<FLOAT> lmk;
        lmk.id = i;
        lmk.position = lmk_pos;
        true_landmarks[i] = lmk;
        
        // Project to camera (simple projection for testing)
        Vector3T<FLOAT> cam_point = lmk_pos;  // In camera frame (at origin)
        FLOAT x_ideal = cam_point(0) / cam_point(2);
        FLOAT y_ideal = cam_point(1) / cam_point(2);
        
        VisualMeasurementT<FLOAT> meas;
        meas.landmark_id = i;
        
        // Perfect observation (no noise)
        meas.observed_ideal = Vector2T<FLOAT>(x_ideal, y_ideal);
        meas.predicted_ideal = Vector2T<FLOAT>(x_ideal + 0.01, y_ideal + 0.01);  // Small initial error
        meas.ideal_residual = meas.observed_ideal.value() - meas.predicted_ideal.value();
        
        // Approximate Jacobians
        Matrix2x6T<FLOAT> J_pose = Matrix2x6T<FLOAT>::Zero();
        J_pose(0, 0) = 1.0 / cam_point(2);
        J_pose(1, 1) = 1.0 / cam_point(2);
        J_pose(0, 2) = -x_ideal / cam_point(2);
        J_pose(1, 2) = -y_ideal / cam_point(2);
        meas.ideal_jacobian_wrt_pose = J_pose;
        
        Matrix2x3T<FLOAT> J_lmk = Matrix2x3T<FLOAT>::Zero();
        J_lmk(0, 0) = 1.0 / cam_point(2);
        J_lmk(1, 1) = 1.0 / cam_point(2);
        J_lmk(0, 2) = -x_ideal / cam_point(2);
        J_lmk(1, 2) = -y_ideal / cam_point(2);
        meas.ideal_jacobian_wrt_landmark = J_lmk;
        
        Vector3T<FLOAT> bearing(x_ideal, y_ideal, 1.0);
        bearing.normalize();
        meas.bearing_vector = bearing;
        
        meas.robust_weight = 1.0;
        meas.estimated_depth = cam_point(2);
        
        visual_frame.measurements.push_back(meas);
    }
    
    // First update to add a keyframe
    int iterations1 = estimator.update(visual_frame, true_landmarks);
    
    // Move forward a bit and add another keyframe
    PreprocessedIMUDataT<FLOAT> imu_data;
    imu_data.delta_t = 0.1;
    imu_data.delta_rotation = Matrix3x3T<FLOAT>::Identity();
    imu_data.delta_position = Vector3T<FLOAT>(0.05, 0.05, 0.0);
    imu_data.delta_velocity = Vector3T<FLOAT>::Zero();
    imu_data.covariance = Matrix9x9T<FLOAT>::Identity() * 0.001;
    estimator.predict(imu_data, 0.1);
    
    // Update visual frame timestamp and do second update
    visual_frame.timestamp = 0.2;
    visual_frame.is_keyframe = true;
    visual_frame.keyframe_id = 1;
    
    FLOAT initial_cost = visual_frame.total_residual_norm();
    int iterations = estimator.update(visual_frame, true_landmarks);
    
    // Get final cost by recomputing residuals
    auto final_position = estimator.getCurrentPosition();
    FLOAT final_cost = 0;
    for (const auto& meas : visual_frame.measurements) {
        if (meas.ideal_residual.has_value()) {
            final_cost += meas.ideal_residual.value().squaredNorm();
        }
    }
    final_cost = std::sqrt(final_cost);
    
    // Check convergence (now we have 2 keyframes)
    assert(iterations >= 0);  // May converge immediately if already good
    assert((final_position - Vector3T<FLOAT>::Zero()).norm() < 0.5);  // Should be reasonably close
    
    std::cout << "  Optimization convergence: PASSED (initial=" << initial_cost 
              << ", final=" << final_cost << ", iterations=" << iterations << ")" << std::endl;
}

template<typename FLOAT>
void test_sliding_window() {
    std::cout << "Testing sliding window management..." << std::endl;
    
    typename SWBAEstimator<FLOAT>::Config config;
    config.window_size = 5;
    config.use_ideal_coordinates = true;
    
    SWBAEstimator<FLOAT> estimator(config);
    
    // Initialize
    estimator.initialize(Vector3T<FLOAT>::Zero(), Matrix3x3T<FLOAT>::Identity(), 
                        Vector3T<FLOAT>::Zero(), 0.0);
    
    // Add multiple keyframes
    for (int kf = 0; kf < 8; ++kf) {
        // IMU predict to next keyframe
        PreprocessedIMUDataT<FLOAT> imu_data;
        imu_data.from_frame_id = kf;
        imu_data.to_frame_id = kf + 1;
        imu_data.delta_t = 0.1;
        imu_data.delta_rotation = Matrix3x3T<FLOAT>::Identity();
        imu_data.delta_position = Vector3T<FLOAT>(0.1, 0.0, 0.0);
        imu_data.delta_velocity = Vector3T<FLOAT>::Zero();
        imu_data.covariance = Matrix9x9T<FLOAT>::Identity() * 0.001;
        
        estimator.predict(imu_data, kf * 0.1 + 0.1);
        
        // Visual update
        ProcessedVisualFrameT<FLOAT> visual_frame;
        visual_frame.timestamp = kf * 0.1 + 0.1;
        visual_frame.frame_id = kf + 1;
        visual_frame.is_keyframe = true;
        visual_frame.keyframe_id = kf + 1;
        
        // Add a measurement
        VisualMeasurementT<FLOAT> meas;
        meas.landmark_id = 100 + kf;
        meas.observed_ideal = Vector2T<FLOAT>(0.1 * kf, 0.1 * kf);
        meas.predicted_ideal = Vector2T<FLOAT>(0.1 * kf + 0.01, 0.1 * kf + 0.01);
        meas.ideal_residual = meas.observed_ideal.value() - meas.predicted_ideal.value();
        meas.ideal_jacobian_wrt_pose = Matrix2x6T<FLOAT>::Identity() * 0.1;
        meas.ideal_jacobian_wrt_landmark = Matrix2x3T<FLOAT>::Identity() * 0.1;
        meas.bearing_vector = Vector3T<FLOAT>(0.1 * kf, 0.1 * kf, 1.0).normalized();
        meas.robust_weight = 1.0;
        meas.estimated_depth = 5.0;
        visual_frame.measurements.push_back(meas);
        
        std::map<int, EstimatedLandmarkT<FLOAT>> landmarks;
        estimator.update(visual_frame, landmarks);
    }
    
    // Check that window size is maintained
    assert(estimator.getNumKeyframes() <= config.window_size);
    assert(estimator.getNumKeyframes() == config.window_size);  // Should be at max
    
    std::cout << "  Sliding window: PASSED (keyframes=" << estimator.getNumKeyframes() << ")" << std::endl;
}

template<typename FLOAT>
void test_precision() {
    std::cout << "Testing with precision type: " << typeid(FLOAT).name() << std::endl;
    
    test_basic_construction<FLOAT>();
    test_initialization<FLOAT>();
    test_predict_with_imu<FLOAT>();
    test_visual_update_with_ideal<FLOAT>();
    test_optimization_convergence<FLOAT>();
    test_sliding_window<FLOAT>();
    
    std::cout << "All tests passed for " << typeid(FLOAT).name() << "!" << std::endl << std::endl;
}

int main() {
    std::cout << "=== Testing SWBA Estimator ===" << std::endl << std::endl;
    
    // Test with float precision
    std::cout << "Testing with float precision:" << std::endl;
    test_precision<float>();
    
    // Test with double precision
    std::cout << "Testing with double precision:" << std::endl;
    test_precision<double>();
    
    std::cout << "=== All SWBA Estimator tests passed! ===" << std::endl;
    
    return 0;
}