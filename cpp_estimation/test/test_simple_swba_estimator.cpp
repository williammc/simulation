#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <chrono>

#include "estimators/simple_swba_estimator.hpp"
#include "simulation_io/preprocessed_interfaces.hpp"

using namespace estimators;
using namespace simulation_io;

// Test helper functions
template<typename FLOAT>
void test_basic_construction() {
    std::cout << "Testing basic construction..." << std::endl;
    
    typename SimpleSWBAEstimator<FLOAT>::Config config;
    config.window_size = 10;
    config.use_keyframes_only = false;
    
    SimpleSWBAEstimator<FLOAT> estimator(config);
    
    assert(estimator.getNumKeyframes() == 0);
    assert(estimator.getNumLandmarks() == 0);
    std::cout << "  Basic construction: PASSED" << std::endl;
}

template<typename FLOAT>
void test_initialization() {
    std::cout << "Testing initialization..." << std::endl;
    
    typename SimpleSWBAEstimator<FLOAT>::Config config;
    SimpleSWBAEstimator<FLOAT> estimator(config);
    
    // Create initial state
    Eigen::Matrix<FLOAT, 3, 1> position(1.0, 2.0, 3.0);
    Eigen::Matrix<FLOAT, 3, 3> rotation = Eigen::Matrix<FLOAT, 3, 3>::Identity();
    Eigen::Matrix<FLOAT, 3, 1> velocity(0.1, 0.0, 0.0);
    
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
void test_predict_with_preintegrated_imu() {
    std::cout << "Testing predict with preintegrated IMU..." << std::endl;
    
    typename SimpleSWBAEstimator<FLOAT>::Config config;
    SimpleSWBAEstimator<FLOAT> estimator(config);
    
    // Initialize at origin
    Eigen::Matrix<FLOAT, 3, 1> position = Eigen::Matrix<FLOAT, 3, 1>::Zero();
    Eigen::Matrix<FLOAT, 3, 3> rotation = Eigen::Matrix<FLOAT, 3, 3>::Identity();
    Eigen::Matrix<FLOAT, 3, 1> velocity = Eigen::Matrix<FLOAT, 3, 1>::Zero();
    
    estimator.initialize(position, rotation, velocity, 0.0);
    
    // Create preintegrated IMU data
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
    imu_data.delta_position = Eigen::Matrix<FLOAT, 3, 1>(0.5, 0.0, 0.0);
    imu_data.delta_velocity = Eigen::Matrix<FLOAT, 3, 1>(0.5, 0.0, 0.0);
    
    // Covariance (9x9 for position, velocity, rotation)
    imu_data.covariance = Eigen::Matrix<FLOAT, 9, 9>::Identity() * 0.001;
    imu_data.num_measurements = 10;
    
    // Predict
    estimator.predict(imu_data);
    
    // Check that position changed (should include gravity effect)
    auto new_position = estimator.getCurrentPosition();
    std::cout << "  New position: " << new_position.transpose() << std::endl;
    
    // Position should have moved due to delta_position and gravity
    // Expected: p = p0 + v0*dt + 0.5*g*dt^2 + R0*delta_p
    Eigen::Matrix<FLOAT, 3, 1> gravity(0, 0, -9.81);
    Eigen::Matrix<FLOAT, 3, 1> expected_pos = position + velocity * imu_data.delta_t + 
                                              0.5 * gravity * imu_data.delta_t * imu_data.delta_t +
                                              rotation * imu_data.delta_position;
    
    FLOAT position_error = (new_position - expected_pos).norm();
    std::cout << "  Position error (with noise): " << position_error << std::endl;
    assert(position_error < 0.1);  // Allow for some noise
    
    // Check rotation changed
    auto new_rotation = estimator.getCurrentRotation();
    assert((new_rotation - rotation).norm() > 0.01);  // Should have rotated
    
    std::cout << "  Predict with preintegrated IMU: PASSED" << std::endl;
}

template<typename FLOAT>
void test_keyframe_creation() {
    std::cout << "Testing keyframe creation..." << std::endl;
    
    typename SimpleSWBAEstimator<FLOAT>::Config config;
    config.window_size = 5;
    config.use_keyframes_only = false;
    config.keyframe_time_threshold = 0.5;
    
    SimpleSWBAEstimator<FLOAT> estimator(config);
    
    // Initialize
    estimator.initialize(Eigen::Matrix<FLOAT, 3, 1>::Zero(), 
                        Eigen::Matrix<FLOAT, 3, 3>::Identity(), 
                        Eigen::Matrix<FLOAT, 3, 1>::Zero(), 0.0);
    
    // Predict with IMU to create keyframes
    for (int i = 0; i < 3; ++i) {
        PreprocessedIMUDataT<FLOAT> imu_data;
        imu_data.from_frame_id = i;
        imu_data.to_frame_id = i + 1;
        imu_data.delta_t = 0.1;
        imu_data.delta_rotation = Eigen::Matrix<FLOAT, 3, 3>::Identity();
        imu_data.delta_position = Eigen::Matrix<FLOAT, 3, 1>(0.1, 0.0, 0.0);
        imu_data.delta_velocity = Eigen::Matrix<FLOAT, 3, 1>::Zero();
        imu_data.covariance = Eigen::Matrix<FLOAT, 9, 9>::Identity() * 0.001;
        imu_data.num_measurements = 5;
        
        estimator.predict(imu_data);
    }
    
    // Check keyframes were created
    assert(estimator.getNumKeyframes() >= 2);  // Should have created keyframes
    
    std::cout << "  Keyframe creation: PASSED (keyframes=" << estimator.getNumKeyframes() << ")" << std::endl;
}

template<typename FLOAT>
void test_visual_update() {
    std::cout << "Testing visual update..." << std::endl;
    
    typename SimpleSWBAEstimator<FLOAT>::Config config;
    config.ideal_coord_weight = 10.0;
    config.max_iterations = 5;
    
    SimpleSWBAEstimator<FLOAT> estimator(config);
    
    // Initialize
    estimator.initialize(Eigen::Matrix<FLOAT, 3, 1>(0, 0, 0),
                        Eigen::Matrix<FLOAT, 3, 3>::Identity(),
                        Eigen::Matrix<FLOAT, 3, 1>::Zero(), 0.0);
    
    // Create visual frame with measurements
    ProcessedVisualFrameT<FLOAT> visual_frame;
    visual_frame.timestamp = 0.1;
    visual_frame.frame_id = 1;
    visual_frame.is_keyframe = false;  // Let estimator decide
    
    // Add some measurements with ideal coordinates
    for (int i = 0; i < 5; ++i) {
        VisualMeasurementT<FLOAT> meas;
        meas.landmark_id = i;
        
        // Ideal coordinates
        FLOAT x_ideal = 0.1 * i;
        FLOAT y_ideal = 0.05 * i;
        meas.observed_ideal = Eigen::Matrix<FLOAT, 2, 1>(x_ideal, y_ideal);
        meas.predicted_ideal = Eigen::Matrix<FLOAT, 2, 1>(x_ideal + 0.01, y_ideal + 0.01);
        meas.ideal_residual = meas.observed_ideal.value() - meas.predicted_ideal.value();
        
        // Simple Jacobians
        meas.ideal_jacobian_wrt_pose = Eigen::Matrix<FLOAT, 2, 6>::Random() * 0.1;
        meas.ideal_jacobian_wrt_landmark = Eigen::Matrix<FLOAT, 2, 3>::Random() * 0.1;
        
        // Bearing vector
        Eigen::Matrix<FLOAT, 3, 1> bearing(x_ideal, y_ideal, 1.0);
        bearing.normalize();
        meas.bearing_vector = bearing;
        
        meas.robust_weight = 1.0;
        meas.estimated_depth = 5.0 + i;
        
        visual_frame.measurements.push_back(meas);
    }
    
    // Provide landmarks
    std::unordered_map<int, Eigen::Matrix<FLOAT, 3, 1>> landmarks;
    for (int i = 0; i < 5; ++i) {
        landmarks[i] = Eigen::Matrix<FLOAT, 3, 1>(i*1.0, i*0.5, 5.0 + i);
    }
    
    // Update
    int iterations = estimator.update(visual_frame, &landmarks);
    
    // Check results
    assert(iterations >= 0);
    assert(estimator.getNumLandmarks() > 0);
    
    std::cout << "  Visual update: PASSED (iterations=" << iterations 
              << ", landmarks=" << estimator.getNumLandmarks() << ")" << std::endl;
}

template<typename FLOAT>
void test_sliding_window() {
    std::cout << "Testing sliding window management..." << std::endl;
    
    typename SimpleSWBAEstimator<FLOAT>::Config config;
    config.window_size = 3;
    config.marginalize_old_keyframes = true;
    
    SimpleSWBAEstimator<FLOAT> estimator(config);
    
    // Initialize
    estimator.initialize(Eigen::Matrix<FLOAT, 3, 1>::Zero(),
                        Eigen::Matrix<FLOAT, 3, 3>::Identity(),
                        Eigen::Matrix<FLOAT, 3, 1>::Zero(), 0.0);
    
    // Add multiple keyframes via IMU predictions
    for (int i = 0; i < 6; ++i) {
        PreprocessedIMUDataT<FLOAT> imu_data;
        imu_data.from_frame_id = i;
        imu_data.to_frame_id = i + 1;
        imu_data.delta_t = 0.5;  // Large enough to trigger keyframe
        imu_data.delta_rotation = Eigen::Matrix<FLOAT, 3, 3>::Identity();
        imu_data.delta_position = Eigen::Matrix<FLOAT, 3, 1>(0.5, 0.0, 0.0);
        imu_data.delta_velocity = Eigen::Matrix<FLOAT, 3, 1>::Zero();
        imu_data.covariance = Eigen::Matrix<FLOAT, 9, 9>::Identity() * 0.001;
        
        estimator.predict(imu_data);
        
        // Call update with nullptr to trigger keyframe creation
        estimator.update(nullptr, nullptr);
    }
    
    // Check window size is maintained
    assert(estimator.getNumKeyframes() <= config.window_size + 1);  // Allow for one extra during transition
    
    std::cout << "  Sliding window: PASSED (keyframes=" << estimator.getNumKeyframes() << ")" << std::endl;
}

template<typename FLOAT>
void test_full_trajectory_output() {
    std::cout << "Testing full trajectory output..." << std::endl;
    
    typename SimpleSWBAEstimator<FLOAT>::Config config;
    SimpleSWBAEstimator<FLOAT> estimator(config);
    
    // Initialize
    estimator.initialize(Eigen::Matrix<FLOAT, 3, 1>::Zero(),
                        Eigen::Matrix<FLOAT, 3, 3>::Identity(),
                        Eigen::Matrix<FLOAT, 3, 1>::Zero(), 0.0);
    
    // Add some poses via IMU
    for (int i = 0; i < 5; ++i) {
        PreprocessedIMUDataT<FLOAT> imu_data;
        imu_data.from_frame_id = i;
        imu_data.to_frame_id = i + 1;
        imu_data.delta_t = 0.1;
        imu_data.delta_rotation = Eigen::Matrix<FLOAT, 3, 3>::Identity();
        imu_data.delta_position = Eigen::Matrix<FLOAT, 3, 1>(0.1, 0.0, 0.0);
        imu_data.delta_velocity = Eigen::Matrix<FLOAT, 3, 1>::Zero();
        imu_data.covariance = Eigen::Matrix<FLOAT, 9, 9>::Identity() * 0.001;
        
        estimator.predict(imu_data);
    }
    
    // Get full trajectory
    auto trajectory = estimator.getFullTrajectory();
    
    // Should have all poses
    assert(trajectory.size() >= 5);
    
    // Check timestamps are increasing
    for (size_t i = 1; i < trajectory.size(); ++i) {
        assert(trajectory[i].timestamp > trajectory[i-1].timestamp);
    }
    
    std::cout << "  Full trajectory output: PASSED (poses=" << trajectory.size() << ")" << std::endl;
}

template<typename FLOAT>
void run_all_tests() {
    std::cout << "\nTesting with precision type: " << typeid(FLOAT).name() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    test_basic_construction<FLOAT>();
    test_initialization<FLOAT>();
    test_predict_with_preintegrated_imu<FLOAT>();
    test_keyframe_creation<FLOAT>();
    test_visual_update<FLOAT>();
    test_sliding_window<FLOAT>();
    test_full_trajectory_output<FLOAT>();
    
    std::cout << "All tests passed for " << typeid(FLOAT).name() << "!\n" << std::endl;
}

int main() {
    std::cout << "=== Testing Simple SWBA Estimator ===" << std::endl;
    
    try {
        // Test with float precision
        run_all_tests<float>();
        
        // Test with double precision
        run_all_tests<double>();
        
        std::cout << "=== All Simple SWBA Estimator tests passed! ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}