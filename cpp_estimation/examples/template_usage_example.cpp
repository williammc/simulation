/**
 * Example demonstrating the use of templated data structures with different precision types.
 * This shows how to use both double and float precision versions of the data structures.
 */

#include <simulation_io/json_io.hpp>
#include <simulation_io/estimator_result_io.hpp>
#include <iostream>
#include <typeinfo>
#include <cmath>

using namespace simulation_io;

// Function template that works with any floating-point precision
template<typename FLOAT>
void create_trajectory_data() {
    std::cout << "\n=== Creating trajectory with " 
              << (std::is_same<FLOAT, double>::value ? "double" : "float") 
              << " precision ===" << std::endl;
    
    // Create simulation data with specified precision
    SimulationDataT<FLOAT> data;
    
    // Set metadata
    data.metadata.version = "1.0";
    data.metadata.trajectory_type = "template_test";
    data.metadata.duration = static_cast<FLOAT>(5.0);
    
    // Add trajectory points
    for (int i = 0; i <= 10; ++i) {
        TrajectoryStateT<FLOAT> state;
        state.timestamp = static_cast<FLOAT>(i * 0.5);
        
        // Simple linear trajectory
        state.position = Vector3T<FLOAT>(
            static_cast<FLOAT>(i),
            static_cast<FLOAT>(i * 0.5),
            static_cast<FLOAT>(1.0)
        );
        
        state.rotation_matrix = Matrix3x3T<FLOAT>::Identity();
        data.trajectory.push_back(state);
    }
    
    // Add landmarks
    for (int i = 0; i < 5; ++i) {
        LandmarkT<FLOAT> landmark;
        landmark.id = i;
        landmark.position = Vector3T<FLOAT>(
            static_cast<FLOAT>(i * 2.0),
            static_cast<FLOAT>(i * 1.5),
            static_cast<FLOAT>(0.0)
        );
        data.landmarks.push_back(landmark);
    }
    
    // Add IMU measurements
    for (int i = 0; i <= 20; ++i) {
        IMUMeasurementT<FLOAT> meas;
        meas.timestamp = static_cast<FLOAT>(i * 0.25);
        meas.accelerometer = Vector3T<FLOAT>(
            static_cast<FLOAT>(0.1),
            static_cast<FLOAT>(0.2),
            static_cast<FLOAT>(9.81)
        );
        meas.gyroscope = Vector3T<FLOAT>::Zero();
        data.imu_measurements.push_back(meas);
    }
    
    std::cout << "Created " << data.trajectory.size() << " trajectory states" << std::endl;
    std::cout << "Created " << data.landmarks.size() << " landmarks" << std::endl;
    std::cout << "Created " << data.imu_measurements.size() << " IMU measurements" << std::endl;
    std::cout << "Size of FLOAT type: " << sizeof(FLOAT) << " bytes" << std::endl;
}

// Function template for creating estimator results
template<typename FLOAT>
void create_estimator_result() {
    std::cout << "\n=== Creating estimator result with " 
              << (std::is_same<FLOAT, double>::value ? "double" : "float") 
              << " precision ===" << std::endl;
    
    EstimatorResultT<FLOAT> result;
    result.estimator_type = EstimatorType::EKF;
    result.runtime_ms = static_cast<FLOAT>(123.45);
    result.iterations = 10;
    result.converged = true;
    result.final_cost = static_cast<FLOAT>(0.001);
    
    // Add some poses
    for (int i = 0; i < 5; ++i) {
        FLOAT t = static_cast<FLOAT>(i);
        Vector3T<FLOAT> pos(t, t * 2, t * 3);
        Matrix3x3T<FLOAT> rot = Matrix3x3T<FLOAT>::Identity();
        
        EstimatedPoseT<FLOAT> pose(t, pos, rot);
        result.trajectory.add_pose(pose);
    }
    
    // Add some landmarks
    for (int i = 0; i < 3; ++i) {
        EstimatedLandmarkT<FLOAT> landmark;
        landmark.id = i;
        landmark.position = Vector3T<FLOAT>(
            static_cast<FLOAT>(i),
            static_cast<FLOAT>(i * 2),
            static_cast<FLOAT>(0)
        );
        result.landmarks.add_landmark(landmark);
    }
    
    std::cout << "Created result with " << result.trajectory.poses.size() << " poses" << std::endl;
    std::cout << "Created result with " << result.landmarks.landmarks.size() << " landmarks" << std::endl;
    std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "Final cost: " << result.final_cost << std::endl;
}

int main() {
    std::cout << "=== Template Usage Example ===" << std::endl;
    std::cout << "This example demonstrates using the templated data structures" << std::endl;
    std::cout << "with different floating-point precisions (float vs double)." << std::endl;
    
    // Use double precision (default)
    create_trajectory_data<double>();
    create_estimator_result<double>();
    
    // Use single precision (float)
    create_trajectory_data<float>();
    create_estimator_result<float>();
    
    // The type aliases still work for backward compatibility
    std::cout << "\n=== Using type aliases (backward compatibility) ===" << std::endl;
    SimulationData data;  // This is SimulationDataT<double>
    EstimatorResult result;  // This is EstimatorResultT<double>
    
    data.metadata.version = "1.0";
    result.runtime_ms = 100.0;
    
    std::cout << "SimulationData is using double: " 
              << std::is_same<SimulationData, SimulationDataT<double>>::value << std::endl;
    std::cout << "EstimatorResult is using double: " 
              << std::is_same<EstimatorResult, EstimatorResultT<double>>::value << std::endl;
    
    return 0;
}