#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "estimators/swba_estimator.hpp"
#include "simulation_io/preprocessed_interfaces.hpp"
#include "simulation_io/estimator_result_io.hpp"

using namespace estimators;
using namespace simulation_io;

int main(int argc, char* argv[]) {
    std::cout << "=== SWBA Estimator Example ===" << std::endl;
    std::cout << "Demonstrating camera-model-independent SWBA with ideal coordinates\n" << std::endl;
    
    // Configure the estimator
    SWBAEstimator<double>::Config config;
    config.window_size = 5;
    config.use_ideal_coordinates = true;
    config.ideal_coordinate_weight = 10.0;
    config.verbose = true;
    
    // Create estimator
    SWBAEstimator<double> estimator(config);
    
    // Initialize at origin
    Eigen::Vector3d initial_position(0, 0, 0);
    Eigen::Matrix3d initial_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d initial_velocity(0, 0, 0);
    estimator.initialize(initial_position, initial_rotation, initial_velocity, 0.0);
    
    // Setup for generating synthetic data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise(0.0, 0.01);
    
    // Create some synthetic landmarks in a circle around the robot
    std::map<int, EstimatedLandmarkT<double>> true_landmarks;
    int num_landmarks = 10;
    double radius = 5.0;
    
    for (int i = 0; i < num_landmarks; ++i) {
        double angle = 2 * M_PI * i / num_landmarks;
        EstimatedLandmarkT<double> lmk;
        lmk.id = i;
        lmk.position = Eigen::Vector3d(radius * cos(angle), radius * sin(angle), 2.0);
        true_landmarks[i] = lmk;
        
        std::cout << "Landmark " << i << " at: " << lmk.position.transpose() << std::endl;
    }
    
    std::cout << "\nSimulating robot motion in a circle...\n" << std::endl;
    
    // Result storage
    EstimatorResultT<double> result;
    result.estimator_type = EstimatorType::SWBA;
    
    // Simulate circular motion
    double motion_radius = 2.0;
    double angular_velocity = 0.2;  // rad/s
    double dt = 0.1;
    int num_steps = 50;
    
    for (int step = 0; step < num_steps; ++step) {
        double t = step * dt;
        double theta = angular_velocity * t;
        
        // Generate IMU data (simplified)
        PreprocessedIMUDataT<double> imu_data;
        imu_data.delta_t = dt;
        
        // Rotation around Z axis
        double dtheta = angular_velocity * dt;
        imu_data.delta_rotation << cos(dtheta), -sin(dtheta), 0,
                                   sin(dtheta),  cos(dtheta), 0,
                                   0, 0, 1;
        
        // Velocity and position changes
        Eigen::Vector3d velocity_body(motion_radius * angular_velocity, 0, 0);
        imu_data.delta_velocity = velocity_body;
        imu_data.delta_position = velocity_body * dt;
        imu_data.covariance = Eigen::Matrix<double, 9, 9>::Identity() * 0.001;
        
        // Predict with IMU
        estimator.predict(imu_data, dt);
        
        // Every 5 steps, generate visual measurements
        if (step % 5 == 0) {
            ProcessedVisualFrameT<double> visual_frame;
            visual_frame.timestamp = t;
            visual_frame.frame_id = step;
            visual_frame.is_keyframe = true;
            visual_frame.keyframe_id = step / 5;
            
            // Get current pose
            auto current_pos = estimator.getCurrentPosition();
            auto current_rot = estimator.getCurrentRotation();
            
            // Generate measurements for visible landmarks
            for (const auto& [id, lmk] : true_landmarks) {
                // Transform landmark to camera frame
                Eigen::Vector3d lmk_cam = current_rot.transpose() * (lmk.position - current_pos);
                
                // Check if landmark is in front of camera
                if (lmk_cam.z() <= 0) continue;
                
                // Generate ideal/normalized coordinates
                double x_ideal = lmk_cam.x() / lmk_cam.z();
                double y_ideal = lmk_cam.y() / lmk_cam.z();
                
                VisualMeasurementT<double> meas;
                meas.landmark_id = id;
                
                // Add measurement noise
                double x_obs = x_ideal + noise(gen);
                double y_obs = y_ideal + noise(gen);
                
                meas.observed_ideal = Eigen::Vector2d(x_obs, y_obs);
                meas.predicted_ideal = Eigen::Vector2d(x_ideal, y_ideal);
                meas.ideal_residual = meas.observed_ideal.value() - meas.predicted_ideal.value();
                
                // Approximate Jacobians
                Eigen::Matrix<double, 2, 6> J_pose = Eigen::Matrix<double, 2, 6>::Zero();
                J_pose(0, 0) = 1.0 / lmk_cam.z();
                J_pose(1, 1) = 1.0 / lmk_cam.z();
                J_pose(0, 2) = -x_ideal / lmk_cam.z();
                J_pose(1, 2) = -y_ideal / lmk_cam.z();
                meas.ideal_jacobian_wrt_pose = J_pose;
                
                // Bearing vector
                Eigen::Vector3d bearing(x_obs, y_obs, 1.0);
                bearing.normalize();
                meas.bearing_vector = bearing;
                
                meas.robust_weight = 1.0;
                meas.estimated_depth = lmk_cam.z();
                
                visual_frame.measurements.push_back(meas);
            }
            
            std::cout << "Step " << step << ": " << visual_frame.measurements.size() 
                     << " visual measurements" << std::endl;
            
            // Update with visual measurements
            int iterations = estimator.update(visual_frame, true_landmarks);
            std::cout << "  Optimization iterations: " << iterations << std::endl;
            
            // Save pose to result
            result.trajectory.add_pose(estimator.getEstimatedPose());
        }
    }
    
    // Get final estimated landmarks
    auto estimated_landmarks = estimator.getEstimatedLandmarks();
    for (const auto& [id, lmk] : estimated_landmarks) {
        result.landmarks.add_landmark(lmk);
    }
    
    // Compute errors
    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "Number of keyframes: " << estimator.getNumKeyframes() << std::endl;
    std::cout << "Number of landmarks: " << estimator.getNumLandmarks() << std::endl;
    
    // Compute landmark errors
    double total_landmark_error = 0;
    int num_compared = 0;
    for (const auto& [id, est_lmk] : estimated_landmarks) {
        if (true_landmarks.find(id) != true_landmarks.end()) {
            double error = (est_lmk.position - true_landmarks[id].position).norm();
            total_landmark_error += error;
            num_compared++;
            std::cout << "Landmark " << id << " error: " << error << " m" << std::endl;
        }
    }
    
    if (num_compared > 0) {
        std::cout << "\nMean landmark error: " << total_landmark_error / num_compared << " m" << std::endl;
    }
    
    // Save result to JSON
    std::string output_file = "swba_example_result.json";
    result.runtime_ms = 0;  // Not measured in this example
    result.iterations = 0;
    result.converged = true;
    result.final_cost = 0;
    
    try {
        EstimatorResultIO::save(result, output_file);
        std::cout << "\nResults saved to: " << output_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving results: " << e.what() << std::endl;
    }
    
    std::cout << "\n=== Example Complete ===" << std::endl;
    
    return 0;
}