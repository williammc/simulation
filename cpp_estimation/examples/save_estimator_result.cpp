#include <simulation_io/estimator_result_io.hpp>
#include <iostream>
#include <cmath>

using namespace simulation_io;

int main() {
    std::cout << "Creating sample estimator result..." << std::endl;
    
    // Create an estimator result
    EstimatorResult result;
    result.estimator_type = EstimatorType::EKF;
    result.runtime_ms = 1523.4;
    result.iterations = 42;
    result.converged = true;
    result.final_cost = 0.0123;
    
    // Add simulation metadata
    result.input_file = "data/trajectories/circle_easy.json";
    result.trajectory_type = "circle";
    result.simulation_duration = 20.0;
    
    // Add custom metadata
    result.metadata["cpp_version"] = "1.0.0";
    result.metadata["optimization_method"] = "Gauss-Newton";
    result.metadata["num_features_tracked"] = 150;
    
    // Create estimated trajectory
    std::cout << "Adding trajectory states..." << std::endl;
    for (int i = 0; i <= 100; ++i) {
        double t = i * 0.1;  // 0 to 10 seconds
        
        // Circular trajectory
        double radius = 5.0;
        double omega = 2.0 * M_PI / 10.0;  // One revolution in 10 seconds
        
        Vector3 position(
            radius * std::cos(omega * t),
            radius * std::sin(omega * t),
            1.0  // Constant height
        );
        
        // Rotation matrix (facing tangent direction)
        double heading = omega * t + M_PI / 2;
        Matrix3x3 rotation;
        rotation << std::cos(heading), -std::sin(heading), 0,
                    std::sin(heading),  std::cos(heading), 0,
                    0, 0, 1;
        
        EstimatedPose pose(t, position, rotation);
        
        // Add velocity
        pose.velocity = Vector3(
            -radius * omega * std::sin(omega * t),
             radius * omega * std::cos(omega * t),
            0
        );
        
        result.trajectory.add_pose(pose);
        
        // Also add to state history (every 10th state)
        if (i % 10 == 0) {
            EstimatorState state;
            state.timestamp = t;
            state.position = position;
            
            // Convert rotation to quaternion
            Eigen::Quaterniond q(rotation);
            state.quaternion = Vector4(q.x(), q.y(), q.z(), q.w());
            state.velocity = pose.velocity;
            
            // Add fake covariance diagonal (decreasing over time)
            VectorX cov_diag(9);
            double uncertainty = 0.1 * std::exp(-0.1 * t);
            cov_diag.fill(uncertainty);
            state.covariance_diagonal = cov_diag;
            
            result.state_history.push_back(state);
        }
    }
    
    // Create estimated landmarks
    std::cout << "Adding landmarks..." << std::endl;
    for (int i = 0; i < 20; ++i) {
        EstimatedLandmark landmark;
        landmark.id = i;
        
        // Place landmarks around the circle
        double angle = 2.0 * M_PI * i / 20.0;
        landmark.position = Vector3(
            8.0 * std::cos(angle),  // Slightly outside the trajectory
            8.0 * std::sin(angle),
            1.0 + 0.5 * std::sin(angle * 3)  // Varying height
        );
        
        // Add uncertainty
        Matrix3x3 cov = Matrix3x3::Identity() * 0.01;
        landmark.covariance = cov;
        
        result.landmarks.add_landmark(landmark);
    }
    
    // Save the result
    std::string output_file = "cpp_estimator_result.json";
    std::cout << "\nSaving result to " << output_file << "..." << std::endl;
    
    try {
        EstimatorResultIO::save(result, output_file);
        std::cout << "Successfully saved!" << std::endl;
        
        // Summary
        std::cout << "\nResult summary:" << std::endl;
        std::cout << "  Algorithm: " << (result.estimator_type == EstimatorType::EKF ? "EKF" : "Unknown") << std::endl;
        std::cout << "  Trajectory poses: " << result.trajectory.poses.size() << std::endl;
        std::cout << "  Landmarks: " << result.landmarks.landmarks.size() << std::endl;
        std::cout << "  State history: " << result.state_history.size() << std::endl;
        std::cout << "  Runtime: " << result.runtime_ms << " ms" << std::endl;
        std::cout << "  Converged: " << (result.converged ? "Yes" : "No") << std::endl;
        
        // Test loading it back
        std::cout << "\nTesting load..." << std::endl;
        EstimatorResult loaded = EstimatorResultIO::load(output_file);
        std::cout << "  Loaded " << loaded.trajectory.poses.size() << " poses" << std::endl;
        std::cout << "  Loaded " << loaded.landmarks.landmarks.size() << " landmarks" << std::endl;
        
        // Verify it can be loaded by Python
        std::cout << "\nThis file should be loadable by Python's EstimatorResultStorage.load_result()" << std::endl;
        std::cout << "Try: ./run.sh evaluate " << output_file << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}