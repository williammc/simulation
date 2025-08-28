/**
 * Test suite for EstimatorResult I/O functionality
 * Tests saving and loading of estimator results with various data types and precisions
 */

#include <simulation_io/estimator_result_io.hpp>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <cassert>
#include <random>

using namespace simulation_io;

// Helper function to compare floating point values with tolerance
template<typename T>
bool approx_equal(T a, T b, T tolerance = 1e-6) {
    return std::abs(a - b) < tolerance;
}

// Helper function to compare Vector3
template<typename FLOAT>
bool vectors_equal(const Vector3T<FLOAT>& a, const Vector3T<FLOAT>& b, FLOAT tolerance = 1e-6) {
    return approx_equal(a.x(), b.x(), tolerance) &&
           approx_equal(a.y(), b.y(), tolerance) &&
           approx_equal(a.z(), b.z(), tolerance);
}

// Helper function to compare Vector4
template<typename FLOAT>
bool quaternions_equal(const Vector4T<FLOAT>& a, const Vector4T<FLOAT>& b, FLOAT tolerance = 1e-6) {
    return approx_equal(a.x(), b.x(), tolerance) &&
           approx_equal(a.y(), b.y(), tolerance) &&
           approx_equal(a.z(), b.z(), tolerance) &&
           approx_equal(a.w(), b.w(), tolerance);
}

// Create a sample EstimatorResult with known values
template<typename FLOAT>
EstimatorResultT<FLOAT> create_test_result() {
    EstimatorResultT<FLOAT> result;
    
    // Set basic metadata
    result.estimator_type = EstimatorType::EKF;
    result.runtime_ms = static_cast<FLOAT>(1234.56);
    result.iterations = 42;
    result.converged = true;
    result.final_cost = static_cast<FLOAT>(0.0123);
    
    // Set optional simulation info
    result.input_file = "test_input.json";
    result.trajectory_type = "figure8";
    result.simulation_duration = static_cast<FLOAT>(10.5);
    
    // Add custom metadata
    result.metadata["test_key"] = "test_value";
    result.metadata["num_features"] = 150;
    result.metadata["float_value"] = 3.14159;
    
    // Create trajectory with various poses
    for (int i = 0; i < 5; ++i) {
        FLOAT t = static_cast<FLOAT>(i * 0.1);
        
        // Create position
        Vector3T<FLOAT> position(
            static_cast<FLOAT>(i * 1.0),
            static_cast<FLOAT>(i * 2.0),
            static_cast<FLOAT>(i * 0.5)
        );
        
        // Create rotation matrix
        FLOAT angle = static_cast<FLOAT>(i * M_PI / 10.0);
        Matrix3x3T<FLOAT> rotation;
        rotation << std::cos(angle), -std::sin(angle), 0,
                    std::sin(angle),  std::cos(angle), 0,
                    0, 0, 1;
        
        EstimatedPoseT<FLOAT> pose(t, position, rotation);
        
        // Add velocity for some poses
        if (i % 2 == 0) {
            pose.velocity = Vector3T<FLOAT>(
                static_cast<FLOAT>(0.5),
                static_cast<FLOAT>(-0.3),
                static_cast<FLOAT>(0.1)
            );
        }
        
        result.trajectory.add_pose(pose);
    }
    
    // Add landmarks
    for (int i = 0; i < 3; ++i) {
        EstimatedLandmarkT<FLOAT> landmark;
        landmark.id = i + 100;  // Use different IDs
        landmark.position = Vector3T<FLOAT>(
            static_cast<FLOAT>(i * 3.0),
            static_cast<FLOAT>(i * -1.5),
            static_cast<FLOAT>(i * 0.2)
        );
        
        // Add descriptor for some landmarks
        if (i == 1) {
            VectorXT<FLOAT> descriptor(4);
            descriptor << 0.1, 0.2, 0.3, 0.4;
            landmark.descriptor = descriptor;
        }
        
        // Add covariance for some landmarks
        if (i == 2) {
            Matrix3x3T<FLOAT> cov = Matrix3x3T<FLOAT>::Identity() * 0.01;
            landmark.covariance = cov;
        }
        
        result.landmarks.add_landmark(landmark);
    }
    
    // Add state history
    for (int i = 0; i < 3; ++i) {
        EstimatorStateT<FLOAT> state;
        state.timestamp = static_cast<FLOAT>(i * 0.2);
        state.position = Vector3T<FLOAT>(
            static_cast<FLOAT>(i),
            static_cast<FLOAT>(i * 2),
            static_cast<FLOAT>(i * 3)
        );
        
        // Create quaternion
        Eigen::Quaternion<FLOAT> q = Eigen::Quaternion<FLOAT>::Identity();
        state.quaternion = Vector4T<FLOAT>(q.x(), q.y(), q.z(), q.w());
        
        // Add velocity for some states
        if (i == 1) {
            state.velocity = Vector3T<FLOAT>(1.0, 2.0, 3.0);
        }
        
        // Add covariance diagonal for some states
        if (i == 2) {
            VectorXT<FLOAT> cov_diag(9);
            cov_diag.setConstant(static_cast<FLOAT>(0.01));
            state.covariance_diagonal = cov_diag;
        }
        
        result.state_history.push_back(state);
    }
    
    return result;
}

// Test basic save and load functionality
template<typename FLOAT>
void test_basic_save_load(const std::string& test_name) {
    std::cout << "\n=== Test: " << test_name << " ===" << std::endl;
    
    // Create test result
    EstimatorResultT<FLOAT> original = create_test_result<FLOAT>();
    
    // Save to file
    std::string filename = "test_result_" + test_name + ".json";
    EstimatorResultIOT<FLOAT>::save(original, filename);
    std::cout << "Saved result to " << filename << std::endl;
    
    // Load from file
    EstimatorResultT<FLOAT> loaded = EstimatorResultIOT<FLOAT>::load(filename);
    std::cout << "Loaded result from " << filename << std::endl;
    
    // Verify basic fields
    assert(loaded.estimator_type == original.estimator_type);
    assert(approx_equal(loaded.runtime_ms, original.runtime_ms, static_cast<FLOAT>(1e-4)));
    assert(loaded.iterations == original.iterations);
    assert(loaded.converged == original.converged);
    assert(approx_equal(loaded.final_cost, original.final_cost, static_cast<FLOAT>(1e-6)));
    
    // Verify optional fields
    assert(loaded.input_file.has_value() == original.input_file.has_value());
    if (loaded.input_file.has_value()) {
        assert(loaded.input_file.value() == original.input_file.value());
    }
    
    assert(loaded.trajectory_type.has_value() == original.trajectory_type.has_value());
    if (loaded.trajectory_type.has_value()) {
        assert(loaded.trajectory_type.value() == original.trajectory_type.value());
    }
    
    assert(loaded.simulation_duration.has_value() == original.simulation_duration.has_value());
    if (loaded.simulation_duration.has_value()) {
        assert(approx_equal(loaded.simulation_duration.value(), 
                           original.simulation_duration.value(), 
                           static_cast<FLOAT>(1e-4)));
    }
    
    // Verify trajectory
    assert(loaded.trajectory.poses.size() == original.trajectory.poses.size());
    for (size_t i = 0; i < original.trajectory.poses.size(); ++i) {
        const auto& orig_pose = original.trajectory.poses[i];
        const auto& load_pose = loaded.trajectory.poses[i];
        
        assert(approx_equal(load_pose.timestamp, orig_pose.timestamp, static_cast<FLOAT>(1e-6)));
        assert(vectors_equal(load_pose.position, orig_pose.position, static_cast<FLOAT>(1e-6)));
        assert(quaternions_equal(load_pose.quaternion, orig_pose.quaternion, static_cast<FLOAT>(1e-6)));
        
        assert(load_pose.velocity.has_value() == orig_pose.velocity.has_value());
        if (orig_pose.velocity.has_value()) {
            assert(vectors_equal(load_pose.velocity.value(), 
                               orig_pose.velocity.value(), 
                               static_cast<FLOAT>(1e-6)));
        }
    }
    
    // Verify landmarks
    assert(loaded.landmarks.landmarks.size() == original.landmarks.landmarks.size());
    for (const auto& [id, orig_lm] : original.landmarks.landmarks) {
        assert(loaded.landmarks.landmarks.count(id) > 0);
        const auto& load_lm = loaded.landmarks.landmarks.at(id);
        
        assert(load_lm.id == orig_lm.id);
        assert(vectors_equal(load_lm.position, orig_lm.position, static_cast<FLOAT>(1e-6)));
    }
    
    // Clean up test file
    std::filesystem::remove(filename);
    std::cout << "Test passed!" << std::endl;
}

// Test empty result
void test_empty_result() {
    std::cout << "\n=== Test: Empty Result ===" << std::endl;
    
    EstimatorResult empty_result;
    empty_result.estimator_type = EstimatorType::UNKNOWN;
    
    std::string filename = "test_empty_result.json";
    EstimatorResultIO::save(empty_result, filename);
    
    EstimatorResult loaded = EstimatorResultIO::load(filename);
    
    assert(loaded.estimator_type == EstimatorType::UNKNOWN);
    assert(loaded.trajectory.poses.empty());
    assert(loaded.landmarks.landmarks.empty());
    assert(loaded.state_history.empty());
    
    std::filesystem::remove(filename);
    std::cout << "Test passed!" << std::endl;
}

// Test large result
void test_large_result() {
    std::cout << "\n=== Test: Large Result ===" << std::endl;
    
    EstimatorResult result;
    result.estimator_type = EstimatorType::SWBA;
    result.runtime_ms = 5678.9;
    
    // Add many poses
    const int num_poses = 1000;
    for (int i = 0; i < num_poses; ++i) {
        double t = i * 0.01;
        Vector3 pos(i * 0.1, i * 0.2, i * 0.3);
        Matrix3x3 rot = Matrix3x3::Identity();
        
        EstimatedPose pose(t, pos, rot);
        result.trajectory.add_pose(pose);
    }
    
    // Add many landmarks
    const int num_landmarks = 500;
    for (int i = 0; i < num_landmarks; ++i) {
        EstimatedLandmark lm(i, Vector3(i, i*2, i*3));
        result.landmarks.add_landmark(lm);
    }
    
    std::string filename = "test_large_result.json";
    EstimatorResultIO::save(result, filename);
    
    EstimatorResult loaded = EstimatorResultIO::load(filename);
    
    assert(loaded.trajectory.poses.size() == num_poses);
    assert(loaded.landmarks.landmarks.size() == num_landmarks);
    assert(approx_equal(loaded.runtime_ms, result.runtime_ms));
    
    // Verify a few samples
    assert(vectors_equal(loaded.trajectory.poses[0].position, 
                        result.trajectory.poses[0].position));
    assert(vectors_equal(loaded.trajectory.poses[num_poses-1].position,
                        result.trajectory.poses[num_poses-1].position));
    
    assert(loaded.landmarks.landmarks.count(0) > 0);
    assert(loaded.landmarks.landmarks.count(num_landmarks-1) > 0);
    
    std::filesystem::remove(filename);
    std::cout << "Test passed! Handled " << num_poses << " poses and " 
              << num_landmarks << " landmarks" << std::endl;
}

// Test metadata preservation
void test_metadata() {
    std::cout << "\n=== Test: Metadata Preservation ===" << std::endl;
    
    EstimatorResult result;
    result.estimator_type = EstimatorType::SRIF;
    
    // Add various types of metadata
    result.metadata["string_value"] = "test_string";
    result.metadata["int_value"] = 42;
    result.metadata["float_value"] = 3.14159;
    result.metadata["bool_value"] = true;
    result.metadata["array_value"] = json::array({1, 2, 3});
    result.metadata["object_value"] = json::object({
        {"nested_key", "nested_value"},
        {"nested_number", 123}
    });
    
    std::string filename = "test_metadata.json";
    EstimatorResultIO::save(result, filename);
    
    EstimatorResult loaded = EstimatorResultIO::load(filename);
    
    // Verify all metadata is preserved
    assert(loaded.metadata.size() == result.metadata.size());
    assert(loaded.metadata["string_value"] == "test_string");
    assert(loaded.metadata["int_value"] == 42);
    assert(loaded.metadata["bool_value"] == true);
    
    // Check float value with tolerance
    double orig_float = result.metadata["float_value"];
    double load_float = loaded.metadata["float_value"];
    assert(approx_equal(load_float, orig_float));
    
    // Check array
    assert(loaded.metadata["array_value"].is_array());
    assert(loaded.metadata["array_value"].size() == 3);
    
    // Check nested object
    assert(loaded.metadata["object_value"].is_object());
    assert(loaded.metadata["object_value"]["nested_key"] == "nested_value");
    assert(loaded.metadata["object_value"]["nested_number"] == 123);
    
    std::filesystem::remove(filename);
    std::cout << "Test passed!" << std::endl;
}

// Test cross-precision compatibility (save as float, load as double)
void test_cross_precision() {
    std::cout << "\n=== Test: Cross-Precision Compatibility ===" << std::endl;
    
    // Create and save with float precision
    EstimatorResultT<float> float_result = create_test_result<float>();
    std::string filename = "test_cross_precision.json";
    EstimatorResultIOT<float>::save(float_result, filename);
    std::cout << "Saved float precision result" << std::endl;
    
    // Load with double precision
    EstimatorResultT<double> double_result = EstimatorResultIOT<double>::load(filename);
    std::cout << "Loaded as double precision result" << std::endl;
    
    // Verify data is preserved (with float tolerance)
    assert(double_result.estimator_type == float_result.estimator_type);
    assert(approx_equal(static_cast<float>(double_result.runtime_ms), 
                       float_result.runtime_ms, 1e-4f));
    assert(double_result.iterations == float_result.iterations);
    assert(double_result.trajectory.poses.size() == float_result.trajectory.poses.size());
    assert(double_result.landmarks.landmarks.size() == float_result.landmarks.landmarks.size());
    
    std::filesystem::remove(filename);
    std::cout << "Test passed!" << std::endl;
}

// Test error handling
void test_error_handling() {
    std::cout << "\n=== Test: Error Handling ===" << std::endl;
    
    // Test loading non-existent file
    try {
        EstimatorResult result = EstimatorResultIO::load("non_existent_file.json");
        assert(false && "Should have thrown exception for non-existent file");
    } catch (const std::runtime_error& e) {
        std::cout << "Correctly caught exception for non-existent file" << std::endl;
    }
    
    // Test saving to invalid path
    try {
        EstimatorResult result;
        EstimatorResultIO::save(result, "/invalid/path/file.json");
        assert(false && "Should have thrown exception for invalid path");
    } catch (const std::runtime_error& e) {
        std::cout << "Correctly caught exception for invalid path" << std::endl;
    }
    
    std::cout << "Test passed!" << std::endl;
}

int main() {
    std::cout << "=== EstimatorResult I/O Test Suite ===" << std::endl;
    std::cout << "Testing save and load functionality for EstimatorResult" << std::endl;
    
    try {
        // Test with double precision
        test_basic_save_load<double>("double_precision");
        
        // Test with float precision
        test_basic_save_load<float>("float_precision");
        
        // Test special cases
        test_empty_result();
        test_large_result();
        test_metadata();
        test_cross_precision();
        test_error_handling();
        
        std::cout << "\n=== All tests passed! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}