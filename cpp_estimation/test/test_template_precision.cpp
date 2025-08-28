/**
 * Test suite for templated data structures with different precisions
 * Verifies that structures work correctly with both float and double
 */

#include <simulation_io/data_structures.hpp>
#include <simulation_io/estimator_result_io.hpp>
#include <iostream>
#include <type_traits>
#include <limits>
#include <cassert>
#include <cmath>

using namespace simulation_io;

// Helper to get type name as string
template<typename T>
std::string type_name() {
    if (std::is_same<T, float>::value) return "float";
    if (std::is_same<T, double>::value) return "double";
    if (std::is_same<T, long double>::value) return "long double";
    return "unknown";
}

// Test basic data structure templates
template<typename FLOAT>
void test_basic_structures() {
    std::cout << "\n=== Testing basic structures with " << type_name<FLOAT>() << " ===" << std::endl;
    
    // Test Vector3T
    {
        Vector3T<FLOAT> vec(1.5, 2.5, 3.5);
        assert(vec.x() == static_cast<FLOAT>(1.5));
        assert(vec.y() == static_cast<FLOAT>(2.5));
        assert(vec.z() == static_cast<FLOAT>(3.5));
        
        Vector3T<FLOAT> vec2 = Vector3T<FLOAT>::Zero();
        assert(vec2.norm() == 0);
        
        std::cout << "  Vector3T<" << type_name<FLOAT>() << "> OK" << std::endl;
    }
    
    // Test Matrix3x3T
    {
        Matrix3x3T<FLOAT> mat = Matrix3x3T<FLOAT>::Identity();
        assert(mat(0, 0) == static_cast<FLOAT>(1));
        assert(mat(1, 1) == static_cast<FLOAT>(1));
        assert(mat(2, 2) == static_cast<FLOAT>(1));
        assert(mat(0, 1) == static_cast<FLOAT>(0));
        
        std::cout << "  Matrix3x3T<" << type_name<FLOAT>() << "> OK" << std::endl;
    }
    
    // Test CameraIntrinsicsT
    {
        CameraIntrinsicsT<FLOAT> intrinsics;
        intrinsics.fx = static_cast<FLOAT>(500.0);
        intrinsics.fy = static_cast<FLOAT>(500.0);
        intrinsics.cx = static_cast<FLOAT>(320.0);
        intrinsics.cy = static_cast<FLOAT>(240.0);
        
        assert(intrinsics.fx == static_cast<FLOAT>(500.0));
        assert(intrinsics.model == "pinhole");
        
        std::cout << "  CameraIntrinsicsT<" << type_name<FLOAT>() << "> OK" << std::endl;
    }
    
    // Test TrajectoryStateT
    {
        TrajectoryStateT<FLOAT> state;
        state.timestamp = static_cast<FLOAT>(1.23);
        state.position = Vector3T<FLOAT>(1, 2, 3);
        state.rotation_matrix = Matrix3x3T<FLOAT>::Identity();
        state.velocity = Vector3T<FLOAT>(0.1, 0.2, 0.3);
        
        assert(state.timestamp == static_cast<FLOAT>(1.23));
        assert(state.position.x() == static_cast<FLOAT>(1));
        assert(state.velocity.has_value());
        assert(state.velocity->y() == static_cast<FLOAT>(0.2));
        
        std::cout << "  TrajectoryStateT<" << type_name<FLOAT>() << "> OK" << std::endl;
    }
    
    // Test LandmarkT
    {
        LandmarkT<FLOAT> landmark;
        landmark.id = 42;
        landmark.position = Vector3T<FLOAT>(10, 20, 30);
        
        assert(landmark.id == 42);
        assert(landmark.position.z() == static_cast<FLOAT>(30));
        assert(landmark.observation_count() == 0);
        
        // Add observation reference
        ObservationRefT<FLOAT> ref;
        ref.timestamp = static_cast<FLOAT>(2.5);
        ref.pixel_u = static_cast<FLOAT>(100);
        ref.pixel_v = static_cast<FLOAT>(200);
        landmark.observation_refs.push_back(ref);
        
        assert(landmark.observation_count() == 1);
        
        std::cout << "  LandmarkT<" << type_name<FLOAT>() << "> OK" << std::endl;
    }
    
    // Test IMUMeasurementT
    {
        IMUMeasurementT<FLOAT> imu;
        imu.timestamp = static_cast<FLOAT>(0.01);
        imu.accelerometer = Vector3T<FLOAT>(0.1, 0.2, 9.81);
        imu.gyroscope = Vector3T<FLOAT>(0.01, 0.02, 0.03);
        
        assert(imu.timestamp == static_cast<FLOAT>(0.01));
        assert(std::abs(imu.accelerometer.z() - static_cast<FLOAT>(9.81)) < 1e-6);
        
        std::cout << "  IMUMeasurementT<" << type_name<FLOAT>() << "> OK" << std::endl;
    }
    
    // Test ImagePointT
    {
        ImagePointT<FLOAT> pixel(320.5, 240.5);
        assert(pixel.u == static_cast<FLOAT>(320.5));
        assert(pixel.v == static_cast<FLOAT>(240.5));
        
        auto vec = pixel.toVector();
        assert(vec.x() == static_cast<FLOAT>(320.5));
        assert(vec.y() == static_cast<FLOAT>(240.5));
        
        auto pixel2 = ImagePointT<FLOAT>::fromVector(vec);
        assert(pixel2.u == pixel.u);
        assert(pixel2.v == pixel.v);
        
        std::cout << "  ImagePointT<" << type_name<FLOAT>() << "> OK" << std::endl;
    }
}

// Test estimator result structures
template<typename FLOAT>
void test_estimator_structures() {
    std::cout << "\n=== Testing estimator structures with " << type_name<FLOAT>() << " ===" << std::endl;
    
    // Test EstimatedPoseT
    {
        Vector3T<FLOAT> pos(1, 2, 3);
        Matrix3x3T<FLOAT> rot = Matrix3x3T<FLOAT>::Identity();
        EstimatedPoseT<FLOAT> pose(static_cast<FLOAT>(1.0), pos, rot);
        
        assert(pose.timestamp == static_cast<FLOAT>(1.0));
        assert(pose.position.x() == static_cast<FLOAT>(1));
        assert(pose.quaternion.w() == static_cast<FLOAT>(1));  // Identity rotation
        
        std::cout << "  EstimatedPoseT<" << type_name<FLOAT>() << "> OK" << std::endl;
    }
    
    // Test EstimatedLandmarkT
    {
        EstimatedLandmarkT<FLOAT> landmark(10, Vector3T<FLOAT>(5, 10, 15));
        assert(landmark.id == 10);
        assert(landmark.position.y() == static_cast<FLOAT>(10));
        
        std::cout << "  EstimatedLandmarkT<" << type_name<FLOAT>() << "> OK" << std::endl;
    }
    
    // Test EstimatedTrajectoryT
    {
        EstimatedTrajectoryT<FLOAT> trajectory;
        trajectory.frame_id = "test_frame";
        
        for (int i = 0; i < 3; ++i) {
            EstimatedPoseT<FLOAT> pose;
            pose.timestamp = static_cast<FLOAT>(i * 0.1);
            pose.position = Vector3T<FLOAT>(i, i*2, i*3);
            pose.quaternion = Vector4T<FLOAT>(0, 0, 0, 1);
            trajectory.add_pose(pose);
        }
        
        assert(trajectory.poses.size() == 3);
        assert(trajectory.poses[1].position.x() == static_cast<FLOAT>(1));
        
        std::cout << "  EstimatedTrajectoryT<" << type_name<FLOAT>() << "> OK" << std::endl;
    }
    
    // Test EstimatorResultT
    {
        EstimatorResultT<FLOAT> result;
        result.runtime_ms = static_cast<FLOAT>(123.45);
        result.iterations = 50;
        result.converged = true;
        result.final_cost = static_cast<FLOAT>(0.001);
        
        assert(result.runtime_ms == static_cast<FLOAT>(123.45));
        assert(!result.converged == false);
        
        std::cout << "  EstimatorResultT<" << type_name<FLOAT>() << "> OK" << std::endl;
    }
}

// Test precision differences
void test_precision_differences() {
    std::cout << "\n=== Testing precision differences ===" << std::endl;
    
    // Create same value in different precisions
    float f_val = 1.0f / 3.0f;
    double d_val = 1.0 / 3.0;
    
    Vector3T<float> f_vec(f_val, f_val, f_val);
    Vector3T<double> d_vec(d_val, d_val, d_val);
    
    std::cout << "  Float precision:  " << std::setprecision(10) << f_vec.x() << std::endl;
    std::cout << "  Double precision: " << std::setprecision(10) << d_vec.x() << std::endl;
    
    // Test that they're different (due to precision)
    assert(std::abs(static_cast<double>(f_vec.x()) - d_vec.x()) > 1e-8);
    
    // Test numeric limits
    std::cout << "  Float epsilon:  " << std::numeric_limits<float>::epsilon() << std::endl;
    std::cout << "  Double epsilon: " << std::numeric_limits<double>::epsilon() << std::endl;
    
    assert(std::numeric_limits<float>::epsilon() > std::numeric_limits<double>::epsilon());
    
    std::cout << "  Precision differences confirmed" << std::endl;
}

// Test type aliases for backward compatibility
void test_type_aliases() {
    std::cout << "\n=== Testing type aliases ===" << std::endl;
    
    // These should all compile and work
    SimulationData data;
    CameraCalibration cam_calib;
    IMUCalibration imu_calib;
    TrajectoryState state;
    Landmark landmark;
    IMUMeasurement imu;
    CameraFrame frame;
    PreintegratedIMUData preint;
    
    // These should be equivalent to double versions
    static_assert(std::is_same<SimulationData, SimulationDataT<double>>::value,
                  "SimulationData should be SimulationDataT<double>");
    static_assert(std::is_same<Vector3, Vector3T<double>>::value,
                  "Vector3 should be Vector3T<double>");
    static_assert(std::is_same<EstimatorResult, EstimatorResultT<double>>::value,
                  "EstimatorResult should be EstimatorResultT<double>");
    
    std::cout << "  All type aliases work correctly" << std::endl;
}

// Test mixed precision usage
void test_mixed_precision() {
    std::cout << "\n=== Testing mixed precision usage ===" << std::endl;
    
    // Create float simulation data
    SimulationDataT<float> float_data;
    float_data.metadata.duration = 10.5f;
    
    TrajectoryStateT<float> f_state;
    f_state.timestamp = 1.0f;
    f_state.position = Vector3T<float>(1.0f, 2.0f, 3.0f);
    float_data.trajectory.push_back(f_state);
    
    // Create double estimator result
    EstimatorResultT<double> double_result;
    double_result.runtime_ms = 123.456;
    
    // Convert float position to double for estimator
    EstimatedPoseT<double> d_pose;
    d_pose.timestamp = static_cast<double>(f_state.timestamp);
    d_pose.position = Vector3T<double>(
        static_cast<double>(f_state.position.x()),
        static_cast<double>(f_state.position.y()),
        static_cast<double>(f_state.position.z())
    );
    d_pose.quaternion = Vector4T<double>(0, 0, 0, 1);
    double_result.trajectory.add_pose(d_pose);
    
    assert(double_result.trajectory.poses.size() == 1);
    assert(double_result.trajectory.poses[0].position.x() == 1.0);
    
    std::cout << "  Mixed precision usage works correctly" << std::endl;
}

// Test memory efficiency
void test_memory_efficiency() {
    std::cout << "\n=== Testing memory efficiency ===" << std::endl;
    
    std::cout << "  Size of Vector3T<float>:  " << sizeof(Vector3T<float>) << " bytes" << std::endl;
    std::cout << "  Size of Vector3T<double>: " << sizeof(Vector3T<double>) << " bytes" << std::endl;
    
    std::cout << "  Size of TrajectoryStateT<float>:  " << sizeof(TrajectoryStateT<float>) << " bytes" << std::endl;
    std::cout << "  Size of TrajectoryStateT<double>: " << sizeof(TrajectoryStateT<double>) << " bytes" << std::endl;
    
    std::cout << "  Size of EstimatorResultT<float>:  " << sizeof(EstimatorResultT<float>) << " bytes" << std::endl;
    std::cout << "  Size of EstimatorResultT<double>: " << sizeof(EstimatorResultT<double>) << " bytes" << std::endl;
    
    // Float versions should be smaller (roughly half for numeric fields)
    assert(sizeof(Vector3T<float>) < sizeof(Vector3T<double>));
    assert(sizeof(TrajectoryStateT<float>) < sizeof(TrajectoryStateT<double>));
    
    std::cout << "  Memory efficiency confirmed" << std::endl;
}

int main() {
    std::cout << "=== Template Precision Test Suite ===" << std::endl;
    std::cout << "Testing templated data structures with different precisions" << std::endl;
    
    try {
        // Test basic structures with both precisions
        test_basic_structures<float>();
        test_basic_structures<double>();
        
        // Test estimator structures with both precisions
        test_estimator_structures<float>();
        test_estimator_structures<double>();
        
        // Test precision-specific features
        test_precision_differences();
        test_type_aliases();
        test_mixed_precision();
        test_memory_efficiency();
        
        std::cout << "\n=== All template precision tests passed! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}