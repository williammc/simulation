#include <simulation_io/json_io.hpp>
#include <iostream>

using namespace simulation_io;

int main() {
    // Create a simple simulation data structure
    SimulationData data;
    
    // Configure metadata
    data.metadata.version = "1.0";
    data.metadata.trajectory_type = "circular";
    data.metadata.duration = 5.0;
    data.metadata.coordinate_system = "ENU";
    
    // Add a simple trajectory
    for (int i = 0; i <= 50; ++i) {
        TrajectoryState state;
        state.timestamp = i * 0.1;
        
        // Circular motion
        double angle = 2.0 * M_PI * state.timestamp / 5.0;
        state.position = Vector3(
            10.0 * std::cos(angle),
            10.0 * std::sin(angle),
            5.0
        );
        
        // Identity rotation matrix (no rotation)
        state.rotation_matrix = Matrix3x3::Identity();
        
        data.trajectory.push_back(state);
    }
    
    // Add some landmarks
    for (int i = 0; i < 10; ++i) {
        Landmark lm;
        lm.id = i;
        lm.position = Vector3(
            i * 2.0 - 9.0,
            i * 1.5 - 7.0,
            0.0
        );
        data.landmarks.push_back(lm);
    }
    
    // Add IMU measurements
    for (int i = 0; i <= 500; ++i) {
        IMUMeasurement meas;
        meas.timestamp = i * 0.01;  // 100 Hz
        
        // Gravity vector
        meas.accelerometer = Vector3(0.0, 0.0, 9.81);
        
        // No rotation
        meas.gyroscope = Vector3::Zero();
        
        data.imu_measurements.push_back(meas);
    }
    
    // Save to file
    std::string output_file = "example_output.json";
    std::cout << "Saving simulation data to " << output_file << "..." << std::endl;
    
    try {
        JsonIO::save(data, output_file);
        std::cout << "Successfully saved!" << std::endl;
        
        // Load it back
        std::cout << "\nLoading simulation data..." << std::endl;
        SimulationData loaded = JsonIO::load(output_file);
        
        std::cout << "Successfully loaded!" << std::endl;
        std::cout << "  Trajectory states: " << loaded.trajectory.size() << std::endl;
        std::cout << "  Landmarks: " << loaded.landmarks.size() << std::endl;
        std::cout << "  IMU measurements: " << loaded.imu_measurements.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}