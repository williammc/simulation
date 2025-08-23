#include <simulation_io/json_io.hpp>
#include <iostream>
#include <string>

using namespace simulation_io;

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_json_file>" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../data/trajectories/circle_easy.json" << std::endl;
        return 1;
    }
    
    std::string filepath = argv[1];
    
    try {
        std::cout << "Loading simulation data from: " << filepath << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Load the JSON file
        SimulationData data = JsonIO::load(filepath);
        
        // Display metadata
        std::cout << "\nMetadata:" << std::endl;
        std::cout << "  Version: " << data.metadata.version << std::endl;
        std::cout << "  Trajectory Type: " << data.metadata.trajectory_type << std::endl;
        std::cout << "  Duration: " << data.metadata.duration << " seconds" << std::endl;
        std::cout << "  Coordinate System: " << data.metadata.coordinate_system << std::endl;
        if (data.metadata.seed.has_value()) {
            std::cout << "  Random Seed: " << data.metadata.seed.value() << std::endl;
        }
        
        // Display calibration info
        std::cout << "\nCalibration:" << std::endl;
        std::cout << "  Cameras: " << data.camera_calibrations.size() << std::endl;
        for (const auto& cam : data.camera_calibrations) {
            std::cout << "    - " << cam.id << " (" << cam.intrinsics.width << "x" 
                      << cam.intrinsics.height << ", " << cam.intrinsics.model << ")" << std::endl;
        }
        std::cout << "  IMUs: " << data.imu_calibrations.size() << std::endl;
        for (const auto& imu : data.imu_calibrations) {
            std::cout << "    - " << imu.id << " (sampling rate: " 
                      << imu.sampling_rate << " Hz)" << std::endl;
        }
        
        // Display ground truth info
        std::cout << "\nGround Truth:" << std::endl;
        std::cout << "  Trajectory States: " << data.trajectory.size() << std::endl;
        if (!data.trajectory.empty()) {
            std::cout << "    Time Range: [" << data.trajectory.front().timestamp 
                      << ", " << data.trajectory.back().timestamp << "] seconds" << std::endl;
            
            // Show first state
            const auto& first = data.trajectory.front();
            std::cout << "    First State:" << std::endl;
            std::cout << "      t=" << first.timestamp 
                      << ", pos=[" << first.position.x << ", " 
                      << first.position.y << ", " << first.position.z << "]" << std::endl;
        }
        std::cout << "  Landmarks: " << data.landmarks.size() << std::endl;
        
        // Display measurements info
        std::cout << "\nMeasurements:" << std::endl;
        std::cout << "  IMU Measurements: " << data.imu_measurements.size() << std::endl;
        if (!data.imu_measurements.empty()) {
            double imu_rate = data.imu_measurements.size() / 
                             (data.imu_measurements.back().timestamp - data.imu_measurements.front().timestamp);
            std::cout << "    Approx. Rate: " << imu_rate << " Hz" << std::endl;
        }
        
        std::cout << "  Camera Frames: " << data.camera_frames.size() << std::endl;
        
        // Count keyframes
        int keyframe_count = 0;
        int total_observations = 0;
        for (const auto& frame : data.camera_frames) {
            if (frame.is_keyframe) {
                keyframe_count++;
            }
            total_observations += frame.observations.size();
        }
        std::cout << "    Keyframes: " << keyframe_count << std::endl;
        std::cout << "    Total Observations: " << total_observations << std::endl;
        
        // Display preintegrated IMU info if present
        if (!data.preintegrated_imu.empty()) {
            std::cout << "  Preintegrated IMU Factors: " << data.preintegrated_imu.size() << std::endl;
            std::cout << "    Between keyframes: ";
            for (size_t i = 0; i < std::min(size_t(3), data.preintegrated_imu.size()); ++i) {
                const auto& preint = data.preintegrated_imu[i];
                std::cout << "[" << preint.from_keyframe_id << "->" << preint.to_keyframe_id << "] ";
            }
            if (data.preintegrated_imu.size() > 3) {
                std::cout << "...";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Successfully loaded Python-generated simulation data!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading file: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}