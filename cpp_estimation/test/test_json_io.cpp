#include <simulation_io/json_io.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cassert>

using namespace simulation_io;

// Helper function to get current timestamp as ISO string
std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

// Helper function to create sample data
SimulationData create_sample_data() {
    SimulationData data;
    
    // Set metadata
    data.metadata.version = "1.0";
    data.metadata.timestamp = get_current_timestamp();
    data.metadata.trajectory_type = "figure8";
    data.metadata.duration = 10.0;
    data.metadata.coordinate_system = "ENU";
    data.metadata.seed = 42;
    
    // Add camera calibration
    CameraCalibration cam_calib;
    cam_calib.id = "cam0";
    cam_calib.intrinsics.model = "pinhole";
    cam_calib.intrinsics.width = 640;
    cam_calib.intrinsics.height = 480;
    cam_calib.intrinsics.fx = 500.0;
    cam_calib.intrinsics.fy = 500.0;
    cam_calib.intrinsics.cx = 320.0;
    cam_calib.intrinsics.cy = 240.0;
    cam_calib.intrinsics.distortion = {0.1, -0.2, 0.0, 0.0, 0.0};
    // Identity transformation for simplicity
    cam_calib.T_BC = Matrix4x4();
    data.camera_calibrations.push_back(cam_calib);
    
    // Add IMU calibration
    IMUCalibration imu_calib;
    imu_calib.id = "imu0";
    imu_calib.accelerometer.noise_density = 0.01;
    imu_calib.accelerometer.random_walk = 0.001;
    imu_calib.gyroscope.noise_density = 0.005;
    imu_calib.gyroscope.random_walk = 0.0005;
    imu_calib.sampling_rate = 200.0;
    data.imu_calibrations.push_back(imu_calib);
    
    // Add trajectory states
    for (int i = 0; i <= 100; ++i) {
        double t = i * 0.1;  // 0 to 10 seconds
        TrajectoryState state;
        state.timestamp = t;
        
        // Figure-8 trajectory
        double omega = 2.0 * M_PI / 10.0;  // One cycle in 10 seconds
        state.position.x = 5.0 * std::sin(omega * t);
        state.position.y = 5.0 * std::sin(2.0 * omega * t);
        state.position.z = 2.0 + 0.5 * std::sin(3.0 * omega * t);
        
        // Simple rotation matrix (rotating around z-axis)
        double angle = omega * t;
        state.rotation_matrix.data[0][0] = std::cos(angle);
        state.rotation_matrix.data[0][1] = -std::sin(angle);
        state.rotation_matrix.data[0][2] = 0.0;
        state.rotation_matrix.data[1][0] = std::sin(angle);
        state.rotation_matrix.data[1][1] = std::cos(angle);
        state.rotation_matrix.data[1][2] = 0.0;
        state.rotation_matrix.data[2][0] = 0.0;
        state.rotation_matrix.data[2][1] = 0.0;
        state.rotation_matrix.data[2][2] = 1.0;
        
        // Velocity (derivative of position)
        state.velocity = Vector3(
            5.0 * omega * std::cos(omega * t),
            10.0 * omega * std::cos(2.0 * omega * t),
            1.5 * omega * std::cos(3.0 * omega * t)
        );
        
        // Angular velocity (rotating around z-axis)
        state.angular_velocity = Vector3(0.0, 0.0, omega);
        
        data.trajectory.push_back(state);
    }
    
    // Add landmarks
    for (int i = 0; i < 50; ++i) {
        Landmark lm;
        lm.id = i;
        // Random positions in a cube
        lm.position.x = -10.0 + 20.0 * (i % 5) / 4.0;
        lm.position.y = -10.0 + 20.0 * ((i / 5) % 5) / 4.0;
        lm.position.z = 0.0 + 5.0 * ((i / 25) % 2);
        
        // Add a simple descriptor
        lm.descriptor = std::vector<double>(128, 0.0);
        for (int j = 0; j < 128; ++j) {
            lm.descriptor.value()[j] = std::sin(i * j * 0.1);
        }
        
        data.landmarks.push_back(lm);
    }
    
    // Add IMU measurements
    for (int i = 0; i <= 2000; ++i) {  // 200 Hz for 10 seconds
        IMUMeasurement meas;
        meas.timestamp = i * 0.005;
        
        // Simulated accelerometer (gravity + motion)
        meas.accelerometer.x = 0.1 * std::sin(meas.timestamp);
        meas.accelerometer.y = 0.1 * std::cos(meas.timestamp);
        meas.accelerometer.z = 9.81 + 0.05 * std::sin(2.0 * meas.timestamp);
        
        // Simulated gyroscope
        meas.gyroscope.x = 0.01 * std::sin(meas.timestamp);
        meas.gyroscope.y = 0.01 * std::cos(meas.timestamp);
        meas.gyroscope.z = 2.0 * M_PI / 10.0;  // Constant rotation around z
        
        data.imu_measurements.push_back(meas);
    }
    
    // Add camera frames with observations
    for (int frame_idx = 0; frame_idx <= 100; ++frame_idx) {  // 10 Hz for 10 seconds
        CameraFrame frame;
        frame.timestamp = frame_idx * 0.1;
        frame.camera_id = "cam0";
        
        // Mark every 10th frame as keyframe
        frame.is_keyframe = (frame_idx % 10 == 0);
        if (frame.is_keyframe) {
            frame.keyframe_id = frame_idx / 10;
        }
        
        // Add observations of visible landmarks
        for (int lm_idx = 0; lm_idx < 50; ++lm_idx) {
            // Simple visibility check (landmarks in front of camera)
            if (lm_idx % 3 == frame_idx % 3) {  // Simulate partial visibility
                CameraObservation obs;
                obs.landmark_id = lm_idx;
                
                // Project to image (simplified)
                obs.pixel.u = 320.0 + 50.0 * std::sin(frame_idx * 0.1 + lm_idx * 0.2);
                obs.pixel.v = 240.0 + 50.0 * std::cos(frame_idx * 0.1 + lm_idx * 0.3);
                
                // Add descriptor
                obs.descriptor = std::vector<double>(128, 0.0);
                for (int j = 0; j < 128; ++j) {
                    obs.descriptor.value()[j] = std::cos(lm_idx * j * 0.1);
                }
                
                frame.observations.push_back(obs);
            }
        }
        
        data.camera_frames.push_back(frame);
    }
    
    // Add preintegrated IMU data between keyframes
    for (int kf_idx = 0; kf_idx < 10; ++kf_idx) {
        PreintegratedIMUData preint;
        preint.from_keyframe_id = kf_idx;
        preint.to_keyframe_id = kf_idx + 1;
        
        // Simulated preintegrated values
        preint.delta_position = Vector3(0.5, 0.3, 0.1);
        preint.delta_velocity = Vector3(0.1, 0.05, 0.02);
        
        // Simple rotation matrix
        double angle = 0.1;
        preint.delta_rotation.data[0][0] = std::cos(angle);
        preint.delta_rotation.data[0][1] = -std::sin(angle);
        preint.delta_rotation.data[0][2] = 0.0;
        preint.delta_rotation.data[1][0] = std::sin(angle);
        preint.delta_rotation.data[1][1] = std::cos(angle);
        preint.delta_rotation.data[1][2] = 0.0;
        preint.delta_rotation.data[2][0] = 0.0;
        preint.delta_rotation.data[2][1] = 0.0;
        preint.delta_rotation.data[2][2] = 1.0;
        
        // Covariance (15x15 flattened)
        preint.covariance = std::vector<double>(225, 0.0);
        for (int i = 0; i < 15; ++i) {
            preint.covariance[i * 15 + i] = 0.001;  // Diagonal elements
        }
        
        preint.dt = 1.0;  // 1 second between keyframes
        preint.num_measurements = 200;  // 200 IMU measurements
        
        // Add optional jacobian
        if (kf_idx % 2 == 0) {
            preint.jacobian = std::vector<double>(225, 0.0);
            for (int i = 0; i < 15; ++i) {
                preint.jacobian.value()[i * 15 + i] = 1.0;  // Identity jacobian
            }
        }
        
        data.preintegrated_imu.push_back(preint);
    }
    
    return data;
}

// Test function to verify data integrity
void verify_data(const SimulationData& original, const SimulationData& loaded) {
    // Check metadata
    assert(original.metadata.version == loaded.metadata.version);
    assert(original.metadata.trajectory_type == loaded.metadata.trajectory_type);
    assert(std::abs(original.metadata.duration - loaded.metadata.duration) < 1e-6);
    assert(original.metadata.coordinate_system == loaded.metadata.coordinate_system);
    if (original.metadata.seed.has_value()) {
        assert(loaded.metadata.seed.has_value());
        assert(original.metadata.seed.value() == loaded.metadata.seed.value());
    }
    
    // Check calibrations
    assert(original.camera_calibrations.size() == loaded.camera_calibrations.size());
    for (size_t i = 0; i < original.camera_calibrations.size(); ++i) {
        const auto& orig = original.camera_calibrations[i];
        const auto& load = loaded.camera_calibrations[i];
        assert(orig.id == load.id);
        assert(orig.intrinsics.model == load.intrinsics.model);
        assert(orig.intrinsics.width == load.intrinsics.width);
        assert(orig.intrinsics.height == load.intrinsics.height);
        assert(std::abs(orig.intrinsics.fx - load.intrinsics.fx) < 1e-6);
        assert(std::abs(orig.intrinsics.fy - load.intrinsics.fy) < 1e-6);
        assert(std::abs(orig.intrinsics.cx - load.intrinsics.cx) < 1e-6);
        assert(std::abs(orig.intrinsics.cy - load.intrinsics.cy) < 1e-6);
    }
    
    assert(original.imu_calibrations.size() == loaded.imu_calibrations.size());
    for (size_t i = 0; i < original.imu_calibrations.size(); ++i) {
        const auto& orig = original.imu_calibrations[i];
        const auto& load = loaded.imu_calibrations[i];
        assert(orig.id == load.id);
        assert(std::abs(orig.accelerometer.noise_density - load.accelerometer.noise_density) < 1e-6);
        assert(std::abs(orig.accelerometer.random_walk - load.accelerometer.random_walk) < 1e-6);
        assert(std::abs(orig.gyroscope.noise_density - load.gyroscope.noise_density) < 1e-6);
        assert(std::abs(orig.gyroscope.random_walk - load.gyroscope.random_walk) < 1e-6);
        assert(std::abs(orig.sampling_rate - load.sampling_rate) < 1e-6);
    }
    
    // Check trajectory
    assert(original.trajectory.size() == loaded.trajectory.size());
    for (size_t i = 0; i < original.trajectory.size(); ++i) {
        const auto& orig = original.trajectory[i];
        const auto& load = loaded.trajectory[i];
        assert(std::abs(orig.timestamp - load.timestamp) < 1e-6);
        assert(std::abs(orig.position.x - load.position.x) < 1e-6);
        assert(std::abs(orig.position.y - load.position.y) < 1e-6);
        assert(std::abs(orig.position.z - load.position.z) < 1e-6);
        // Check rotation matrix
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                assert(std::abs(orig.rotation_matrix.data[r][c] - load.rotation_matrix.data[r][c]) < 1e-6);
            }
        }
    }
    
    // Check landmarks
    assert(original.landmarks.size() == loaded.landmarks.size());
    for (size_t i = 0; i < original.landmarks.size(); ++i) {
        const auto& orig = original.landmarks[i];
        const auto& load = loaded.landmarks[i];
        assert(orig.id == load.id);
        assert(std::abs(orig.position.x - load.position.x) < 1e-6);
        assert(std::abs(orig.position.y - load.position.y) < 1e-6);
        assert(std::abs(orig.position.z - load.position.z) < 1e-6);
    }
    
    // Check IMU measurements
    assert(original.imu_measurements.size() == loaded.imu_measurements.size());
    for (size_t i = 0; i < std::min(size_t(10), original.imu_measurements.size()); ++i) {
        const auto& orig = original.imu_measurements[i];
        const auto& load = loaded.imu_measurements[i];
        assert(std::abs(orig.timestamp - load.timestamp) < 1e-6);
        assert(std::abs(orig.accelerometer.x - load.accelerometer.x) < 1e-6);
        assert(std::abs(orig.accelerometer.y - load.accelerometer.y) < 1e-6);
        assert(std::abs(orig.accelerometer.z - load.accelerometer.z) < 1e-6);
        assert(std::abs(orig.gyroscope.x - load.gyroscope.x) < 1e-6);
        assert(std::abs(orig.gyroscope.y - load.gyroscope.y) < 1e-6);
        assert(std::abs(orig.gyroscope.z - load.gyroscope.z) < 1e-6);
    }
    
    // Check camera frames
    assert(original.camera_frames.size() == loaded.camera_frames.size());
    for (size_t i = 0; i < std::min(size_t(10), original.camera_frames.size()); ++i) {
        const auto& orig = original.camera_frames[i];
        const auto& load = loaded.camera_frames[i];
        assert(std::abs(orig.timestamp - load.timestamp) < 1e-6);
        assert(orig.camera_id == load.camera_id);
        assert(orig.is_keyframe == load.is_keyframe);
        if (orig.keyframe_id.has_value()) {
            assert(load.keyframe_id.has_value());
            assert(orig.keyframe_id.value() == load.keyframe_id.value());
        }
        assert(orig.observations.size() == load.observations.size());
    }
    
    // Check preintegrated IMU data
    assert(original.preintegrated_imu.size() == loaded.preintegrated_imu.size());
    for (size_t i = 0; i < original.preintegrated_imu.size(); ++i) {
        const auto& orig = original.preintegrated_imu[i];
        const auto& load = loaded.preintegrated_imu[i];
        assert(orig.from_keyframe_id == load.from_keyframe_id);
        assert(orig.to_keyframe_id == load.to_keyframe_id);
        assert(std::abs(orig.delta_position.x - load.delta_position.x) < 1e-6);
        assert(std::abs(orig.delta_position.y - load.delta_position.y) < 1e-6);
        assert(std::abs(orig.delta_position.z - load.delta_position.z) < 1e-6);
        assert(std::abs(orig.delta_velocity.x - load.delta_velocity.x) < 1e-6);
        assert(std::abs(orig.delta_velocity.y - load.delta_velocity.y) < 1e-6);
        assert(std::abs(orig.delta_velocity.z - load.delta_velocity.z) < 1e-6);
        assert(std::abs(orig.dt - load.dt) < 1e-6);
        assert(orig.num_measurements == load.num_measurements);
        assert(orig.covariance.size() == load.covariance.size());
        if (orig.jacobian.has_value()) {
            assert(load.jacobian.has_value());
            assert(orig.jacobian.value().size() == load.jacobian.value().size());
        }
    }
}

int main() {
    try {
        std::cout << "Creating sample simulation data..." << std::endl;
        SimulationData data = create_sample_data();
        
        std::cout << "Created data with:" << std::endl;
        std::cout << "  - " << data.trajectory.size() << " trajectory states" << std::endl;
        std::cout << "  - " << data.landmarks.size() << " landmarks" << std::endl;
        std::cout << "  - " << data.imu_measurements.size() << " IMU measurements" << std::endl;
        std::cout << "  - " << data.camera_frames.size() << " camera frames" << std::endl;
        std::cout << "  - " << data.preintegrated_imu.size() << " preintegrated IMU factors" << std::endl;
        std::cout << "  - " << data.camera_calibrations.size() << " camera calibrations" << std::endl;
        std::cout << "  - " << data.imu_calibrations.size() << " IMU calibrations" << std::endl;
        
        // Count keyframes
        int keyframe_count = 0;
        for (const auto& frame : data.camera_frames) {
            if (frame.is_keyframe) keyframe_count++;
        }
        std::cout << "  - " << keyframe_count << " keyframes" << std::endl;
        
        // Save to JSON
        std::string filename = "test_simulation_data.json";
        std::cout << "\nSaving to " << filename << "..." << std::endl;
        JsonIO::save(data, filename);
        std::cout << "Save completed successfully!" << std::endl;
        
        // Load from JSON
        std::cout << "\nLoading from " << filename << "..." << std::endl;
        SimulationData loaded_data = JsonIO::load(filename);
        std::cout << "Load completed successfully!" << std::endl;
        
        std::cout << "\nLoaded data with:" << std::endl;
        std::cout << "  - " << loaded_data.trajectory.size() << " trajectory states" << std::endl;
        std::cout << "  - " << loaded_data.landmarks.size() << " landmarks" << std::endl;
        std::cout << "  - " << loaded_data.imu_measurements.size() << " IMU measurements" << std::endl;
        std::cout << "  - " << loaded_data.camera_frames.size() << " camera frames" << std::endl;
        std::cout << "  - " << loaded_data.preintegrated_imu.size() << " preintegrated IMU factors" << std::endl;
        
        // Count keyframes in loaded data
        int loaded_keyframe_count = 0;
        for (const auto& frame : loaded_data.camera_frames) {
            if (frame.is_keyframe) loaded_keyframe_count++;
        }
        std::cout << "  - " << loaded_keyframe_count << " keyframes" << std::endl;
        std::cout << "  - " << loaded_data.camera_calibrations.size() << " camera calibrations" << std::endl;
        std::cout << "  - " << loaded_data.imu_calibrations.size() << " IMU calibrations" << std::endl;
        
        // Verify data integrity
        std::cout << "\nVerifying data integrity..." << std::endl;
        verify_data(data, loaded_data);
        std::cout << "Data verification passed!" << std::endl;
        
        // Display some sample data
        std::cout << "\nSample trajectory state (t=0):" << std::endl;
        if (!loaded_data.trajectory.empty()) {
            const auto& state = loaded_data.trajectory[0];
            std::cout << "  Timestamp: " << state.timestamp << std::endl;
            std::cout << "  Position: [" << state.position.x << ", " 
                      << state.position.y << ", " << state.position.z << "]" << std::endl;
            std::cout << "  Rotation Matrix: [" << state.rotation_matrix.data[0][0] << ", " 
                      << state.rotation_matrix.data[0][1] << ", " << state.rotation_matrix.data[0][2] << "]" << std::endl;
        }
        
        std::cout << "\nSample landmark:" << std::endl;
        if (!loaded_data.landmarks.empty()) {
            const auto& lm = loaded_data.landmarks[0];
            std::cout << "  ID: " << lm.id << std::endl;
            std::cout << "  Position: [" << lm.position.x << ", " 
                      << lm.position.y << ", " << lm.position.z << "]" << std::endl;
        }
        
        std::cout << "\nAll tests passed successfully!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}