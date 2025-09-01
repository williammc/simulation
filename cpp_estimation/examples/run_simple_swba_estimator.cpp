/**
 * run_simple_swba_estimator.cpp
 * 
 * Command-line executable for running simplified C++ SWBA estimator on simulation data.
 * Based on Python simple_swba_vio.py implementation.
 * 
 * Usage:
 *   run_simple_swba_estimator --input simulation.json --output output_dir/ [--config config.yaml]
 */

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdlib>

#include <nlohmann/json.hpp>
#include <Eigen/Core>

#include "estimators/simple_swba_estimator.hpp"
#include "simulation_io/data_structures.hpp"
#include "simulation_io/json_io.hpp"
#include "simulation_io/preprocessed_interfaces.hpp"
#include "simulation_io/estimator_result_io.hpp"

using namespace estimators;
using namespace simulation_io;
using json = nlohmann::json;

// Simple command-line argument parser
struct Arguments {
    std::string input_file;
    std::string output_dir = "output/slam";
    std::string config_file;
    bool verbose = false;
    bool use_ideal_coordinates = true;
    
    bool parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--input" && i + 1 < argc) {
                input_file = argv[++i];
            } else if (arg == "--output" && i + 1 < argc) {
                output_dir = argv[++i];
            } else if (arg == "--config" && i + 1 < argc) {
                config_file = argv[++i];
            } else if (arg == "--verbose" || arg == "-v") {
                verbose = true;
            } else if (arg == "--no-ideal") {
                use_ideal_coordinates = false;
            } else if (arg == "--help" || arg == "-h") {
                return false;
            }
        }
        return !input_file.empty();
    }
    
    void print_usage(const char* program_name) {
        std::cout << "Usage: " << program_name << " --input <simulation.json> [options]\n";
        std::cout << "\nOptions:\n";
        std::cout << "  --input FILE       Input simulation JSON file (required)\n";
        std::cout << "  --output DIR       Output directory (default: output/slam)\n";
        std::cout << "  --config FILE      Configuration YAML/JSON file\n";
        std::cout << "  --verbose, -v      Enable verbose output\n";
        std::cout << "  --no-ideal         Disable ideal coordinate usage\n";
        std::cout << "  --help, -h         Show this help message\n";
    }
};

// Visual preprocessor to convert observations to ideal coordinates
class SimpleVisualPreprocessor {
private:
    CameraCalibration camera_calib_;
    bool use_ideal_;
    
public:
    SimpleVisualPreprocessor(const CameraCalibration& calib, bool use_ideal = true) 
        : camera_calib_(calib), use_ideal_(use_ideal) {}
    
    ProcessedVisualFrame process_frame(
        const CameraFrame& raw_frame,
        const Eigen::Vector3d& current_position,
        const Eigen::Matrix3d& current_rotation,
        const std::map<int, EstimatedLandmark>& landmarks) {
        
        ProcessedVisualFrame processed;
        processed.timestamp = raw_frame.timestamp;
        processed.frame_id = static_cast<int>(raw_frame.timestamp * 1000);  // Convert to ms as ID
        processed.is_keyframe = raw_frame.is_keyframe;
        processed.keyframe_id = raw_frame.keyframe_id;
        processed.camera_id = raw_frame.camera_id;
        
        // Get camera intrinsics from calibration
        double fx = camera_calib_.intrinsics.fx;
        double fy = camera_calib_.intrinsics.fy;
        double cx = camera_calib_.intrinsics.cx;
        double cy = camera_calib_.intrinsics.cy;
        
        // Process each observation
        for (const auto& obs : raw_frame.observations) {
            VisualMeasurement meas;
            meas.landmark_id = obs.landmark_id;
            
            // Observed pixel
            meas.observed_pixel = Eigen::Vector2d(obs.pixel.u, obs.pixel.v);
            
            // Find landmark
            auto lmk_it = landmarks.find(obs.landmark_id);
            if (lmk_it == landmarks.end()) {
                // Skip if landmark not found
                continue;
            }
            
            // Project landmark to get predicted pixel
            Eigen::Vector3d lmk_cam = current_rotation.transpose() * 
                                     (lmk_it->second.position - current_position);
            
            if (lmk_cam.z() <= 0) {
                // Behind camera
                continue;
            }
            
            double x_norm = lmk_cam.x() / lmk_cam.z();
            double y_norm = lmk_cam.y() / lmk_cam.z();
            
            // Apply distortion if needed (simplified - assuming no distortion)
            double u_pred = fx * x_norm + cx;
            double v_pred = fy * y_norm + cy;
            
            meas.predicted_pixel = Eigen::Vector2d(u_pred, v_pred);
            meas.residual = meas.observed_pixel - meas.predicted_pixel;
            meas.pixel_covariance = Eigen::Matrix2d::Identity() * 4.0;  // 2 pixel std dev
            
            // Ideal/normalized coordinates
            if (use_ideal_) {
                // Convert pixel to ideal coordinates
                double x_obs_ideal = (meas.observed_pixel.x() - cx) / fx;
                double y_obs_ideal = (meas.observed_pixel.y() - cy) / fy;
                meas.observed_ideal = Eigen::Vector2d(x_obs_ideal, y_obs_ideal);
                
                meas.predicted_ideal = Eigen::Vector2d(x_norm, y_norm);
                meas.ideal_residual = meas.observed_ideal.value() - meas.predicted_ideal.value();
                meas.ideal_covariance = Eigen::Matrix2d::Identity() * (4.0 / (fx * fx));
                
                // Compute Jacobians (simplified)
                Eigen::Matrix<double, 2, 6> J_pose = Eigen::Matrix<double, 2, 6>::Zero();
                double z_inv = 1.0 / lmk_cam.z();
                double z_inv2 = z_inv * z_inv;
                
                // Jacobian w.r.t. position (in camera frame)
                J_pose(0, 0) = -z_inv;
                J_pose(0, 2) = x_norm * z_inv;
                J_pose(1, 1) = -z_inv;
                J_pose(1, 2) = y_norm * z_inv;
                
                // Jacobian w.r.t. rotation (simplified)
                J_pose(0, 3) = x_norm * y_norm;
                J_pose(0, 4) = -(1 + x_norm * x_norm);
                J_pose(0, 5) = y_norm;
                J_pose(1, 3) = 1 + y_norm * y_norm;
                J_pose(1, 4) = -x_norm * y_norm;
                J_pose(1, 5) = -x_norm;
                
                meas.ideal_jacobian_wrt_pose = J_pose;
                
                // Jacobian w.r.t. landmark
                Eigen::Matrix<double, 2, 3> J_lmk = Eigen::Matrix<double, 2, 3>::Zero();
                J_lmk(0, 0) = z_inv;
                J_lmk(0, 2) = -x_norm * z_inv;
                J_lmk(1, 1) = z_inv;
                J_lmk(1, 2) = -y_norm * z_inv;
                
                meas.ideal_jacobian_wrt_landmark = J_lmk;
            }
            
            // Bearing vector
            Eigen::Vector3d bearing(x_norm, y_norm, 1.0);
            bearing.normalize();
            meas.bearing_vector = bearing;
            
            meas.estimated_depth = lmk_cam.z();
            meas.robust_weight = 1.0;
            
            // Set information matrix if available
            if (use_ideal_ && meas.ideal_covariance.has_value()) {
                meas.information_matrix = meas.ideal_covariance.value().inverse();
            }
            
            processed.measurements.push_back(meas);
        }
        
        return processed;
    }
};

int main(int argc, char* argv[]) {
    // Parse arguments
    Arguments args;
    if (!args.parse(argc, argv)) {
        args.print_usage(argv[0]);
        return 1;
    }
    
    std::cout << "=== C++ Simple SWBA Estimator ===" << std::endl;
    std::cout << "Input:  " << args.input_file << std::endl;
    std::cout << "Output: " << args.output_dir << std::endl;
    
    // Create output directory
    std::filesystem::create_directories(args.output_dir);
    
    // Load simulation data
    std::cout << "\nLoading simulation data..." << std::endl;
    
    SimulationData sim_data;
    try {
        std::cout << "  Opening file: " << args.input_file << std::endl;
        sim_data = JsonIO::load(args.input_file);
        std::cout << "  Successfully loaded!" << std::endl;
        std::cout << "  Trajectory states: " << sim_data.trajectory.size() << std::endl;
        std::cout << "  Landmarks: " << sim_data.landmarks.size() << std::endl;
        std::cout << "  Camera frames: " << sim_data.camera_frames.size() << std::endl;
        std::cout << "  Preintegrated IMU: " << sim_data.preintegrated_imu.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading simulation data: " << e.what() << std::endl;
        return 1;
    }
    
    // Configure estimator (matching Python SWBA)
    SimpleSWBAEstimator<double>::Config config;
    config.window_size = 10;
    config.marginalize_old_keyframes = true;
    
    // Keyframe selection parameters
    config.use_keyframes_only = false;  // Use internal keyframe selection
    config.keyframe_time_threshold = 0.5;  // seconds
    config.keyframe_translation_threshold = 0.2;  // meters
    config.keyframe_rotation_threshold = 0.2;  // radians
    
    // Optimization parameters
    config.max_iterations = 10;
    config.convergence_threshold = 1e-4;
    config.damping_factor = 0.01;
    
    // Visual measurement parameters
    config.ideal_coord_weight = 10.0;
    config.pixel_coord_weight = 1.0;
    config.min_measurements = 5;
    
    // Debug/logging
    config.verbose = args.verbose;
    config.debug_enabled = true;
    
    // Load config file if provided
    if (!args.config_file.empty()) {
        std::cout << "Loading config from: " << args.config_file << std::endl;
        // TODO: Load YAML/JSON config
    }
    
    // Create estimator
    SimpleSWBAEstimator<double> estimator(config);
    
    // Initialize from first ground truth state
    if (!sim_data.trajectory.empty()) {
        const auto& init_state = sim_data.trajectory[0];
        
        // Use rotation matrix directly (already in TrajectoryState)
        Eigen::Matrix3d init_rotation = init_state.rotation_matrix;
        
        // Check if velocity is available
        Eigen::Vector3d init_velocity = Eigen::Vector3d::Zero();
        if (init_state.velocity.has_value()) {
            init_velocity = init_state.velocity.value();
            std::cout << "Initial velocity available: " << init_velocity.transpose() << std::endl;
        } else {
            std::cout << "WARNING: No initial velocity, using zero!" << std::endl;
        }
        
        estimator.initialize(
            init_state.position,
            init_rotation,
            init_velocity,
            init_state.timestamp
        );
        
        std::cout << "\nInitialized at t=" << init_state.timestamp 
                 << ", pos: " << init_state.position.transpose() 
                 << ", vel: " << init_velocity.transpose() << std::endl;
    } else {
        std::cerr << "Error: No ground truth states for initialization" << std::endl;
        return 1;
    }
    
    // Convert landmarks to estimator format
    std::map<int, EstimatedLandmark> landmarks_map;
    for (const auto& lmk : sim_data.landmarks) {
        EstimatedLandmark elark;
        elark.id = lmk.id;
        elark.position = lmk.position;
        landmarks_map[lmk.id] = elark;
    }
    
    // Setup visual preprocessor
    CameraCalibration camera_calib = sim_data.camera_calibrations.empty() ? 
                                     CameraCalibration() : sim_data.camera_calibrations[0];
    SimpleVisualPreprocessor preprocessor(camera_calib, args.use_ideal_coordinates);
    
    // Process data
    std::cout << "\nProcessing..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Important: Process ALL frames, not just keyframes
    // This matches Python simple_swba_vio.py behavior
    size_t frame_idx = 0;
    size_t imu_idx = 0;
    int total_iterations = 0;
    
    std::cout << "Total IMU segments: " << sim_data.preintegrated_imu.size() << std::endl;
    std::cout << "Total camera frames: " << sim_data.camera_frames.size() << std::endl;
    
    // Process IMU and camera frames in sequence
    for (const auto& imu_data : sim_data.preintegrated_imu) {
        // Convert to preprocessed format
        PreprocessedIMUData processed_imu;
        processed_imu.from_frame_id = imu_data.from_frame_id;
        processed_imu.to_frame_id = imu_data.to_frame_id;
        processed_imu.delta_position = imu_data.delta_position;
        processed_imu.delta_velocity = imu_data.delta_velocity;
        processed_imu.delta_rotation = imu_data.delta_rotation;
        processed_imu.delta_t = imu_data.dt;
        
        // Convert flattened covariance vector to 9x9 matrix
        if (imu_data.covariance.size() >= 81) {
            processed_imu.covariance = Eigen::Map<const Eigen::Matrix<double, 9, 9, Eigen::RowMajor>>(
                imu_data.covariance.data()
            );
        } else {
            processed_imu.covariance = Eigen::Matrix<double, 9, 9>::Identity() * 0.001;
        }
        
        processed_imu.num_measurements = imu_data.num_measurements;
        
        // Predict with IMU
        estimator.predict(processed_imu, processed_imu.delta_t);
        
        // Process corresponding camera frame (if available)
        if (frame_idx < sim_data.camera_frames.size()) {
            const auto& camera_frame = sim_data.camera_frames[frame_idx];
            
            // Get current state for preprocessing
            auto current_pos = estimator.getCurrentPosition();
            auto current_rot = estimator.getCurrentRotation();
            
            // Process visual frame
            auto processed_frame = preprocessor.process_frame(
                camera_frame, current_pos, current_rot, landmarks_map
            );
            
            // Update estimator (will decide internally if this is a keyframe)
            int iterations = estimator.update(processed_frame, landmarks_map);
            total_iterations += iterations;
            
            if (args.verbose) {
                std::cout << "  Frame " << frame_idx << " at t=" << camera_frame.timestamp
                         << ": " << processed_frame.measurements.size() 
                         << " measurements, " << iterations << " iterations" << std::endl;
            }
            
            frame_idx++;
        }
        
        // Show progress
        if ((imu_idx + 1) % 10 == 0 || args.verbose) {
            std::cout << "  Processed " << (imu_idx + 1) << "/" << sim_data.preintegrated_imu.size() 
                     << " IMU factors, " << frame_idx << " camera frames" << std::endl;
        }
        imu_idx++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "\nEstimation complete in " << runtime_ms << " ms" << std::endl;
    std::cout << "  Processed frames: " << frame_idx << std::endl;
    std::cout << "  Keyframes: " << estimator.getNumKeyframes() << std::endl;
    std::cout << "  Landmarks: " << estimator.getNumLandmarks() << std::endl;
    std::cout << "  Total iterations: " << total_iterations << std::endl;
    
    // Prepare result
    EstimatorResult result;
    result.estimator_type = EstimatorType::SWBA;
    result.runtime_ms = static_cast<double>(runtime_ms);
    result.iterations = total_iterations;
    result.converged = true;
    result.final_cost = 0;
    
    // Add metadata
    result.metadata["cpp_implementation"] = json(true);
    result.metadata["estimator_variant"] = json("simple_swba");
    result.metadata["ideal_coordinates"] = json(args.use_ideal_coordinates);
    result.metadata["window_size"] = json(config.window_size);
    result.metadata["use_keyframes_only"] = json(config.use_keyframes_only);
    result.metadata["keyframe_time_threshold"] = json(config.keyframe_time_threshold);
    result.metadata["keyframe_translation_threshold"] = json(config.keyframe_translation_threshold);
    result.metadata["keyframe_rotation_threshold"] = json(config.keyframe_rotation_threshold);
    result.metadata["num_frames_processed"] = json(frame_idx);
    
    // Add simulation info
    result.input_file = args.input_file;
    result.trajectory_type = "unknown";  // Could extract from filename
    result.simulation_duration = sim_data.trajectory.empty() ? 0 :
        sim_data.trajectory.back().timestamp;
    
    // Extract full trajectory (includes ALL frames, not just keyframes)
    auto full_trajectory = estimator.getFullTrajectory();
    for (const auto& pose : full_trajectory) {
        result.trajectory.add_pose(pose);
    }
    
    std::cout << "Trajectory contains " << full_trajectory.size() << " poses" << std::endl;
    
    // Get estimated landmarks
    auto estimated_landmarks = estimator.getEstimatedLandmarks();
    for (const auto& [id, lmk] : estimated_landmarks) {
        result.landmarks.add_landmark(lmk);
    }
    
    // Save result
    std::string output_file = args.output_dir + "/cpp_simple_swba_result.json";
    std::cout << "\nSaving results to: " << output_file << std::endl;
    
    try {
        EstimatorResultIO::save(result, output_file);
        std::cout << "✓ Results saved successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ Error saving results: " << e.what() << std::endl;
        return 1;
    }
    
    // Print summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Runtime:         " << runtime_ms << " ms" << std::endl;
    std::cout << "Frames:          " << frame_idx << std::endl;
    std::cout << "Keyframes:       " << estimator.getNumKeyframes() << std::endl;
    std::cout << "Trajectory size: " << full_trajectory.size() << " poses" << std::endl;
    std::cout << "Landmarks:       " << result.landmarks.landmarks.size() << std::endl;
    std::cout << "Output:          " << output_file << std::endl;
    
    return 0;
}