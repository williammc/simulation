#ifndef SIMULATION_IO_JSON_IO_HPP
#define SIMULATION_IO_JSON_IO_HPP

#include "data_structures.hpp"
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <limits>

namespace simulation_io {

using json = nlohmann::json;

// JSON conversion helpers for Eigen types
namespace detail {

inline json vector3_to_json(const Vector3& v) {
    return json::array({v.x(), v.y(), v.z()});
}

inline Vector3 json_to_vector3(const json& j) {
    if (!j.is_array() || j.size() != 3) {
        throw std::runtime_error("Invalid Vector3 JSON format");
    }
    return Vector3(j[0], j[1], j[2]);
}

inline json matrix3x3_to_json(const Matrix3x3& m) {
    json result = json::array();
    for (int i = 0; i < 3; ++i) {
        json row = json::array();
        for (int j = 0; j < 3; ++j) {
            row.push_back(m(i, j));
        }
        result.push_back(row);
    }
    return result;
}

inline Matrix3x3 json_to_matrix3x3(const json& j) {
    if (!j.is_array() || j.size() != 3) {
        throw std::runtime_error("Invalid Matrix3x3 JSON format");
    }
    Matrix3x3 m;
    for (int i = 0; i < 3; ++i) {
        if (!j[i].is_array() || j[i].size() != 3) {
            throw std::runtime_error("Invalid Matrix3x3 row format");
        }
        for (int j_ = 0; j_ < 3; ++j_) {
            m(i, j_) = j[i][j_];
        }
    }
    return m;
}

inline json matrix4x4_to_json(const Matrix4x4& m) {
    json result = json::array();
    for (int i = 0; i < 4; ++i) {
        json row = json::array();
        for (int j = 0; j < 4; ++j) {
            row.push_back(m(i, j));
        }
        result.push_back(row);
    }
    return result;
}

inline Matrix4x4 json_to_matrix4x4(const json& j) {
    if (!j.is_array() || j.size() != 4) {
        throw std::runtime_error("Invalid Matrix4x4 JSON format");
    }
    Matrix4x4 m;
    for (int i = 0; i < 4; ++i) {
        if (!j[i].is_array() || j[i].size() != 4) {
            throw std::runtime_error("Invalid Matrix4x4 row format");
        }
        for (int j_ = 0; j_ < 4; ++j_) {
            m(i, j_) = j[i][j_];
        }
    }
    return m;
}

inline VectorX json_to_vectorx(const json& j, int expected_size = -1) {
    if (!j.is_array()) {
        throw std::runtime_error("Invalid VectorX JSON format");
    }
    
    // Handle 2D array (needs flattening) or 1D array
    if (!j.empty() && j[0].is_array()) {
        // It's a 2D array, flatten it
        std::vector<double> flat;
        for (const auto& row : j) {
            for (const auto& val : row) {
                flat.push_back(val);
            }
        }
        VectorX v(flat.size());
        for (size_t i = 0; i < flat.size(); ++i) {
            v(i) = flat[i];
        }
        return v;
    } else {
        // It's already a 1D array
        VectorX v(j.size());
        for (size_t i = 0; i < j.size(); ++i) {
            v(i) = j[i];
        }
        return v;
    }
}

} // namespace detail

// SimulationData JSON serialization
class JsonIO {
public:
    static void save(const SimulationData& data, const std::string& filepath) {
        json j;
        
        // Metadata
        j["metadata"]["version"] = data.metadata.version;
        j["metadata"]["timestamp"] = data.metadata.timestamp;
        j["metadata"]["trajectory_type"] = data.metadata.trajectory_type;
        j["metadata"]["duration"] = data.metadata.duration;
        j["metadata"]["coordinate_system"] = data.metadata.coordinate_system;
        if (data.metadata.seed.has_value()) {
            j["metadata"]["seed"] = data.metadata.seed.value();
        }
        j["metadata"]["units"]["position"] = data.metadata.units.position;
        j["metadata"]["units"]["rotation"] = data.metadata.units.rotation;
        j["metadata"]["units"]["time"] = data.metadata.units.time;
        
        // Calibration
        j["calibration"]["cameras"] = json::array();
        for (const auto& cam : data.camera_calibrations) {
            json cam_json;
            cam_json["id"] = cam.id;
            cam_json["model"] = cam.intrinsics.model;
            cam_json["width"] = cam.intrinsics.width;
            cam_json["height"] = cam.intrinsics.height;
            cam_json["intrinsics"]["fx"] = cam.intrinsics.fx;
            cam_json["intrinsics"]["fy"] = cam.intrinsics.fy;
            cam_json["intrinsics"]["cx"] = cam.intrinsics.cx;
            cam_json["intrinsics"]["cy"] = cam.intrinsics.cy;
            cam_json["distortion"] = cam.intrinsics.distortion;
            cam_json["T_BC"] = detail::matrix4x4_to_json(cam.T_BC);
            j["calibration"]["cameras"].push_back(cam_json);
        }
        
        j["calibration"]["imus"] = json::array();
        for (const auto& imu : data.imu_calibrations) {
            json imu_json;
            imu_json["id"] = imu.id;
            imu_json["accelerometer"]["noise_density"] = imu.accelerometer.noise_density;
            imu_json["accelerometer"]["random_walk"] = imu.accelerometer.random_walk;
            imu_json["gyroscope"]["noise_density"] = imu.gyroscope.noise_density;
            imu_json["gyroscope"]["random_walk"] = imu.gyroscope.random_walk;
            imu_json["sampling_rate"] = imu.sampling_rate;
            j["calibration"]["imus"].push_back(imu_json);
        }
        
        // Ground truth trajectory
        if (!data.trajectory.empty()) {
            j["groundtruth"]["trajectory"] = json::array();
            for (const auto& state : data.trajectory) {
                json state_json;
                state_json["timestamp"] = state.timestamp;
                state_json["position"] = detail::vector3_to_json(state.position);
                state_json["rotation_matrix"] = detail::matrix3x3_to_json(state.rotation_matrix);
                if (state.velocity.has_value()) {
                    state_json["velocity"] = detail::vector3_to_json(state.velocity.value());
                }
                if (state.angular_velocity.has_value()) {
                    state_json["angular_velocity"] = detail::vector3_to_json(state.angular_velocity.value());
                }
                j["groundtruth"]["trajectory"].push_back(state_json);
            }
        } else {
            j["groundtruth"]["trajectory"] = nullptr;
        }
        
        // Ground truth landmarks
        if (!data.landmarks.empty()) {
            j["groundtruth"]["landmarks"] = json::array();
            for (const auto& lm : data.landmarks) {
                json lm_json;
                lm_json["id"] = lm.id;
                lm_json["position"] = detail::vector3_to_json(lm.position);
                if (lm.descriptor.has_value()) {
                    lm_json["descriptor"] = lm.descriptor.value();
                }
                j["groundtruth"]["landmarks"].push_back(lm_json);
            }
        } else {
            j["groundtruth"]["landmarks"] = nullptr;
        }
        
        // IMU measurements
        if (!data.imu_measurements.empty()) {
            j["measurements"]["imu"] = json::array();
            for (const auto& meas : data.imu_measurements) {
                json meas_json;
                meas_json["timestamp"] = meas.timestamp;
                meas_json["accelerometer"] = detail::vector3_to_json(meas.accelerometer);
                meas_json["gyroscope"] = detail::vector3_to_json(meas.gyroscope);
                j["measurements"]["imu"].push_back(meas_json);
            }
        } else {
            j["measurements"]["imu"] = nullptr;
        }
        
        // Camera measurements
        if (!data.camera_frames.empty()) {
            j["measurements"]["camera_frames"] = json::array();
            for (const auto& frame : data.camera_frames) {
                json frame_json;
                frame_json["timestamp"] = frame.timestamp;
                frame_json["camera_id"] = frame.camera_id;
                frame_json["observations"] = json::array();
                
                for (const auto& obs : frame.observations) {
                    json obs_json;
                    obs_json["landmark_id"] = obs.landmark_id;
                    obs_json["pixel"] = json::array({obs.pixel.u, obs.pixel.v});
                    if (obs.descriptor.has_value()) {
                        obs_json["descriptor"] = obs.descriptor.value();
                    }
                    frame_json["observations"].push_back(obs_json);
                }
                
                // Add keyframe information
                frame_json["is_keyframe"] = frame.is_keyframe;
                if (frame.keyframe_id.has_value()) {
                    frame_json["keyframe_id"] = frame.keyframe_id.value();
                }
                
                j["measurements"]["camera_frames"].push_back(frame_json);
            }
        } else {
            j["measurements"]["camera_frames"] = json::array();
        }
        
        // Preintegrated IMU measurements
        if (!data.preintegrated_imu.empty()) {
            j["measurements"]["preintegrated_imu"] = json::array();
            for (const auto& preint : data.preintegrated_imu) {
                json preint_json;
                preint_json["from_keyframe_id"] = preint.from_keyframe_id;
                preint_json["to_keyframe_id"] = preint.to_keyframe_id;
                preint_json["delta_position"] = detail::vector3_to_json(preint.delta_position);
                preint_json["delta_velocity"] = detail::vector3_to_json(preint.delta_velocity);
                preint_json["delta_rotation"] = detail::matrix3x3_to_json(preint.delta_rotation);
                
                // Convert VectorX to std::vector for JSON
                std::vector<double> cov_vec(preint.covariance.data(), 
                                           preint.covariance.data() + preint.covariance.size());
                preint_json["covariance"] = cov_vec;
                
                preint_json["dt"] = preint.dt;
                preint_json["num_measurements"] = preint.num_measurements;
                
                if (preint.jacobian.has_value()) {
                    std::vector<double> jac_vec(preint.jacobian.value().data(),
                                               preint.jacobian.value().data() + preint.jacobian.value().size());
                    preint_json["jacobian"] = jac_vec;
                }
                j["measurements"]["preintegrated_imu"].push_back(preint_json);
            }
        } else {
            j["measurements"]["preintegrated_imu"] = json::array();
        }
        
        // Write to file with pretty printing
        std::ofstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filepath);
        }
        file << std::setw(2) << j << std::endl;
        file.close();
    }
    
    static SimulationData load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + filepath);
        }
        
        json j;
        file >> j;
        file.close();
        
        SimulationData data;
        
        // Load metadata
        if (j.contains("metadata")) {
            const auto& meta = j["metadata"];
            if (meta.contains("version")) data.metadata.version = meta["version"];
            if (meta.contains("timestamp")) data.metadata.timestamp = meta["timestamp"];
            if (meta.contains("trajectory_type")) data.metadata.trajectory_type = meta["trajectory_type"];
            if (meta.contains("duration")) data.metadata.duration = meta["duration"];
            if (meta.contains("coordinate_system")) data.metadata.coordinate_system = meta["coordinate_system"];
            if (meta.contains("seed") && !meta["seed"].is_null()) {
                data.metadata.seed = meta["seed"];
            }
            
            if (meta.contains("units")) {
                if (meta["units"].contains("position")) data.metadata.units.position = meta["units"]["position"];
                if (meta["units"].contains("rotation")) data.metadata.units.rotation = meta["units"]["rotation"];
                if (meta["units"].contains("time")) data.metadata.units.time = meta["units"]["time"];
            }
        }
        
        // Load calibration
        if (j.contains("calibration")) {
            const auto& calib = j["calibration"];
            
            // Camera calibrations
            if (calib.contains("cameras") && calib["cameras"].is_array()) {
                for (const auto& cam_json : calib["cameras"]) {
                    CameraCalibration cam;
                    cam.id = cam_json["id"];
                    cam.intrinsics.model = cam_json["model"];
                    cam.intrinsics.width = cam_json["width"];
                    cam.intrinsics.height = cam_json["height"];
                    cam.intrinsics.fx = cam_json["intrinsics"]["fx"];
                    cam.intrinsics.fy = cam_json["intrinsics"]["fy"];
                    cam.intrinsics.cx = cam_json["intrinsics"]["cx"];
                    cam.intrinsics.cy = cam_json["intrinsics"]["cy"];
                    if (cam_json.contains("distortion")) {
                        cam.intrinsics.distortion = cam_json["distortion"].get<std::vector<double>>();
                    }
                    if (cam_json.contains("T_BC")) {
                        cam.T_BC = detail::json_to_matrix4x4(cam_json["T_BC"]);
                    }
                    data.camera_calibrations.push_back(cam);
                }
            }
            
            // IMU calibrations
            if (calib.contains("imus") && calib["imus"].is_array()) {
                for (const auto& imu_json : calib["imus"]) {
                    IMUCalibration imu;
                    imu.id = imu_json["id"];
                    imu.accelerometer.noise_density = imu_json["accelerometer"]["noise_density"];
                    imu.accelerometer.random_walk = imu_json["accelerometer"]["random_walk"];
                    imu.gyroscope.noise_density = imu_json["gyroscope"]["noise_density"];
                    imu.gyroscope.random_walk = imu_json["gyroscope"]["random_walk"];
                    imu.sampling_rate = imu_json["sampling_rate"];
                    data.imu_calibrations.push_back(imu);
                }
            }
        }
        
        // Load ground truth
        if (j.contains("groundtruth")) {
            const auto& gt = j["groundtruth"];
            
            // Trajectory
            if (gt.contains("trajectory") && gt["trajectory"].is_array()) {
                for (const auto& state_json : gt["trajectory"]) {
                    TrajectoryState state;
                    state.timestamp = state_json["timestamp"];
                    state.position = detail::json_to_vector3(state_json["position"]);
                    
                    // Support both rotation_matrix and legacy quaternion format
                    if (state_json.contains("rotation_matrix")) {
                        state.rotation_matrix = detail::json_to_matrix3x3(state_json["rotation_matrix"]);
                    } else if (state_json.contains("quaternion")) {
                        // Legacy quaternion support - convert to rotation matrix
                        std::cerr << "Warning: Legacy quaternion format detected, using identity rotation" << std::endl;
                        state.rotation_matrix = Matrix3x3::Identity();
                    }
                    
                    if (state_json.contains("velocity")) {
                        state.velocity = detail::json_to_vector3(state_json["velocity"]);
                    }
                    if (state_json.contains("angular_velocity")) {
                        state.angular_velocity = detail::json_to_vector3(state_json["angular_velocity"]);
                    }
                    data.trajectory.push_back(state);
                }
            }
            
            // Landmarks
            if (gt.contains("landmarks") && gt["landmarks"].is_array()) {
                for (const auto& lm_json : gt["landmarks"]) {
                    Landmark lm;
                    lm.id = lm_json["id"];
                    lm.position = detail::json_to_vector3(lm_json["position"]);
                    if (lm_json.contains("descriptor")) {
                        lm.descriptor = lm_json["descriptor"].get<std::vector<double>>();
                    }
                    data.landmarks.push_back(lm);
                }
            }
        }
        
        // Load measurements
        if (j.contains("measurements")) {
            const auto& meas = j["measurements"];
            
            // IMU measurements
            if (meas.contains("imu") && meas["imu"].is_array()) {
                for (const auto& meas_json : meas["imu"]) {
                    IMUMeasurement imu_meas;
                    imu_meas.timestamp = meas_json["timestamp"];
                    imu_meas.accelerometer = detail::json_to_vector3(meas_json["accelerometer"]);
                    imu_meas.gyroscope = detail::json_to_vector3(meas_json["gyroscope"]);
                    data.imu_measurements.push_back(imu_meas);
                }
            }
            
            // Camera frames
            if (meas.contains("camera_frames") && meas["camera_frames"].is_array()) {
                int frame_index = 0;
                for (const auto& frame_json : meas["camera_frames"]) {
                    CameraFrame frame;
                    frame.timestamp = frame_json["timestamp"];
                    frame.camera_id = frame_json["camera_id"];
                    
                    // Load keyframe information
                    if (frame_json.contains("is_keyframe")) {
                        frame.is_keyframe = frame_json["is_keyframe"];
                    } else {
                        frame.is_keyframe = false;
                    }
                    if (frame_json.contains("keyframe_id") && !frame_json["keyframe_id"].is_null()) {
                        frame.keyframe_id = frame_json["keyframe_id"];
                    }
                    
                    if (frame_json.contains("observations") && frame_json["observations"].is_array()) {
                        int obs_index = 0;
                        for (const auto& obs_json : frame_json["observations"]) {
                            CameraObservation obs;
                            obs.landmark_id = obs_json["landmark_id"];
                            const auto& pixel = obs_json["pixel"];
                            obs.pixel = ImagePoint(pixel[0], pixel[1]);
                            if (obs_json.contains("descriptor")) {
                                obs.descriptor = obs_json["descriptor"].get<std::vector<double>>();
                            }
                            frame.observations.push_back(obs);
                            obs_index++;
                        }
                    }
                    data.camera_frames.push_back(frame);
                    frame_index++;
                }
            }
            
            // Preintegrated IMU measurements
            if (meas.contains("preintegrated_imu") && meas["preintegrated_imu"].is_array()) {
                for (const auto& preint_json : meas["preintegrated_imu"]) {
                    PreintegratedIMUData preint;
                    preint.from_keyframe_id = preint_json["from_keyframe_id"];
                    preint.to_keyframe_id = preint_json["to_keyframe_id"];
                    preint.delta_position = detail::json_to_vector3(preint_json["delta_position"]);
                    preint.delta_velocity = detail::json_to_vector3(preint_json["delta_velocity"]);
                    
                    // Handle delta_rotation 
                    if (preint_json["delta_rotation"].is_array()) {
                        if (!preint_json["delta_rotation"].empty() && 
                            preint_json["delta_rotation"][0].is_array()) {
                            // It's a matrix
                            preint.delta_rotation = detail::json_to_matrix3x3(preint_json["delta_rotation"]);
                        } else if (preint_json["delta_rotation"].size() == 9) {
                            // It's a flattened matrix
                            Matrix3x3 m;
                            const auto& flat = preint_json["delta_rotation"];
                            for (int i = 0; i < 3; ++i) {
                                for (int j = 0; j < 3; ++j) {
                                    m(i, j) = flat[i * 3 + j];
                                }
                            }
                            preint.delta_rotation = m;
                        }
                    }
                    
                    // Handle covariance (could be 2D or flattened)
                    preint.covariance = detail::json_to_vectorx(preint_json["covariance"], 225);
                    
                    preint.dt = preint_json["dt"];
                    preint.num_measurements = preint_json["num_measurements"];
                    
                    // Handle jacobian if present
                    if (preint_json.contains("jacobian") && !preint_json["jacobian"].is_null()) {
                        preint.jacobian = detail::json_to_vectorx(preint_json["jacobian"], 225);
                    }
                    
                    data.preintegrated_imu.push_back(preint);
                }
            }
        }
        
        // Post-process: Populate observation references in landmarks
        // This creates a reverse mapping from landmarks to their observations
        populateLandmarkObservations(data);
        
        return data;
    }
    
private:
    static void populateLandmarkObservations(SimulationData& data) {
        // Create a map from landmark_id to landmark index for quick lookup
        std::unordered_map<int, size_t> landmark_id_to_index;
        for (size_t i = 0; i < data.landmarks.size(); ++i) {
            landmark_id_to_index[data.landmarks[i].id] = i;
        }
        
        // Clear any existing observation references
        for (auto& landmark : data.landmarks) {
            landmark.observation_refs.clear();
        }
        
        // Iterate through all camera frames and build observation references
        for (size_t frame_idx = 0; frame_idx < data.camera_frames.size(); ++frame_idx) {
            const auto& frame = data.camera_frames[frame_idx];
            
            for (size_t obs_idx = 0; obs_idx < frame.observations.size(); ++obs_idx) {
                const auto& obs = frame.observations[obs_idx];
                
                // Find the corresponding landmark
                auto it = landmark_id_to_index.find(obs.landmark_id);
                if (it != landmark_id_to_index.end()) {
                    // Create observation reference
                    ObservationRef ref;
                    ref.camera_id = frame.camera_id;
                    ref.timestamp = frame.timestamp;
                    ref.frame_index = static_cast<int>(frame_idx);
                    ref.observation_index = static_cast<int>(obs_idx);
                    ref.pixel_u = obs.pixel.u;
                    ref.pixel_v = obs.pixel.v;
                    ref.keyframe_id = frame.keyframe_id;
                    
                    // Add to landmark's observation list
                    data.landmarks[it->second].observation_refs.push_back(ref);
                }
            }
        }
        
        // Optional: Print statistics about observations
        if (!data.landmarks.empty()) {
            size_t total_obs = 0;
            size_t keyframe_obs = 0;
            size_t max_obs = 0;
            size_t min_obs = std::numeric_limits<size_t>::max();
            
            for (const auto& landmark : data.landmarks) {
                size_t obs_count = landmark.observation_count();
                size_t kf_count = landmark.keyframe_observation_count();
                
                total_obs += obs_count;
                keyframe_obs += kf_count;
                max_obs = std::max(max_obs, obs_count);
                min_obs = std::min(min_obs, obs_count);
            }
            
            double avg_obs = static_cast<double>(total_obs) / data.landmarks.size();
            
            std::cout << "Landmark observation statistics:" << std::endl;
            std::cout << "  Total landmarks: " << data.landmarks.size() << std::endl;
            std::cout << "  Total observations: " << total_obs << std::endl;
            std::cout << "  Keyframe observations: " << keyframe_obs << std::endl;
            std::cout << "  Average observations per landmark: " << avg_obs << std::endl;
            std::cout << "  Max observations: " << max_obs << std::endl;
            std::cout << "  Min observations: " << min_obs << std::endl;
        }
    }
};

} // namespace simulation_io

#endif // SIMULATION_IO_JSON_IO_HPP