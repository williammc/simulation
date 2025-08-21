#ifndef SIMULATION_IO_JSON_IO_HPP
#define SIMULATION_IO_JSON_IO_HPP

#include "data_structures.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace simulation_io {

using json = nlohmann::json;

// JSON conversion helpers
namespace detail {

inline json vector3_to_json(const Vector3& v) {
    return json::array({v.x, v.y, v.z});
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
            row.push_back(m.data[i][j]);
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
            m.data[i][j_] = j[i][j_];
        }
    }
    return m;
}

inline json matrix4x4_to_json(const Matrix4x4& m) {
    json result = json::array();
    for (int i = 0; i < 4; ++i) {
        json row = json::array();
        for (int j = 0; j < 4; ++j) {
            row.push_back(m.data[i][j]);
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
            m.data[i][j_] = j[i][j_];
        }
    }
    return m;
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
                j["measurements"]["camera_frames"].push_back(frame_json);
            }
        } else {
            j["measurements"]["camera_frames"] = json::array();
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
            if (meta.contains("seed")) data.metadata.seed = meta["seed"];
            
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
                        // For simplicity, we'll just use identity matrix for legacy files
                        // In production, you'd want to implement quaternion_to_rotation_matrix
                        std::cerr << "Warning: Legacy quaternion format detected, using identity rotation" << std::endl;
                        state.rotation_matrix = Matrix3x3(); // Identity
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
                for (const auto& frame_json : meas["camera_frames"]) {
                    CameraFrame frame;
                    frame.timestamp = frame_json["timestamp"];
                    frame.camera_id = frame_json["camera_id"];
                    
                    if (frame_json.contains("observations") && frame_json["observations"].is_array()) {
                        for (const auto& obs_json : frame_json["observations"]) {
                            CameraObservation obs;
                            obs.landmark_id = obs_json["landmark_id"];
                            const auto& pixel = obs_json["pixel"];
                            obs.pixel = ImagePoint(pixel[0], pixel[1]);
                            if (obs_json.contains("descriptor")) {
                                obs.descriptor = obs_json["descriptor"].get<std::vector<double>>();
                            }
                            frame.observations.push_back(obs);
                        }
                    }
                    data.camera_frames.push_back(frame);
                }
            }
        }
        
        return data;
    }
};

} // namespace simulation_io

#endif // SIMULATION_IO_JSON_IO_HPP