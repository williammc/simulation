#ifndef SIMULATION_IO_ESTIMATOR_RESULT_IO_HPP
#define SIMULATION_IO_ESTIMATOR_RESULT_IO_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <optional>
#include <map>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <random>

namespace simulation_io {

using json = nlohmann::json;

// Template-based Eigen types
template<typename FLOAT>
using Vector3T = Eigen::Matrix<FLOAT, 3, 1>;

template<typename FLOAT>
using Vector4T = Eigen::Matrix<FLOAT, 4, 1>;

template<typename FLOAT>
using Matrix3x3T = Eigen::Matrix<FLOAT, 3, 3>;

template<typename FLOAT>
using Matrix4x4T = Eigen::Matrix<FLOAT, 4, 4>;

template<typename FLOAT>
using VectorXT = Eigen::Matrix<FLOAT, Eigen::Dynamic, 1>;

template<typename FLOAT>
using MatrixXT = Eigen::Matrix<FLOAT, Eigen::Dynamic, Eigen::Dynamic>;

// Default double precision types for backward compatibility
using Vector3 = Vector3T<double>;
using Vector4 = Vector4T<double>;
using Matrix3x3 = Matrix3x3T<double>;
using Matrix4x4 = Matrix4x4T<double>;
using VectorX = VectorXT<double>;
using MatrixX = MatrixXT<double>;

// Estimator types matching Python EstimatorType enum
enum class EstimatorType {
    UNKNOWN,
    EKF,
    SWBA,
    SRIF
};

// Estimated pose at a point in time
template<typename FLOAT>
struct EstimatedPoseT {
    FLOAT timestamp;
    Vector3T<FLOAT> position;
    Vector4T<FLOAT> quaternion;  // [x, y, z, w] format
    std::optional<Vector3T<FLOAT>> velocity;
    
    EstimatedPoseT() : timestamp(0), position(Vector3T<FLOAT>::Zero()), 
                       quaternion(Vector4T<FLOAT>(0, 0, 0, 1)) {}
    
    EstimatedPoseT(FLOAT t, const Vector3T<FLOAT>& p, const Matrix3x3T<FLOAT>& R) 
        : timestamp(t), position(p) {
        // Convert rotation matrix to quaternion
        Eigen::Quaternion<FLOAT> q(R);
        quaternion = Vector4T<FLOAT>(q.x(), q.y(), q.z(), q.w());
    }
    
    // Convert to JSON
    json to_json() const {
        json j;
        j["timestamp"] = timestamp;
        j["position"] = {position.x(), position.y(), position.z()};
        j["quaternion"] = {quaternion.x(), quaternion.y(), 
                           quaternion.z(), quaternion.w()};
        if (velocity.has_value()) {
            j["velocity"] = {velocity->x(), velocity->y(), velocity->z()};
        } else {
            j["velocity"] = nullptr;
        }
        return j;
    }
};

// Estimated landmark
template<typename FLOAT>
struct EstimatedLandmarkT {
    int id;
    Vector3T<FLOAT> position;
    std::optional<VectorXT<FLOAT>> descriptor;
    std::optional<Matrix3x3T<FLOAT>> covariance;
    
    EstimatedLandmarkT() : id(-1), position(Vector3T<FLOAT>::Zero()) {}
    
    EstimatedLandmarkT(int id_, const Vector3T<FLOAT>& pos) : id(id_), position(pos) {}
    
    // Convert to JSON
    json to_json() const {
        json j;
        j["id"] = id;
        j["position"] = {position.x(), position.y(), position.z()};
        
        if (descriptor.has_value()) {
            std::vector<double> desc_vec(descriptor->data(), 
                                         descriptor->data() + descriptor->size());
            j["descriptor"] = desc_vec;
        } else {
            j["descriptor"] = nullptr;
        }
        
        if (covariance.has_value()) {
            std::vector<double> cov_vec(covariance->data(), 
                                        covariance->data() + 9);
            j["covariance"] = cov_vec;
        } else {
            j["covariance"] = nullptr;
        }
        
        return j;
    }
};

// Estimator state at a point in time (for state history)
template<typename FLOAT>
struct EstimatorStateT {
    FLOAT timestamp;
    Vector3T<FLOAT> position;
    Vector4T<FLOAT> quaternion;
    std::optional<Vector3T<FLOAT>> velocity;
    std::optional<VectorXT<FLOAT>> covariance_diagonal;  // Store only diagonal for efficiency
    
    // Convert to compact JSON format
    json to_json() const {
        json j;
        j["t"] = timestamp;
        j["p"] = {position.x(), position.y(), position.z()};
        j["q"] = {quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()};
        
        if (velocity.has_value()) {
            j["v"] = {velocity->x(), velocity->y(), velocity->z()};
        }
        
        if (covariance_diagonal.has_value()) {
            std::vector<double> cov_diag(covariance_diagonal->data(),
                                         covariance_diagonal->data() + covariance_diagonal->size());
            j["cov_diag"] = cov_diag;
        }
        
        return j;
    }
};

// Trajectory (collection of poses)
template<typename FLOAT>
struct EstimatedTrajectoryT {
    std::string frame_id;
    std::vector<EstimatedPoseT<FLOAT>> poses;
    
    EstimatedTrajectoryT() : frame_id("world") {}
    
    // Add a pose
    void add_pose(const EstimatedPoseT<FLOAT>& pose) {
        poses.push_back(pose);
    }
    
    // Convert to JSON (format compatible with Python)
    json to_json() const {
        json j;
        j["frame_id"] = frame_id;
        
        json poses_array = json::array();
        for (const auto& pose : poses) {
            poses_array.push_back(pose.to_json());
        }
        j["poses"] = poses_array;
        
        return j;
    }
};

// Map of landmarks
template<typename FLOAT>
struct EstimatedMapT {
    std::string frame_id;
    std::map<int, EstimatedLandmarkT<FLOAT>> landmarks;
    
    EstimatedMapT() : frame_id("world") {}
    
    // Add a landmark
    void add_landmark(const EstimatedLandmarkT<FLOAT>& landmark) {
        landmarks[landmark.id] = landmark;
    }
    
    // Convert to JSON (format compatible with Python)
    json to_json() const {
        json j;
        j["frame_id"] = frame_id;
        
        json landmarks_array = json::array();
        for (const auto& [id, landmark] : landmarks) {
            landmarks_array.push_back(landmark.to_json());
        }
        j["landmarks"] = landmarks_array;
        
        return j;
    }
};

// Main estimator result structure
template<typename FLOAT>
struct EstimatorResultT {
    // Core components
    EstimatedTrajectoryT<FLOAT> trajectory;
    EstimatedMapT<FLOAT> landmarks;
    std::vector<EstimatorStateT<FLOAT>> state_history;
    
    // Runtime information
    FLOAT runtime_ms;
    int iterations;
    bool converged;
    FLOAT final_cost;
    
    // Metadata
    EstimatorType estimator_type;
    std::map<std::string, json> metadata;
    
    // Optional simulation info
    std::optional<std::string> input_file;
    std::optional<std::string> trajectory_type;
    std::optional<FLOAT> simulation_duration;
    
    EstimatorResultT() 
        : runtime_ms(0), iterations(0), converged(false), 
          final_cost(0), estimator_type(EstimatorType::UNKNOWN) {}
};

using EstimatorResult = EstimatorResultT<double>;

// Main I/O class for estimator results
template<typename FLOAT>
class EstimatorResultIOT {
public:
    // Save estimator result to JSON file
    static void save(const EstimatorResultT<FLOAT>& result, const std::string& filepath) {
        json j;
        
        // Generate unique IDs
        std::string run_id = generate_uuid();
        std::string timestamp = get_current_timestamp();
        
        // Metadata
        j["run_id"] = run_id;
        j["timestamp"] = timestamp;
        j["algorithm"] = estimator_type_to_string(result.estimator_type);
        
        // Configuration (minimal since C++ doesn't have full config)
        j["configuration"] = {
            {"estimator_type", estimator_type_to_string(result.estimator_type)},
            {"cpp_implementation", true}  // Flag to indicate C++ origin
        };
        
        // Results
        j["results"] = {
            {"runtime_ms", result.runtime_ms},
            {"iterations", result.iterations},
            {"converged", result.converged},
            {"final_cost", result.final_cost},
            {"num_poses", result.trajectory.poses.size()},
            {"num_landmarks", result.landmarks.landmarks.size()},
            {"metadata", result.metadata}
        };
        
        // Metrics (empty, will be computed by Python evaluation)
        j["metrics"] = json::object();
        
        // Simulation metadata if available
        if (result.input_file.has_value() || 
            result.trajectory_type.has_value() || 
            result.simulation_duration.has_value()) {
            j["simulation"] = json::object();
            if (result.input_file.has_value()) {
                j["simulation"]["input_file"] = result.input_file.value();
            }
            if (result.trajectory_type.has_value()) {
                j["simulation"]["trajectory_type"] = result.trajectory_type.value();
            }
            if (result.simulation_duration.has_value()) {
                j["simulation"]["duration"] = result.simulation_duration.value();
            }
        }
        
        // Estimated trajectory
        j["estimated_trajectory"] = result.trajectory.to_json();
        
        // Estimated landmarks
        j["estimated_landmarks"] = result.landmarks.to_json();
        
        // State history (compact format)
        json state_history = json::array();
        for (const auto& state : result.state_history) {
            state_history.push_back(state.to_json());
        }
        j["state_history"] = state_history;
        
        // Write to file
        std::ofstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filepath);
        }
        file << j.dump(2);  // Pretty print with indent of 2
        file.close();
    }
    
    // Load estimator result from JSON file (for reading back)
    static EstimatorResultT<FLOAT> load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        
        json j;
        file >> j;
        file.close();
        
        EstimatorResultT<FLOAT> result;
        
        // Parse algorithm type
        if (j.contains("algorithm")) {
            result.estimator_type = string_to_estimator_type(j["algorithm"]);
        }
        
        // Parse results
        if (j.contains("results")) {
            const auto& res = j["results"];
            result.runtime_ms = res.value("runtime_ms", 0.0);
            result.iterations = res.value("iterations", 0);
            result.converged = res.value("converged", false);
            result.final_cost = res.value("final_cost", 0.0);
            
            if (res.contains("metadata") && res["metadata"].is_object()) {
                result.metadata = res["metadata"];
            }
        }
        
        // Parse simulation metadata
        if (j.contains("simulation")) {
            const auto& sim = j["simulation"];
            if (sim.contains("input_file") && !sim["input_file"].is_null()) {
                result.input_file = sim["input_file"];
            }
            if (sim.contains("trajectory_type") && !sim["trajectory_type"].is_null()) {
                result.trajectory_type = sim["trajectory_type"];
            }
            if (sim.contains("duration") && !sim["duration"].is_null()) {
                result.simulation_duration = sim["duration"];
            }
        }
        
        // Parse trajectory
        if (j.contains("estimated_trajectory")) {
            const auto& traj = j["estimated_trajectory"];
            result.trajectory.frame_id = traj.value("frame_id", "world");
            
            if (traj.contains("poses") && traj["poses"].is_array()) {
                for (const auto& pose_json : traj["poses"]) {
                    EstimatedPoseT<FLOAT> pose;
                    pose.timestamp = pose_json["timestamp"];
                    
                    const auto& pos = pose_json["position"];
                    pose.position = Vector3T<FLOAT>(pos[0], pos[1], pos[2]);
                    
                    const auto& quat = pose_json["quaternion"];
                    pose.quaternion = Vector4T<FLOAT>(quat[0], quat[1], quat[2], quat[3]);
                    
                    if (pose_json.contains("velocity") && !pose_json["velocity"].is_null()) {
                        const auto& vel = pose_json["velocity"];
                        pose.velocity = Vector3T<FLOAT>(vel[0], vel[1], vel[2]);
                    }
                    
                    result.trajectory.poses.push_back(pose);
                }
            }
        }
        
        // Parse landmarks
        if (j.contains("estimated_landmarks")) {
            const auto& lmks = j["estimated_landmarks"];
            result.landmarks.frame_id = lmks.value("frame_id", "world");
            
            if (lmks.contains("landmarks") && lmks["landmarks"].is_array()) {
                for (const auto& lmk_json : lmks["landmarks"]) {
                    EstimatedLandmarkT<FLOAT> landmark;
                    landmark.id = lmk_json["id"];
                    
                    const auto& pos = lmk_json["position"];
                    landmark.position = Vector3T<FLOAT>(pos[0], pos[1], pos[2]);
                    
                    result.landmarks.landmarks[landmark.id] = landmark;
                }
            }
        }
        
        return result;
    }
    
private:
    // Convert estimator type to string
    static std::string estimator_type_to_string(EstimatorType type) {
        switch (type) {
            case EstimatorType::EKF: return "ekf";
            case EstimatorType::SWBA: return "swba";
            case EstimatorType::SRIF: return "srif";
            default: return "unknown";
        }
    }
    
    // Convert string to estimator type
    static EstimatorType string_to_estimator_type(const std::string& str) {
        if (str == "ekf") return EstimatorType::EKF;
        if (str == "swba") return EstimatorType::SWBA;
        if (str == "srif") return EstimatorType::SRIF;
        return EstimatorType::UNKNOWN;
    }
    
    // Generate UUID (simplified version)
    static std::string generate_uuid() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 15);
        
        std::stringstream ss;
        for (int i = 0; i < 8; ++i) {
            ss << std::hex << dis(gen);
        }
        return ss.str();
    }
    
    // Get current timestamp in ISO format
    static std::string get_current_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");
        return ss.str();
    }
};

using EstimatorResultIO = EstimatorResultIOT<double>;

// Add type aliases for other templated structs for backward compatibility
using EstimatedPose = EstimatedPoseT<double>;
using EstimatedLandmark = EstimatedLandmarkT<double>;
using EstimatorState = EstimatorStateT<double>;
using EstimatedTrajectory = EstimatedTrajectoryT<double>;
using EstimatedMap = EstimatedMapT<double>;

} // namespace simulation_io

#endif // SIMULATION_IO_ESTIMATOR_RESULT_IO_HPP