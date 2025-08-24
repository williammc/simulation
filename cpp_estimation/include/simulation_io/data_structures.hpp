#ifndef SIMULATION_IO_DATA_STRUCTURES_HPP
#define SIMULATION_IO_DATA_STRUCTURES_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <string>
#include <optional>
#include <map>
#include <cstdint>

namespace simulation_io {

// Use Eigen types for linear algebra
using Vector3 = Eigen::Vector3d;
using Matrix3x3 = Eigen::Matrix3d;
using Matrix4x4 = Eigen::Matrix4d;
using VectorX = Eigen::VectorXd;
using MatrixX = Eigen::MatrixXd;

// Calibration structures
struct CameraIntrinsics {
    double fx, fy, cx, cy;
    int width, height;
    std::string model;
    std::vector<double> distortion;
    
    CameraIntrinsics() : fx(0), fy(0), cx(0), cy(0), width(0), height(0), model("pinhole") {}
};

struct CameraCalibration {
    std::string id;
    CameraIntrinsics intrinsics;
    Matrix4x4 T_BC;  // Body to Camera transformation
    
    CameraCalibration() : T_BC(Matrix4x4::Identity()) {}
};

struct IMUNoiseParams {
    double noise_density;
    double random_walk;
    
    IMUNoiseParams() : noise_density(0), random_walk(0) {}
    IMUNoiseParams(double nd, double rw) : noise_density(nd), random_walk(rw) {}
};

struct IMUCalibration {
    std::string id;
    IMUNoiseParams accelerometer;
    IMUNoiseParams gyroscope;
    double sampling_rate;
    
    IMUCalibration() : sampling_rate(0) {}
};

// Trajectory structures
struct TrajectoryState {
    double timestamp;
    Vector3 position;
    Matrix3x3 rotation_matrix;
    std::optional<Vector3> velocity;
    std::optional<Vector3> angular_velocity;
    
    TrajectoryState() : timestamp(0), position(Vector3::Zero()), rotation_matrix(Matrix3x3::Identity()) {}
};

// Forward declaration
struct ImagePoint;

// Observation reference for landmarks (populated during JSON loading)
struct ObservationRef {
    std::string camera_id;       // Which camera observed this landmark
    double timestamp;             // When it was observed
    int frame_index;             // Index in camera_frames vector
    int observation_index;       // Index in frame.observations vector
    std::optional<int> keyframe_id;  // Keyframe ID if this is a keyframe observation
    
    // We'll store pixel coordinates directly to avoid dependency issues
    double pixel_u, pixel_v;
    
    ObservationRef() : timestamp(0), frame_index(-1), observation_index(-1), pixel_u(0), pixel_v(0) {}
};

// Landmark structure
struct Landmark {
    int id;
    Vector3 position;
    std::optional<std::vector<double>> descriptor;
    
    // Temporary member for tracking observations (populated during JSON loading)
    // This makes it easier to access all observations of this landmark
    std::vector<ObservationRef> observation_refs;
    
    Landmark() : id(-1), position(Vector3::Zero()) {}
    Landmark(int id_, const Vector3& pos) : id(id_), position(pos) {}
    
    // Helper to get total observation count
    size_t observation_count() const { return observation_refs.size(); }
    
    // Helper to get keyframe observation count
    size_t keyframe_observation_count() const {
        size_t count = 0;
        for (const auto& ref : observation_refs) {
            if (ref.keyframe_id.has_value()) count++;
        }
        return count;
    }
};

// Measurement structures
struct IMUMeasurement {
    double timestamp;
    Vector3 accelerometer;
    Vector3 gyroscope;
    
    IMUMeasurement() : timestamp(0), accelerometer(Vector3::Zero()), gyroscope(Vector3::Zero()) {}
};

struct ImagePoint {
    double u, v;
    
    ImagePoint() : u(0), v(0) {}
    ImagePoint(double u_, double v_) : u(u_), v(v_) {}
    
    Eigen::Vector2d toVector() const {
        return Eigen::Vector2d(u, v);
    }
    
    static ImagePoint fromVector(const Eigen::Vector2d& v) {
        return ImagePoint(v.x(), v.y());
    }
};

struct CameraObservation {
    int landmark_id;
    ImagePoint pixel;
    std::optional<std::vector<double>> descriptor;
    
    CameraObservation() : landmark_id(-1) {}
};

struct CameraFrame {
    double timestamp;
    std::string camera_id;
    std::vector<CameraObservation> observations;
    bool is_keyframe;
    std::optional<int> keyframe_id;
    
    CameraFrame() : timestamp(0), is_keyframe(false) {}
};

// Preintegrated IMU data structure
struct PreintegratedIMUData {
    int from_keyframe_id;
    int to_keyframe_id;
    Vector3 delta_position;
    Vector3 delta_velocity;
    Matrix3x3 delta_rotation;
    VectorX covariance;  // Flattened covariance matrix (15x15 -> 225 elements)
    double dt;
    int num_measurements;
    std::optional<VectorX> jacobian;  // Flattened jacobian matrix
    
    PreintegratedIMUData() 
        : from_keyframe_id(-1), 
          to_keyframe_id(-1), 
          delta_position(Vector3::Zero()),
          delta_velocity(Vector3::Zero()),
          delta_rotation(Matrix3x3::Identity()),
          covariance(VectorX::Zero(225)),
          dt(0), 
          num_measurements(0) {}
};

// Metadata structure
struct Metadata {
    std::string version;
    std::string timestamp;
    std::string trajectory_type;
    double duration;
    std::string coordinate_system;
    std::optional<int> seed;
    
    struct Units {
        std::string position;
        std::string rotation;
        std::string time;
        
        Units() : position("meters"), rotation("rotation_matrix"), time("seconds") {}
    } units;
    
    Metadata() : version("1.0"), trajectory_type("unknown"), duration(0), coordinate_system("ENU") {}
};

// Main simulation data container
struct SimulationData {
    Metadata metadata;
    std::vector<CameraCalibration> camera_calibrations;
    std::vector<IMUCalibration> imu_calibrations;
    std::vector<TrajectoryState> trajectory;
    std::vector<Landmark> landmarks;
    std::vector<IMUMeasurement> imu_measurements;
    std::vector<CameraFrame> camera_frames;
    std::vector<PreintegratedIMUData> preintegrated_imu;
    
    SimulationData() = default;
};

} // namespace simulation_io

#endif // SIMULATION_IO_DATA_STRUCTURES_HPP