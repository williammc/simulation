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

// Template-based Eigen types for linear algebra

template<typename FLOAT>
using Vector2T = Eigen::Matrix<FLOAT, 2, 1>;

template<typename FLOAT>
using Vector3T = Eigen::Matrix<FLOAT, 3, 1>;

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
using Matrix3x3 = Matrix3x3T<double>;
using Matrix4x4 = Matrix4x4T<double>;
using VectorX = VectorXT<double>;
using MatrixX = MatrixXT<double>;

// Calibration structures
template<typename FLOAT>
struct CameraIntrinsicsT {
    FLOAT fx, fy, cx, cy;
    int width, height;
    std::string model;
    std::vector<FLOAT> distortion;
    
    CameraIntrinsicsT() : fx(0), fy(0), cx(0), cy(0), width(0), height(0), model("pinhole") {}
};

using CameraIntrinsics = CameraIntrinsicsT<double>;

template<typename FLOAT>
struct CameraCalibrationT {
    std::string id;
    CameraIntrinsicsT<FLOAT> intrinsics;
    Matrix4x4T<FLOAT> T_BC;  // Body to Camera transformation
    
    CameraCalibrationT() : T_BC(Matrix4x4T<FLOAT>::Identity()) {}
};

using CameraCalibration = CameraCalibrationT<double>;

template<typename FLOAT>
struct IMUNoiseParamsT {
    FLOAT noise_density;
    FLOAT random_walk;
    
    IMUNoiseParamsT() : noise_density(0), random_walk(0) {}
    IMUNoiseParamsT(FLOAT nd, FLOAT rw) : noise_density(nd), random_walk(rw) {}
};

using IMUNoiseParams = IMUNoiseParamsT<double>;

template<typename FLOAT>
struct IMUCalibrationT {
    std::string id;
    IMUNoiseParamsT<FLOAT> accelerometer;
    IMUNoiseParamsT<FLOAT> gyroscope;
    FLOAT sampling_rate;
    
    IMUCalibrationT() : sampling_rate(0) {}
};

using IMUCalibration = IMUCalibrationT<double>;

// Trajectory structures
template<typename FLOAT>
struct TrajectoryStateT {
    FLOAT timestamp;
    Vector3T<FLOAT> position;
    Matrix3x3T<FLOAT> rotation_matrix;
    std::optional<Vector3T<FLOAT>> velocity;
    std::optional<Vector3T<FLOAT>> angular_velocity;
    
    TrajectoryStateT() : timestamp(0), position(Vector3T<FLOAT>::Zero()), rotation_matrix(Matrix3x3T<FLOAT>::Identity()) {}
};

using TrajectoryState = TrajectoryStateT<double>;

// Forward declaration
template<typename FLOAT> struct ImagePointT;

// Observation reference for landmarks (populated during JSON loading)
template<typename FLOAT>
struct ObservationRefT {
    std::string camera_id;       // Which camera observed this landmark
    FLOAT timestamp;             // When it was observed
    int frame_index;             // Index in camera_frames vector
    int observation_index;       // Index in frame.observations vector
    std::optional<int> keyframe_id;  // Keyframe ID if this is a keyframe observation
    
    // We'll store pixel coordinates directly to avoid dependency issues
    FLOAT pixel_u, pixel_v;
    
    ObservationRefT() : timestamp(0), frame_index(-1), observation_index(-1), pixel_u(0), pixel_v(0) {}
};

using ObservationRef = ObservationRefT<double>;

// Landmark structure
template<typename FLOAT>
struct LandmarkT {
    int id;
    Vector3T<FLOAT> position;
    std::optional<std::vector<FLOAT>> descriptor;
    
    // Temporary member for tracking observations (populated during JSON loading)
    // This makes it easier to access all observations of this landmark
    std::vector<ObservationRefT<FLOAT>> observation_refs;
    
    LandmarkT() : id(-1), position(Vector3T<FLOAT>::Zero()) {}
    LandmarkT(int id_, const Vector3T<FLOAT>& pos) : id(id_), position(pos) {}
    
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

using Landmark = LandmarkT<double>;

// Measurement structures
template<typename FLOAT>
struct IMUMeasurementT {
    FLOAT timestamp;
    Vector3T<FLOAT> accelerometer;
    Vector3T<FLOAT> gyroscope;
    
    IMUMeasurementT() : timestamp(0), accelerometer(Vector3T<FLOAT>::Zero()), gyroscope(Vector3T<FLOAT>::Zero()) {}
};

using IMUMeasurement = IMUMeasurementT<double>;

template<typename FLOAT>
struct ImagePointT {
    FLOAT u, v;
    
    ImagePointT() : u(0), v(0) {}
    ImagePointT(FLOAT u_, FLOAT v_) : u(u_), v(v_) {}
    
    Eigen::Matrix<FLOAT, 2, 1> toVector() const {
        return Eigen::Matrix<FLOAT, 2, 1>(u, v);
    }
    
    static ImagePointT fromVector(const Eigen::Matrix<FLOAT, 2, 1>& v) {
        return ImagePointT(v.x(), v.y());
    }
};

using ImagePoint = ImagePointT<double>;

template<typename FLOAT>
struct CameraObservationT {
    int landmark_id;
    ImagePointT<FLOAT> pixel;
    std::optional<std::vector<FLOAT>> descriptor;
    std::optional<Vector2T<FLOAT>> ideal_coordinates;  // Normalized/ideal coordinates from simulation
    
    CameraObservationT() : landmark_id(-1) {}
};

using CameraObservation = CameraObservationT<double>;

template<typename FLOAT>
struct CameraFrameT {
    FLOAT timestamp;
    std::string camera_id;
    std::vector<CameraObservationT<FLOAT>> observations;
    bool is_keyframe;
    std::optional<int> keyframe_id;
    
    CameraFrameT() : timestamp(0), is_keyframe(false) {}
};

using CameraFrame = CameraFrameT<double>;

// Preintegrated IMU data structure
template<typename FLOAT>
struct PreintegratedIMUDataT {
    int from_frame_id;
    int to_frame_id;
    Vector3T<FLOAT> delta_position;
    Vector3T<FLOAT> delta_velocity;
    Matrix3x3T<FLOAT> delta_rotation;
    VectorXT<FLOAT> covariance;  // Flattened covariance matrix (15x15 -> 225 elements)
    FLOAT dt;
    int num_measurements;
    std::optional<VectorXT<FLOAT>> jacobian;  // Flattened jacobian matrix
    
    PreintegratedIMUDataT() 
        : from_frame_id(-1), 
          to_frame_id(-1), 
          delta_position(Vector3T<FLOAT>::Zero()),
          delta_velocity(Vector3T<FLOAT>::Zero()),
          delta_rotation(Matrix3x3T<FLOAT>::Identity()),
          covariance(VectorXT<FLOAT>::Zero(225)),
          dt(0), 
          num_measurements(0) {}
};

using PreintegratedIMUData = PreintegratedIMUDataT<double>;

// Metadata structure
template<typename FLOAT>
struct MetadataT {
    std::string version;
    std::string timestamp;
    std::string trajectory_type;
    FLOAT duration;
    std::string coordinate_system;
    std::optional<int> seed;
    
    struct Units {
        std::string position;
        std::string rotation;
        std::string time;
        
        Units() : position("meters"), rotation("rotation_matrix"), time("seconds") {}
    } units;
    
    MetadataT() : version("1.0"), trajectory_type("unknown"), duration(0), coordinate_system("ENU") {}
};

using Metadata = MetadataT<double>;

// Main simulation data container
template<typename FLOAT>
struct SimulationDataT {
    MetadataT<FLOAT> metadata;
    std::vector<CameraCalibrationT<FLOAT>> camera_calibrations;
    std::vector<IMUCalibrationT<FLOAT>> imu_calibrations;
    std::vector<TrajectoryStateT<FLOAT>> trajectory;
    std::vector<LandmarkT<FLOAT>> landmarks;
    std::vector<IMUMeasurementT<FLOAT>> imu_measurements;
    std::vector<CameraFrameT<FLOAT>> camera_frames;
    std::vector<PreintegratedIMUDataT<FLOAT>> preintegrated_imu;
    
    SimulationDataT() = default;
};

using SimulationData = SimulationDataT<double>;

} // namespace simulation_io

#endif // SIMULATION_IO_DATA_STRUCTURES_HPP