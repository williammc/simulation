#ifndef SIMULATION_IO_DATA_STRUCTURES_HPP
#define SIMULATION_IO_DATA_STRUCTURES_HPP

#include <vector>
#include <array>
#include <string>
#include <optional>
#include <map>
#include <cstdint>

namespace simulation_io {

// Basic data types
struct Vector3 {
    double x, y, z;
    
    Vector3() : x(0), y(0), z(0) {}
    Vector3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    std::vector<double> to_vector() const {
        return {x, y, z};
    }
    
    static Vector3 from_vector(const std::vector<double>& v) {
        if (v.size() != 3) {
            throw std::runtime_error("Vector3 requires exactly 3 elements");
        }
        return Vector3(v[0], v[1], v[2]);
    }
};

struct Quaternion {
    double w, x, y, z;
    
    Quaternion() : w(1), x(0), y(0), z(0) {}
    Quaternion(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}
    
    std::vector<double> to_vector() const {
        return {w, x, y, z};
    }
    
    static Quaternion from_vector(const std::vector<double>& v) {
        if (v.size() != 4) {
            throw std::runtime_error("Quaternion requires exactly 4 elements");
        }
        return Quaternion(v[0], v[1], v[2], v[3]);
    }
};

struct Matrix3x3 {
    std::array<std::array<double, 3>, 3> data;
    
    Matrix3x3() {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                data[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    
    std::vector<std::vector<double>> to_vector() const {
        std::vector<std::vector<double>> result(3);
        for (int i = 0; i < 3; ++i) {
            result[i] = std::vector<double>(data[i].begin(), data[i].end());
        }
        return result;
    }
    
    static Matrix3x3 from_vector(const std::vector<std::vector<double>>& v) {
        if (v.size() != 3) {
            throw std::runtime_error("Matrix3x3 requires 3 rows");
        }
        Matrix3x3 m;
        for (int i = 0; i < 3; ++i) {
            if (v[i].size() != 3) {
                throw std::runtime_error("Matrix3x3 requires 3 columns");
            }
            for (int j = 0; j < 3; ++j) {
                m.data[i][j] = v[i][j];
            }
        }
        return m;
    }
};

struct Matrix4x4 {
    std::array<std::array<double, 4>, 4> data;
    
    Matrix4x4() {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                data[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    
    std::vector<std::vector<double>> to_vector() const {
        std::vector<std::vector<double>> result(4);
        for (int i = 0; i < 4; ++i) {
            result[i] = std::vector<double>(data[i].begin(), data[i].end());
        }
        return result;
    }
    
    static Matrix4x4 from_vector(const std::vector<std::vector<double>>& v) {
        if (v.size() != 4) {
            throw std::runtime_error("Matrix4x4 requires 4 rows");
        }
        Matrix4x4 m;
        for (int i = 0; i < 4; ++i) {
            if (v[i].size() != 4) {
                throw std::runtime_error("Matrix4x4 requires 4 columns");
            }
            for (int j = 0; j < 4; ++j) {
                m.data[i][j] = v[i][j];
            }
        }
        return m;
    }
};

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
    Quaternion quaternion;
    std::optional<Vector3> velocity;
    std::optional<Vector3> angular_velocity;
    
    TrajectoryState() : timestamp(0) {}
};

// Landmark structure
struct Landmark {
    int id;
    Vector3 position;
    std::optional<std::vector<double>> descriptor;
    
    Landmark() : id(-1) {}
    Landmark(int id_, const Vector3& pos) : id(id_), position(pos) {}
};

// Measurement structures
struct IMUMeasurement {
    double timestamp;
    Vector3 accelerometer;
    Vector3 gyroscope;
    
    IMUMeasurement() : timestamp(0) {}
};

struct ImagePoint {
    double u, v;
    
    ImagePoint() : u(0), v(0) {}
    ImagePoint(double u_, double v_) : u(u_), v(v_) {}
    
    std::vector<double> to_vector() const {
        return {u, v};
    }
    
    static ImagePoint from_vector(const std::vector<double>& v) {
        if (v.size() != 2) {
            throw std::runtime_error("ImagePoint requires exactly 2 elements");
        }
        return ImagePoint(v[0], v[1]);
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
    
    CameraFrame() : timestamp(0) {}
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
        
        Units() : position("meters"), rotation("quaternion_wxyz"), time("seconds") {}
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
    
    SimulationData() = default;
};

} // namespace simulation_io

#endif // SIMULATION_IO_DATA_STRUCTURES_HPP