/**
 * Mock C++ SLAM Estimator for Testing Binary Integration
 * 
 * This program mimics a real SLAM estimator by:
 * 1. Reading simulation data from JSON
 * 2. Adding noise to simulate estimation
 * 3. Writing results in the expected format
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <cmath>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

using json = nlohmann::json;
using Matrix3d = Eigen::Matrix3d;

// Structure for 3D position
struct Position3D {
    double x, y, z;
    
    Position3D operator+(const Position3D& noise) const {
        return {x + noise.x, y + noise.y, z + noise.z};
    }
};

// Helper functions for rotation matrix operations
Matrix3d quaternionToMatrix(double w, double x, double y, double z) {
    // Normalize quaternion first
    double norm = std::sqrt(w*w + x*x + y*y + z*z);
    if (norm > 1e-6) {
        w /= norm; x /= norm; y /= norm; z /= norm;
    }
    
    Matrix3d R;
    R << 1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
         2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
         2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y);
    return R;
}

Matrix3d addNoiseToRotation(const Matrix3d& R, double noise_level, std::mt19937& gen) {
    // Add small noise via axis-angle perturbation
    std::normal_distribution<> dist(0.0, noise_level * 0.1);
    
    // Small rotation vector
    Eigen::Vector3d omega(dist(gen), dist(gen), dist(gen));
    double angle = omega.norm();
    
    if (angle < 1e-8) {
        return R;
    }
    
    // Rodrigues formula for small rotation
    Eigen::Vector3d axis = omega / angle;
    Eigen::Matrix3d K;
    K << 0, -axis.z(), axis.y(),
         axis.z(), 0, -axis.x(),
         -axis.y(), axis.x(), 0;
    
    Matrix3d delta_R = Matrix3d::Identity() + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
    
    // Apply perturbation
    return delta_R * R;
}

// Add Gaussian noise to a position
Position3D addNoise(const Position3D& pos, double noise_level, std::mt19937& gen) {
    std::normal_distribution<> dist(0.0, noise_level);
    return {
        pos.x + dist(gen),
        pos.y + dist(gen),
        pos.z + dist(gen)
    };
}


int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string input_file = "";
    std::string output_file = "estimation_result.json";
    double noise_level = 0.01;
    bool simulate_failure = false;
    int delay_ms = 100;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--noise" && i + 1 < argc) {
            noise_level = std::stod(argv[++i]);
        } else if (arg == "--fail") {
            simulate_failure = true;
        } else if (arg == "--delay" && i + 1 < argc) {
            delay_ms = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Mock C++ SLAM Estimator\n"
                      << "Usage: " << argv[0] << " --input <file> [options]\n"
                      << "Options:\n"
                      << "  --input <file>    Input JSON file (required)\n"
                      << "  --output <file>   Output JSON file (default: estimation_result.json)\n"
                      << "  --noise <level>   Noise level (default: 0.01)\n"
                      << "  --delay <ms>      Processing delay in ms (default: 100)\n"
                      << "  --fail            Simulate failure\n"
                      << "  --help            Show this help\n";
            return 0;
        }
    }
    
    if (input_file.empty()) {
        std::cerr << "Error: --input is required\n";
        return 1;
    }
    
    // Simulate failure if requested
    if (simulate_failure) {
        std::cerr << "Error: Simulated failure\n";
        return 1;
    }
    
    // Simulate processing delay
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    
    try {
        // Read input JSON
        std::ifstream input(input_file);
        if (!input.is_open()) {
            std::cerr << "Error: Cannot open input file: " << input_file << std::endl;
            return 1;
        }
        
        json input_data;
        input >> input_data;
        input.close();
        
        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Process trajectory - add noise to ground truth
        json estimated_trajectory = json::array();
        if (input_data.contains("trajectory")) {
            for (const auto& point : input_data["trajectory"]) {
                json est_point;
                est_point["timestamp"] = point["timestamp"];
                
                // Add noise to position
                Position3D pos = {
                    point["position"][0].get<double>(),
                    point["position"][1].get<double>(),
                    point["position"][2].get<double>()
                };
                Position3D noisy_pos = addNoise(pos, noise_level, gen);
                est_point["position"] = {noisy_pos.x, noisy_pos.y, noisy_pos.z};
                
                // Handle rotation - convert from quaternion if present, otherwise use rotation_matrix
                Matrix3d R = Matrix3d::Identity();
                
                if (point.contains("rotation_matrix")) {
                    // Direct rotation matrix input
                    const auto& R_json = point["rotation_matrix"];
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            R(i, j) = R_json[i][j].get<double>();
                        }
                    }
                } else if (point.contains("quaternion")) {
                    // Legacy quaternion format - convert to rotation matrix
                    double w = point["quaternion"][0].get<double>();
                    double x = point["quaternion"][1].get<double>();
                    double y = point["quaternion"][2].get<double>();
                    double z = point["quaternion"][3].get<double>();
                    R = quaternionToMatrix(w, x, y, z);
                }
                
                // Add noise to rotation
                Matrix3d noisy_R = addNoiseToRotation(R, noise_level, gen);
                
                // Output as rotation matrix (list of lists)
                std::vector<std::vector<double>> R_list(3, std::vector<double>(3));
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        R_list[i][j] = noisy_R(i, j);
                    }
                }
                est_point["rotation_matrix"] = R_list;
                
                // Copy velocity if present and not null
                if (point.contains("velocity") && !point["velocity"].is_null()) {
                    Position3D vel = {
                        point["velocity"][0].get<double>(),
                        point["velocity"][1].get<double>(),
                        point["velocity"][2].get<double>()
                    };
                    Position3D noisy_vel = addNoise(vel, noise_level * 0.5, gen);
                    est_point["velocity"] = {noisy_vel.x, noisy_vel.y, noisy_vel.z};
                }
                
                estimated_trajectory.push_back(est_point);
            }
        }
        
        // Process landmarks - add noise to positions
        json estimated_landmarks = json::array();
        if (input_data.contains("landmarks")) {
            for (const auto& landmark : input_data["landmarks"]) {
                json est_landmark;
                est_landmark["id"] = landmark["id"];
                
                Position3D pos = {
                    landmark["position"][0].get<double>(),
                    landmark["position"][1].get<double>(),
                    landmark["position"][2].get<double>()
                };
                Position3D noisy_pos = addNoise(pos, noise_level * 2.0, gen);
                est_landmark["position"] = {noisy_pos.x, noisy_pos.y, noisy_pos.z};
                
                estimated_landmarks.push_back(est_landmark);
            }
        }
        
        // Create output JSON in the format expected by EstimatorResultStorage
        json output_data;
        output_data["metadata"] = {
            {"estimator", "mock_cpp"},
            {"version", "1.0.0"},
            {"noise_level", noise_level},
            {"timestamp", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())}
        };
        
        output_data["estimated_trajectory"] = estimated_trajectory;
        output_data["estimated_landmarks"] = estimated_landmarks;
        
        // Add mock runtime information
        output_data["runtime_ms"] = delay_ms;
        output_data["iterations"] = 10;
        output_data["converged"] = true;
        output_data["final_cost"] = 0.001 + noise_level;
        
        // Write output JSON
        std::ofstream output(output_file);
        if (!output.is_open()) {
            std::cerr << "Error: Cannot open output file: " << output_file << std::endl;
            return 1;
        }
        
        output << std::setw(2) << output_data << std::endl;
        output.close();
        
        std::cout << "Successfully processed " << input_file << std::endl;
        std::cout << "Output written to " << output_file << std::endl;
        std::cout << "Processed " << estimated_trajectory.size() << " poses and " 
                  << estimated_landmarks.size() << " landmarks" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}