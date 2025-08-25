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

using json = nlohmann::json;

// Structure for 3D position
struct Position3D {
    double x, y, z;
    
    Position3D operator+(const Position3D& noise) const {
        return {x + noise.x, y + noise.y, z + noise.z};
    }
};

// Structure for quaternion
struct Quaternion {
    double w, x, y, z;
    
    void normalize() {
        double norm = std::sqrt(w*w + x*x + y*y + z*z);
        if (norm > 1e-6) {
            w /= norm; x /= norm; y /= norm; z /= norm;
        }
    }
};

// Add Gaussian noise to a position
Position3D addNoise(const Position3D& pos, double noise_level, std::mt19937& gen) {
    std::normal_distribution<> dist(0.0, noise_level);
    return {
        pos.x + dist(gen),
        pos.y + dist(gen),
        pos.z + dist(gen)
    };
}

// Add small noise to quaternion
Quaternion addNoiseToQuaternion(const Quaternion& q, double noise_level, std::mt19937& gen) {
    std::normal_distribution<> dist(0.0, noise_level * 0.1);
    Quaternion noisy = {
        q.w + dist(gen),
        q.x + dist(gen),
        q.y + dist(gen),
        q.z + dist(gen)
    };
    noisy.normalize();
    return noisy;
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
                
                // Add noise to quaternion if present
                if (point.contains("quaternion")) {
                    Quaternion q = {
                        point["quaternion"][0].get<double>(),
                        point["quaternion"][1].get<double>(),
                        point["quaternion"][2].get<double>(),
                        point["quaternion"][3].get<double>()
                    };
                    Quaternion noisy_q = addNoiseToQuaternion(q, noise_level, gen);
                    est_point["quaternion"] = {noisy_q.w, noisy_q.x, noisy_q.y, noisy_q.z};
                }
                
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