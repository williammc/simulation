#include <simulation_io/json_io.hpp>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace simulation_io;
using json = nlohmann::json;

int main(int argc, char* argv[]) {
    std::string filepath;
    
    // Accept input path from command line, or use default
    if (argc > 1) {
        filepath = argv[1];
    } else {
        filepath = "../../data/trajectories/circle_easy.json";
        std::cout << "Usage: " << argv[0] << " <json_file>" << std::endl;
        std::cout << "Using default: " << filepath << std::endl;
    }
    
    std::cout << "Testing Python data compatibility..." << std::endl;
    std::cout << "Loading: " << filepath << std::endl;
    
    // First, let's load raw JSON to debug
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Cannot open file!" << std::endl;
        return 1;
    }
    
    json j;
    try {
        file >> j;
        std::cout << "JSON parsed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return 1;
    }
    file.close();
    
    // Now test specific sections
    try {
        std::cout << "\nTesting metadata..." << std::endl;
        if (j.contains("metadata")) {
            const auto& meta = j["metadata"];
            std::cout << "  version: " << meta.value("version", "missing") << std::endl;
            if (meta.contains("seed")) {
                std::cout << "  seed null? " << meta["seed"].is_null() << std::endl;
            } else {
                std::cout << "  seed: not present" << std::endl;
            }
        }
        
        std::cout << "\nTesting calibration..." << std::endl;
        if (j.contains("calibration")) {
            const auto& calib = j["calibration"];
            
            // Test cameras
            if (calib.contains("cameras") && calib["cameras"].is_array()) {
                std::cout << "  Cameras: " << calib["cameras"].size() << std::endl;
                for (const auto& cam : calib["cameras"]) {
                    std::cout << "    id: " << cam["id"] << std::endl;
                    std::cout << "    width type: " << cam["width"].type_name() << std::endl;
                    std::cout << "    distortion is array? " << cam["distortion"].is_array() << std::endl;
                    std::cout << "    T_BC is array? " << cam["T_BC"].is_array() << std::endl;
                }
            }
            
            // Test IMUs
            if (calib.contains("imus") && calib["imus"].is_array()) {
                std::cout << "  IMUs: " << calib["imus"].size() << std::endl;
                for (const auto& imu : calib["imus"]) {
                    std::cout << "    id: " << imu["id"] << std::endl;
                    std::cout << "    sampling_rate type: " << imu["sampling_rate"].type_name() << std::endl;
                    std::cout << "    accelerometer type: " << imu["accelerometer"].type_name() << std::endl;
                }
            }
        }
        
        std::cout << "\nTesting camera frames..." << std::endl;
        if (j.contains("measurements") && j["measurements"].contains("camera_frames")) {
            const auto& frames = j["measurements"]["camera_frames"];
            std::cout << "  Total frames: " << frames.size() << std::endl;
            
            // Check first few frames for keyframe_id
            int count = 0;
            for (const auto& frame : frames) {
                if (frame.contains("keyframe_id")) {
                    std::cout << "    Frame " << count << " keyframe_id is null? " 
                              << frame["keyframe_id"].is_null() << std::endl;
                }
                count++;
                if (count >= 3) break;
            }
        }
        
        std::cout << "\nNow trying full load with JsonIO..." << std::endl;
        try {
            SimulationData data = JsonIO::load(filepath);
            std::cout << "SUCCESS! Loaded:" << std::endl;
            std::cout << "  - Trajectory states: " << data.trajectory.size() << std::endl;
            std::cout << "  - Landmarks: " << data.landmarks.size() << std::endl;
            std::cout << "  - Camera frames: " << data.camera_frames.size() << std::endl;
            std::cout << "  - IMU measurements: " << data.imu_measurements.size() << std::endl;
            std::cout << "  - Preintegrated IMU: " << data.preintegrated_imu.size() << std::endl;
        } catch (const json::exception& je) {
            std::cerr << "JsonIO load failed with JSON error: " << je.what() << std::endl;
            std::cerr << "Error ID: " << je.id << std::endl;
            throw;
        } catch (const std::exception& e) {
            std::cerr << "JsonIO load failed: " << e.what() << std::endl;
            throw;
        }
        
    } catch (const json::exception& e) {
        std::cerr << "JSON error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}