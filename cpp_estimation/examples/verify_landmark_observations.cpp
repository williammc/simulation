/**
 * Simple verification that landmark observation references are properly populated
 */

#include "simulation_io/json_io.hpp"
#include "simulation_io/data_structures.hpp"
#include <iostream>
#include <cassert>

using namespace simulation_io;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <json_file>" << std::endl;
        return 1;
    }
    
    std::string json_file = argv[1];
    
    try {
        // Load the simulation data
        std::cout << "\n=== Loading simulation data from: " << json_file << " ===" << std::endl;
        SimulationData data = JsonIO::load(json_file);
        
        std::cout << "\n=== Data Summary ===" << std::endl;
        std::cout << "Total landmarks: " << data.landmarks.size() << std::endl;
        std::cout << "Total camera frames: " << data.camera_frames.size() << std::endl;
        
        // Count total observations
        size_t total_observations = 0;
        for (const auto& frame : data.camera_frames) {
            total_observations += frame.observations.size();
        }
        std::cout << "Total observations: " << total_observations << std::endl;
        
        std::cout << "\n=== Landmark Observation Details ===" << std::endl;
        
        // Show details for first few landmarks
        int landmarks_to_show = std::min(5, static_cast<int>(data.landmarks.size()));
        
        for (int i = 0; i < landmarks_to_show; ++i) {
            const auto& landmark = data.landmarks[i];
            
            std::cout << "\nLandmark " << landmark.id << ":" << std::endl;
            std::cout << "  Position: [" << landmark.position.x() << ", " 
                      << landmark.position.y() << ", " << landmark.position.z() << "]" << std::endl;
            std::cout << "  Total observations: " << landmark.observation_count() << std::endl;
            std::cout << "  Keyframe observations: " << landmark.keyframe_observation_count() << std::endl;
            
            // Show first few observations
            int obs_to_show = std::min(3, static_cast<int>(landmark.observation_refs.size()));
            for (int j = 0; j < obs_to_show; ++j) {
                const auto& ref = landmark.observation_refs[j];
                std::cout << "    Obs " << j << ": frame=" << ref.frame_index 
                          << ", camera=" << ref.camera_id
                          << ", time=" << ref.timestamp 
                          << ", pixel=[" << ref.pixel_u << ", " << ref.pixel_v << "]";
                if (ref.keyframe_id.has_value()) {
                    std::cout << ", keyframe_id=" << ref.keyframe_id.value();
                }
                std::cout << std::endl;
            }
            
            if (landmark.observation_refs.size() > obs_to_show) {
                std::cout << "    ... and " << (landmark.observation_refs.size() - obs_to_show) 
                          << " more observations" << std::endl;
            }
        }
        
        // Verify consistency
        std::cout << "\n=== Verification ===" << std::endl;
        
        bool all_consistent = true;
        size_t total_refs = 0;
        
        for (const auto& landmark : data.landmarks) {
            total_refs += landmark.observation_refs.size();
            
            // Verify each reference points to valid data
            for (const auto& ref : landmark.observation_refs) {
                // Check frame index is valid
                if (ref.frame_index < 0 || ref.frame_index >= static_cast<int>(data.camera_frames.size())) {
                    std::cerr << "ERROR: Invalid frame index " << ref.frame_index 
                              << " for landmark " << landmark.id << std::endl;
                    all_consistent = false;
                    continue;
                }
                
                const auto& frame = data.camera_frames[ref.frame_index];
                
                // Check observation index is valid
                if (ref.observation_index < 0 || ref.observation_index >= static_cast<int>(frame.observations.size())) {
                    std::cerr << "ERROR: Invalid observation index " << ref.observation_index 
                              << " for landmark " << landmark.id << std::endl;
                    all_consistent = false;
                    continue;
                }
                
                const auto& obs = frame.observations[ref.observation_index];
                
                // Verify the observation actually refers to this landmark
                if (obs.landmark_id != landmark.id) {
                    std::cerr << "ERROR: Observation landmark_id mismatch. Expected " << landmark.id 
                              << " but got " << obs.landmark_id << std::endl;
                    all_consistent = false;
                }
                
                // Verify pixel coordinates match
                if (std::abs(obs.pixel.u - ref.pixel_u) > 1e-6 || 
                    std::abs(obs.pixel.v - ref.pixel_v) > 1e-6) {
                    std::cerr << "ERROR: Pixel coordinate mismatch for landmark " << landmark.id << std::endl;
                    all_consistent = false;
                }
                
                // Verify timestamp matches
                if (std::abs(frame.timestamp - ref.timestamp) > 1e-6) {
                    std::cerr << "ERROR: Timestamp mismatch for landmark " << landmark.id << std::endl;
                    all_consistent = false;
                }
            }
        }
        
        std::cout << "Total observation references: " << total_refs << std::endl;
        std::cout << "Expected (should match total observations): " << total_observations << std::endl;
        
        if (total_refs != total_observations) {
            std::cerr << "WARNING: Total references doesn't match total observations!" << std::endl;
            all_consistent = false;
        }
        
        if (all_consistent) {
            std::cout << "\n✓ All observation references are consistent!" << std::endl;
        } else {
            std::cerr << "\n✗ Found inconsistencies in observation references!" << std::endl;
            return 1;
        }
        
        // Show observation distribution
        if (!data.landmarks.empty()) {
            std::cout << "\n=== Observation Distribution ===" << std::endl;
            
            std::map<size_t, int> obs_histogram;
            for (const auto& landmark : data.landmarks) {
                obs_histogram[landmark.observation_count()]++;
            }
            
            std::cout << "Observations per landmark:" << std::endl;
            for (const auto& [count, num_landmarks] : obs_histogram) {
                std::cout << "  " << count << " observations: " << num_landmarks << " landmarks" << std::endl;
            }
        }
        
        std::cout << "\n=== Success! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}