/**
 * Test that landmark observation references are properly populated
 */

#include <gtest/gtest.h>
#include "simulation_io/json_io.hpp"
#include "simulation_io/data_structures.hpp"
#include <fstream>

using namespace simulation_io;

// Create a minimal test JSON with landmarks and observations
std::string createTestJson() {
    nlohmann::json j;
    
    // Minimal metadata
    j["metadata"]["version"] = "1.0";
    j["metadata"]["timestamp"] = "2024-01-01";
    
    // Create some landmarks
    j["groundtruth"]["landmarks"] = nlohmann::json::array();
    for (int i = 0; i < 3; ++i) {
        nlohmann::json lm;
        lm["id"] = i;
        lm["position"] = {i * 1.0, i * 2.0, i * 3.0};
        j["groundtruth"]["landmarks"].push_back(lm);
    }
    
    // Create camera frames with observations
    j["measurements"]["camera_frames"] = nlohmann::json::array();
    
    // Frame 1: Observes landmarks 0 and 1
    nlohmann::json frame1;
    frame1["timestamp"] = 0.0;
    frame1["camera_id"] = "cam0";
    frame1["is_keyframe"] = true;
    frame1["keyframe_id"] = 0;
    frame1["observations"] = nlohmann::json::array();
    
    nlohmann::json obs1;
    obs1["landmark_id"] = 0;
    obs1["pixel"] = {100.0, 200.0};
    frame1["observations"].push_back(obs1);
    
    nlohmann::json obs2;
    obs2["landmark_id"] = 1;
    obs2["pixel"] = {150.0, 250.0};
    frame1["observations"].push_back(obs2);
    
    j["measurements"]["camera_frames"].push_back(frame1);
    
    // Frame 2: Observes landmarks 1 and 2
    nlohmann::json frame2;
    frame2["timestamp"] = 0.1;
    frame2["camera_id"] = "cam0";
    frame2["is_keyframe"] = false;
    frame2["observations"] = nlohmann::json::array();
    
    nlohmann::json obs3;
    obs3["landmark_id"] = 1;
    obs3["pixel"] = {160.0, 260.0};
    frame2["observations"].push_back(obs3);
    
    nlohmann::json obs4;
    obs4["landmark_id"] = 2;
    obs4["pixel"] = {180.0, 280.0};
    frame2["observations"].push_back(obs4);
    
    j["measurements"]["camera_frames"].push_back(frame2);
    
    // Frame 3: Observes all landmarks (keyframe)
    nlohmann::json frame3;
    frame3["timestamp"] = 0.2;
    frame3["camera_id"] = "cam0";
    frame3["is_keyframe"] = true;
    frame3["keyframe_id"] = 1;
    frame3["observations"] = nlohmann::json::array();
    
    for (int i = 0; i < 3; ++i) {
        nlohmann::json obs;
        obs["landmark_id"] = i;
        obs["pixel"] = {200.0 + i * 10, 300.0 + i * 10};
        frame3["observations"].push_back(obs);
    }
    
    j["measurements"]["camera_frames"].push_back(frame3);
    
    return j.dump(2);
}

TEST(LandmarkObservations, PopulateObservationReferences) {
    // Create temporary test file
    const std::string test_file = "/tmp/test_landmark_obs.json";
    {
        std::ofstream file(test_file);
        file << createTestJson();
        file.close();
    }
    
    // Load the data
    SimulationData data = JsonIO::load(test_file);
    
    // Verify landmarks were loaded
    ASSERT_EQ(data.landmarks.size(), 3);
    
    // Verify camera frames were loaded
    ASSERT_EQ(data.camera_frames.size(), 3);
    
    // Check landmark 0: should have 2 observations (frames 0 and 2)
    EXPECT_EQ(data.landmarks[0].observation_count(), 2);
    EXPECT_EQ(data.landmarks[0].keyframe_observation_count(), 2); // Both are keyframes
    
    // Verify observation details for landmark 0
    const auto& lm0_refs = data.landmarks[0].observation_refs;
    ASSERT_EQ(lm0_refs.size(), 2);
    
    // First observation (frame 0, observation 0)
    EXPECT_EQ(lm0_refs[0].frame_index, 0);
    EXPECT_EQ(lm0_refs[0].observation_index, 0);
    EXPECT_EQ(lm0_refs[0].camera_id, "cam0");
    EXPECT_NEAR(lm0_refs[0].timestamp, 0.0, 1e-6);
    EXPECT_TRUE(lm0_refs[0].keyframe_id.has_value());
    EXPECT_EQ(lm0_refs[0].keyframe_id.value(), 0);
    EXPECT_NEAR(lm0_refs[0].pixel_u, 100.0, 1e-6);
    EXPECT_NEAR(lm0_refs[0].pixel_v, 200.0, 1e-6);
    
    // Second observation (frame 2, observation 0)
    EXPECT_EQ(lm0_refs[1].frame_index, 2);
    EXPECT_EQ(lm0_refs[1].observation_index, 0);
    EXPECT_TRUE(lm0_refs[1].keyframe_id.has_value());
    EXPECT_EQ(lm0_refs[1].keyframe_id.value(), 1);
    
    // Check landmark 1: should have 3 observations (all frames)
    EXPECT_EQ(data.landmarks[1].observation_count(), 3);
    EXPECT_EQ(data.landmarks[1].keyframe_observation_count(), 2); // 2 keyframes
    
    // Check landmark 2: should have 2 observations (frames 1 and 2)
    EXPECT_EQ(data.landmarks[2].observation_count(), 2);
    EXPECT_EQ(data.landmarks[2].keyframe_observation_count(), 1); // 1 keyframe
    
    // Verify we can access observations through the references
    for (const auto& landmark : data.landmarks) {
        for (const auto& ref : landmark.observation_refs) {
            // Verify the reference points to valid data
            ASSERT_LT(ref.frame_index, static_cast<int>(data.camera_frames.size()));
            ASSERT_GE(ref.frame_index, 0);
            
            const auto& frame = data.camera_frames[ref.frame_index];
            ASSERT_LT(ref.observation_index, static_cast<int>(frame.observations.size()));
            ASSERT_GE(ref.observation_index, 0);
            
            const auto& obs = frame.observations[ref.observation_index];
            EXPECT_EQ(obs.landmark_id, landmark.id);
            
            // Verify pixel coordinates match
            EXPECT_NEAR(obs.pixel.u, ref.pixel_u, 1e-6);
            EXPECT_NEAR(obs.pixel.v, ref.pixel_v, 1e-6);
        }
    }
    
    // Clean up
    std::remove(test_file.c_str());
}

TEST(LandmarkObservations, EmptyLandmarks) {
    nlohmann::json j;
    j["metadata"]["version"] = "1.0";
    j["groundtruth"]["landmarks"] = nlohmann::json::array();
    j["measurements"]["camera_frames"] = nlohmann::json::array();
    
    const std::string test_file = "/tmp/test_empty_landmarks.json";
    {
        std::ofstream file(test_file);
        file << j.dump(2);
        file.close();
    }
    
    SimulationData data = JsonIO::load(test_file);
    
    EXPECT_EQ(data.landmarks.size(), 0);
    EXPECT_EQ(data.camera_frames.size(), 0);
    
    std::remove(test_file.c_str());
}

TEST(LandmarkObservations, NoObservations) {
    nlohmann::json j;
    j["metadata"]["version"] = "1.0";
    
    // Landmarks but no observations
    j["groundtruth"]["landmarks"] = nlohmann::json::array();
    for (int i = 0; i < 3; ++i) {
        nlohmann::json lm;
        lm["id"] = i;
        lm["position"] = {i * 1.0, 0.0, 0.0};
        j["groundtruth"]["landmarks"].push_back(lm);
    }
    
    j["measurements"]["camera_frames"] = nlohmann::json::array();
    
    const std::string test_file = "/tmp/test_no_obs.json";
    {
        std::ofstream file(test_file);
        file << j.dump(2);
        file.close();
    }
    
    SimulationData data = JsonIO::load(test_file);
    
    ASSERT_EQ(data.landmarks.size(), 3);
    
    // All landmarks should have zero observations
    for (const auto& landmark : data.landmarks) {
        EXPECT_EQ(landmark.observation_count(), 0);
        EXPECT_EQ(landmark.keyframe_observation_count(), 0);
        EXPECT_TRUE(landmark.observation_refs.empty());
    }
    
    std::remove(test_file.c_str());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}