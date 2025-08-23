cmake_minimum_required(VERSION 3.14)
project(SimulationIO VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
include(FetchContent)

# Fetch nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

# Fetch Eigen
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

# Make both available
FetchContent_MakeAvailable(json Eigen)

# Create header-only library
add_library(simulation_io INTERFACE)
target_include_directories(simulation_io INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(simulation_io INTERFACE 
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)
target_compile_features(simulation_io INTERFACE cxx_std_17)

# Add alias for the library
add_library(SimulationIO::simulation_io ALIAS simulation_io)

# Option to build tests
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Option to build examples
option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation (optional, can be commented out if not needed)
# install(DIRECTORY include/simulation_io
#     DESTINATION include
# )
# 
# install(TARGETS simulation_io
#     EXPORT SimulationIOTargets
# )
# 
# install(EXPORT SimulationIOTargets
#     FILE SimulationIOTargets.cmake
#     NAMESPACE SimulationIO::
#     DESTINATION lib/cmake/SimulationIO
# )

# Create package configuration file (optional)
# include(CMakePackageConfigHelpers)
# 
# configure_package_config_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SimulationIOConfig.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/SimulationIOConfig.cmake"
#     INSTALL_DESTINATION lib/cmake/SimulationIO
# )
# 
# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/SimulationIOConfigVersion.cmake"
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY AnyNewerVersion
# )
# 
# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/SimulationIOConfig.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/SimulationIOConfigVersion.cmake"
#     DESTINATION lib/cmake/SimulationIO
# )