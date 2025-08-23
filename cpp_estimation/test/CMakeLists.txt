add_executable(test_json_io test_json_io.cpp)
target_link_libraries(test_json_io PRIVATE simulation_io)

add_executable(test_load_python test_load_python.cpp)
target_link_libraries(test_load_python PRIVATE simulation_io)

# Add test
add_test(NAME test_json_io COMMAND test_json_io)

# Copy test data if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_data.json")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_data.json
                   ${CMAKE_CURRENT_BINARY_DIR}/test_data.json COPYONLY)
endif()