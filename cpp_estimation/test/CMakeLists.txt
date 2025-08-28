add_executable(test_json_io test_json_io.cpp)
target_link_libraries(test_json_io PRIVATE simulation_io)

add_executable(test_load_python test_load_python.cpp)
target_link_libraries(test_load_python PRIVATE simulation_io)

add_executable(test_estimator_result_io test_estimator_result_io.cpp)
target_link_libraries(test_estimator_result_io PRIVATE simulation_io)

add_executable(test_template_precision test_template_precision.cpp)
target_link_libraries(test_template_precision PRIVATE simulation_io)

add_executable(test_swba_estimator test_swba_estimator.cpp)
target_link_libraries(test_swba_estimator PRIVATE simulation_io)

# Add tests
add_test(NAME test_json_io COMMAND test_json_io)
add_test(NAME test_estimator_result_io COMMAND test_estimator_result_io)
add_test(NAME test_template_precision COMMAND test_template_precision)
add_test(NAME test_swba_estimator COMMAND test_swba_estimator)

# Copy test data if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_data.json")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_data.json
                   ${CMAKE_CURRENT_BINARY_DIR}/test_data.json COPYONLY)
endif()