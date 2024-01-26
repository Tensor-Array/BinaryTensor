enable_language(C CXX)

add_executable(binarytensor_test main.cpp)
target_include_directories(binarytensor_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(binarytensor_test PRIVATE BinaryTensor::Core)
