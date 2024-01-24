enable_language(C CXX)

add_executable(binarytensor_test main.cpp)
target_include_directories(binarytensor_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(binarytensor_test PRIVATE BinaryTensor::Core)
include(CTest)
add_test(
    NAME TestInstantiator
    COMMAND binarytensor_test)

install(
    TARGETS binarytensor_test
    EXPORT BinaryTensorTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT runtime)

