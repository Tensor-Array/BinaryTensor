cmake_minimum_required(VERSION 3.8.0)

project(BinaryTensor)

include(GNUInstallDirs)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/src/binary_tensor"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT headers
    FILES_MATCHING
    PATTERN "*.hh"
    PATTERN "*.h"
    )

add_subdirectory("src/binary_tensor/core")
add_subdirectory("test")

set(CPACK_PACKAGE_NAME "BinaryTensor")
set(CPACK_PACKAGE_VENDOR "TensorArray-Creators")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY)

set(CPACK_PACKAGE_VERSION "0.1.0")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "A machine learning libraries")

install(
    EXPORT BinaryTensorTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BinaryTensor
    NAMESPACE BinaryTensor::
    FILE BinaryTensorTargets.cmake
    )

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "Config.cmake.in"
    "BinaryTensorConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BinaryTensor
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
    )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/BinaryTensorConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
    )

### Install Config and ConfigVersion files
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/BinaryTensorConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/BinaryTensorConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BinaryTensor
    )

include(CPack)
cpack_add_component(libraries
DISPLAY_NAME "Libraries"
GROUP Development)

cpack_add_component(
    headers
    DISPLAY_NAME "C++ Headers"
    GROUP Development)

cpack_add_component_group(
    Development
    EXPANDED
    PROJECT_DESCRIPTION
    "All of the tools you'll ever need to develop software")
