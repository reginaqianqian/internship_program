cmake_minimum_required(VERSION 3.15)

file(READ version VERSION_CONTENT)
string(STRIP "${VERSION_CONTENT}" VERSION_CONTENT)
project(tactile_sdk VERSION ${VERSION_CONTENT})

set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

include(FetchContent)

option(BUILD_TEST "" OFF)
option(BUILD_EXAMPLE "" OFF)
option(BUILD_PY "" OFF)

set(INFER_ENGINE_TYPES "none" "cuda" "coreml")
set(INFER_ENGINE "none" CACHE STRING "infer engine")
set_property(CACHE INFER_ENGINE PROPERTY STRINGS ${INFER_ENGINE_TYPES})

if(NOT ${INFER_ENGINE} IN_LIST INFER_ENGINE_TYPES)
    message(FATAL_ERROR "bad engine: ${INFER_ENGINE}")
endif()

# install prefix
# set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

if(BUILD_TEST)
    enable_testing()
    find_package(Catch2 REQUIRED)
    # FetchContent_Declare(Catch2 
    #     GIT_REPOSITORY https://salsa.debian.org/debian/catch2.git         
    #     GIT_TAG debian/3.4.0-2) 
    # FetchContent_MakeAvailable(Catch2)  
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
endif()

# build tactile_sdk
add_subdirectory(module/tactile_sdk)

if(BUILD_EXAMPLE)
    add_subdirectory(example_cpp)
endif()

# build python wrapper
if(BUILD_PY)
    add_subdirectory(py)
endif()

# install config
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/config
  DESTINATION etc
)

# cpack

if(APPLE)
    set(CPACK_GENERATOR productbuild)
    set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${VERSION_CONTENT})
    set(CPACK_PACKAGE_VENDOR "sharpa")
    set(CPACK_PACKAGE_VERSION ${VERSION_CONTENT})
    set(CPACK_PRODUCTBUILD_IDENTIFIER "com.sharpa.${PROJECT_NAME}")  # Reverse-DNS format [6](@ref)
    set(CPACK_COMPONENTS_ALL Unspecified)
    # set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/path/to/icon.icns")  # Optional installer icon
elseif(UNIX AND NOT APPLE)
    set(CPACK_GENERATOR DEB)
    set(CPACK_DEBIAN_PACKAGE_NAME ${PROJECT_NAME})
    set(CPACK_DEBIAN_PACKAGE_VERSION ${VERSION_CONTENT})
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)  # or arm64, etc.
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER qiyue)
    set(CPACK_DEBIAN_PACKAGE_DESCRIPTION tactile_sdk)
endif()

set(CPACK_SET_DESTDIR ON)
set(CPACK_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
include(CPack)

# uninstall
if(NOT TARGET uninstall)
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
            IMMEDIATE @ONLY)
    add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
