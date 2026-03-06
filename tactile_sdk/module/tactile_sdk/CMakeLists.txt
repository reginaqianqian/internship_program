
# find libs
# Use ZLIB from GetZlib if available (cross-compilation), otherwise find system package
if(NOT TARGET ZLIB::ZLIB)
    find_package(ZLIB REQUIRED)
endif()
find_package(PkgConfig REQUIRED)
if (NOT CMAKE_BUILD_EXTERNAL)
    find_package(spdlog REQUIRED)
    find_package(nlohmann_json REQUIRED)
    find_package(Eigen3 REQUIRED)
    add_subdirectory(3rd_party/msgpack11)
    find_package(fmt REQUIRED)
    set(MSGPACK11_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/msgpack11)
endif()
set(USE_OPENCV ON)
if(INFER_ENGINE STREQUAL "none")
    message("NOT NEED OPENCV!")
    set(USE_OPENCV OFF)
else()
    message("NEED OPENCV!")
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC)

if(NOT INFER_ENGINE STREQUAL "cuda")
    list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/trt.cpp")
endif()
if(NOT INFER_ENGINE STREQUAL "coreml")
    list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/coreml.cpp")
endif()

if(NOT USE_OPENCV)
    list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/drift_detect.cpp")
endif()



set(NAME tactile_sdk)

# version file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/build_info.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/src/build_info.h
)

add_library(${NAME} SHARED ${SRC} ${MSGPACK11_SOURCES})
target_compile_options(${NAME} PUBLIC -fPIC -std=c++17)
list(FIND INFER_ENGINE_TYPES "${INFER_ENGINE}" INFER_ENGINE_ID)
target_compile_definitions(${NAME} PUBLIC INFER_ENGINE=${INFER_ENGINE_ID})

target_link_libraries(${NAME} PRIVATE
    ${MSGPACK11_LIBRARIES}
    fmt::fmt
    ZLIB::ZLIB 
    spdlog::spdlog
    Eigen3::Eigen 
    ${MSGPACK11_INTERFACE_TARGET}
)
set_source_files_properties(${MSGPACK11_SOURCES} PROPERTIES COMPILE_FLAGS "-fno-rtti")

# target_compile_definitions(${NAME} PUBLIC NEED_OPENCV=${USE_OPENCV})
if(USE_OPENCV)
    add_definitions(-DUSE_OPENCV)
    find_package(OpenCV REQUIRED)
    target_link_libraries(${NAME} PRIVATE ${OpenCV_LIBS})
endif()
if(NOT CMAKE_BUILD_EXTERNAL)
    target_link_libraries(${NAME} PRIVATE nlohmann_json::nlohmann_json)
    if(APPLE)
        find_package(libjpeg-turbo REQUIRED)
        target_link_libraries(${NAME} PRIVATE libjpeg-turbo::turbojpeg)
    elseif(UNIX AND NOT APPLE)
        pkg_check_modules(TURBOJPEG REQUIRED libturbojpeg)
        target_link_libraries(${NAME} PRIVATE ${TURBOJPEG_LIBRARIES})
    endif()
else()
    # 获取turbojpeg的全局属性
    get_property(TURBOJPEG_INCLUDE_DIRS GLOBAL PROPERTY TURBOJPEG_INCLUDE_DIRS)
    get_property(TURBOJPEG_LIBRARIES GLOBAL PROPERTY TURBOJPEG_LIBRARIES)
    # 输出调试信息
    message(STATUS "tactile_sdk: TURBOJPEG_INCLUDE_DIRS from global property = ${TURBOJPEG_INCLUDE_DIRS}")
    message(STATUS "tactile_sdk: TURBOJPEG_LIBRARIES from global property = ${TURBOJPEG_LIBRARIES}")

    target_link_libraries(${NAME} PRIVATE  msgpack11::msgpack11 ${TURBOJPEG_LIBRARIES})
endif()


if(INFER_ENGINE STREQUAL "cuda")
    find_package(CUDA REQUIRED)
    target_link_libraries(${NAME} PRIVATE
        ${CUDA_LIBRARIES}
        nvinfer
        nvonnxparser
    )
elseif(INFER_ENGINE STREQUAL "coreml")
    find_package(onnxruntime REQUIRED)
    target_link_libraries(${NAME} PRIVATE
        onnxruntime::onnxruntime
    )
endif()

target_include_directories(${NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${TURBOJPEG_INCLUDE_DIRS}
        ${MSGPACK11_INCLUDE_DIRS}
)

install(TARGETS ${NAME} LIBRARY DESTINATION lib)

file(GLOB HEADER ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
install(FILES ${HEADER} DESTINATION include)

if(BUILD_TEST)
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/test SRC_TEST)

    if(NOT INFER_ENGINE STREQUAL "cuda")
        list(REMOVE_ITEM SRC_TEST "${CMAKE_CURRENT_SOURCE_DIR}/test/test_trt.cpp")
    endif()

    add_executable(${NAME}_test ${SRC_TEST})

    target_link_libraries(${NAME}_test PRIVATE
        ${NAME}
        Catch2::Catch2WithMain
        ${MSGPACK11_LIBRARIES}
        fmt::fmt
    )

    if(INFER_ENGINE STREQUAL "cuda")
        target_link_libraries(${NAME} PRIVATE
            ${CUDA_LIBRARIES}
            nvinfer
            nvonnxparser
        )
    endif()

    target_include_directories(${NAME}_test PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${MSGPACK11_INCLUDE_DIRS}
    )
    catch_discover_tests(${NAME}_test)
endif()
