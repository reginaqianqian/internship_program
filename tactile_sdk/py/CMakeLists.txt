
# find pybind11
execute_process(
    COMMAND pybind11-config --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
find_package(pybind11 CONFIG REQUIRED)

file(GLOB srcFiles ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
pybind11_add_module(tactile_sdk_py ${srcFiles})
# set_target_properties(tactile_sdk_py PROPERTIES OUTPUT_NAME sharpa/tactile/tactile_sdk_py)
target_link_libraries(tactile_sdk_py PRIVATE tactile_sdk)

# build py wheel
add_custom_target(tactile_sdk_wheel ALL
        COMMAND sh build.sh ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/build.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "building python wheel"
        )
add_dependencies(tactile_sdk_wheel tactile_sdk_py)
