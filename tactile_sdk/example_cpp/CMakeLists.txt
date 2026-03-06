
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

add_executable(example-offline example_offline.cpp)
target_include_directories(example-offline PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(example-offline PRIVATE
    tactile_sdk
    ${OpenCV_LIBRARIES}
    fmt::fmt
)

add_executable(example-fetch example_fetch.cpp)
target_include_directories(example-fetch PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(example-fetch PRIVATE
    tactile_sdk
    ${OpenCV_LIBRARIES}
    fmt::fmt
)

add_executable(example-callback example_callback.cpp)
target_include_directories(example-callback PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(example-callback PRIVATE
    tactile_sdk
    ${OpenCV_LIBRARIES}
    fmt::fmt
)

add_executable(example-fetch-cuda example_fetch_cuda.cpp)
target_include_directories(example-fetch-cuda PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(example-fetch-cuda PRIVATE
    tactile_sdk
    ${OpenCV_LIBRARIES}
    fmt::fmt
)

add_executable(example-callback-cuda example_callback_cuda.cpp)
target_include_directories(example-callback-cuda PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(example-callback-cuda PRIVATE
    tactile_sdk
    ${OpenCV_LIBRARIES}
    fmt::fmt
)
