cmake_minimum_required(VERSION 3.2)
project(msgpack11 VERSION 0.0.9 LANGUAGES CXX C)

option(MSGPACK11_BUILD_TESTS "Build unit tests" OFF)
option(MSGPACK11_STANDALONE_BUILD "Build as static lib" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSGPACK11_STANDALONE_BUILD)

  add_library(msgpack11 msgpack11.cpp)
  target_include_directories(msgpack11 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(msgpack11 PRIVATE -fno-rtti)
  if(NOT MSVC)
    target_compile_options(msgpack11 PRIVATE -Wall -Wextra -Werror)
  endif()
  configure_file("msgpack11.pc.in" "msgpack11.pc" @ONLY)

  if (MSGPACK11_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
  endif()

  install(TARGETS msgpack11 DESTINATION lib)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/msgpack11.hpp" DESTINATION include)
  install(FILES "${CMAKE_BINARY_DIR}/msgpack11.pc" DESTINATION lib/pkgconfig)

else()
    message(STATUS "msgpack11: Included as source dependency (not building library)")

    # 创建一个 INTERFACE 库，用于传递编译设置
    add_library(msgpack11_interface INTERFACE)
    target_include_directories(msgpack11_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(msgpack11_interface INTERFACE cxx_std_11)

    #后续可以考虑开启，把警告当出错来对待，提高代码质量，这个会应用到整个项目的编码检查中
    # if(NOT MSVC)
    #     target_compile_options(msgpack11_interface INTERFACE -Wall -Wextra -Werror)
    # endif()
    target_compile_definitions(msgpack11_interface INTERFACE MSGPACK11_FROM_SOURCE)

    # 导出源文件，供主项目使用 导出到父作用域！
    set(MSGPACK11_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/msgpack11.cpp
        PARENT_SCOPE  
    )
    set(MSGPACK11_INTERFACE_TARGET msgpack11_interface PARENT_SCOPE)

endif()