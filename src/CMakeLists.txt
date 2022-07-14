cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(CORE_API)

enable_language(C)
enable_language(CXX)
set(CMAKE_DEBUG_POSTFIX _d)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(SOURCE_API ${CMAKE_SOURCE_DIR}/src/Blurer.cpp ${CMAKE_SOURCE_DIR}/src/Blurer.h 
        ${CMAKE_SOURCE_DIR}/src/Frame_Blurer.cpp  ${CMAKE_SOURCE_DIR}/src/Video_Renderer.cpp
        ${CMAKE_SOURCE_DIR}/src/Video_Stream.cpp)
set(${CMAKE_TOOLCHAIN_FILE} "build/conan_toolchain.cmake")


include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
set(CONAN_DISABLE_CHECK_COMPILER TRUE)
conan_basic_setup()


add_library(CORE_API SHARED ${SOURCE_API})
target_link_libraries(CORE_API PUBLIC ${CONAN_LIBS})

target_compile_features(CORE_API PRIVATE cxx_std_20)


if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.10")
endif()

if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_DLL)
endif()