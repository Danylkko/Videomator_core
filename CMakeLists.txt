cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
set(PROJECT_NAME core)

add_subdirectory(src)
add_subdirectory(test)





# project(CORE_API)

# enable_language(C)
# enable_language(CXX)
# set(CMAKE_DEBUG_POSTFIX _d)
# set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
# set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
# set(SOURCE_API src/Blurer.cpp src/Blurer.h)

# include(${CMAKE_SOURCE_DIR}/conanbuildinfo.cmake)
# conan_basic_setup()


# add_library(CORE_API SHARED ${SOURCE_API})
# target_compile_features(CORE_API PRIVATE cxx_std_20)

# target_link_libraries(CORE_API PUBLIC ${CONAN_LIBS})

# if(APPLE)
#     set(CMAKE_OSX_DEPLOYMENT_TARGET "10.10")
# endif()

# if(WIN32)
#     target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_DLL)
# endif()