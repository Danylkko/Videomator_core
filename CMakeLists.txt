cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
set(PROJECT_NAME core)
project(core)

add_subdirectory(src)
add_subdirectory(test)

#FILE(COPY "${CMAKE_SOURCE_DIR}/frozen_east_text_detection.pb" DESTINATION "${PROJECT_BINARY_DIR}")