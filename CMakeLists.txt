cmake_minimum_required(VERSION 2.8)
project(CVproject)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES src/main.cpp src/daugman.cpp src/daugman.h)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
