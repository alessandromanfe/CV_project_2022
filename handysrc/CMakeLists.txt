cmake_minimum_required(VERSION 2.8)
project(Handy)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB source
     "handysrc/*.h"
     "handysrc/*.cpp"
)
add_executable(${PROJECT_NAME} ${source} )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
