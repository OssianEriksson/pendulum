cmake_minimum_required(VERSION 3.21)

project(pendulum VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 20)

set(CURSES_NEED_NCURSES TRUE)

find_package(Eigen3 REQUIRED)
find_package(Matplot++ CONFIG REQUIRED)
find_package(Curses REQUIRED)

add_executable(pendulum src/pendulum.cpp)
target_link_libraries(pendulum PRIVATE Eigen3::Eigen Matplot++::cimg Matplot++::matplot ${CURSES_LIBRARIES})
target_include_directories(pendulum PRIVATE ${CURSES_INCLUDE_DIRS})
target_compile_options(pendulum PRIVATE ${CURSES_CFLAGS})
