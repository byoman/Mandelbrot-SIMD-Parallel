cmake_minimum_required (VERSION 3.6)
project (ProgParallel)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/modules")

find_package(Threads REQUIRED)
find_package(SDL2 REQUIRED)
find_package(benchmark REQUIRED)
find_package(TBB REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS " -O3 -W -march=native -Wall -mavx -mavx2 -mfma -ltbb" )
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx -mavx2 -mfma -ltbb -march=native -s")

include_directories(lib/include)
include("modules/FindTBB.cmake")

add_library(renderer src/render.hpp src/render.cpp)
add_executable(view src/view.cpp)
target_include_directories(view PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(view renderer ${SDL2_LIBRARIES} ${TBB_LIBRARIES} tbb Threads::Threads)

add_executable(bench src/bench.cpp)
target_link_libraries(bench renderer tbb benchmark::benchmark)

