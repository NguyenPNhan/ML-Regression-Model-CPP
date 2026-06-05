cmake_minimum_required(VERSION 3.16)

project(MyMLProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(COMMON_SOURCES
  utils/pandas.cpp
  utils/model_selection.cpp
  utils/matlib.cpp
  utils/metrics.cpp
)

file(GLOB ALGO_DIRS RELATIVE ${CMAKE_SOURCE_DIR} */main.cpp)

foreach(MAIN_FILE ${ALGO_DIRS})
  get_filename_component(ALGO_NAME ${MAIN_FILE} DIRECTORY)

  add_executable(${ALGO_NAME}
    ${MAIN_FILE}
    ${COMMON_SOURCES}
  )

  target_include_directories(${ALGO_NAME}
    PRIVATE
    utils
    class
  )
endforeach()