cmake_minimum_required(VERSION 3.14)
project(live-subtitle LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# whisper.cpp path (configurable via -DWHISPER_CPP_DIR=...)
set(WHISPER_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../whisper.cpp"
    CACHE PATH "Path to whisper.cpp source directory")

if(NOT EXISTS "${WHISPER_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "whisper.cpp not found at: ${WHISPER_CPP_DIR}\n"
        "Set WHISPER_CPP_DIR to the correct path:\n"
        "  cmake -DWHISPER_CPP_DIR=/path/to/whisper.cpp ..")
endif()

# whisper.cpp options - set before add_subdirectory
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2           ON  CACHE BOOL "" FORCE)

add_subdirectory(${WHISPER_CPP_DIR} ${CMAKE_BINARY_DIR}/whisper.cpp)

find_package(SDL2 REQUIRED)
find_package(Threads REQUIRED)

add_executable(live-subtitle
    src/main.cpp
    ${WHISPER_CPP_DIR}/examples/common.cpp
    ${WHISPER_CPP_DIR}/examples/common-sdl.cpp
)

target_include_directories(live-subtitle PRIVATE
    ${WHISPER_CPP_DIR}/include
    ${WHISPER_CPP_DIR}/ggml/include
    ${WHISPER_CPP_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${SDL2_INCLUDE_DIRS}
)

target_link_libraries(live-subtitle PRIVATE
    whisper
    ${SDL2_LIBRARIES}
    Threads::Threads
)
