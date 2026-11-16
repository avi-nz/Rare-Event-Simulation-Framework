cmake_minimum_required(VERSION 3.20)

project(rareevent LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(RAREEVENT_BUILD_PYTHON     "Build the Python extension module" ${SKBUILD})
option(RAREEVENT_BUILD_TESTS      "Build the C++ unit tests"          OFF)
option(RAREEVENT_BUILD_BENCHMARKS "Build the C++ benchmarks"          OFF)

# ---------------------------------------------------------------------------
# C++ core library  (cpp/include + cpp/src)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE RAREEVENT_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/*.cpp")

if(RAREEVENT_CORE_SOURCES)
    add_library(rareevent_core STATIC ${RAREEVENT_CORE_SOURCES})
    target_include_directories(rareevent_core
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
    # Needed because the static library is linked into a shared Python module.
    set_target_properties(rareevent_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
    # No .cpp files yet (or everything is header-only templates).
    add_library(rareevent_core INTERFACE)
    target_include_directories(rareevent_core
        INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
endif()
add_library(rareevent::core ALIAS rareevent_core)

# ---------------------------------------------------------------------------
# Python bindings  (bindings/ -> rareevent/_core)
# ---------------------------------------------------------------------------
if(RAREEVENT_BUILD_PYTHON)
    file(GLOB_RECURSE RAREEVENT_BINDING_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cpp")

    if(RAREEVENT_BINDING_SOURCES)
        find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
        find_package(pybind11 CONFIG REQUIRED)

        pybind11_add_module(_core MODULE ${RAREEVENT_BINDING_SOURCES})
        target_link_libraries(_core PRIVATE rareevent::core)

        # Installs next to python/rareevent/__init__.py inside the wheel.
        install(TARGETS _core LIBRARY DESTINATION rareevent)
    else()
        message(STATUS "No files in bindings/ yet - skipping the Python extension module.")
    endif()
endif()

# ---------------------------------------------------------------------------
# C++ tests and benchmarks (each folder gets its own CMakeLists.txt later)
# ---------------------------------------------------------------------------
if(RAREEVENT_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(cpp/tests)
endif()

if(RAREEVENT_BUILD_BENCHMARKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/benchmarks/CMakeLists.txt")
    add_subdirectory(cpp/benchmarks)
endif()
