cmake_minimum_required(VERSION 3.12)
project(AiraCore VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

# Platform detection
if(AIRLYS_OS)
    message(STATUS "[AiraCore] Targeting Airlys OS Bare-Metal")
    add_definitions(-DAIRLYS_OS)
    set(HAL_SOURCE src/hal/hal_airlys.c)
elseif(ANDROID)
    message(STATUS "[AiraCore] Targeting Android NDK / Bionic")
    set(HAL_SOURCE src/hal/hal_android.c)
elseif(WIN32)
    message(STATUS "[AiraCore] Targeting Microsoft Windows")
    set(HAL_SOURCE src/hal/hal_windows.c)
else()
    message(STATUS "[AiraCore] Targeting GNU/Linux POSIX")
    set(HAL_SOURCE src/hal/hal_linux.c)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/O2 /W3 /arch:AVX2)
else()
    add_compile_options(-O3 -Wall -Wextra -funroll-loops)
endif()

# Core Library
add_library(aira_static STATIC
    src/aira_math.c
    src/aira_tokenizer.c
    ${HAL_SOURCE}
)

# CLI Binary
add_executable(aira-core
    src/aira_core.c
)
target_link_libraries(aira-core PRIVATE aira_static m)

# Unit Test
add_executable(test_math
    tests/test_math.c
)
target_link_libraries(test_math PRIVATE aira_static m)
