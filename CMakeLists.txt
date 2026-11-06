cmake_minimum_required(VERSION 3.25)
project(drum_along VERSION 0.1.0.1 DESCRIPTION "A drummer that listens" LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Global build settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Link the C runtime statically so the released plug-in works on a machine
# without the Visual C++ redistributable. Set before the SDK, which reads it too.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(SMTG_USE_STATIC_CRT ON CACHE BOOL "" FORCE)

# ---------------------------------------------------------------------------
# Our own compile options, applied per target so they never leak into the
# VST3 SDK. The network is small dense math, so we let the compiler
# vectorise aggressively; AVX2 + FMA is a safe baseline for any DAW machine.
# ---------------------------------------------------------------------------
function(drumalong_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /arch:AVX2 /fp:fast /EHsc
                                                 $<$<CONFIG:Release>:/O2>)
        target_compile_definitions(${target} PRIVATE _USE_MATH_DEFINES NOMINMAX)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -mavx2 -mfma -ffast-math
                                                 $<$<CONFIG:Release>:-O3>)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Core: the network, the learned layer, weights I/O and the streaming engine.
# ---------------------------------------------------------------------------
add_library(drumalong_core STATIC
    core/src/WeightProgrammer.cpp
    core/src/WeightsIO.cpp
    core/src/BeatNet.cpp
    core/src/Network.cpp
    core/src/DrumAlongEngine.cpp
)
target_include_directories(drumalong_core PUBLIC core/include)
# nlohmann/json (MIT), vendored single header; only WeightsIO.cpp uses it.
target_include_directories(drumalong_core PRIVATE third_party)
drumalong_compile_options(drumalong_core)

# ---------------------------------------------------------------------------
# Offline test harness: WAV in, MIDI + debug CSV out. Streams the file through
# the engine in host-sized blocks so the code path is identical to the plugin.
# ---------------------------------------------------------------------------
add_executable(wav2mid tools/wav2mid/main.cpp)
target_link_libraries(wav2mid PRIVATE drumalong_core)
drumalong_compile_options(wav2mid)
if(MSVC)
    target_compile_definitions(wav2mid PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# ---------------------------------------------------------------------------
# VST3 plug-in (Steinberg VST 3 SDK, MIT licence, in third_party/vst3sdk).
# ---------------------------------------------------------------------------
option(DRUMALONG_BUILD_VST3 "Build the VST3 plug-in" ON)
set(VST3_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vst3sdk")

if(DRUMALONG_BUILD_VST3 AND EXISTS "${VST3_SDK_DIR}/CMakeLists.txt")
    # Only what the plug-in needs: no examples, no VSTGUI, no admin-only
    # symlink into Program Files. Steinberg's validator runs after each build.
    set(SMTG_ENABLE_VST3_PLUGIN_EXAMPLES  OFF CACHE BOOL "" FORCE)
    set(SMTG_ENABLE_VST3_HOSTING_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SMTG_ENABLE_VSTGUI_SUPPORT        OFF CACHE BOOL "" FORCE)
    set(SMTG_CREATE_PLUGIN_LINK           OFF CACHE BOOL "" FORCE)
    set(SMTG_RUN_VST_VALIDATOR            ON  CACHE BOOL "" FORCE)
    set(vst3sdk_SOURCE_DIR "${VST3_SDK_DIR}")
    add_subdirectory(${VST3_SDK_DIR} ${CMAKE_BINARY_DIR}/vst3sdk)
    smtg_enable_vst3_sdk()

    smtg_add_vst3plugin(drum_along
        plugin/source/pluginids.h
        plugin/source/version.h
        plugin/source/processor.h
        plugin/source/processor.cpp
        plugin/source/controller.h
        plugin/source/controller.cpp
        plugin/source/factory.cpp
    )
    target_link_libraries(drum_along PRIVATE sdk drumalong_core)
    target_include_directories(drum_along PRIVATE plugin/source)
    if(MSVC)
        target_compile_definitions(drum_along PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
        target_sources(drum_along PRIVATE plugin/resource/info.rc)
    endif()

    # Ship the weights inside the bundle: Contents/Resources/weights/...
    smtg_target_add_plugin_resources(drum_along
        RESOURCES "${CMAKE_CURRENT_SOURCE_DIR}/weights/drummer.json"
        OUTPUT_SUBDIRECTORY weights)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/weights/beatnet.json")
        smtg_target_add_plugin_resources(drum_along
            RESOURCES "${CMAKE_CURRENT_SOURCE_DIR}/weights/beatnet.json"
            OUTPUT_SUBDIRECTORY weights)
    endif()
    file(GLOB DRUMALONG_STYLES "${CMAKE_CURRENT_SOURCE_DIR}/weights/styles/*.json")
    smtg_target_add_plugin_resources(drum_along
        RESOURCES ${DRUMALONG_STYLES}
        OUTPUT_SUBDIRECTORY weights/styles)

    # Test host: plays a WAV through the built .vst3 and records its MIDI, so
    # the plug-in can be compared event for event with wav2mid.
    # Like Steinberg's validator, it compiles the plug provider and the
    # platform module loader itself (they are not part of sdk_hosting).
    add_executable(vst3_check
        tools/vst3_check/main.cpp
        ${VST3_SDK_DIR}/public.sdk/source/vst/hosting/plugprovider.cpp
        $<$<BOOL:${SMTG_WIN}>:${VST3_SDK_DIR}/public.sdk/source/vst/hosting/module_win32.cpp>
    )
    target_link_libraries(vst3_check PRIVATE sdk_hosting)
    if(MSVC)
        target_compile_definitions(vst3_check PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    endif()
endif()
