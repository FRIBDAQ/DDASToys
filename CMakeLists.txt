cmake_minimum_required(VERSION 3.13)

project(DDASToys VERSION 6.5.0 LANGUAGES CXX C)

enable_testing()

# =============================================================================
# Options
# =============================================================================

option(DDASTOYS_CUDA_DEV
    "Build the experimental CUDA GPU fit editor plugins: libFitEditorCudaAnalytic 
    (analytic fit via the CUDA fit engine), libFitEditorDE (differential evolution 
    fit), and libFitEditorCudaPSO (particle-swarm fit). 
    Requires nvcc + libCudaOptimize." OFF)
option(DDASTOYS_MLINFERENCE
    "Build the ML inference fit editor (requires LibTorch)" ON)
option(DDASTOYS_TRACEVIEW
    "Build the TraceView Qt5 diagnostic GUI (requires Qt5 >= 5.11)" ON)
option(DDASTOYS_DOCS
    "Build Doxygen source docs and DocBook user manual" ON)
option(DDASTOYS_TIMING
    "Compile ENABLE_TIMING into FitEditors for inference profiling" OFF)

set(DDASTOYS_CUDA_MAXPOINTS 1024 CACHE STRING
    "Maximum trace length compiled into the CUDA fit engine (MAXPOINTS macro)")

# =============================================================================
# Language standards
# =============================================================================

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# All objects go into shared libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# RPATH policy
# =============================================================================

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)   # also adds DAQLIB, UFMTLIB, etc.
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# =============================================================================
# Compile definitions
# =============================================================================

if(DDASTOYS_TIMING)
    add_compile_definitions(ENABLE_TIMING)
endif()

# =============================================================================
# NSCLDAQ — read directly from the DAQROOT environment variable
#
# No find_package or find module.  daqsetup.bash sets DAQROOT; we use it as-is.
# UnifiedFormat is bundled at ${DAQROOT}/unifiedformat in NSCLDAQ 12.2+.
# =============================================================================

if("$ENV{DAQROOT}" STREQUAL "")
    message(FATAL_ERROR
        "DAQROOT is not set in the environment.\n"
        "Source daqsetup.bash for NSCLDAQ >= 12.2-000 before running cmake:\n"
        "  source /path/to/nscldaq/12.2-000/daqsetup.bash")
endif()

set(DAQROOT  "$ENV{DAQROOT}")
set(UFMTROOT "${DAQROOT}/unifiedformat"
    CACHE PATH "UnifiedFormat installation root (default: \${DAQROOT}/unifiedformat)")

set(DAQINC  "${DAQROOT}/include")
set(DAQLIB  "${DAQROOT}/lib")
set(UFMTINC "${UFMTROOT}/include")
set(UFMTLIB "${UFMTROOT}/lib")

message(STATUS "NSCLDAQ  : ${DAQROOT}")
message(STATUS "UFMTROOT : ${UFMTROOT}")

# Two INTERFACE IMPORTED targets cover all use cases:
#
#   NSCLDAQ::Headers  — include paths only, no link.
#     Used by ddastoys_unpacker_objs (needs DataFormat.h) and
#     FitEditorAnalytic/Template/MLInference (need CBuiltRingItemEditor.h).
#     The plugins must NOT link NSCLDAQ libs; EventEditor resolves them.
#
#   NSCLDAQ::NSCLDAQ  — include paths + all runtime libraries.
#     Used by ddastoys_ring_objs, eeconverter, and traceview.

add_library(NSCLDAQ::Headers INTERFACE IMPORTED GLOBAL)
target_include_directories(NSCLDAQ::Headers INTERFACE
    "${DAQINC}"
    "${UFMTINC}"
)

add_library(NSCLDAQ::NSCLDAQ INTERFACE IMPORTED GLOBAL)
target_include_directories(NSCLDAQ::NSCLDAQ INTERFACE
    "${DAQINC}"
    "${UFMTINC}"
)
target_link_libraries(NSCLDAQ::NSCLDAQ INTERFACE
    "${DAQLIB}/libdataformat.so"
    "${DAQLIB}/libException.so"
    "${DAQLIB}/liburl.so"
    "${DAQLIB}/libdaqio.so"
    "${UFMTLIB}/libAbstractFormat.so"
)

# =============================================================================
# CUDA fitting (optional)
# =============================================================================

if(DDASTOYS_CUDA_DEV)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_path(CUDAOPTIMIZE_INCLUDE_DIR
        NAMES DE_Optimizer.h
        HINTS /usr/opt/libcudaoptimize/include
        DOC "libCudaOptimize include directory")
    find_library(CUDAOPTIMIZE_LIB
        NAMES CudaOptimize
        HINTS /usr/opt/libcudaoptimize/lib
        DOC "libCudaOptimize shared library")

    if(NOT CUDAOPTIMIZE_INCLUDE_DIR OR NOT CUDAOPTIMIZE_LIB)
        message(FATAL_ERROR
            "DDASTOYS_CUDA_DEV=ON but libCudaOptimize was not found under "
            "/usr/opt/libcudaoptimize.  Set CUDAOPTIMIZE_INCLUDE_DIR and "
            "CUDAOPTIMIZE_LIB to point at your installation.")
    endif()

    # NOTE: the CUDA macro is applied per-target (only to the analytic sources
    # of libFitEditorCudaAnalytic), NOT globally, so the normal CPU
    # libFitEditorAnalytic keeps using the serial fit engine. MAXPOINTS is set
    # on the CUDA object libraries below.
    message(STATUS "CUDA enabled — MAXPOINTS=${DDASTOYS_CUDA_MAXPOINTS}")
endif()

# =============================================================================
# Required external packages
# =============================================================================

# --- GNU Scientific Library ---
# FindGSL is built into CMake (since 3.2); provides GSL::gsl, GSL::gslcblas.
# Only lmfit_analytic.cpp and lmfit_template.cpp call GSL at link time.
# functions_analytic.cpp and functions_template.cpp use only <cmath>.
find_package(GSL 2.4 REQUIRED)

# --- CERN ROOT ---
find_package(ROOT REQUIRED COMPONENTS Core Tree RIO Hist Graf Gpad)
include(${ROOT_USE_FILE})
message(STATUS "ROOT ${ROOT_VERSION} at ${ROOTSYS}")

# --- CppUnit (unit tests) ---
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(CPPUNIT cppunit)
endif()
if(NOT CPPUNIT_FOUND)
    find_library(CPPUNIT_LIBRARIES  NAMES cppunit)
    find_path   (CPPUNIT_INCLUDE_DIRS cppunit/TestRunner.h)
    if(CPPUNIT_LIBRARIES AND CPPUNIT_INCLUDE_DIRS)
        set(CPPUNIT_FOUND TRUE)
    endif()
endif()
if(NOT CPPUNIT_FOUND)
    message(WARNING "CppUnit not found — unit tests will be skipped.")
endif()

# --- gengetopt ---
find_program(GENGETOPT_EXECUTABLE gengetopt)
if(NOT GENGETOPT_EXECUTABLE)
    message(WARNING
        "gengetopt not found — eeconverter will not be built.  "
        "Install gengetopt or set GENGETOPT_EXECUTABLE.")
endif()

# =============================================================================
# Optional packages
# =============================================================================

if(DDASTOYS_MLINFERENCE)
    find_package(Torch QUIET)
    if(NOT Torch_FOUND)
        set(DDASTOYS_MLINFERENCE OFF)
        message(WARNING
            "LibTorch not found — ML inference editor disabled.  "
            "Pass -DCMAKE_PREFIX_PATH=/path/to/libtorch to enable it.")
    else()
        message(STATUS "LibTorch: ${TORCH_INSTALL_PREFIX}")
    endif()
endif()

if(DDASTOYS_TRACEVIEW)
    find_package(Qt5 5.11 COMPONENTS Widgets QUIET)
    if(NOT Qt5_FOUND)
        set(DDASTOYS_TRACEVIEW OFF)
        message(WARNING "Qt5 >= 5.11 not found — TraceView disabled.")
    else()
        message(STATUS "Qt5 ${Qt5_VERSION}")
    endif()
endif()

if(DDASTOYS_DOCS)
    find_package(Doxygen QUIET)
    find_program(DOCBOOK2HTML_EXE NAMES docbook2html db2html)
    find_program(DOCBOOK2PDF_EXE  NAMES docbook2pdf  db2pdf)
    if(NOT Doxygen_FOUND AND NOT DOCBOOK2HTML_EXE)
        set(DDASTOYS_DOCS OFF)
        message(WARNING "Neither Doxygen nor docbook2html found — docs disabled.")
    endif()
endif()

# =============================================================================
# DDASFormat submodule
# =============================================================================

# DDASFormat declares option(BUILD_DOCS ...).  Disable it here so it doesn't
# create a conflicting `docs` target; use DDASTOYS_DOCS for our own docs.
set(BUILD_DOCS OFF CACHE BOOL
    "DDASFormat docs (managed by DDASToys via DDASTOYS_DOCS)" FORCE)

add_subdirectory(DDASFormat)

# DDASFormat's CMakeLists.txt never calls target_include_directories, so
# patch the target here to make `target_link_libraries(... DDASFormat)` work.
target_include_directories(DDASFormat
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/DDASFormat>
        $<INSTALL_INTERFACE:include>
)

# =============================================================================
# Internal OBJECT libraries
#
# In CMake >= 3.12, target_link_libraries(consumer PRIVATE obj_lib) includes
# the compiled objects AND propagates the OBJECT library's PUBLIC requirements.
#
# =============================================================================

# --- Configuration + profiling ---
add_library(ddastoys_config_objs OBJECT
    Configuration.cpp
    profiling.cpp
)
target_include_directories(ddastoys_config_objs PUBLIC ${CMAKE_SOURCE_DIR})

# --- Analytic function evaluation — NO GSL (functions_analytic.cpp uses only <cmath>) ---
add_library(ddastoys_analytic_funcs_objs OBJECT
    functions_analytic.cpp
)
target_include_directories(ddastoys_analytic_funcs_objs PUBLIC ${CMAKE_SOURCE_DIR})

# --- Template function evaluation — NO GSL (functions_template.cpp uses only <cmath>) ---
add_library(ddastoys_template_funcs_objs OBJECT
    functions_template.cpp
)
target_include_directories(ddastoys_template_funcs_objs PUBLIC ${CMAKE_SOURCE_DIR})

# --- DDASFitHit unpacker ---
# DDASFitHitUnpacker.cpp uses <DataFormat.h> for ufmt:: ring-item structs.
# Those are plain C structs — no libdataformat link needed — but the header
# path is.  NSCLDAQ::Headers provides it without pulling in any libraries.
add_library(ddastoys_unpacker_objs OBJECT
    DDASFitHitUnpacker.cpp
)
target_include_directories(ddastoys_unpacker_objs
    PUBLIC
        "${CMAKE_SOURCE_DIR}"
        "${UFMTINC}"   # UnifiedFormat DataFormat.h — search here first
        "${DAQINC}"
)
target_link_libraries(ddastoys_unpacker_objs
    PUBLIC DDASFormat NSCLDAQ::Headers
)

# --- NSCLDAQ ring-item processor ---
# CRingItemProcessor.cpp constructs real NSCLDAQ ring item objects — the
# final binary needs the runtime libraries, propagated via NSCLDAQ::NSCLDAQ.
add_library(ddastoys_ring_objs OBJECT
    CRingItemProcessor.cpp
)
target_include_directories(ddastoys_ring_objs PUBLIC 
        ${CMAKE_SOURCE_DIR}         
        "${DAQINC}"
        "${UFMTINC}"
)
target_link_libraries(ddastoys_ring_objs
    PUBLIC NSCLDAQ::NSCLDAQ DDASFormat
)

# --- CUDA objects (optional, DDASTOYS_CUDA_DEV) ---
# Two independent implementations, split so each dev plugin links only what it
# needs:
#   ddastoys_cuda_engine_objs : CudaFitEngine1/2 — GPU LM residual/Jacobian
#                               offload, used by libFitEditorCudaAnalytic.
#                               Raw CUDA, no libCudaOptimize.
#   ddastoys_cuda_de/pso_objs : cudafit1/2 — libCudaOptimize global optimizers,
#                               used by libFitEditorCudaDE/PSO.
if(DDASTOYS_CUDA_DEV)
    add_library(ddastoys_cuda_engine_objs OBJECT
        CudaFitEngineAnalytic.cu
    )
    set_target_properties(ddastoys_cuda_engine_objs PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_definitions(ddastoys_cuda_engine_objs PRIVATE MAXPOINTS=${DDASTOYS_CUDA_MAXPOINTS})
    target_include_directories(ddastoys_cuda_engine_objs PRIVATE ${CMAKE_SOURCE_DIR})

    # DE variant (default):
    add_library(ddastoys_cuda_de_objs OBJECT cudafit_analytic.cu)
    set_target_properties(ddastoys_cuda_de_objs PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_definitions(ddastoys_cuda_de_objs  PRIVATE MAXPOINTS=${DDASTOYS_CUDA_MAXPOINTS})
    target_include_directories(ddastoys_cuda_de_objs  PRIVATE ${CMAKE_SOURCE_DIR} ${CUDAOPTIMIZE_INCLUDE_DIR})
    target_link_libraries(ddastoys_cuda_de_objs  PUBLIC "${CUDAOPTIMIZE_LIB}")

    # PSO variant (adds USE_PSO):
    add_library(ddastoys_cuda_pso_objs OBJECT cudafit_analytic.cu)
    set_target_properties(ddastoys_cuda_pso_objs PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_definitions(ddastoys_cuda_pso_objs PRIVATE MAXPOINTS=${DDASTOYS_CUDA_MAXPOINTS} USE_PSO)
    target_include_directories(ddastoys_cuda_pso_objs PRIVATE ${CMAKE_SOURCE_DIR} ${CUDAOPTIMIZE_INCLUDE_DIR})
    target_link_libraries(ddastoys_cuda_pso_objs PUBLIC "${CUDAOPTIMIZE_LIB}")
endif()

# =============================================================================
# Public shared library: libDDASFitHitUnpacker.so
# =============================================================================

add_library(DDASFitHitUnpacker SHARED)
target_link_libraries(DDASFitHitUnpacker
    PRIVATE ddastoys_unpacker_objs DDASFormat
)
target_link_options(DDASFitHitUnpacker PRIVATE "LINKER:-z,defs")

# =============================================================================
# FitEditor plugin shared libraries
#
# Compile against NSCLDAQ headers (via ${DAQINC}/${UFMTINC}).
# Do NOT link NSCLDAQ libs — EventEditor resolves those symbols at load time.
# Exact object lists mirror the Makefile.
# =============================================================================

# --- libFitEditorAnalytic.so (CPU serial fit) ---
add_library(FitEditorAnalytic SHARED
    FitEditorAnalytic.cpp
    CFitEngine.cpp
    lmfit_analytic.cpp
    SerialFitEngineAnalytic.cpp
)
target_include_directories(FitEditorAnalytic PRIVATE ${DAQINC} ${UFMTINC})
target_link_libraries(FitEditorAnalytic
    PRIVATE
        ddastoys_config_objs
        ddastoys_analytic_funcs_objs
        DDASFormat
        GSL::gsl GSL::gslcblas
)

# --- libFitEditorTemplate.so ---
add_library(FitEditorTemplate SHARED
    FitEditorTemplate.cpp
    lmfit_template.cpp
)
target_include_directories(FitEditorTemplate PRIVATE ${DAQINC} ${UFMTINC})
target_link_libraries(FitEditorTemplate
    PRIVATE
        ddastoys_config_objs
        ddastoys_template_funcs_objs
        DDASFormat
        GSL::gsl GSL::gslcblas
)

# --- libFitEditorMLInference.so (optional) ---
# GSL linked per Makefile blanket CXXLDFLAGS (even though not directly used).
if(DDASTOYS_MLINFERENCE)
    add_library(FitEditorMLInference SHARED
        FitEditorMLInference.cpp
        mlinference.cpp
    )
    target_include_directories(FitEditorMLInference PRIVATE ${DAQINC} ${UFMTINC})
    target_link_libraries(FitEditorMLInference
        PRIVATE
            ddastoys_config_objs
            ddastoys_analytic_funcs_objs
            DDASFormat
            ${TORCH_LIBRARIES}
    )
    set_target_properties(FitEditorMLInference PROPERTIES CXX_STANDARD 14)
endif()

# --- Experimental CUDA plugins (analytic-style GPU fits) ---
# Three additional analytic plugins, built alongside the normal CPU 
# libFitEditorAnalytic (which is unaffected):
#   libFitEditorCudaAnalytic.so : the same analytic editor/lmfit sources compiled
#     with -DCUDA, so lmfit_analytic uses CudaFitEngine1/2 (GPU LM residual/
#     Jacobian offload) instead of the serial engine. No new C++ code.
#   libFitEditorCudaDE.so : the FitEditorCuda editor driving the libCudaOptimize
#     differential evolution optimizer (cudafit1/2), always doing both single- and 
#     double-pulse fits with the trace kept resident on the GPU between them.
#   libFitEditorCudaPSO.so : the FitEditorCuda editor driving the libCudaOptimize
#     swarm optimizer (cudafit1/2), always doing both single- and double-pulse
#     fits with the trace kept resident on the GPU between them.
if(DDASTOYS_CUDA_DEV)
    # libFitEditorCudaAnalytic.so — analytic fit via the CUDA fit engine.
    # No SerialFitEngineAnalytic.cpp: with -DCUDA, lmfit_analytic selects
    # CudaFitEngine1/2 (provided by ddastoys_cuda_engine_objs).
    add_library(FitEditorCudaAnalytic SHARED
        FitEditorAnalytic.cpp
        CFitEngine.cpp
        lmfit_analytic.cpp
    )
    target_compile_definitions(FitEditorCudaAnalytic PRIVATE CUDA)
    target_include_directories(FitEditorCudaAnalytic PRIVATE ${DAQINC} ${UFMTINC})
    target_link_libraries(FitEditorCudaAnalytic
        PRIVATE
            ddastoys_config_objs
            ddastoys_analytic_funcs_objs
            ddastoys_cuda_engine_objs
            DDASFormat
            GSL::gsl GSL::gslcblas
    )

    add_library(FitEditorCudaDE SHARED FitEditorCuda.cpp)
    target_include_directories(FitEditorCudaDE PRIVATE ${DAQINC} ${UFMTINC})
    target_link_libraries(FitEditorCudaDE PRIVATE
        ddastoys_config_objs ddastoys_analytic_funcs_objs
        ddastoys_cuda_de_objs DDASFormat)

    add_library(FitEditorCudaPSO SHARED FitEditorCuda.cpp)   # unchanged name
    target_include_directories(FitEditorCudaPSO PRIVATE ${DAQINC} ${UFMTINC})
    target_link_libraries(FitEditorCudaPSO PRIVATE
        ddastoys_config_objs ddastoys_analytic_funcs_objs
        ddastoys_cuda_pso_objs DDASFormat)
endif()

# =============================================================================
# Subdirectories
# =============================================================================

if(GENGETOPT_EXECUTABLE)
    add_subdirectory(EEConverter)
else()
    message(STATUS "Skipping EEConverter — gengetopt not found.")
endif()

if(DDASTOYS_TRACEVIEW)
    add_subdirectory(TraceView)
endif()

if(DDASTOYS_DOCS)
    add_subdirectory(Docs)
endif()

if(CPPUNIT_FOUND)
    add_subdirectory(tests)
endif()

# =============================================================================
# Install
# =============================================================================

set(_install_libs DDASFormat DDASFitHitUnpacker FitEditorAnalytic FitEditorTemplate)
if(DDASTOYS_MLINFERENCE)
    list(APPEND _install_libs FitEditorMLInference)
  endif()
if(DDASTOYS_CUDA_DEV)
    list(APPEND _install_libs FitEditorCudaAnalytic FitEditorCudaPSO)
endif()

install(TARGETS ${_install_libs} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

# Public headers (all *.h from root except Asserts.h, CRingItemProcessor.h, profiling.h)
set(_public_headers
    ${CMAKE_SOURCE_DIR}/CFitEngine.h
    ${CMAKE_SOURCE_DIR}/Configuration.h
    ${CMAKE_SOURCE_DIR}/DDASFitHit.h
    ${CMAKE_SOURCE_DIR}/DDASFitHitUnpacker.h
    ${CMAKE_SOURCE_DIR}/FitEditorAnalytic.h
    ${CMAKE_SOURCE_DIR}/FitEditorTemplate.h
    ${CMAKE_SOURCE_DIR}/fit_extensions.h
    ${CMAKE_SOURCE_DIR}/functions_analytic.h
    ${CMAKE_SOURCE_DIR}/functions_template.h
    ${CMAKE_SOURCE_DIR}/jacobian_analytic.h
    ${CMAKE_SOURCE_DIR}/lmfit_analytic.h
    ${CMAKE_SOURCE_DIR}/lmfit_template.h
)
if(DDASTOYS_MLINFERENCE)
    list(APPEND _public_headers
        ${CMAKE_SOURCE_DIR}/FitEditorMLInference.h
        ${CMAKE_SOURCE_DIR}/mlinference.h
    )
endif()
if(DDASTOYS_CUDA_DEV)
    list(APPEND _public_headers
        ${CMAKE_SOURCE_DIR}/FitEditorCuda.h
        ${CMAKE_SOURCE_DIR}/cudafit_analytic.cuh
    )
endif()

install(FILES ${_public_headers} DESTINATION include)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "DDASToys ${PROJECT_VERSION} — build configuration")
message(STATUS "  Install prefix : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  DAQROOT        : ${DAQROOT}")
message(STATUS "  UFMTROOT       : ${UFMTROOT}")
message(STATUS "  ROOT           : ${ROOT_VERSION}")
message(STATUS "  GSL            : ${GSL_VERSION}")
message(STATUS "  CUDA plugins   : ${DDASTOYS_CUDA_DEV}")
message(STATUS "  ML inference   : ${DDASTOYS_MLINFERENCE}")
message(STATUS "  TraceView      : ${DDASTOYS_TRACEVIEW}")
message(STATUS "  Documentation  : ${DDASTOYS_DOCS}")
message(STATUS "  Timing macros  : ${DDASTOYS_TIMING}")
message(STATUS "")
