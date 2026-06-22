# =============================================================================
# EEConverter
#
# Builds two artifacts:
#
#   libDDASRootFitFormat.so — ROOT-serializable DDAS hit/event classes, plus
#     the rootcling dictionary (.cxx), pre-compiled module (.pcm), and rootmap
#     that ROOT needs for transparent I/O of these types.
#
#   eeconverter — reads NSCLDAQ event files, unpacks DDAS hits with fit
#     extensions, and writes them into a ROOT TTree.
#
# =============================================================================

# =============================================================================
# Generate converterargs.{c,h} with gengetopt
# =============================================================================

set(_ggo_src "${CMAKE_CURRENT_SOURCE_DIR}/converterargs.ggo")
set(_ggo_c   "${CMAKE_CURRENT_BINARY_DIR}/converterargs.c")
set(_ggo_h   "${CMAKE_CURRENT_BINARY_DIR}/converterargs.h")

add_custom_command(
    OUTPUT  ${_ggo_c} ${_ggo_h}
    COMMAND "${GENGETOPT_EXECUTABLE}"
            --input=${_ggo_src}
            --file-name=converterargs
            --output-dir=${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${_ggo_src}
    COMMENT "Generating converterargs.{c,h}"
)

# =============================================================================
# libDDASRootFitFormat.so — ROOT-serializable hit / event classes
# =============================================================================

add_library(DDASRootFitFormat SHARED
    DDASRootFitEvent.cpp
    DDASRootFitHit.cpp
    # The rootcling-generated dictionary source is appended by
    # ROOT_GENERATE_DICTIONARY when MODULE DDASRootFitFormat is specified.
)

# Include paths needed to compile DDASRootFitHit.cpp / DDASRootFitEvent.cpp:
#   - EEConverter dir  — DDASRootFitHit.h, DDASRootFitEvent.h, LinkDef.h
#   - Project root     — DDASFitHit.h, fit_extensions.h
#   - DDASFormat       — DDASHit.h  (propagated via DDASFormat target)
target_include_directories(DDASRootFitFormat
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# TObject symbols in ROOT::Core are resolved at runtime from the exe's link.
# We add ROOT::Core explicitly so the RPATH and pkg-config info are correct,
# and so consumers that find_package(DDASToys) get ROOT propagated.
target_link_libraries(DDASRootFitFormat
    PUBLIC
        DDASFormat
        ROOT::Core       # TObject base class
)

# rootcling invocation via ROOT's CMake integration.
# ROOT_GENERATE_DICTIONARY:
#   - Argument 1      — dictionary name; the generated source is ${name}.cxx
#   - MODULE          — attaches the generated source to this existing target
#   - LINKDEF         — the #pragma link directives file
#   - OPTIONS         — extra -I flags for rootcling (it runs independently
#                       of CMake's include lists, so we must repeat them)
#
# Outputs placed in CMAKE_CURRENT_BINARY_DIR:
#   G__DDASRootFitFormat.cxx          (compiled into DDASRootFitFormat)
#   DDASRootFitFormat_rdict.pcm       (ROOT pre-compiled module)
#   DDASRootFitFormat.rootmap         (class → library mapping)
ROOT_GENERATE_DICTIONARY(G__DDASRootFitFormat
    # Headers to introspect:
    ${CMAKE_CURRENT_SOURCE_DIR}/DDASRootFitEvent.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DDASRootFitHit.h
    ${CMAKE_SOURCE_DIR}/DDASFitHit.h
    MODULE  DDASRootFitFormat
    LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/LinkDef.h
    OPTIONS
        "-I${CMAKE_SOURCE_DIR}"
        "-I${CMAKE_SOURCE_DIR}/DDASFormat"
        "-I${CMAKE_CURRENT_SOURCE_DIR}"
)

# =============================================================================
# eeconverter executable
# =============================================================================

add_executable(eeconverter
    eeconverter.cpp
    "${_ggo_c}"           # generated C file (plain C, project has LANGUAGES C)
    CEventProcessor.cpp
    ProcessToRootSink.cpp
    RootFileDataSink.cpp
    StringsToIntegers.cpp
)

target_include_directories(eeconverter
    PRIVATE
        "${DAQINC}"                     # FIRST: all ring-item headers + DataFormat.h with global ufmt aliases
        "${UFMTINC}"                    # SECOND: ufmt DataFormat.h (guard fires; DAQINC version already ran)
        "${CMAKE_CURRENT_BINARY_DIR}"   # converterargs.h (generated)
        "${CMAKE_CURRENT_SOURCE_DIR}"   # local EEConverter headers
        # ${CMAKE_SOURCE_DIR} comes from ddastoys_ring_objs PUBLIC include
)

target_link_libraries(eeconverter
    PRIVATE
        # Object libraries from the project root
        ddastoys_unpacker_objs  # DDASFitHitUnpacker (+ DDASFormat + NSCLDAQ::Headers)
        ddastoys_ring_objs      # CRingItemProcessor (+ NSCLDAQ::NSCLDAQ + DDASFormat)

        # ROOT format library (provides DDASRootFitHit, DDASRootFitEvent,
        # and the dictionary registration)
        DDASRootFitFormat

        # DDASFormat (propagated via ring/unpacker objs, but explicit is robust)
        DDASFormat

        # All NSCLDAQ runtime libraries (propagated via ring_objs, explicit here
        # for clarity)
        NSCLDAQ::NSCLDAQ

        # ROOT runtime (RootFileDataSink creates TFile and TTree)
        ROOT::Core
        ROOT::Tree
        ROOT::RIO
)

# =============================================================================
# Install
# =============================================================================

install(TARGETS DDASRootFitFormat
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# The .pcm and .rootmap must live beside the .so so ROOT can find them.
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/DDASRootFitFormat_rdict.pcm"
    "${CMAKE_CURRENT_BINARY_DIR}/DDASRootFitFormat.rootmap"
    DESTINATION lib
)

# ROOT also looks for the .pcm next to the running executable.
# Create a symlink ${prefix}/bin/DDASRootFitFormat_rdict.pcm in ../lib/…
install(CODE
    "execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
         \"\${CMAKE_INSTALL_PREFIX}/lib/DDASRootFitFormat_rdict.pcm\"
         \"\${CMAKE_INSTALL_PREFIX}/bin/DDASRootFitFormat_rdict.pcm\")"
)

install(TARGETS eeconverter RUNTIME DESTINATION bin)

# Public EEConverter headers (DDASRootFitHit.h and DDASRootFitEvent.h)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/DDASRootFitHit.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/DDASRootFitEvent.h"
    DESTINATION include
)
