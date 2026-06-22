# =============================================================================
# TraceView
#
# A lightweight Qt5 + ROOT GUI for inspecting DDAS traces and their fits.
# Only built when Qt5 >= 5.11 is available (checked in top-level before this
# subdirectory is added).
#
# Source mapping (from traceview.pro LIBS line):
#   Links ../DDASFitHitUnpacker.o → ddastoys_unpacker_objs
#   Links ../Configuration.o     → ddastoys_config_objs (+ profiling, harmless)
#   Links ../functions_analytic.o → ddastoys_analytic_funcs_objs
#   Links ../functions_template.o → ddastoys_template_funcs_objs
#   Links ../CRingItemProcessor.o → ddastoys_ring_objs
# =============================================================================

set(CMAKE_AUTOMOC ON)

add_executable(traceview
    traceview.cpp
    QTraceView.cpp
    QHitData.cpp
    QRootCanvas.cpp
    TraceViewProcessor.cpp
    DDASDecoder.cpp
    FitManager.cpp
)

# Same DAQINC-before-UFMTINC ordering: TraceViewProcessor.cpp and
# DDASDecoder.cpp use NSCLDAQ ring-item classes.
target_include_directories(traceview
    PRIVATE
        "${DAQINC}"
        "${UFMTINC}"
)

target_link_libraries(traceview
    PRIVATE
        # --- Project-root object libraries ---
        # Each brings its compiled objects AND propagates its PUBLIC include
        # paths and link requirements to traceview.

        ddastoys_unpacker_objs        # DDASFitHitUnpacker (+ DDASFormat headers)
        ddastoys_ring_objs            # CRingItemProcessor (+ NSCLDAQ::NSCLDAQ)
        ddastoys_config_objs          # Configuration (+ profiling, minor excess)
        ddastoys_analytic_funcs_objs  # functions_analytic for FitManager
        ddastoys_template_funcs_objs  # functions_template for FitManager

        # --- DDASFormat runtime ---
        DDASFormat

        # --- Qt5 ---
        Qt5::Widgets   # pulls in all Qt5 Widgets headers and libQt5Widgets

        # --- CERN ROOT ---
        # traceview uses: TApplication, TSystem, TCanvas, TH1D,
        #                 TLegend, TStyle, TVirtualX (ROOT↔Qt event loop)
        ROOT::Core     # TApplication, TSystem, TVirtualX
        ROOT::Hist     # TH1D
        ROOT::Graf     # TLegend, TStyle
        ROOT::Gpad     # TCanvas, TPad

        # --- NSCLDAQ + UnifiedFormat ---
        # Also propagated by ddastoys_ring_objs, but listed explicitly for
        # clarity: DDASDecoder.cpp and TraceViewProcessor.cpp use full NSCLDAQ
        # ring-item classes and FragmentIndex.
        NSCLDAQ::NSCLDAQ
)

# =============================================================================
# Install
# =============================================================================
install(TARGETS traceview RUNTIME DESTINATION bin)