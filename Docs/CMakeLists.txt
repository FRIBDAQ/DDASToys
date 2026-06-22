# =============================================================================
# Docs
#
# Builds two documentation sets that mirror the original Docs/Makefile:
#
#   doxygen_docs  — Doxygen HTML source documentation
#     Input: project root, EEConverter, TraceView (as specified in Doxyfile)
#     Output: ${CMAKE_CURRENT_BINARY_DIR}/sourcedocs/
#
#   manual_docs   — DocBook user manual (HTML + PDF)
#     Input:  Docs/manual.xml
#     Output: ${CMAKE_CURRENT_BINARY_DIR}/manual/
#
#   docs          — umbrella target that builds whichever of the above are
#                   available.
#
# The existing Docs/Doxyfile uses relative INPUT/OUTPUT paths that only work
# when doxygen is run from the Docs/ source directory.  We copy the Doxyfile
# to the binary directory and append CMake-expanded absolute path overrides at
# the end.  Doxygen uses the last definition of any duplicated key, so the
# appended lines win without modifying the checked-in file.
# =============================================================================

set(_has_docs FALSE)

# =============================================================================
# Doxygen
# =============================================================================
if(Doxygen_FOUND)
    set(_doxyfile_in  "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
    set(_doxyfile_out "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
    set(_doxy_outdir  "${CMAKE_CURRENT_BINARY_DIR}/sourcedocs")

    # Copy the existing, hand-tuned Doxyfile verbatim …
    configure_file("${_doxyfile_in}" "${_doxyfile_out}" COPYONLY)

    # … then append overrides for the paths that change between in-tree and
    # out-of-tree builds.  Doxygen's last-definition-wins rule makes this work.
    file(APPEND "${_doxyfile_out}"
        "\n# ---- CMake out-of-tree overrides (appended at configure time) ----\n"
        "INPUT = \"${CMAKE_SOURCE_DIR}\" "
              "\"${CMAKE_SOURCE_DIR}/EEConverter\" "
              "\"${CMAKE_SOURCE_DIR}/TraceView\"\n"
        "OUTPUT_DIRECTORY = \"${_doxy_outdir}\"\n"
        # Keep the Doxyfile's PROJECT_NUMBER in sync with the CMake version.
        "PROJECT_NUMBER = ${PROJECT_VERSION}\n"
    )

    add_custom_target(doxygen_docs
        COMMAND "${DOXYGEN_EXECUTABLE}" "${_doxyfile_out}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        BYPRODUCTS "${_doxy_outdir}"
        COMMENT "Running Doxygen — output: ${_doxy_outdir}"
        VERBATIM
    )

    install(DIRECTORY "${_doxy_outdir}/"
        DESTINATION "share/sourcedocs"
        OPTIONAL   # docs not rebuilt automatically; don't break `make install`
    )

    set(_has_docs TRUE)
    set(_docs_deps doxygen_docs)
endif()

# =============================================================================
# DocBook manual (HTML + PDF)
# =============================================================================
set(_manual_xml "${CMAKE_CURRENT_SOURCE_DIR}/manual.xml")
set(_manual_dir "${CMAKE_CURRENT_BINARY_DIR}/manual")
set(_manual_deps "")

if(DOCBOOK2HTML_EXE)
    set(_manual_html "${_manual_dir}/manual.html")
    add_custom_command(
        OUTPUT  "${_manual_html}"
        COMMAND "${DOCBOOK2HTML_EXE}"
                -o "${_manual_dir}"
                "${_manual_xml}"
        DEPENDS "${_manual_xml}"
        COMMENT "Building DocBook HTML manual"
        VERBATIM
    )
    list(APPEND _manual_deps "${_manual_html}")
    set(_has_docs TRUE)
endif()

if(DOCBOOK2PDF_EXE)
    set(_manual_pdf "${_manual_dir}/manual.pdf")
    add_custom_command(
        OUTPUT  "${_manual_pdf}"
        COMMAND "${DOCBOOK2PDF_EXE}"
                -o "${_manual_dir}"
                "${_manual_xml}"
        DEPENDS "${_manual_xml}"
        COMMENT "Building DocBook PDF manual"
        VERBATIM
    )
    list(APPEND _manual_deps "${_manual_pdf}")
    set(_has_docs TRUE)
endif()

if(_manual_deps)
    add_custom_target(manual_docs
        DEPENDS ${_manual_deps}
        COMMENT "DocBook manual built in ${_manual_dir}"
    )
    list(APPEND _docs_deps manual_docs)

    install(DIRECTORY "${_manual_dir}/"
        DESTINATION "share/manual"
        OPTIONAL
    )
endif()

# =============================================================================
# Umbrella target
# =============================================================================
if(_has_docs)
    add_custom_target(docs ALL DEPENDS ${_docs_deps}
        COMMENT "All DDASToys documentation")
else()
    message(WARNING
        "Docs subdirectory added but neither Doxygen nor docbook2html was "
        "found — no documentation will be generated.")
endif()