# =============================================================================
# Unit tests
#
# Source breakdown:
#   TestRunner.cpp         — CppUnit boilerplate main()
#   fithittests.cpp        — tests DDASFitHit struct/class behaviour
#   fitunpackertests.cpp   — tests DDASFitHitUnpacker; uses <DataFormat.h>
#                            for ufmt::RingItem structs (header-only, no lib)
#   configurationtests.cpp — tests the Configuration file parser
#
# Link deps:
#   cppunit, DDASFormat, GSL
#   NSCLDAQ::Headers for <DataFormat.h> include path
#   ddastoys_unpacker_objs provides DDASFitHitUnpacker.o
#   ddastoys_config_objs provides Configuration.o (+ profiling, negligible)
#
# CppUnit variables (CPPUNIT_LIBRARIES, CPPUNIT_INCLUDE_DIRS) are set by
# pkg_check_modules or the fallback find_library in the top-level file.
# =============================================================================

# Source files live in the project root, not in a tests/ subdirectory.
add_executable(unittests_ddastoys
    ${CMAKE_CURRENT_SOURCE_DIR}/TestRunner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fithittests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fitunpackertests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/configurationtests.cpp
)

target_include_directories(unittests_ddastoys
    PRIVATE
        ${CPPUNIT_INCLUDE_DIRS}
        # ${CMAKE_SOURCE_DIR} is inherited via ddastoys_config_objs /
        # ddastoys_unpacker_objs PUBLIC include paths — no need to repeat.
)

target_link_libraries(unittests_ddastoys
    PRIVATE
        # Object libraries — provide the compiled units under test
        ddastoys_unpacker_objs   # DDASFitHitUnpacker (+ DDASFormat + NSCLDAQ::Headers)
        ddastoys_config_objs     # Configuration (+ profiling, negligible)

        # External libraries — mirrors Makefile: -lcppunit $(CXXLDFLAGS)
        ${CPPUNIT_LIBRARIES}
        DDASFormat
        GSL::gsl
        GSL::gslcblas
)

# Register with CTest.
# DDASFormat's own unittests are also registered (DDASFormat calls
# enable_testing() + add_test() in its own CMakeLists.txt).
add_test(
    NAME    DDASToysTests
    COMMAND unittests_ddastoys
)
