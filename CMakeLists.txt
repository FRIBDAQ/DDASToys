# CMake requirement:

cmake_minimum_required(VERSION 3.13)

# Package:

project(DDASToys VERSION 6.1.000)

# Set the CXX standard for the project:

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Subdirectories with libraries:

# Configure subproject into <subproject-build-dir>
execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/DDASFormat DDASFormat WORKING_DIRECTORY DDASFormat)
# Build and install subproject
execute_process(COMMAND ${MAKE} all install)

#add_subdirectory(DDASFormat)
#install(TARGETS DDASFormat DESTINATION ${CMAKE_INSTALL_PREFIX}/DDASFormat)

# DDASFitHitUnpacker.so:

#set(ddasfithitunpacker_sources DDASFitHitUnpacker.cpp)
#set(ddasfithitunpacker_headers DDASFitHitUnpacker.h)
#add_library(DDASFitHitUnpacker SHARED ${ddasfithitunpacker_sources})
#target_compile_options(DDASFitHitUnpacker PRIVATE -g -O2)
#target_link_options(DDASFitHitUnpacker PRIVATE -g)

# Package installation:

#install(FILES ${ddasfithitunpacker_headers} DESTINATION include)
#install(TARGETS DDASFitHitUnpacker LIBRARY DESTINATION lib)
