# CMake requirement:

cmake_minimum_required(VERSION 3.13)

# Package:

project(DDASToys VERSION 6.1.000)

# Set the CXX standard for the project:

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Subdirectories with libraries:

add_subdirectory(DDASFormat)


# DDASFitHitUnpacker.so:

#set(ddasfithitunpacker_sources DDASFitHitUnpacker.cpp)
#set(ddasfithitunpacker_headers DDASFitHitUnpacker.h)
#add_library(DDASFitHitUnpacker SHARED ${ddasfithitunpacker_sources})
#target_compile_options(DDASFitHitUnpacker PRIVATE -g -O2)
#target_link_options(DDASFitHitUnpacker PRIVATE -g)

# Package installation:

#install(FILES ${ddasfithitunpacker_headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
#install(TARGETS DDASFitHitUnpacker LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
