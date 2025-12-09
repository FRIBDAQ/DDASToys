/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Aaron Chester
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/**
 * @file profiling.h
 * @brief Defines some useful structs for profiling inference
 */

#ifndef PROFILING_H
#define PROFILING_H

#include <chrono>
#include <string>
#include <vector>

/** @namespace ddastoys */
namespace ddastoys {
    
    /**
     * @struct Timer
     * @brief  Timer class using `std::chrono` to profile. 
     * @details
     * Example use of this struct:
     * @code
     * Timer timer;
     * ... user code to profile ...
     * double time = timer.elapsed();
     * @code
     */
    struct Timer {
	std::chrono::high_resolution_clock::time_point s_start; //!< Start time

	/**
	 * @brief Constructor
	 * @details
	 * Constructs object, sets start time to `now`
	 */
	Timer() : s_start(std::chrono::high_resolution_clock::now()) {};

	/**
	 * @brief Get the elapsed time in microseconds
	 * @return Elapsed time in microseconds
	 */
	double elapsed() {
	    auto end = std::chrono::high_resolution_clock::now();
	    return std::chrono::duration<double, std::micro>(end - s_start).count();
	};
    };

    /**
     * @struct Stats
     * @brief Compute stats for e.g. series of timings
     * @details
     * Example use of this class with a timer:
     * @code
     * Stats stats;
     * while (profiling) {
     *     Timer timer;
     *     ... some code to profile ...
     *     double time = timer.elapsed();
     *     stats.addData(time);
     * }
     * stats.compute();
     * stats.print();
     * @code
     */
    struct Stats {
	double s_mean;   //!< Mean of s_data when `compute()` is called
	double s_stddev; //!< Stddev of s_data when `compute()` is called
	double s_min;    //!< Max value of s_data when `compute()` is called
	double s_max;    //!< Min value of s_data when `compute()` is called
	std::vector<double> s_data; //!< Data for stats computation

	/**
	 * @brief Constructor
	 * @details
	 * Resets member variables to 0, clears data vector
	 */
	Stats() { reset(); };
    
	/** @brief Compute stats from list of times */
	void compute();
	/** @brief Print the current stats */
	void print(std::string label="======== Stats ========");
	/**
	 * @brief Add data to the current data set
	 * @param data Data to add
	 */
	void addData(double data) { s_data.push_back(data); };
	/**
	 * @brief Get the size of the stats vector
	 * @return Size of the stats vector
	 */
	size_t size() { return s_data.size(); };
	/** @brief Reset stats and clear data vector */
	void reset();
    };
}

#endif
