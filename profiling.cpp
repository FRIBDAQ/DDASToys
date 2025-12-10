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
 * @file profiling.cpp
 * @brief Implement profiling functions
 */

#include "profiling.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace ddastoys {
    
    double
    Timer::elapsed()
    {
	auto end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::micro>(end - s_start).count();
    }
    
    void
    Stats::compute()
    {
	s_mean = std::accumulate(s_data.begin(), s_data.end(), 0.0);
	s_mean /= s_data.size();

	double sumSq = 0;
	for (const auto d : s_data) {
	    sumSq += (d - s_mean)*(d - s_mean);
	}
	s_stddev = std::sqrt(sumSq)/s_data.size();

	s_min = *std::min_element(s_data.begin(), s_data.end());
	s_max = *std::max_element(s_data.begin(), s_data.end());
    }

    void
    Stats::print(std::string label)
    {
	std::cout << label << std::endl;
	std::cout << "  Mean:   " << s_mean << " us" << std::endl;
	std::cout << "  Std:    " << s_stddev << " us" << std::endl;
	std::cout << "  Min:    " << s_min << " us" << std::endl;
	std::cout << "  Max:    " << s_max << " us" << std::endl;
    }

    void
    Stats::reset()
    {
	s_mean = 0;
	s_stddev = 0;
	s_min = 0;
	s_max = 0;
	s_data.clear();
    }
    
} // namespace ddastoys
