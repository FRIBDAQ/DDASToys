/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  FitEngine.cpp
 *  @brief: Implement base class.
 */
#include "jacobian.h"

/**
 * FitEngine constructor
 *   Just marshall the x/y points.
 */
FitEngine::FitEngine(std::vector<std::pair<uint16_t, uint16_t>>& data)
{
    for (size_t i = 0; i < data.size(); i++) {
        x.push_back(data[i].first);
        y.push_back(data[i].second);
    }
}
