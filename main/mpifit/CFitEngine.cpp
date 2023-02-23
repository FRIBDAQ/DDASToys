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

/** 
 * @file  CFitEngine.cpp
 * @brief Implement constructor for CFitEngine base class.
 */

/** @addtogroup AnalyticFit
 * @{
 */

#include "CFitEngine.h"

/**
 * Marshall the x/y points into the coordinate vectors.
 */
CFitEngine::CFitEngine(std::vector<std::pair<uint16_t, uint16_t>>& data)
{
  for (size_t i = 0; i < data.size(); i++) {
    x.push_back(data[i].first);
    y.push_back(data[i].second);
  }
}

/** @} */
