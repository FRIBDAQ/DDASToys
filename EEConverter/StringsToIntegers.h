/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2005.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Author:
             Ron Fox
             NSCL
             Michigan State University
             East Lansing, MI 48824-1321
*/

/** 
 * @file  StringsToIntegers.h
 * @brief Definition of functions to convert strings of comma-delimited 
 * integers to a vector of ints.
 */

#ifndef STRINGSTOINTEGERS_H
#define STRINGSTOINTEGERS_H

#ifndef __STL_STRING
#include <string>
#ifndef __STL_STRING
#define __STL_STRING
#endif
#endif

#ifndef __STL_VECTOR
#include <vector>
#ifndef __STL_VECTOR
#define __STL_VECTOR
#endif
#endif

#ifndef __CINVALIDARGUMENTEXCEPTION_H
#include <CInvalidArgumentException.h>
#endif

/**
 * @brief aA unbound function that takes a comma separated list of integer 
 * (in string form) and converts them into a vector of ints. *
 * @param items  Stringified comma separated list of integers. *
 * @return std::vector<int>  Ordered vector of the integers decoded from the 
 * string. *
 * @throw CInvalidArgumentException Throws back to the caller if the item type
 * is not a convertable type.
 */
std::vector<int> stringListToIntegers(std::string items);

#endif
