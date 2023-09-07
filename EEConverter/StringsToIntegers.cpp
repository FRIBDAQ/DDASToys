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
 * @file StringsToInteger.cpp
 * @brief Functions to convert lists of comma-delimited integers to a vector 
 * of ints.
 */

#include "StringsToIntegers.h"

#include <map>

#include <DataFormat.h>

using namespace std;

static bool initialized = false;
static map<string, int>  textToInt;

/**
 * @brief Initialize map of ring item types.
 */
static void initialize()
{
    textToInt["BEGIN_RUN"]           = BEGIN_RUN;
    textToInt["END_RUN"]             = END_RUN;
    textToInt["PAUSE_RUN"]           = PAUSE_RUN;
    textToInt["RESUME_RUN"]          = RESUME_RUN;
    textToInt["PACKET_TYPE"]         = PACKET_TYPES;
    textToInt["MONITORED_VARIABLES"] = MONITORED_VARIABLES;
    textToInt["PERIODIC_SCALERS"]    = PERIODIC_SCALERS;
    textToInt["PHYSICS_EVENT"]       = PHYSICS_EVENT;
    textToInt["PHYSICS_EVENT_COUNT"] = PHYSICS_EVENT_COUNT;

}

/**
 * @brief Convert a single stringified number to an integer and return it.
 *
 * @param aNumber Stringified number to convert to an integer type.
 *
 * @return The converted number.
 *
 * @throw CInvalidArgumentException If the input string is not an integer or 
 * symbolic item type.
 */
static int
convertOne(string aNumber)
{
    char *end;

    int value = strtol(aNumber.c_str(), &end, 0);
    if (*end != '\0') {
	if (textToInt.find(aNumber) != textToInt.end()) {
	    return textToInt[aNumber];
	}
	else {
	    string whyBad  = " must be an integer or a symbolic item type but was ";
	    whyBad += aNumber;
	    throw CInvalidArgumentException(aNumber, whyBad, string("converting a list to integers"));
	}
    }
  
    return value;
}

/**
 * @details
 * This is most useful in decoding things like: 
 *  
 * @verbatim
 *   ... --exclude=1,2,3 ...
 * @endverbatim 
 *
 */
vector<int>
stringListToIntegers(string items) 
{
    size_t start = 0;
    vector<int> result;
    size_t comma;

    if (!initialized) {
	initialize();
    }

    while ((comma = items.find(string(","), start)) != string::npos) {
	string aNumber;
	aNumber.assign(items, start, comma-start);
	try {
	    result.push_back(convertOne(aNumber));
	}
	catch (CException& e) {
	    throw;
	}
	start = comma + 1;
    }
    
    // There's one last string that does not terminate in a comma:

    string aNumber;
    aNumber.assign(items, start, items.size()-start);
    result.push_back(convertOne(aNumber));

    return result;
}
