/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins 
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  DtRange.h 
 *  @brief: Header with DTRange data types.
 */
#ifndef DTRANGE_H
#define DTRANGE_H

#include <functions.h>


/**
 *  This is the format of an event.  Note that the arrays shown have the
 *  number of elements that there are pulses in the generated event:
 *
 */



struct Event {
    uint32_t               s_isDouble;           // True if event is double.
    DDAS::HitExtension     s_fitinfo;            // Fit results.
    
    double                 s_actualOffset;       // Actual offset.
    DDAS::PulseDescription s_pulses[];           // One or two elements.
};



#endif