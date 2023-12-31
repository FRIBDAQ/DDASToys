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

/** 
 * @file  RootExtensions.cpp
 * @brief Trivial implementations of the RootExtensions struct.
 */

#include "RootExtensions.h"

#include "fit_extensions.h"

///
// RootPulseDescription
//

/**
 * @details
 * Just zero out all values.
 */
RootPulseDescription::RootPulseDescription() :
    position(0.0), amplitude(0.0), steepness(0.0), decayTime(0.0)
{}

RootPulseDescription&
RootPulseDescription::operator=(const DDAS::PulseDescription& rhs)
{
    position = rhs.position;
    amplitude = rhs.amplitude;
    steepness = rhs.steepness;
    decayTime = rhs.decayTime;
    
    return *this;
}

///
// RootFit1Info
//

/**
 * @details
 * Just initialize everything to zero again.
 */
RootFit1Info::RootFit1Info() :
    chiSquare(0.0), offset(0.0), iterations(0), fitStatus(0)
{}

RootFit1Info&
RootFit1Info::operator=(const DDAS::fit1Info& rhs)
{
    iterations = rhs.iterations;
    fitStatus  = rhs.fitStatus;
    chiSquare  = rhs.chiSquare;
    pulse      = rhs.pulse;
    offset     = rhs.offset;
    
    return *this;
}

///
// RootFit2Info
//

RootFit2Info::RootFit2Info() :
    chiSquare(0.0), offset(0.0), iterations(0), fitStatus(0)
{
    RootPulseDescription dummy; // Zeroes.
    pulses[0]  = dummy;
    pulses[1]  = dummy;
}

RootFit2Info&
RootFit2Info::operator=(const DDAS::fit2Info& rhs)
{
    iterations = rhs.iterations;
    fitStatus  = rhs.fitStatus;
    chiSquare  = rhs.chiSquare;
    offset     = rhs.offset;
    pulses[0]  = rhs.pulses[0];
    pulses[1]  = rhs.pulses[1];
    
    return *this;
}

///
// RootExtension
//

RootHitExtension::RootHitExtension() :
    haveExtension(false)  // default member constructors do the rest.
{}

RootHitExtension&
RootHitExtension::operator=(const DDAS::HitExtension& rhs)
{
    haveExtension = true;
    onePulseFit = rhs.onePulseFit;
    twoPulseFit = rhs.twoPulseFit;
    
    return *this;
}
