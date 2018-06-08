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

/** @file:  RootExtensions.cpp
 *  @brief: Trivial implementations of the RootExtensions struct.
 */
#include "RootExtensions.h"

/*----------------------------------------------------------------------------
 * RootPulseDescription
 */

ClassImp(RootPulseDescription);

/**
 * Constructor
 *   Just zero out all values.
 */
RootPulseDescription::RootPulseDescription() :
    position(0.0),
    amplitude(0.0),
    steepness(0.0),
    decayTime(0.0)
{
    
}
/**
 *  Assignment from a DDAS::PulseDescription  this is how this struct.
 *  is intended to get is values.
 *
 * @param rhs  = reference to a DDAS::PulseDescription that will provide
 *               our values.
 * @return *this - to support assignment chaining.
 */
RootPulseDescription&
RootPulseDescription::operator=(const DDAS::PulseDescription& rhs)
{
    position = rhs.position;
    amplitude = rhs.amplitude;
    steepness = rhs.steepness;
    decayTime = rhs.decayTime;
    
    return *this;
}

/*------------------------------------------------------------------------
 *  RootFitInfo:
 */

ClassImp(RootFit1Info);

/**
 *  constructor .. just initialize everything to zero again.
 */
RootFit1Info::RootFit1Info() :
    iterations(0),
    fitStatus(0),
    chiSquare(0.0),
    offset(0.0)
{}

/**
 * Assignment from a DDAS::fit1Info
 * @param rhs - References the DDAS::fitInfo from which we'll get our values.
 * @return *this
 */
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

/*----------------------------------------------------------------------------
 *  RootFit2Info
 */
ClassImp(RootFit2Info);

/**
 * constructor.
 */
RootFit2Info::RootFit2Info() :
    iterations(0),
    fitStatus(0),
    chiSquare(0.0),
    offset(0.0)
{
    RootPulseDescription dummy;           // Zeroes.
    pulses[0]  = dummy;
    pulses[1]  = dummy;
}
/**
 * Assignment from a DDAS::fit2Info:
 *
 * @param rhs - references the DDAS::fit2Info from which we get our values.
 * @return *this.
 */
RootFit2Info&
RootFit2Info::operator=(const DDAS::fit2Info& rhs)
{
    iterations = rhs.iterations;
    fitStatus  = rhs.fitStatus;
    chiSquare  = rhs.chiSquare;
    offset     = rhs.offset;
    pulses[0]  = rhs.pulses[0];        // Not worth a loop.
    pulses[1]  = rhs.pulses[1];
    
    return *this;
}

/*---------------------------------------------------------------------------
 *  RootHitExtension.
 */

ClassImp(RootHitExtension);

/**
 * constructor
 */
RootHitExtension::RootHitExtension() :
    haveExtension(false)  // default member constructors do the rest.
{
}

/**
 * assignment from a DDAS::HitExtension
 *
 * @param rhs  - References the DDAS::HitExtension that gives us  our values.
 * @return *this;
 */
RootHitExtension&
RootHitExtension::operator=(const DDAS::HitExtension& rhs)
{
    haveExtension = true;
    onePulseFit = rhs.onePulseFit;
    twoPulseFit = rhs.twoPulseFit;
    
    return *this;
}