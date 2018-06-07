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


/** @file:  RootExtensions.h
 *  @brief: Provide streamable versions of the event extensions defined in
 *          functions.h
 */

#ifndef ROOTEXTENSIONS_H
#define ROOTEXTENSIONS_H

#include "TObject.h"
#include "functions.h"


/**
 * These structs mirror those having to do with the fit extensions to DDAS
 * data, but they support Root serialization and, therefore, ca be put into
 * Root files/tree leaves.
 */

// Root 'mirror' of DDAS::PulseDescription

struct RootPulseDescription : public TObject
{
   // Data members.
   
   double position;         // Where the pusle is.
   double amplitude;        // Pulse amplitude
   double steepness;        // Logistic steepness factor.
   double decayTime;        // Decay time constant.
   
   // Methods:
   
   RootPulseDecription();
   RootPulseDescription& operator=(const DDAS::PulseDescription& rhs);
   
   // Add root members to support streaming.
   
   ClassDef(RootPulseDescription, 1);
};

// Root 'mirror' of fit1Info.

struct RootFit1Info : public TObject
{
    // Data members:
    
   unsigned iterations;     // Iterations for fit to converge
   unsigned fitStatus;      // fit status from GSL.
   double chiSquare;
   RootPulseDescription pulse;
   double  offset;          // Constant offset.
   
   // Methods:
   
   RootFit1Info();
   RootFit1Info& operator=(const DDAS::fit1Info& rhs);
   
   // Root stuff.
   
   ClassDef(RootFit1Info, 1);
};

struct RootFit2Info : public TObject
{
    // Data members:
    
   unsigned iterations;          // Iterations needed to converge.
   unsigned fitStatus;           // Fit status from GSL
   double chiSquare; 
   RootPulseDescription pulses[2];  // The two pulses
   double offset;               // Ofset on which they siyt.
   
   // Methods
   
   RootFit2Info();
   RootFit2Info& operator=(const DDAS::fit2Info& rhs);
   
   ClassDef(RootFit2Info, 1);
};

struct RootHitExtension : public TObject
{
    RootFit1Info onePulseFit;
    RootFit2Info twoPulseFit;
    
    RootHitExtension()
    RootHitExtension& operator=(const DDAS::HitExtension& rhs);
    
    ClassDef(RootHitExtension, 1);
};

#endif