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

/** @file:  fit_extensions.h
 *  @brief: Define structs used by fitting functions and to extend DDAS hits
 */

#ifndef FIT_EXTENSIONS_H
#define FIT_EXTENSIONS_H

#include <cstdint>

namespace DDAS {

  // Describes a single pulse without an offset
  struct PulseDescription {  
    double position;  // Where the pusle is.
    double amplitude; // Pulse amplitude
    double steepness; // Logistic steepness factor.
    double decayTime; // Decay time constant.    
  };

  // Full fitting information for the single pulse
  struct fit1Info { // Info from single pulse fit:
    unsigned iterations; // Iterations for fit to converge
    unsigned fitStatus;  // Fit status from GSL.
    double chiSquare;
    PulseDescription pulse;
    double  offset;      // Constant offset.
  };
    
  // Full fitting information for the double pulse
  struct fit2Info { // Info from double pulse fit:
    unsigned iterations; // Iterations needed to converge.
    unsigned fitStatus;  // Fit status from GSL
    double chiSquare; 
    PulseDescription pulses[2]; // The two pulses
    double offset; // Shared constant offset
  };

  // For good measure here's what we append to a DDAS Hit that's
  // had its trace fitted
  struct HitExtension { // Data added to hits with traces:
    fit1Info onePulseFit;
    fit2Info twoPulseFit;
  };
  
}

typedef struct _nullExtension {
  std::uint32_t s_size;
  _nullExtension() : s_size(sizeof(std::uint32_t)) {}
} nullExtension, *pNullExtension;

typedef struct _FitInfo {
  std::uint32_t  s_size;
  DDAS::HitExtension s_extension;
  _FitInfo();
} FitInfo, *pFitInfo;

#endif
