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

/** @file:  functions.h
 *  @brief: Provide code to evaluate various functions for the DDAS Fit.
 */
#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <vector>
#include <stdint.h>

namespace DDAS {
double logistic(double A, double k, double x1, double x);
double decay(double A, double k, double x1, double x);
double switchOn(double x1, double x);

double singlePulse(
    double A1, double k1, double k2, double x1, double C, double x
);
double doublePulse(
    double A1, double k1, double k2, double x1,
    double A2, double k3, double k4, double x2,
    double C, double x
);

double chiSquare1(
    double A1, double k1, double k2, double x1, double C,
    const std::vector<uint16_t>& trace, int low = 0 , int high = -1
);

double chiSquare2(
    double A1, double k1, double k2, double x1,
    double A2, double k3, double k4, double x2,
    double C,    
    const std::vector<uint16_t>& trace, int low = 0, int high = -1
);

void writeTrace(
   const char* filename, const char* title,
   const std::vector<uint16_t>& trace
);
void writeTrace2(
   const char* filename, const char* title,
   const std::vector<uint16_t>& trace1, const std::vector<uint16_t>& trace2
);
// The following structs are used by the fitting
// functions.

// Describes a single pulse without an offset.

struct PulseDescription {  
   double position;         // Where the pusle is.
   double amplitude;        // Pulse amplitude
   double steepness;        // Logistic steepness factor.
   double decayTime;        // Decay time constant.    
};

// Full fitting information for the single pulse:
    
struct fit1Info {           // Info from single pulse fit:
   unsigned iterations;     // Iterations for fit to converge
   unsigned fitStatus;      // fit status from GSL.
   double chiSquare;
   PulseDescription pulse;
   double  offset;          // Constant offset.

};
    
// Full fitting information for the double pulse:

struct fit2Info {                // info from double pulse fit:
   unsigned iterations;          // Iterations needed to converge.
   unsigned fitStatus;           // Fit status from GSL
   double chiSquare; 
   PulseDescription pulses[2];  // The two pulses
   double offset;               // Ofset on which they sit.
};

// For good measure here's what we append to a DDAS Hit that's
// had its trace fitted.

struct HitExtension {     // Data added to hits with traces:
    fit1Info onePulseFit;
    fit2Info twoPulseFit;
};
// This struct is passed around the fitting subsystem to jacobian
// and function evaluators.
//
struct GslFitParameters {
   const std::vector<uint16_t>* s_pTrace;
   unsigned               s_low;        // limits of the fit.
   unsigned               s_high;       // Low/high both inclusive.
};
  

void lmfit1(
   fit1Info* pResult, std::vector<uint16_t>& trace,
   const std::pair<unsigned, unsigned>& limits
);
void lmfit2(
   fit2Info* pResult, std::vector<uint16_t>& trace,
   const std::pair<unsigned, unsigned>& limits,
   fit1Info* pSinglePulseFit = nullptr
);

};
#endif