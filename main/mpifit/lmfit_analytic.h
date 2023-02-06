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

/** @file:  lmfit_analytic.h
 *  @brief: Define the fitting functions and data structures for L-M fits.
 */

#ifndef LMFIT_ANALYTIC_H
#define LMFIT_ANALYTIC_H

#include <vector>

#include "fit_extensions.h"

namespace DDAS {
  namespace AnalyticFit {

  // This struct is passed around the fitting subsystem to Jacobian and
  // function evaluators
  struct GslFitParameters {
    const std::vector<std::pair<uint16_t, uint16_t>>* s_pPoints;
  };
    
  void lmfit1(
	      fit1Info* pResult, std::vector<uint16_t>& trace,
	      const std::pair<unsigned, unsigned>& limits,
	      uint16_t saturation = 0xffff
	      );
  
  void lmfit2(
	      fit2Info* pResult, std::vector<uint16_t>& trace,
	      const std::pair<unsigned, unsigned>& limits,
	      fit1Info* pSinglePulseFit = nullptr,
	      uint16_t saturation = 0xffff
	      );

  void lmfit2fixedT(
		    fit2Info* pResult, std::vector<uint16_t>& trace,
		    const std::pair<unsigned, unsigned>& limits,
		    fit1Info* pSinglePulseFit = nullptr,
		    uint16_t saturation = 0xffff
		    );

  }
};

#endif
