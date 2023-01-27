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

/*
  Edited to perform template fitting.
  -- ASC 9/19/2019
*/

/** @file:  functions.h
 *  @brief: Provide code to evaluate various functions for the DDAS Fit.
 */

#ifndef FUNCTIONS_TEMPLATE_H
#define FUNCTIONS_TEMPLATE_H

#include <vector>
#include <cstdint>

namespace DDAS {
  namespace TemplateFit {
  
    double singlePulse(
		       double S1, double x1, double C, double x,
		       const std::vector<double>& trace_template
		       );

    double doublePulse(
		       double S1, double x1, double S2, double x2,
		       double C, double x,
		       const std::vector<double>& trace_template
		       );
  
    double chiSquare1(
		      double S1, double x1, double C,
		      const std::vector<std::pair<std::uint16_t,
		                                    std::uint16_t> >& points,
		      const std::vector<double>& trace_template
		      );

    double chiSquare2(
		      double S1, double x1, double S2, double x2, double C,
		      const std::vector<std::pair<std::uint16_t,
		                                    std::uint16_t> >& points,
		      const std::vector<double>& trace_template
		      );
  }
};
  
#endif
