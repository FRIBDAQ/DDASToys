/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Aaron Chester
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** 
 * @file  functions_template.h
 * @brief Define functions used to fit DDAS pulses using a trace template.
 * @note All functions are in the DDAS::TemplateFit namespace
 */

#ifndef FUNCTIONS_TEMPLATE_H
#define FUNCTIONS_TEMPLATE_H

#include <vector>
#include <cstdint>

/** @namespace ddastoys */
namespace ddastoys {
    /** @namespace ddastoys::templatefit */
    namespace templatefit {
	
	/**
	 * @ingroup template
	 * @{
	 */
	
	/**
	 * @brief Evaluate the single pulse fit function at a given point.
	 * @param A1 Scaling factor for the template fit.
	 * @param x1 Offset with respect to the template.
	 * @param C  Constant baseline.
	 * @param x  Position at which to evaluate this function.
	 * @param trace_template Template trace used to fit the data.
	 * @return Single pulse function evaluated at x.
	 */
	double singlePulse(
	    double S1, double x1, double C, double x,
	    const std::vector<double>& trace_template
	    );
	/** 
	 * @brief Evaluate the double pulse fit function at a given point.
	 * @param A1 Scaling factor for the template fit.
	 * @param x1 Offset with respect to the template.
	 * @param A2 Scaling factor for the template fit.
	 * @param x2 Offset with respect to the template.
	 * @param C  Constant baseline.
	 * @param x  Position at which to evaluate this function.
	 * @param trace_template  Template trace used to fit the data.
	 * @return double  Double pulse function evaluated at x.
	 */
	double doublePulse(
	    double S1, double x1, double S2, double x2,
	    double C, double x,
	    const std::vector<double>& trace_template
	    );
	/**
	 * @brief Computes the chi-square goodness of a specific 
	 * parameterization of a single pulse canonical form with respect 
	 * to a trace.
	 * @param A1 Scaling factor for the template fit.
	 * @param x1 Offset with respect to the template.
	 * @param C  Constant baseline.
	 * @param points Set of x, y data points which contribute to the total 
	 *   chi-square value.
	 * @param trace_template Template trace used to fit the data.
	 * @return The chi-square goodness-of-fit statistic.
	 */
	double chiSquare1(
	    double S1, double x1, double C,
	    const std::vector<std::pair<std::uint16_t, std::uint16_t> >& points,
	    const std::vector<double>& trace_template
	    );
	/**
	 * @brief Computes the chi-square goodness of a specific 
	 * parameterization of a double pulse canonical form with respect 
	 * to a trace.
	 * @param A1 Scaling factor for the template fit
	 * @param x1 Offset with respect to the template
	 * @param A2 Scaling factor for the template fit
	 * @param x2 Offset with respect to the template
	 * @param C  Constant baseline
	 * @param points Set of x, y data points which contribute to the total 
	 *  chi-square value
	 * @param trace_template Template trace used to fit the data.
	 * @return double  The chi-square goodness-of-fit statistic.
	 */
	double chiSquare2(
	    double S1, double x1, double S2, double x2, double C,
	    const std::vector<std::pair<std::uint16_t, std::uint16_t> >& points,
	    const std::vector<double>& trace_template
	    );

	/** @} */
    }
};
  
#endif
