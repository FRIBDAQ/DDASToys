/** 
 * @file  functions_template.cpp
 * @brief Implement functions used to fit DDAS pulses using a trace template.
 * @note All functions are in the DDAS::TemplateFit namespace
 */

#include "functions_template.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>

/**
 * @details
 * Evaluate the value of a single pulse in accordance with our canonical 
 * functional form. The form is a logistic rise with an exponential decay 
 * that sits on top of a constant offset.
 */
double
DDAS::TemplateFit::singlePulse(
    double A1, double x1, double C, double x,
    const std::vector<double>& trace_template
    )
{
    double value = 0;                      // Template trace fit value
    int ishift = static_cast<int>(x - x1); // Phase-shifted sample number
    double intershift = x - x1 - ishift;   // Shift for interpolation
    double inter = 0.;                     // Interpolation term
    int last = static_cast<int>(trace_template.size()-1);

    if (ishift < 0) {
	value = C;
    } else if(ishift >= 0 && ishift < last) { // Good range
	inter = intershift*(trace_template[ishift+1]-trace_template[ishift]);
	value = C + A1*(trace_template[ishift] + inter);
    } else {
	inter = intershift*(trace_template[last] - trace_template[last-1]);
	value = C + A1*(trace_template[last] + inter);
    }
  
    return value;  
}

/** 
 * @details
 * Evaluate the canonical form of a double pulse. This is done by summing 
 * two single pulses. The constant term is thrown into the first pulse. 
 * The second pulse gets a constant term of 0.
 */
double
DDAS::TemplateFit::doublePulse(
    double A1, double x1, double A2, double x2, double C, double x,
    const std::vector<double>& trace_template
    )
{
    double p1 = singlePulse(A1, x1, C, x, trace_template);
    double p2 = singlePulse(A2, x2, 0.0, x, trace_template);
    
    return p1 + p2;  
}

/**
 * @details
 * Neyman's chi-square value is computed from a passed set of (x, y) data.
 */
double
DDAS::TemplateFit::chiSquare1(
    double A1, double x1, double C,
    const std::vector<std::pair<std::uint16_t, std::uint16_t> >& points,
    const std::vector<double>& trace_template
    )
{    
    double result = 0.0;
    for(size_t i = 0; i < points.size(); i++) {
	double x = points[i].first;
	double y = points[i].second;
	double pulse = singlePulse(A1, x1, C, x, trace_template);
	double diff = y - pulse;
	if (y != 0.0) {
	    result += (diff/y)*diff;  // This order may control overflows
	    if (std::fpclassify(result) == FP_ZERO) result =  0.0;
	}
    }
  
    return result;
}

/**
 * @details
 * Neyman's chi-square value is computed from a passed set of (x, y) data.
 */
double
DDAS::TemplateFit::chiSquare2(
    double A1, double x1, double A2, double x2, double C,
    const std::vector<std::pair<std::uint16_t, std::uint16_t> >& points,
    const std::vector<double>& trace_template
    )
{    
    double result = 0.0;
    for(size_t i = 0; i < points.size(); i++) {
	double x = points[i].first;
	double y = points[i].second;
	double pulse = doublePulse(A1, x1, A2, x2, C, x, trace_template);
	double diff = y - pulse;
	if (y != 0.0) {
	    result += (diff/y)*diff;  // This order may control overflows
	    if (std::fpclassify(result) == FP_ZERO) result =  0.0;
	}
    }
  
    return result;
}
