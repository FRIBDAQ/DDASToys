
/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins
	     Aaron Chester
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** 
 * @file  functions_analytic.h
 * @brief Provide code to evaluate various functions used to fit DDAS pulses.
 * @note  All functions are in the ddastoys::analyticfit namespace.
 */

#ifndef FUNCTIONS_ANALYTIC_H
#define FUNCTIONS_ANALYTIC_H

#include <vector>
#include <cstdint>

/** @namespace ddastoys */
namespace ddastoys {
    /** @namespace ddastoys::analyticfit */
    namespace analyticfit {
	
	/**
	 * @ingroup analytic
	 * @{
	 */
	
	/**
	 * @brief Evaluate a logistic function for the specified parameters 
	 * and point.
	 * @param A  Amplitude of the signal.
	 * @param k  Steepness of the signal (related to the rise time).
	 * @param x1 Mid-point of the rise of the logistic.
	 * @param x  Location at which to evaluate the function.
	 * @return Function value at x.
	 */
	double logistic(double A, double k, double x1, double x);
	/**
	 * @brief Evaluate an exponential decay for the specific parameters 
	 * and point.
	 * @param A  Amplitude of the signal
	 * @param k  Decay constant of the signal.
	 * @param x1 Position of the pulse.
	 * @param x  Where to evaluate the signal.
	 * @return Function value at x.
	 */
	double decay(double A, double k, double x1, double x);
	/**
	 * @brief Logistic switch to turn on a function evaluation at a 
	 * given point.
	 * @param x1 Switch on point. The switch is "on" for values greater 
	 *   than x1.
	 * @param x  Point in space at which to evaluate the switch.
	 * @return The switch value: very nearly 0.0 left of x1 and very 
	 *   nearly 1.0 to the right of x1.
	 */
	double switchOn(double x1, double x);
	/**
	 * @brief Evaluate the single pulse fit function at a given point.
	 * @param A1 Pulse amplitiude.
	 * @param k1 Logistic rise steepness.
	 * @param k2 Exponential decay time constant.
	 * @param x1 Logistic position.
	 * @param C  Constant offset.
	 * @param x  Position at which to evaluate the function.
	 * @return double Single pulse function evaluated at x.
	 */
	double singlePulse(
	    double A1, double k1, double k2, double x1,
	    double C, double x
	    );
	/**
	 * @brief Evaluate the double pulse fit function at a given point.
	 * @param A1 Amplitude of the first pulse.
	 * @param k1 Steepness of first pulse rise.
	 * @param k2 Decay time of the first pulse.
	 * @param x1 Position of the first pulse.
	 * @param A2 Amplitude of the second pulse.
	 * @param k3 Steepness of second pulse rise.
	 * @param k4 Decay time of second pulse.
	 * @param x2 Position of second pulse.
	 * @param C  Constant offset the pulses sit on.
	 * @param x  Position at which to evaluate the pulse.
	 * @return Double pulse function evaluated at x.
	 */
	double doublePulse(
	    double A1, double k1, double k2, double x1,
	    double A2, double k3, double k4, double x2,
	    double C, double x
	    );
	/**
	 * @brief Calculate the pulse amplitude corrected for the ballistic 
	 * deficit imposed by the exponential decay.
	 * @param A  The scaling term of the pulse.
	 * @param k1 The steepness term of the logistic.
	 * @param k2 The fall time term of the decay.
	 * @param x0 The position of the pulse.
	 * @return The corrected amplitude of the fitted pulse.
	 * @retval -1 If \f$k_1 \leq 0\f$ or \f$k_1 \leq 0\f$.
	 * @retval -2 If \f$k_1/k_2 \leq 1\f$.
	 */
	double pulseAmplitude(double A, double k1, double k2, double x0);
	/**
	 * @brief Computes the chi-square goodness-of-fit for a specific 
	 * parameterization of a single pulse canonical form with respect to 
	 * a trace.
	 * @param A1 Amplitude of pulse
	 * @param k1 Steepness of pulse rise.
	 * @param k2 Decay time of pulse fall.
	 * @param x1 Position of the pulse.
	 * @param C  Constant offset of the trace.
	 * @param trace Trace to compute the chi-square value with respect to.
	 * @param low, high Region of interest over which to compute the 
	 *   chi-square value.
	 * @note high = -1 will set the high limit to the last sample in the 
	 *   trace.
	 * @return The chi-square goodness-of-fit statistic.
	 */
	double chiSquare1(
	    double A1, double k1, double k2, double x1, double C,
	    const std::vector<uint16_t>& trace,
	    int low = 0 , int high = -1
	    );
	/**
	 * @brief Computes the chi-square goodness-of-fit for a specific 
	 * parameterization of a single pulse canonical form with respect 
	 * to a trace.
	 * @param A1 Amplitude of pulse
	 * @param k1 Steepness of pulse rise.
	 * @param k2 Decay time of pulse fall.
	 * @param x1 Position of the pulse.
	 * @param C  Constant offset of the trace.
	 * @param points Set of (x, y) data points which contribute to the 
	 *   total chi-square value
   	 * @return The chi-square goodness-of-fit statistic.
	 */
	double chiSquare1(
	    double A1, double k1, double k2, double x1, double C,
	    const std::vector<std::pair<uint16_t, uint16_t> >& points
	    );
	/**
	 * @brief Computes the chi-square goodness of a specific 
	 * parameterization of a double pulse canonical form with respect 
	 * to a trace.
	 * @param A1 Amplitude of the first pulse.
	 * @param k1 Steepness of first pulse rise.
	 * @param k2 Decay time of the first pulse.
	 * @param x1 Position of the first pulse.
	 * @param A2 Amplitude of the second pulse.
	 * @param k3 Steepness of second pulse rise.
	 * @param k4 Decay time of second pulse.
	 * @param x2 Position of second pulse.
	 * @param C  Constant offset the pulses sit on.
	 * @param trace Trace to compute the chisquare with respect to.
	 * @param low, high Region of interest over which to compute the 
	 *   chi-square value.  
	 * @note high = -1 will set the high limit to the last sample in 
	 *   the trace.
	 * @return The chi-square goodness-of-fit statistic.
	 */
	double chiSquare2(
	    double A1, double k1, double k2, double x1,
	    double A2, double k3, double k4, double x2,
	    double C,    
	    const std::vector<uint16_t>& trace,
	    int low = 0, int high = -1
	    );    
	/**
	 * @brief Computes the chi-square goodness of a specific 
	 * parameterization of a double pulse canonical form with respect 
	 * to a trace.
	 * @param A1 Amplitude of the first pulse.
	 * @param k1 Steepness of first pulse rise.
	 * @param k2 Decay time of the first pulse.
	 * @param x1 Position of the first pulse.
	 * @param A2 Amplitude of the second pulse.
	 * @param k3 Steepness of second pulse rise.
	 * @param k4 Decay time of second pulse.
	 * @param x2 Position of second pulse.
	 * @param C  Constant offset the pulses sit on.
	 * @param points Set of x, y data points which contribute to the total 
	 *   chi-square value
	 * @return The chi-square goodness-of-fit statistic.
	 */
	double chiSquare2(
	    double A1, double k1, double k2, double x1,
	    double A2, double k3, double k4, double x2,
	    double C,
	    const std::vector<std::pair<uint16_t, uint16_t> >& points
	    );
	/**
	 * @brief Write a single trace to a file.
	 *  @param filename  Where to write.
	 *  @param title     Title string.
	 *  @param trace     The trace data.
	 */	
	void writeTrace(
	    const char* filename, const char* title,
	    const std::vector<uint16_t>& trace
	    );
	/**
	 * @brief Write two traces to a file.
	 * @param filename Where to write.
	 * @param title    Title string.
	 * @param t1       The first trace.
	 * @param t2       The second trace.
	 */
	void writeTrace2(
	    const char* filename, const char* title,
	    const std::vector<uint16_t>& trace1,
	    const std::vector<uint16_t>& trace2
	    );

	/** @} */
	
    } // namespace ddastoys::analyticfit
} // namespace ddastoys

#endif
