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
 * @file  functions_analytic.cpp
 * @brief Implement analytic functions used to fit DDAS pulses.
 * @note  All functions are in the DDAS::AnalyticFit namespace.
 */

#include "functions_analytic.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>

/**
 * @brief Evaluate a logistic function for the specified parameters and point.
 *
 * A logistic function is a function with a sigmoidal shape. We use it to fit 
 * the rising edge of signals DDAS digitizes from detectors. See e.g. 
 * https://en.wikipedia.org/wiki/Logistic_function for a discussion of this
 * function.
 *
 * @param A   Amplitude of the signal.
 * @param k   Steepness of the signal (related to the rise time).
 * @param x1  Mid point of the rise of the logistic.
 * @param x   Location at which to evaluate the function.
 *
 * @return double  Function value at x
 */
double
DDAS::AnalyticFit::logistic(double A, double k, double x1, double x)
{
  return A/(1+exp(-k*(x-x1)));
}

/**
 * @brief Evaluate an exponential decay for the specific parameters and point.
 *
 * Signals from detectors usually have a falling shape that approximates an 
 * exponential. This function evaluates this decay at some point.
 *
 *@param A   Amplitude of the signal
 *@param k   Decay constant of the signal.
 *@param x1  Position of the pulse.
 *@param x   Where to evaluate the signal.
 *
 *@return double  Function value at x
 */
double
DDAS::AnalyticFit::decay(double A, double k, double x1, double x)
{
  return A*(exp(-k*(x-x1)));
}

/**
 * @brief Logistic switch to turn on a function evaluation at a given point.
 *
 * Provides a multipier that can be used to turn on a function at a point in 
 * space. This is provided for GPUs that may want to define functions with 
 * conditional chunks without the associated flow control divergence 
 * conditionals would require. This is implemented as a _very_ steep logistic 
 * with rise centered at the point in question.
 *
 *  @param x1  Switch on point. The switch is "on" for values greater than x1.
 *  @param x   Point in space at which to evaluate the switch.
 *
 *  @return double  The switch value: very nearly 0.0 left of x1 and very 
 *                  nearly 1.0 to the right of x1.
 */
double
DDAS::AnalyticFit::switchOn(double x1, double x)
{
  return logistic(1.0, 10000.0, x1, x);
}

/**
 * @brief Evaluate the single pulse fit function at a given point.
 *
 * Evaluate the value of a single pulse in accordance with our canonical 
 * functional form. The form is a logistic rise with an exponential decay 
 * that sits on top of a constant offset.
 *
 * @param A1  Pulse amplitiude.
 * @param k1  Logistic rise steepness.
 * @param k2  Exponential decay time constant.
 * @param x1  Logistic position.
 * @param C   Constant offset.
 * @param x   Position at which to evaluate the function.
 * 
 * @return double  Single pulse function evaluated at x.
 */
double
DDAS::AnalyticFit::singlePulse(
			       double A1, double k1, double k2, double x1,
			       double C, double x
			       )
{
  return (logistic(A1, k1, x1, x) * decay(1.0, k2, x1, x)) + C;
}

/**
 * @brief Evaluate the double pulse fit function at a given point.
 *
 * Evaluate the canonical form of a double pulse. This is done by summing 
 * two single pulses. The constant term is thrown into the first pulse. 
 * The second pulse gets a constant term of 0.
 *
 * @param A1    Amplitude of the first pulse.
 * @param k1    Steepness of first pulse rise.
 * @param k2    Decay time of the first pulse.
 * @param x1    position of the first pulse.
 *
 * @param A2    Amplitude of the second pulse.
 * @param k3    Steepness of second pulse rise.
 * @param k4    Decay time of second pulse.
 * @param x2    position of second pulse.
 *
 * @param C     Constant offset the pulses sit on.
 * @param x     position at which to evaluate the pulse.
 *
 * @return double  Double pulse function evaluated at x.
 */
double
DDAS::AnalyticFit::doublePulse(
			       double A1, double k1, double k2, double x1,
			       double A2, double k3, double k4, double x2,
			       double C, double x    
			       )
{
  double p1 = singlePulse(A1, k1, k2, x1, C, x);
  double p2 = singlePulse(A2, k3, k4, x2, 0.0, x);
  return p1 + p2;
}

/**
 * @brief Calculate the pulse amplitude corrected for the ballistic deficit 
 * imposed by the exponential decay.
 *
 * The A1 term in a pulse fit is not actually the "true" pulse amplitude. 
 * The effect of the exponential decay is already important causing A1 to 
 * over-estimate the amplitude.
 *
 * This function computes the "true" amplitude of a pulse given its parameters.
 * This is done by noting that the deriviative of the pulse has a zero at:
 * 
 *   x = x0 + ln(k1/k2 - 1)/k1
 *
 * We plug that position back into the pulse to get the amplitude.
 *
 * @param A   The scaling term of the pulse.
 * @param k1  The steepness term of the logistic.
 * @param k2  The fall time term of the decay.
 * @param x0  The position of the pulse.
 *
 * @return double  The corrected amplitude of the fitted pulse.
 *
 * @retval -1  If k1/k2 <= 1.
 */
double
DDAS::AnalyticFit::pulseAmplitude(double A, double k1, double k2, double x0)
{
  double frac = k1/k2;
  if (frac <= 1.0) {
    return -1; 
  }
  double pos = x0 + log(frac-1.0)/k1;
  return singlePulse(A, k1, k2, x0, 0.0, pos);
}

/**
 * @warning This function is a wrapper for DDAS:AnalyticFit::pulseAmplitude and 
 * issues a warning if it is called. It exists for backwards compatability and 
 * should not be used. The correct function in the  DDAS::AnalyticFit namespace
 * should be called instead.
 *
 * @brief Calculate the pulse amplitude corrected for the ballistic deficit 
 * imposed by the exponential decay.
 *
 * The A1 term in a pulse fit is not actually the "true" pulse amplitude. 
 * The effect of the exponential decay is already important causing A1 to 
 * over-estimate the amplitude.
 *
 * This function computes the "true" amplitude of a pulse given its parameters.
 * This is done by noting that the deriviative of the pulse has a zero at:
 * 
 *   x = x0 + ln(k1/k2 - 1)/k1
 *
 * We plug that position back into the pulse to get the amplitude.
 *
 * @param A   The scaling term of the pulse.
 * @param k1  The steepness term of the logistic.
 * @param k2  The fall time term of the decay.
 * @param x0  The position of the pulse.
 *
 * @return double  DDAS::AnalyticFit::pulseAmplitude(A, k1, k2, x0);
 * @retval -1  If k1/k2 <= 1.
 */
double pulseAmplitude(double A, double k1, double k2, double x0)
{
  static bool warned(false);
  if(!warned) {
    std::cerr << "WARNING the pulseAmplitude function is in the DDAS::AnalyticFit namespace\n";
    std::cerr << "It should be called as DDAS::AnalyticFit::pulseAmplitude(...);\n";
    warned = true;
  }
  return  DDAS::AnalyticFit::pulseAmplitude(A, k1, k2, x0);
}

/**
 * @brief Computes the chi-square goodness-of-fit for a specific 
 * parameterization of a single pulse canonical form with respect to a trace.
 *
 * The chi-square value is calculated based on the passed limits low, high.
 *
 * @param A1  Amplitude of pulse
 * @param k1  Steepness of pulse rise.
 * @param k2  Decay time of pulse fall.
 * @param x1  Position of the pulse.
 * @param C   Constant offset of the trace.
 * @param trace  Trace to compute the chi-square value with respect to.
 * @param low, high  Region of interest over which to compute the chi-square 
 *                   value.
 *
 * @note high = -1 will set the high limit to the last sample in the trace.
 *
 * @return double  The chi-square goodness-of-fit statistic.
 */
double
DDAS::AnalyticFit::chiSquare1(
			      double A1, double k1, double k2,double x1,
			      double C,
			      const std::vector<std::uint16_t>& trace,
			      int low, int high
			      )
{
  if (high == -1) high = trace.size() - 1;
    
  double result = 0.0;
  for (int i = low; i <= high; i++) {
    double x = i;
    double y = trace[i];
    double pulse = singlePulse(A1, k1, k2, x1 ,C, x);  // Fitted pulse.
    double diff = y-pulse;
    if (y != 0.0) {
      result += (diff/y)*diff;  // This order may control overflows
      if (std::fpclassify(result) == FP_ZERO) result =  0.0;
    }
  }
  
  return result;
}

/**
 * @brief Computes the chi-square goodness-of-fit for a specific 
 * parameterization of a single pulse canonical form with respect to a trace.
 *
 * The chi-square value is calculated from a passed set of (x, y) points.
 *
 * @param A1  Amplitude of pulse
 * @param k1  Steepness of pulse rise.
 * @param k2  Decay time of pulse fall.
 * @param x1  Position of the pulse.
 * @param C   Constant offset of the trace.
 * @param points  Set of x, y data points which contribute to the total 
 *                chi-square value
 *
 * @return double  The chi-square goodness-of-fit statistic.
 */
double
DDAS::AnalyticFit::chiSquare1(
           double A1, double k1, double k2, double x1,
           double C,
           const std::vector<std::pair<std::uint16_t, std::uint16_t> >& points
			      )
{    
  double result = 0.0;
  for  (size_t i = 0; i < points.size(); i++) {
    double x = points[i].first;
    double y = points[i].second;
    double pulse = singlePulse(A1, k1, k2, x1 ,C, x);  // Fitted pulse.
    double diff = y-pulse;
    if (y != 0.0) {
      result += (diff/y)*diff;  // This order may control overflows
      if (std::fpclassify(result) == FP_ZERO) result =  0.0;
    }
  }
  
  return result;
}

/**
 * @brief Computes the chi-square goodness of a specific parameterization of a
 * double pulse canonical form with respect to a trace.
 *
 * The chi-square values is calculated based on the passed limits low, high.
 *
 * @param A1  Amplitude of the first pulse.
 * @param k1  Steepness of first pulse rise.
 * @param k2  Decay time of the first pulse.
 * @param x1  Position of the first pulse.
 *
 * @param A2  Amplitude of the second pulse.
 * @param k3  Steepness of second pulse rise.
 * @param k4  Decay time of second pulse.
 * @param x2  Position of second pulse.
 *
 * @param C    Constant offset the pulses sit on.
 * @param trace  Trace to compute the chisquare with respect to.
 * @param low, high  Region of interest over which to compute the chi-square 
 *                   value.
 *  
 * @note high = -1 will set the high limit to the last sample in the trace.
 *
 * @return double  The chi-square goodness-of-fit statistic.
 */
double
DDAS::AnalyticFit::chiSquare2(
			      double A1, double k1, double k2, double x1,
			      double A2, double k3, double k4, double x2,
			      double C,    
			      const std::vector<std::uint16_t>& trace,
			      int low, int high
			      )
{
  double result = 0.0;
  if (high == -1) high = trace.size() -1;
    
  for (int i = low; i <= high; i++) {
    double x = i;
    double y = trace[i];
    double pulse = doublePulse(A1, k1, k2, x1, A2, k3, k4, x2, C, x);
    double diff = y - pulse;
    if (std::fpclassify(diff) == FP_ZERO) diff = 0.0;
    if (y != 0.0) {
      result += (diff/y)*diff;  // This order may control overflows
      if (std::fpclassify(result) == FP_ZERO) result =  0.0;
    }
        
        
  }
    
  return result;
}

/**
 * @brief Computes the chi-square goodness of a specific parameterization of a
 * double pulse canonical form with respect to a trace.
 *
 * The chi-square value is calculated from a passed set of (x, y) points.
 *
 * @param A1  Amplitude of the first pulse.
 * @param k1  Steepness of first pulse rise.
 * @param k2  Decay time of the first pulse.
 * @param x1  Position of the first pulse.
 *
 * @param A2  Amplitude of the second pulse.
 * @param k3  Steepness of second pulse rise.
 * @param k4  Decay time of second pulse.
 * @param x2  Position of second pulse.
 *
 * @param C    Constant offset the pulses sit on.
 * @param points  Set of x, y data points which contribute to the total 
 *                chi-square value
 *
 * @return double  The chi-square goodness-of-fit statistic.
 */
double
DDAS::AnalyticFit::chiSquare2(
         double A1, double k1, double k2, double x1,
         double A2, double k3, double k4, double x2,
         double C,
         const std::vector<std::pair<std::uint16_t, std::uint16_t> >& points
			      )
{
  double result = 0.0;
    
  for (size_t i = 0; i < points.size(); i++) {
    double x = points[i].first;
    double y = points[i].second;
    double pulse = doublePulse(A1, k1, k2, x1, A2, k3, k4, x2, C, x);
    double diff = y - pulse;
    if (std::fpclassify(diff) == FP_ZERO) diff = 0.0;
    if (y != 0.0) {
      result += (diff/y)*diff;  // This order may control overflows
      if (std::fpclassify(result) == FP_ZERO) result =  0.0;
    }
  }
    
  return result;
}

/**
 * @brief Write  a single trace to file.
 *
 *  @param filename  Where to write.
 *  @param title     Title string.
 *  @param trace     The trace data.
 */
void
DDAS::AnalyticFit::writeTrace(
			      const char* filename, const char* title,
			      const std::vector<std::uint16_t>& trace
			      )
{
  std::ofstream o(filename);
  o << title << std::endl;    
  for (size_t i = 0; i < trace.size(); i++) {
    o << i << " " << trace[i] << std::endl;
  }
}

/**
 * @brief Write two traces to file.
 *
 * Same as writeTrace, but the two traces and the chi square component of the 
 * difference between trace1, trace2
 *
 * @param filename  Where to write.
 * @param title     Title string.
 * @param t1        The first trace.
 * @param t2        The second trace.
 *
 * @note The traces must be the same length.
 */
void
DDAS::AnalyticFit::writeTrace2(
			       const char* filename, const char* title,
			       const std::vector<std::uint16_t>& t1,
			       const std::vector<std::uint16_t>& t2
			       )
{
  std::ofstream o(filename);    
  o << title << std::endl;
  for (size_t i = 0; i < t1.size(); i++) {
    std::uint16_t diff = t1[i] - t2[i];
    o << i << " " << t1[i] << " " << t2[i]
      << " " << diff*diff/t1[i] << std::endl;
  }
}

