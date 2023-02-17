/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file  SerialFitEngineAnalytic.cpp
 *  @brief Implement the serial fit engines for single and double pulse fits.
 */

#include "jacobian_analytic.h"

#include <cmath>

#include "functions_analytic.h"

// Single pulse fit parameter indices:

static const int P1A_INDEX(0);
static const int P1K1_INDEX(1);
static const int P1K2_INDEX(2);
static const int P1X1_INDEX(3);
static const int P1C_INDEX(4);
static const int P1_PARAM_COUNT(5);

static const int P2A1_INDEX(0);
static const int P2K1_INDEX(1);
static const int P2K2_INDEX(2);
static const int P2X1_INDEX(3);
 
static const int P2A2_INDEX(4);
static const int P2K3_INDEX(5);
static const int P2K4_INDEX(6);
static const int P2X2_INDEX(7);
static const int P2C_INDEX(8);

using namespace DDAS;

//////////////////////////////////////////////////////////////////////////////
// Partial derivative functions that are common:

/**
 * Returns the partial derivative of a single pulse with respect to the
 * amplitude evaluated at a point
 *
 * @param k1  Current guess at rise steepness param (log(81)/risetime90).
 * @param k2  Current guess at the decay time constant.
 * @param x1  Current guess at pulse position.
 * @param x   x value at which to evaluate all this.
 * @param w   Weight for the point 
 * @param erise, efall  Pre-computed common exponential terms needed to 
 *                      evaluate the derivative.
 *
 * @return double  Value of (dP1/dA)(x)/w
 */
static double
dp1dA(double k1, double k2, double x1, double x, double w,
      double erise, double efall)
{
  double d = efall;             // decay(1.0, k2, x1, x);
  double l = 1.0/(1.0 + erise); // logistic(1.0, k1, x1, x);
  return d*l / w;
}

/**
 * Partial of single pulse with respect to the rise time constant k1.
 *
 * @param A   Current guess at amplitude.
 * @param k1  Current guess at rise steepness param (log(81)/risetime90).
 * @param k2  Current guess at the decay time constant.
 * @param x1  Current guess at pulse position.
 * @param x   x value at which to evaluate all this.
 * @param w   Weight for the point 
 * @param erise, efall  Pre-computed common exponential terms needed to 
 *                      evaluate the derivative.
 *
 * @return double  Value of (dP1/dk1)(x)/w
 */
static double
dp1dk1(double A, double k1, double k2, double x1, double x, double w,
       double erise, double efall)
{
  double d1 =   A*efall; // decay(A, k2, x1, x)
  double d2 =   erise;   // part of logistic deriv.
  double num = d1*d2*(x - x1);
  double l   =  1.0/(1.0 + erise); //  logistic(1.0, k1, x1, x)
    
  return (num*l*l)/w;
}

/**
 * Partial of a single pulse with respect to the decay time constant.
 *
 * @param A   Current guess at amplitude.
 * @param k1  Current guess at rise steepness param (log(81)/risetime90).
 * @param k2  Current guess at the decay time constant.
 * @param x1  Current guess at pulse position.
 * @param x   x value at which to evaluate all this.
 * @param w   Weight for the point 
 * @param erise, efall  Pre-computed common exponential terms needed to 
 *                      evaluate the derivative.
 *
 * @return double  Value of (dP1/dk2)(x)/w
 */
static double
dp1dk2(double A, double k1, double k2, double x1, double x, double w,
       double erise, double efall)
{
  double d1 = A*efall; // decay(A, k2, x1, x)
  double num = d1*(x1 - x);
  double l = 1.0/(1.0 + erise); // logistic(1.0, k1, x1, x)
    
  return (num*l)/w;
}

/**
 * Partial derivative of a single pulse with respect to the time at the 
 * middle of the pulse's rise.
 *
 * @param A   Current guess at amplitude.
 * @param k1  Current guess at rise steepness param (log(81)/risetime90).
 * @param k2  Current guess at the decay time constant.
 * @param x1  Current guess at pulse position.
 * @param x   x value at which to evaluate all this.
 * @param w   Weight for the point 
 * @param erise, efall  Pre-computed common exponential terms needed to 
 *                      evaluate the derivative.
 *
 * @return double  Value of (dP1/dk2)(x)/w
 */
static double
dp1dx1(double A, double k1, double k2, double x1, double x, double w,
       double erise, double efall)
{
  double dk1 = erise;             // decay(1.0, k1, x1, x)
  double dk2 = efall;             // decay(1.0, k2, x1, x)
  double l   = 1.0/(1.0 + erise); // logistic(1.0, k1, x1, x)
    
  double left = A*k2*dk2*l;
  double right = A*k1*dk1*dk2*l*l;
    
  return (left - right)/w;
}

/**
 * Partial derivative of single pulse with respect to the constant term
 * evaluated at a point.
 *
 * @param A   Current guess at amplitude.
 * @param k1  Current guess at rise steepness param (log(81)/risetime90).
 * @param k2  Current guess at the decay time constant.
 * @param x1  Current guess at pulse position.
 * @param x   x value at which to evaluate all this.
 * @param w   weight for the point 
 *
 * @return double  Value of (dP1/dC)(x)/w
 */
static double
dp1dC(double A, double k1, double k2, double x1, double x, double w)
{
    
  return 1.0/w;
}

//////////////////////////////////////////////////////////////////////////////
// Serial fit engine 1
//   Implementation of fit engine for single pulse serial execution.
//

/**
 * Construct the fit engine and set the input data. Delegates to base class 
 * construction.
 *
 * @param data  The trace data.
 */
SerialFitEngine1::SerialFitEngine1(std::vector<std::pair<std::uint16_t, std::uint16_t>>& data) :
  CFitEngine(data)
{}

/**
 * @brief Compute the vector of residuals.
 *
 * @param[in] p   Current fit parameters.
 * @param[out] r  Vector of residual values.
 */
void
SerialFitEngine1::residuals(const gsl_vector* p, gsl_vector* r)
{
  // Extract the fit parameters:
    
  double A  = gsl_vector_get(p, P1A_INDEX);
  double k1 = gsl_vector_get(p, P1K1_INDEX);
  double k2 = gsl_vector_get(p, P1K2_INDEX);
  double x1 = gsl_vector_get(p, P1X1_INDEX);
  double C  = gsl_vector_get(p, P1C_INDEX);

  // x, y are vectors that are the actual trace, and have the same number
  // of elements. 
  size_t npts = x.size();
  for (size_t i =0; i < npts; i++) {
    double xi = x[i];
    double yactual = y[i];        
    double fitted  = DDAS::AnalyticFit::singlePulse(A, k1, k2, x1, C, xi);
    gsl_vector_set(r, i, (fitted - yactual));
  }
}

/**
 * @brief Compute the Jacobian matrix.
 *
 * Compute the Jacobian matrix of the fit with respect to current values of 
 * the fit parameters.
 *
 * @param[in] p   Current fit parameterization.
 * @param[out] J  The Jacobian for this iteration of the fit.
 */
void
SerialFitEngine1::jacobian(const gsl_vector* p, gsl_matrix* J)
{
  // Note:
  // - the weights are 1.0.
  // - the x/y base class members are the trace.
  
  double A   = gsl_vector_get(p, P1A_INDEX);
  double k1  = gsl_vector_get(p, P1K1_INDEX);
  double k2  = gsl_vector_get(p, P1K2_INDEX);
  double x1  = gsl_vector_get(p, P1X1_INDEX);
    
  size_t npts = x.size();
  for (size_t i =0; i < npts; i++) {
    double xi = x[i];
    double erise = exp(-k1*(xi - x1)); // these are common subexpressions
    double efall = exp(-k2*(xi - x1)); // we can factor out here:
        
    // Compute the partials:
        
    double Ai   = dp1dA(k1, k2, x1, xi, 1.0, erise, efall); // Weights are 1.0.
    double k1i  = dp1dk1(A, k1, k2, x1, xi, 1.0, erise, efall);
    double k2i  = dp1dk2(A, k1, k2, x1, xi, 1.0, erise, efall);
    double x1i  = dp1dx1(A, k1, k2, x1, xi, 1.0, erise, efall);
    double Ci   = dp1dC(A, k1, k2, x1, xi, 1.0);
        
    // Stuff them into the proper elements of the jacobian matrix.
        
    gsl_matrix_set(J, i, P1A_INDEX, Ai);
    gsl_matrix_set(J, i, P1K1_INDEX, k1i);
    gsl_matrix_set(J, i, P1K2_INDEX, k2i);
    gsl_matrix_set(J, i, P1X1_INDEX, x1i);
    gsl_matrix_set(J, i, P1C_INDEX, Ci);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Implement SerialFitEngine2 - the serialized fit engine for double
// pulses.

/**
 * @brief Construct the fit engine and set the input data.
 *
 * Delegates to base class construction.
 *
 * @param data  The trace.
 */
SerialFitEngine2::SerialFitEngine2(std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data) :
  CFitEngine(data) {}

/**
 * @brief Compute the residual vector.
 *
 * @param[in] p   Current parameter set.
 * @param[out] r  Pointwise residuals.
 */
void
SerialFitEngine2::residuals(const gsl_vector* p, gsl_vector* r)
{
  // Pull out the current fit parameterization>
    
  double A1    = gsl_vector_get(p, P2A1_INDEX);   // Pulse 1.
  double k1    = gsl_vector_get(p, P2K1_INDEX);
  double k2    = gsl_vector_get(p, P2K2_INDEX);
  double x1    = gsl_vector_get(p, P2X1_INDEX);    
    
  double A2    = gsl_vector_get(p, P2A2_INDEX);   // Pulse 2.
  double k3    = gsl_vector_get(p, P2K3_INDEX);
  double k4    = gsl_vector_get(p, P2K4_INDEX);
  double x2    = gsl_vector_get(p, P2X2_INDEX);
    
  double C     = gsl_vector_get(p, P2C_INDEX);    // constant.
    
  size_t npts = x.size();
  for (size_t i = 0; i < npts; i++) {
    double xc = x[i];
    double yc = y[i];
    double p  = DDAS::AnalyticFit::doublePulse(A1, k1, k2, x1, A2, k3, k4, x2, C, xc);
    gsl_vector_set(r, i, (p - yc));
  }
}

/**
 * @brief Compute the Jacobian matrix.
 *
 * Compute the Jacobian matrix of the fit with respect to current values of 
 * the fit parameters.
 *
 * @param[in] p   The current parameters.
 * @param[out] J  The Jacobian matrix to fill in.
 */
void
SerialFitEngine2::jacobian(const gsl_vector* p, gsl_matrix* J)
{
  // Fish the current fit parameters from p:
    
  double A1    = gsl_vector_get(p, P2A1_INDEX); // Pulse 1.
  double k1    = gsl_vector_get(p, P2K1_INDEX);
  double k2    = gsl_vector_get(p, P2K2_INDEX);
  double x1    = gsl_vector_get(p, P2X1_INDEX);    
    
  double A2    = gsl_vector_get(p, P2A2_INDEX); // Pulse 2.
  double k3    = gsl_vector_get(p, P2K3_INDEX);
  double k4    = gsl_vector_get(p, P2K4_INDEX);
  double x2    = gsl_vector_get(p, P2X2_INDEX);
        
  size_t npts = x.size();
  for (size_t i = 0; i < npts; i++) {
    double xc = x[i];
        
    double erise1 = exp(-k1*(xc - x1));
    double efall1 = exp(-k2*(xc - x1));
        
    double erise2 = exp(-k3*(xc - x2));
    double efall2 = exp(-k4*(xc - x2));
        
    gsl_matrix_set(
	   J, i, P2A1_INDEX, dp1dA(k1, k2, x1, xc, 1.0, erise1, efall1)
		   );
    gsl_matrix_set(
	   J, i, P2K1_INDEX, dp1dk1(A1, k1, k2, x1, xc, 1.0, erise1, efall1)
		   );
    gsl_matrix_set(
	   J, i, P2K2_INDEX, dp1dk2(A1, k1, k2, x1, xc, 1.0, erise1, efall1)
		   );
    gsl_matrix_set(
	   J, i, P2X1_INDEX, dp1dx1(A1, k1, k2, x1, xc, 1.0, erise1, efall1)
		   );
        
    // For pulse 2 elements:  A1->A2, k1 -> k3, k2 -> k4, x1 -> x2
        
    gsl_matrix_set(
	   J, i, P2A2_INDEX, dp1dA(k3, k4, x2, xc, 1.0, erise2, efall2)
		   );
    gsl_matrix_set(
	   J, i, P2K3_INDEX, dp1dk1(A2, k3, k4, x2, xc, 1.0, erise2, efall2)
		   );
    gsl_matrix_set(
	   J, i, P2K4_INDEX, dp1dk2(A2, k3, k4, x2, xc, 1.0, erise2, efall2)
		   );
    gsl_matrix_set(
	   J, i, P2X2_INDEX, dp1dx1(A2, k3, k4, x2, xc, 1.0, erise2, efall2)
		   );
        
    // Don't forget the constant term
        
    gsl_matrix_set(
	   J, i, P2C_INDEX, dp1dC(A1, k1, k2, x1, xc, 1.0)
		   ); // Need to make the function call if weights are != 1
  }
    
}
