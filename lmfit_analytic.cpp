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
 * @file  lmfit_analytic.cpp
 * @brief Implementation of analytic fitting functions we use in GSL's LM 
 * fitter.
 */

/** @addtogroup AnalyticFit
 * @{
 */

#include "lmfit_analytic.h"

#include <algorithm>
#include <stdexcept>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_multifit_nlin.h>

#include "functions_analytic.h"
#include "jacobian_analytic.h"

// Constants used in deriving estimates of k1, k2:

static const double ln2(log(2));
static const double ln3over4(log(3) - log(4));
static const double ln9(log(9));

const double DDAS_FALLBACK_K1 = 0.1; //!< Fallback for pulse risetime.
const double DDAS_FALLBACK_K2 = 0.1; //!< Fallback for pulse decay.

const int SINGLE_MAXITERATIONS = 50;  //!< Max iterations for single pulse fit
const int DOUBLE_MAXITERATIONS = 200; //!< Max iterations for double pulse fit

// Single pulse fit parameter indices:

static const int P1A_INDEX(0);
static const int P1K1_INDEX(1);
static const int P1K2_INDEX(2);
static const int P1X1_INDEX(3);
static const int P1C_INDEX(4);
static const int P1_PARAM_COUNT(5);

// Double pulse fit with all parameters free:

static const int P2A1_INDEX(0);
static const int P2K1_INDEX(1);
static const int P2K2_INDEX(2);
static const int P2X1_INDEX(3); 
static const int P2A2_INDEX(4);
static const int P2K3_INDEX(5);
static const int P2K4_INDEX(6);
static const int P2X2_INDEX(7);
static const int P2C_INDEX(8); 
static const int P2_PARAM_COUNT(9);

// Double pulse fit with same time parameters parameter indices

static const int P2FTA1_INDEX(0);
static const int P2FTK1_INDEX(1);
static const int P2FTK2_INDEX(2);
static const int P2FTX1_INDEX(3);
static const int P2FTA2_INDEX(4);
static const int P2FTX2_INDEX(5); 
static const int P2FTC_INDEX(6); 
static const int P2FT_PARAM_COUNT(7);
 

/*------------------------------------------------------------------
 * Utility functions.
 */

/**
 * @brief Given a trace and a saturation value, returns the vector of 
 * sample-no, sample-value pairs that are below a saturation value.
 *
 * @param[out] points     Reduced trace.
 * @param[in]  trace      Raw trace.
 * @param[in]  low, high  Limits of the trace to reduce.
 * @param[in]  saturation Saturation level.
 */
static void reduceTrace(
    std::vector<std::pair<std::uint16_t, std::uint16_t>>& points,
    int low, int high,
    const std::vector<std::uint16_t>& trace, std::uint16_t saturation)
{
    for (int i = low; i <= high; i++) {
	if (trace[i] < saturation) {
	    points.push_back(
		std::pair<std::uint16_t, std::uint16_t>(i, trace[i])
		);
	}
    }
    
}

/*-----------------------------------------------------------------------------
 *  Fitting via Levenberg-Marquardt in the GSL.
 *
 *  First we need some useful functions to compute Jacobian elements:
 */

/**
 * @brief Returns the partial derivative of a single pulse with respect to the
 * amplitude evaluated at a point.
 *
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x value at which to evaluate all this.
 * @param w  Weight for the point.
 *
 * @return Value of (dP1/dA)(x)/w
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
 * @brief Partial of single pulse with respect to the rise time constant k1.
 *
 * @param A  Current guess at amplitude.
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x value at which to evaluate all this.
 * @param w  Weight for the point.
 *
 * @return Value of (dP1/dk1)(x)/w
 */
static double
dp1dk1(double A, double k1, double k2, double x1, double x, double w,
       double erise, double efall)
{
    double d1  = A*efall;           // decay(A, k2, x1, x);  
    double d2  = erise;             // decay(1.0, k1, x1,  x);
    double num = d1*d2*(x - x1);
    double l   = 1.0/(1.0 + erise); // logistic(1.0, k1, x1, x);   
     
    return (num*l*l)/w;
}

/**
 * @brief Partial of a single pulse with respect to the decay time constant.
 *
 * @param A  Current guess at amplitude.
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x value at which to evaluate all this.
 * @param w  Weight for the point.
 *
 * @return Value of (dP1/dk2)(x)/w.
 */
static double
dp1dk2(double A, double k1, double k2, double x1, double x, double w,
       double erise, double efall)
{
    double d1  = A*efall;           // decay(A, k2, x1, x);
    double num = d1*(x1 - x);
    double l   = 1.0/(1.0 + erise); // logistic(1.0, k1, x1, x);
    
    return (num*l)/w;
}

/**
 * @brief Partial of a single pulse with respect to the time at the middle
 * of the pulse's rise.
 *
 * @param A  Current guess at amplitude.
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x value at which to evaluate all this.
 * @param w  Weight for the point .
 *
 * @return Value of (dP1/dk2)(x)/w.
 */
static double
dp1dx1(double A, double k1, double k2, double x1, double x, double w,
       double erise, double efall)
{
    double dk1 = erise;             // decay(1.0, k1, x1, x);
    double dk2 = efall;             // decay (1.0, k2, x1, x);
    double l   = 1.0/(1.0 + erise); // logistic(1.0, k1, x1, x);
    
    double left  = A*k2*dk2*l;
    double right = A*k1*dk1*dk2*l*l;
    
    return (left - right)/w;
}

/**
 * @brief Partial derivative of single pulse with respect to the constant term
 * evaluated at a point.
 *
 * @param A  Current guess at amplitude.
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x value at which to evaluate all this.
 * @param w  Weight for the point 
 *
 * @return Value of (dP1/dC)(x)/w.
 */
static double
dp1dC(double A, double k1, double k2, double x1, double x, double w)
{    
    return 1.0/w;
}

/*
 * GSL's LM  fitter requires that we provide:
 *   -  A function that can produce the residuals (gsl_p1Residuals).
 *   -  A function that can compute the Jacobians (gsl_p1Jacobian).
 *   -  A function that can do both (gsl_p1Compute)
 *
 * We need make a choice about the allocation of gsl/vector/matrix indices
 * to the fit parameters A, k1, k2, x1 and C:
 */
 
/**
 * @brief Compute the vector of residuals applied to the data points for the 
 * specified parameters.
 *
 * @param[in]  p     Current parameters of the fit (see indices above).
 * @param[in]  pData Actually a pointer to a GslFitParameters struct.
 * @param[out] r     Function residuals for each data point.
 *
 * @return Status of the computation (GSL_SUCCESS).  
 *
 * @note GPU implementation hint: This function is nicely data parallel.
 */
static int
gsl_p1Residuals(const gsl_vector* p, void* pData, gsl_vector* r)
{
    // Note all points are weighted by 1.0 in this computation.
    CFitEngine* pEngine = reinterpret_cast<CFitEngine*>(pData);
    pEngine->residuals(p, r);
    
    return GSL_SUCCESS;
}

/**
 * @brief Compute the Jacobian matrix  partial derivatives evaluated at each
 * data point.
 *
 * @param[in]  p     Current fit parameterization.
 * @param[in]  pdata Actually a pointer to a GslFitParameters struct.
 * @param[out] J     The Jacobian for this iteration of the fit.
 * 
 * @return GSL_SUCCESS if all computations work.
 *
 * @note GPU Implementation hint: This function is nicely data parallel.
 */
static int
gsl_p1Jacobian(const gsl_vector* p, void* pData, gsl_matrix* J)
{
    // Note: the weights will all be 1.0.
    CFitEngine* pEngine = reinterpret_cast<CFitEngine*>(pData);
    pEngine->jacobian(p, J);
  
    return GSL_SUCCESS;
}

/**
 * @brief Compute function calls gsl_p1Residuals and gsl_p1Jacobian.
 *
 * @param[in]  p      Current fit parameterization.
 * @param[in]  pData  Pointer to the vector of raw data points (trace).
 * @param[out] resids Computed vector of residuals.
 * @param[out] J      Computed Jacobian matrix.
 *
 * @return GSL_SUCCESS completion status.
 *
 * @note GPU implementation hint: That both functions are nicely data parallel
 *   and can run in parallel w.r.t. to each other.
 */
static int
gsl_p1Compute(const gsl_vector* p, void*pData, gsl_vector* resids, gsl_matrix* J)
{
    CFitEngine* pEngine = reinterpret_cast<CFitEngine*>(pData);    
    pEngine->residuals(p, resids);
    pEngine->jacobian(p, J);   
    return GSL_SUCCESS;
}

/**
 * @brief Estimate the decay constant for a single trace given that we know 
 * where the maximum is.
 *
 * @details
 * We use that k = ln2/(x-x0) when x is the point at half max and that 
 * k = ln(3/4)/(x0-x) when x is the point at 3/4 max.
 *
 * We search from the max channel and if we're lucky we find both the 3/4
 * and the 1/2 values for x-x0 -- then we take the average as our estimate.
 * If we only find the k for 3/4 we use it. If we find neither we fall back 
 * to 0.1 as originally.
 *
 * @param x0    Channel of the trace maximum starting the search.
 * @param C0    Estimate for the constant.
 * @param trace The trace.
 *
 * @note For now assume that the 3/4 and 1/2 points are inside the AOI if
 *   they exist. In the future could confine the search to the AOI.
 * @note For now we're not correcting for flat-tops. 
 */
static double
estimateK2(int x0, double C0, const std::vector<std::uint16_t>& trace)
{
    double khalf = -1.0; // Assume k is positive.
    double k34   = -1.0;
    int    maxval = trace[x0] - C0;
    
    for (size_t i = x0; i < trace.size(); i++) {
        
	// If we've not computed k34 and we crossed the 3/4 threshold
	// compute k34:        
	double v = trace[i] - C0;
	if ((k34 < 0.0) && (v < 0.75*maxval)) {
	    k34 = ln3over4/(x0 - i);
	}

	// Similarly for khalf:        
	if ((khalf < 0.0) && (v < 0.5*maxval)) {
	    khalf = ln2/(i - x0);            
	    // Note that we've got both so we can quit:            
	    break;
	}
    }
  
    //  Deal with the three potential cases:
    
    if ((k34 > 0) && (khalf > 0)) return 0.5*(k34 + khalf); // Average of both.
    if (k34 > 0) return k34; // Only have k34.
    return DDAS_FALLBACK_K2; // Desparation measures.
}

/**
 * @brief Estimate a value for the steepness parameter of the rising side 
 * of the pulse. 
 *
 * @details
 * We approximate xmax as 0.9 of maximum and use that ln(9)/(xmax - xhalf) 
 * gives k where xhalf is the position of 1/2 height. If we can't find xhalf 
 * then we fall back on a guess of 0.1.
 *
 * @todo Should allow this fallback value to be an input parameter.
 *
 * @param xmax  Where the maximum point is.
 * @param C0    Estimate for the constant offset.
 * @param trace The trace data.
 *
 * @return The risetime estimate.
 *
 * @note For now assume that the half position is within the AOI and are not 
 * correcting for flattops.
 */
double
estimateK1(int xmax, double C0, const std::vector<std::uint16_t>& trace)
{
    double max = trace[xmax] - C0; // Background subtracted max.
    
    // Hunt for 1/2 max position:    
    for (int i = xmax; i >= 0; i--) {
	double v = trace[i] - C0;
	if (v < 0.5*max) {
	    return ln9/(xmax - i);
	}
    }
    
    return DDAS_FALLBACK_K1; // Fall back value.
}

/**
 * @details
 * Implements GSL's Levenburg-Marquardt fitter to determine best-fit parameters
 * by chi-square minimization.
 */
void
DDAS::AnalyticFit::lmfit1(
    fit1Info* pResult, std::vector<std::uint16_t>& trace,
    const std::pair<unsigned, unsigned>& limits, std::uint16_t saturation
    )
{
    unsigned low  = limits.first;
    unsigned high = limits.second;
    
    // Produce the set of x/y points that are to be fit.  This is the trace
    // within the limits and with points at or above saturation removed:    
    std::vector<std::pair<std::uint16_t, std::uint16_t>> points;
    reduceTrace(points, low, high, trace, saturation);
#ifdef CUDA
    CudaFitEngine1 engine(points);
#else
    SerialFitEngine1 engine(points);
#endif    
    unsigned npts = points.size();
    
    const gsl_multifit_fdfsolver_type* method = gsl_multifit_fdfsolver_lmsder;
    gsl_multifit_fdfsolver* solver;
    gsl_multifit_function_fdf function;
    gsl_vector* initialGuess;
    
    // Make the solver workspace:    
    solver = gsl_multifit_fdfsolver_alloc(method, npts, P1_PARAM_COUNT);
    if (solver == nullptr) {
	throw std::runtime_error("lmfit1 Unable to allocate fit solver workspace");
    }
    
    // Fill in function/data pointers:    
    function.f   = gsl_p1Residuals;
    function.df  = gsl_p1Jacobian;
    function.fdf = gsl_p1Compute;
    function.n   = npts;
    function.p   = P1_PARAM_COUNT;
    
    function.params = &engine;
    
    // Make the initial parameter guesses.
    
    // A0/X0 wil be determined by the maximum point on the trace.
    // Note that the guesses don't correct for flattops.
    // Hopefully the fits themselves will iron that all out.  
    initialGuess = gsl_vector_alloc(P1_PARAM_COUNT);
    
    double   max = 0;
    unsigned maxchan = 0;
    for (unsigned i = low; i <= high; i++) {
	if (trace[i] > max) {
	    max = trace[i];
	    maxchan = i;
	}
    }   
    
    double A0  = max;
    double C0  = fmin(trace[low], trace[high]); 
    A0 -= C0;
    double K10 = estimateK1(maxchan, C0, trace);
    double K20 = estimateK2(maxchan, C0, trace);
    double X10 = static_cast<double>(maxchan);   
    
    gsl_vector_set(initialGuess, P1A_INDEX, A0);
    gsl_vector_set(initialGuess, P1K1_INDEX, K10);
    gsl_vector_set(initialGuess, P1K2_INDEX, K20);
    gsl_vector_set(initialGuess, P1X1_INDEX, X10);
    gsl_vector_set(initialGuess, P1C_INDEX, C0);
    
    // Final solver setup:    
    gsl_multifit_fdfsolver_set(solver, &function, initialGuess);
    
    // Iterate until there's either convergence or the iteration count is hit.  
    int status;
    unsigned iteration(0);
    do {
	iteration++;        
	status = gsl_multifit_fdfsolver_iterate(solver);
	if (status != 0) break;        
	status = gsl_multifit_test_delta(
	    solver->dx, solver->x, 1.0e-4, 1.0e-4
	    );
    } while((status == GSL_CONTINUE) && (iteration < SINGLE_MAXITERATIONS));
    
    // Fish the values out of the solvers.    
    double A  = gsl_vector_get(solver->x, P1A_INDEX);
    double K1 = gsl_vector_get(solver->x, P1K1_INDEX);
    double K2 = gsl_vector_get(solver->x, P1K2_INDEX);
    double X1 = gsl_vector_get(solver->x, P1X1_INDEX);
    double C  = gsl_vector_get(solver->x, P1C_INDEX);
    
    double ChiSquare = chiSquare1(A, K1 , K2, X1, C, points);
    
    // Set the result struct from the fit parameters and the chi square.    
    pResult->iterations = iteration;
    pResult->fitStatus     = status;
    pResult->chiSquare = ChiSquare;
    pResult->pulse.position = X1;
    pResult->pulse.amplitude = A;
    pResult->pulse.steepness = K1;
    pResult->pulse.decayTime = K2;
    pResult->offset          = C;
    
    gsl_multifit_fdfsolver_free(solver);
    gsl_vector_free(initialGuess);
}

/*
 * For the double pulse case we have a constant and a parameters for two
 * independently variable pulses. The first pulse is characterized by A1, k2,
 * k2, x1, and the second by A2, k3, k4, x2. The indices of these in the GSL
 * vector/matrix cols are defined at the top of this file.
 */

/*
 * @brief Compute the residuals for the double pulse case. Note that we use
 * weight of 1.0 for each point.
 *
 * @param[in]  p     The fit parameters (see indices above).
 * @param[in]  pData Actually a pointer to a GslFitParameters object.
 * @param[out] r     Pointer to the vector to contain the residuals.
 * 
 * @return GSL_SUCCESS can't really fail.
 */
static int
gsl_p2Residuals(const gsl_vector* p, void* pData, gsl_vector* r)
{
    CFitEngine* pEngine = reinterpret_cast<CFitEngine*>(pData);
    pEngine->residuals(p, r);
  
    return GSL_SUCCESS;
}
/**
 * @brief Compute the jacobian matrix. 
 *
 * @details 
 * Rows for each data point, columns for each partial derivative of the 
 * function evaluated at that point. Note that all the terms look exactly 
 * like those for the single pulse case, there are just almost twice as many.
 *
 * @param[in]  p     gsl_vector of current fit parameters.
 * @param[in]  pData Actully a pointer to a GslFitParameterse struct.
 * @param[out] j     gsl_matrix into which the jacobian will be computed.
 *
 * @return GSL_SUCCESS, I don't see how this can fail.
 */
static int
gsl_p2Jacobian(const gsl_vector* p, void* pData, gsl_matrix* j)
{
    // Note: weights are all set to 1.0
    CFitEngine* pEngine = reinterpret_cast<CFitEngine*>(pData);
    pEngine->jacobian(p, j);
  
    return GSL_SUCCESS;
}

/**
 * @brief Compute function for double pulse fits which evaluates the Jacobian
 * matrix and calculates the fit residuals.
 *
 * @param[in]  p      The Current fit parameterization.
 * @param[in]  pData  The fit engine which performs the calculation.
 * @param[out] resids Will have residuals stored here.
 * @param[out] J      Will have the Jacobian elements stored here.
 *
 * @result GSL_SUCCESS, can only be that.
 */
static int
gsl_p2Compute(const gsl_vector* p, void* pData, gsl_vector* resids, gsl_matrix* J)
{
    CFitEngine* pEngine = reinterpret_cast<CFitEngine*>(pData);
    pEngine->residuals(p, resids);
    pEngine->jacobian(p, J);
  
    return GSL_SUCCESS;
}

/**
 * @details
 * Implements GSL's Levenburg-Marquardt fitter to determine best-fit parameters
 * by chi-square minimization.
 */
void
DDAS::AnalyticFit::lmfit2(
    fit2Info* pResult, std::vector<std::uint16_t>& trace,
    const std::pair<unsigned, unsigned>& limits,
    fit1Info* pSinglePulseFit, std::uint16_t saturation
    )
{
    unsigned low = limits.first;
    unsigned high = limits.second;    
    
    // Now produce a set of x/y points to be fit from the trace,
    // limits and saturation value:    
    std::vector<std::pair<std::uint16_t, std::uint16_t> > points;
    reduceTrace(points, low, high, trace, saturation);
    int npts = points.size(); // Number of points to fit.  
#ifdef CUDA
    CudaFitEngine2 engine(points);
#else
    SerialFitEngine2 engine(points);
#endif
  
    // Set up basic solver stuff:    
    const gsl_multifit_fdfsolver_type* method = gsl_multifit_fdfsolver_lmsder;
    gsl_multifit_fdfsolver* solver;
    gsl_multifit_function_fdf function;
    gsl_vector* initialGuess;
    
    // Make the solver workspace:    
    solver = gsl_multifit_fdfsolver_alloc(method, npts, P2_PARAM_COUNT);
    if (solver == nullptr) {
	throw std::runtime_error("lmfit2 Unable to allocate fit solver workspace");
    }    
    
    // Fill in the function data pointers:    
    function.f   = gsl_p2Residuals;
    function.df  = gsl_p2Jacobian;
    function.fdf = gsl_p2Compute;
    function.n   = npts;
    function.p   = P2_PARAM_COUNT;
    
    function.params = &engine;    
    
    // For LM, initial parameter guesses are not that sensitive (I think).    
    initialGuess = gsl_vector_alloc(P2_PARAM_COUNT);
    
    // Use Fit with one pulse to get initial guesses:
    // Since often double pulse fits are done after a single pulse fit the user
    // _may_ provide the results of that fit... which still may be nonsense
    // and require a backwards fit    
    fit1Info fit1;
    if (!pSinglePulseFit) {
	lmfit1(&fit1, trace, limits);
    } else {
	fit1 = *pSinglePulseFit; // Use what's passed in if possbible.
    }  
    
    double C0 = fit1.offset;
    double A0 = fit1.pulse.amplitude;
    double K10 = fit1.pulse.steepness;
    double K20 = fit1.pulse.decayTime;
    double X10 = fit1.pulse.position;
    double A10;
    double K30;
    double K40;
    double X20;
  
    if ((A0 < 0) || (K10 < 0) || (K20 < 0) || (X10 < 0)) {
	std::vector<std::uint16_t> reversed = trace;
	std::reverse(reversed.begin(), reversed.end());
	std::pair<unsigned, unsigned> revLimits;
	revLimits.second = trace.size() - limits.first;
	revLimits.first  = revLimits.second - (limits.second - limits.first);
	lmfit1(&fit1, reversed, revLimits, saturation);
       
	C0  = fit1.offset;
	A0  = fit1.pulse.amplitude;
	K10 = -fit1.pulse.steepness;
	K20 = -fit1.pulse.decayTime;
	X10 = npts - fit1.pulse.position;
    }
  
    // For estimates of the second trace, use the maximum value and position
    // of a trace with the single pulse subtracted. for the amplitude and
    // position estimates. Since signal rise times are determined by the
    // detector/electronics. Use the estimates gotten from the single pulse
    // fit. Note that for our random test cases that may fall over for some
    // cases where the pulses have radically differing rise/fall times.

    // Note, we only consider the region of interest of the trace fitted.
    
    // Note as well that we don't correct for flattops on these estamtors
    // The fit _should_ converge things to the right answers.
    
    int maxchannel = 0;
    double maxvalue = -1.0e6;
    for (unsigned i = limits.first; i <= limits.second; i++) {      
	double single = singlePulse(A0, K10, K20, X10, C0, (double)i);
	double diff = trace[i] - single;
	if (diff > maxvalue) {
	    maxvalue = diff;
	    maxchannel = i;
	}
    }
        
    A10 = maxvalue;
    K30 = K10;
    K40 = K20;
    X20 = maxchannel;        

    gsl_vector_set(initialGuess, P2A1_INDEX, A0);    // Left pulse
    gsl_vector_set(initialGuess, P2K1_INDEX, K10);
    gsl_vector_set(initialGuess, P2K2_INDEX, K20);
    gsl_vector_set(initialGuess, P2X1_INDEX, X10);
  
    gsl_vector_set(initialGuess, P2A2_INDEX, A10);   // Right pulse.
    gsl_vector_set(initialGuess, P2K3_INDEX, K30);
    gsl_vector_set(initialGuess, P2K4_INDEX, K40);
    gsl_vector_set(initialGuess, P2X2_INDEX, X20);
    
    gsl_vector_set(initialGuess, P2C_INDEX, C0);  

    // Final fitter/solvers setup:    
    gsl_multifit_fdfsolver_set(solver, &function, initialGuess);
        
    int status;
    unsigned iteration(0);
    do {
	iteration++;        
	status = gsl_multifit_fdfsolver_iterate(solver);
	if (status != 0) break;        
	status = gsl_multifit_test_delta(
	    solver->dx, solver->x, 1.0e-4, 1.0e-4
	    );        
    } while((status == GSL_CONTINUE) && (iteration < DOUBLE_MAXITERATIONS));

    // Fish our results and compute the chi square:
    
    double A1 = gsl_vector_get(solver->x, P2A1_INDEX);  // Pulse 1 
    double K1 = gsl_vector_get(solver->x, P2K1_INDEX);
    double K2 = gsl_vector_get(solver->x, P2K2_INDEX);
    double X1 = gsl_vector_get(solver->x, P2X1_INDEX);
    
    double A2 = gsl_vector_get(solver->x, P2A2_INDEX);  // Pulse 2
    double K3 = gsl_vector_get(solver->x, P2K3_INDEX);
    double K4 = gsl_vector_get(solver->x, P2K4_INDEX);
    double X2 = gsl_vector_get(solver->x, P2X2_INDEX);
    
    double C  = gsl_vector_get(solver->x, P2C_INDEX);  
        
    double ChiSquare =
	chiSquare2(A1, K1, K2, X1, A2, K3, K4, X2, C, points);
    
    pResult->iterations = iteration;
    pResult->fitStatus  = status;
    pResult->chiSquare  = ChiSquare;
    pResult->offset     = C;
    
    pResult->pulses[0].position  = X1;   // Pulse 1.
    pResult->pulses[0].amplitude = A1;
    pResult->pulses[0].steepness = K1;
    pResult->pulses[0].decayTime = K2;
    
    pResult->pulses[1].position  = X2;   // Pulse 1.
    pResult->pulses[1].amplitude = A2;
    pResult->pulses[1].steepness = K3;
    pResult->pulses[1].decayTime = K4;    
    
    gsl_multifit_fdfsolver_free(solver);    
    gsl_vector_free(initialGuess);
}

/*---------------------------------------------------------------------------
 * Double pulse with the time constants between the pulses constrained to be the
 * same.
 */
 
/*
 * @brief Compute the residuals for the double pulse case. Note that we use
 * weight of 1.0 for each point.
 *
 * @param[in]  p     The fit parameters (see indices above).
 * @param[in]  pData Actuall a pointer to a GslFitParameters object.
 * @param[out] r     Pointer to the vector to contain the residuals.
 *
 * @return GSL_SUCCESS can't really fail.
 */
static int
gsl_p2ftResiduals(const gsl_vector* p, void* pData, gsl_vector* r)
{
    // Pull out the current fit parameterization    
    double A1    = gsl_vector_get(p, P2FTA1_INDEX);   // Pulse 1.
    double k1    = gsl_vector_get(p, P2FTK1_INDEX);
    double k2    = gsl_vector_get(p, P2FTK2_INDEX);
    double x1    = gsl_vector_get(p, P2FTX1_INDEX);
    
    
    double A2    = gsl_vector_get(p, P2FTA2_INDEX);   // Pulse 2.
    double k3    = gsl_vector_get(p, P2FTK1_INDEX);   // k3 = k1
    double k4    = gsl_vector_get(p, P2FTK2_INDEX);   // k4 = k2
    double x2    = gsl_vector_get(p, P2FTX2_INDEX);
    
    double C     = gsl_vector_get(p, P2FTC_INDEX);    // constant.
    
    // Recast pData as a reference to the trace:    
    DDAS::AnalyticFit::GslFitParameters* pParams
	= reinterpret_cast<DDAS::AnalyticFit::GslFitParameters*>(pData);
    const std::vector<
	std::pair<std::uint16_t, std::uint16_t>
	>& points(*pParams->s_pPoints);
    
    // Compute double pulse residuals for each point:    
    for (size_t i = 0; i < points.size(); i++) {
	double x = points[i].first;
	double y = points[i].second;
	double p = DDAS::AnalyticFit::doublePulse(A1, k1, k2, x1,
						  A2, k3, k4, x2,
						  C, x);
	gsl_vector_set(r, i, (p - y)); // divided by w = 1.0.
    }
    
    return GSL_SUCCESS;
}

/**
 * @brief Compute the Jacobian matrix.
 *
 * @details
 * Rows for each data point, columns for each partial derivative of the 
 * function evaluated at that point. Note that all the terms look exactly 
 * like those for the single pulse case, there are just almost twice as many.
 *
 * @param[in]  p     gsl_vector of current fit parameters.
 * @param[in]  pData Actually a pointer to a GslFitParameterse struct.
 * @param[out] j     gsl_matrix into which the Jacobian will be computed.
 *
 * @result GSL_SUCCESS I don't see how this can fail (flw).
 */
static int
gsl_p2ftJacobian(const gsl_vector* p, void* pData, gsl_matrix* j)
{
    // Note: all weights are set to 0.
  
    // Fish the current fit parameters from p:    
    double A1    = gsl_vector_get(p, P2FTA1_INDEX);   // Pulse 1.
    double k1    = gsl_vector_get(p, P2FTK1_INDEX);
    double k2    = gsl_vector_get(p, P2FTK2_INDEX);
    double x1    = gsl_vector_get(p, P2FTX1_INDEX);
    
    
    double A2    = gsl_vector_get(p, P2FTA2_INDEX);   // Pulse 2.
    double k3    = gsl_vector_get(p, P2FTK1_INDEX);   // k3 = k1
    double k4    = gsl_vector_get(p, P2FTK2_INDEX);   // k4 = k2
    double x2    = gsl_vector_get(p, P2FTX2_INDEX);
        
    // Recast pData as a reference to the trace vector:    
    DDAS::AnalyticFit::GslFitParameters* pParams
	= reinterpret_cast<DDAS::AnalyticFit::GslFitParameters*>(pData);
    const std::vector<
	std::pair<std::uint16_t, std::uint16_t>
	>& points(*pParams->s_pPoints);
   
    // Loop over the data points producing the Jacobian for each point.
    // Note we can re-use the partial derivative functions for the single pulse
    // case since the two pulses are independent and have identical functional
    // forms.  
    for (size_t i = 0; i < points.size(); i++) {
	double x = points[i].first;
        
	// Pulse 1 terms (A, k1, k2, x1):        
	// Compute some reuseable exponentials:        
	double erise1 = exp(-k1*(x - x1));
	double efall1 = exp(-k2*(x - x1));
        
	double erise2 = exp(-k3*(x - x2));
	double efall2 = exp(-k4*(x - x2));
        
	gsl_matrix_set(
	    j, i, P2FTA1_INDEX, dp1dA(k1, k2, x1, x, 1.0, erise1, efall1)
	    );
	gsl_matrix_set(
	    j, i, P2FTK1_INDEX,
	    dp1dk1(A1, k1, k2, x1, x, 1.0, erise1, efall1) +
	    dp1dk1(A2, k1, k2, x2, x, 1.0, erise2, efall2)
	    );
	gsl_matrix_set(
	    j, i, P2FTK2_INDEX,
	    dp1dk2(A1, k1, k2, x1, x, 1.0, erise1, efall1) +
	    dp1dk2(A2, k1, k2, x2, x, 1.0, erise2, efall2)
	    );
	gsl_matrix_set(
	    j, i, P2FTX1_INDEX, dp1dx1(A1, k1, k2, x1, x, 1.0, erise1, efall1)
	    );
        
	// For pulse 2 elements:  A1->A2,  x1 -> x2        
	gsl_matrix_set(
	    j, i, P2FTA2_INDEX, dp1dA(k3, k4, x2, x, 1.0, erise2, efall2)
	    );
	gsl_matrix_set(
	    j, i, P2FTX2_INDEX, dp1dx1(A2, k3, k4, x2, x, 1.0, erise2, efall2)
	    );
        
	// Don't forget the constant term        
	gsl_matrix_set(j, i, P2FTC_INDEX, dp1dC(A1, k1, k2, x1, x, 1.0));     
    }
    
    
    
    return GSL_SUCCESS;
}

/**
 * @brief Compute method for double pulse fits that calls the functions to both
 * evaluate the Jacobian and calculate the fit residuals.
 *
 * @param[in]  p      The current fit parameterization.
 * @param[in]  pData  std::vector<std::uint16_t>*
 * @param[out] resids Will have residuals stored here.
 * @param[out] J      Will have the Jacobian elements stored here.
 *
 * @result GSL_SUCCESS can only be that.
 *
 */
static int
gsl_p2ftCompute(const gsl_vector* p, void* pData,
		gsl_vector* resids, gsl_matrix* J)
{
    gsl_p2ftResiduals(p, pData, resids);
    gsl_p2ftJacobian(p, pData, J);
    return GSL_SUCCESS;
}

/**
 * @details
 * Implements GSL's Levenburg-Marquardt fitter to determine best-fit parameters
 * by chi-square minimization.
 */
void
DDAS::AnalyticFit::lmfit2fixedT(
    fit2Info* pResult, std::vector<std::uint16_t>& trace,
    const std::pair<unsigned, unsigned>& limits,
    fit1Info* pSinglePulseFit, std::uint16_t saturation
    )
{
    unsigned low = limits.first;
    unsigned high = limits.second;    
    
    // Now produce a set of x/y points to be fit from the trace,
    // limits and saturation value:    
    std::vector<std::pair<std::uint16_t, std::uint16_t> > points;
    reduceTrace(points, low, high, trace, saturation);
    int npts = points.size(); // Number of points to fit.
    
    // Set up basic solver stuff:    
    const gsl_multifit_fdfsolver_type* method = gsl_multifit_fdfsolver_lmsder;
    gsl_multifit_fdfsolver*     solver;
    gsl_multifit_function_fdf  function;
    gsl_vector*                initialGuess;
    
    // Make the solver workspace:    
    solver = gsl_multifit_fdfsolver_alloc(method, npts, P2FT_PARAM_COUNT);
    if (solver == nullptr) {
	throw std::runtime_error("lmfit2fixed Unable to allocate fit solver workspace");
    }    
    
    // Fill in the function data pointers:    
    function.f   = gsl_p2ftResiduals;
    function.df  = gsl_p2ftJacobian;
    function.fdf = gsl_p2ftCompute;
    function.n   = npts;
    function.p   = P2FT_PARAM_COUNT;
    
    DDAS::AnalyticFit::GslFitParameters params = {&points};
    
    function.params = &params;    
    
    // For LM, initial parameter guesses are not that sensitive (I think).    
    initialGuess = gsl_vector_alloc(P2FT_PARAM_COUNT);
    
    // Use Fit with one pulse to get initial guesses:
    // Since often double pulse fits are done after a single pulse fit the user
    // _may_ provide the results of that fit... which still may be nonsense
    // and require a backwards fit    
    fit1Info fit1;
    if (!pSinglePulseFit) {
	lmfit1(&fit1, trace, limits);
    } else {
	fit1 = *pSinglePulseFit; // Use what's passed in if  possbible.
    }    
    
    double C0 = fit1.offset;
    double A0 = fit1.pulse.amplitude;
    double K10 = fit1.pulse.steepness;
    double K20 = fit1.pulse.decayTime;
    double X10 = fit1.pulse.position;
    double A10;
    double X20;
    
    if ((A0 < 0) || (K10 < 0) || (K20 < 0) || (X10 < 0)) {       
	std::vector<std::uint16_t> reversed = trace;
	std::reverse(reversed.begin(), reversed.end());
	std::pair<unsigned, unsigned> revLimits;
	revLimits.second = trace.size() - limits.first;
	revLimits.first  = revLimits.second - (limits.second - limits.first);
	lmfit1(&fit1, reversed, revLimits, saturation);
       
	C0  = fit1.offset;
	A0  = fit1.pulse.amplitude;
	K10 = -fit1.pulse.steepness;
	K20 = -fit1.pulse.decayTime;
	X10 = npts - fit1.pulse.position;
       
       
    }
    
    // For estimates of the second trace, use the maximum value and position
    // of a trace with the single pulse subtracted. For the amplitude and
    // position estimates. Since signal rise times are determined by the
    // detector/electronics. Use the estimates gotten from the single pulse fit.
    
    // Note that for our random test cases that may fall over for some
    // cases where the pulses have radically differing rise/fall times.

    // Note, we only consider the region of interest of the trace fitted.
    
    // Note as well that we don't correct for flattops on these estamtors
    // The fit _should_ converge things to the right answers.    
    int maxchannel = 0;
    double maxvalue = -1.0e6;
    for (unsigned i = limits.first; i <= limits.second; i++)  {      
	double single = singlePulse(A0, K10, K20, X10, C0, (double)i);
	double diff = trace[i] - single;
	if (maxvalue < diff) {
	    maxvalue = diff;
	    maxchannel = i;
	}
    }
        
    A10 = maxvalue;
    X20 = maxchannel;        

    gsl_vector_set(initialGuess, P2FTA1_INDEX, A0);    // Left pulse
    gsl_vector_set(initialGuess, P2FTK1_INDEX, K10);
    gsl_vector_set(initialGuess, P2FTK2_INDEX, K20);
    gsl_vector_set(initialGuess, P2FTX1_INDEX, X10);
    
    gsl_vector_set(initialGuess, P2FTA2_INDEX, A10);   // Right pulse.
    gsl_vector_set(initialGuess, P2FTX2_INDEX, X20);
    
    gsl_vector_set(initialGuess, P2FTC_INDEX, C0);
    
    // Final fitter/solvers setup:    
    gsl_multifit_fdfsolver_set(solver, &function, initialGuess);

    // Iterate until converged or we hit the iteration limit:
    int status;
    unsigned iteration(0);
    do {
	iteration++;        
	status = gsl_multifit_fdfsolver_iterate(solver);
	if (status != 0) break;        
	status = gsl_multifit_test_delta(
	    solver->dx, solver->x, 1.0e-4, 1.0e-4
	    );        
    } while((status == GSL_CONTINUE) && (iteration < DOUBLE_MAXITERATIONS));

    // Fish our results and compute the chi square:    
    double A1 = gsl_vector_get(solver->x, P2FTA1_INDEX);  // Pulse 1 
    double K1 = gsl_vector_get(solver->x, P2FTK1_INDEX);
    double K2 = gsl_vector_get(solver->x, P2FTK2_INDEX);
    double X1 = gsl_vector_get(solver->x, P2FTX1_INDEX);
    
    double A2 = gsl_vector_get(solver->x, P2FTA2_INDEX);  // Pulse 2
    double K3 = gsl_vector_get(solver->x, P2FTK1_INDEX);
    double K4 = gsl_vector_get(solver->x, P2FTK2_INDEX);
    double X2 = gsl_vector_get(solver->x, P2FTX2_INDEX);
    
    double C  = gsl_vector_get(solver->x, P2FTC_INDEX);
    
    double ChiSquare =
	chiSquare2(A1, K1, K2, X1, A2, K3, K4, X2, C, points);
    
    pResult->iterations = iteration;
    pResult->fitStatus  = status;
    pResult->chiSquare  = ChiSquare;
    pResult->offset     = C;
    
    pResult->pulses[0].position  = X1;   // Pulse 1.
    pResult->pulses[0].amplitude = A1;
    pResult->pulses[0].steepness = K1;
    pResult->pulses[0].decayTime = K2;
    
    pResult->pulses[1].position  = X2;   // Pulse 1.
    pResult->pulses[1].amplitude = A2;
    pResult->pulses[1].steepness = K3;
    pResult->pulses[1].decayTime = K4;
    
    
    gsl_multifit_fdfsolver_free(solver);    
    gsl_vector_free(initialGuess);
}

/** @} */
