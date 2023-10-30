/** 
 * @file  lmfit_template.cpp
 * @brief Implementation of template fitting functions we use in GSL's LM 
 * fitter. 
 * @note Fit functions are in the DDAS::TemplateFit namespace
 */

#include "lmfit_template.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_multifit_nlinear.h> // Updated nonlinear solver

#include "functions_template.h"

static const int SINGLE_MAXITERS(50);  //!< Max iterations for single pulse fit.
static const int DOUBLE_MAXITERS(200); //!< Max iterations for double pulse fit.

// Single pulse fit parameter indices:
static const int P1A1_INDEX(0);
static const int P1X1_INDEX(1);
static const int P1C_INDEX(2);
static const int P1_PARAM_COUNT(3);

// Double pulse fit with all parameters free:
static const int P2A1_INDEX(0);
static const int P2X1_INDEX(1);
static const int P2A2_INDEX(2);
static const int P2X2_INDEX(3);
static const int P2C_INDEX(4);
static const int P2_PARAM_COUNT(5);

static const int BASELINE(8); //!< Samples for estimating the baseline (unused).

/*------------------------------------------------------------------
 * Utility functions.
 */

/**
 * @brief Given a trace and a saturation value, returns the vector of 
 * sample-no, sample-value pairs that are below a saturation value.
 *
 * @param[out] points      Reduced trace.
 * @param[in]  low, high   Limits of the trace to reduce.
 * @param[in]  trace       Raw trace.
 * @param[in]  saturation  Saturation level.
 */
static void
reduceTrace(
    std::vector<std::pair<uint16_t, uint16_t>>& points,
    unsigned low, unsigned high,
    const std::vector<uint16_t>& trace, uint16_t saturation
    )
{
    for (unsigned i = low; i <= high; i++) {
	if (trace[i] < saturation) {
	    points.push_back(
		std::pair<uint16_t, uint16_t>(i, trace[i])
		);
	}
    }
    
}

/**
 * @brief Compute the vector of residuals applied to the data points for the 
 * specified parameters.
 *
 * @param[in]  p     Current parameters of the fit.
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
    DDAS::TemplateFit::GslFitParameters* pParams
	= reinterpret_cast<DDAS::TemplateFit::GslFitParameters*>(pData);
    
    // Pull the fit parameterization from p:  
    double A1  = gsl_vector_get(p, P1A1_INDEX);
    double x1  = gsl_vector_get(p, P1X1_INDEX);
    double C   = gsl_vector_get(p, P1C_INDEX);
    
    // Convert the raw data into its proper form:  
    const std::vector<
	std::pair<uint16_t, uint16_t>
	>& points(*pParams->s_pPoints); // Data
    const std::vector<double>& trtmp(*pParams->s_pTraceTemplate); // Template
  
    // Now loop over all the data points, filling in r with the weighted
    // residuals (m - y) where m is the model data.
    for (size_t i = 0; i<points.size(); i++) {
	double x = points[i].first;
	double y = points[i].second;
	double m = DDAS::TemplateFit::singlePulse(A1, x1, C, x, trtmp);
	gsl_vector_set(r, i, (m - y)); // Weighted by 1.0
    }
    
    return GSL_SUCCESS;
}

/**
 * @details
 * GSL non-linear least-squares fitter with Levenburg-Marquardt method used 
 * to solve the "trust-region problem" of approximating the objective function
 * around its minimum. See 
 * https://www.gnu.org/software/gsl/doc/html/index.html.
 */
void
DDAS::TemplateFit::lmfit1(
    fit1Info* pResult, std::vector<uint16_t>& trace,
    std::vector<double>& traceTemplate,
    unsigned alignPoint,
    const std::pair<unsigned, unsigned>& limits,
    uint16_t saturation
    )
{
    unsigned low  = limits.first;
    unsigned high = limits.second;
    
    // Produce the set of x/y points that are to be fit.  This is the trace
    // within the limits and with points at or above saturation removed  
    std::vector<std::pair<uint16_t, uint16_t>> points;
    reduceTrace(points, low, high, trace, saturation);    
    unsigned npts = points.size(); // Number of points for the fit

    // Nonlinear least squares fitting in GSL 2.5 is done by approximating
    // the objective function g(x) by some low-order approximation in the
    // vicinity of some point x (a "trust region method"). Aspects of the
    // iteration and the methods for solving the trust region problem are 
    // provided in the gsl_multifit_nlinear_parameters struct. See
    // https://www.gnu.org/software/gsl/doc/html/index.html for details.  
    const gsl_multifit_nlinear_type* method = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_workspace*  solver;
    gsl_multifit_nlinear_fdf         function;
    gsl_multifit_nlinear_parameters  function_params
	= gsl_multifit_nlinear_default_parameters();
    function_params.trs = gsl_multifit_nlinear_trs_lm; // Set the algorithm
    gsl_vector*                      initialGuess;

    // Make the solver workspace  
    solver = gsl_multifit_nlinear_alloc(
	method, &function_params, npts, P1_PARAM_COUNT
	);
    if (!solver) {
	throw std::runtime_error(
	    "lmfit1 Unable to allocate fit solver workspace"
	    );
    }
   
    // Fill in function/data pointers:  
    function.f   = gsl_p1Residuals;
    function.df  = nullptr; // Finite difference method from gsl2.5
    function.n   = npts;
    function.p   = P1_PARAM_COUNT;
    DDAS::TemplateFit::GslFitParameters params;
    params.s_pPoints = &points;
    params.s_pTraceTemplate = &traceTemplate;
    function.params = &params;
    
    // Make the initial parameter guesses. A0/X0 wil be determined by the
    // maximum point on the trace. Note that the guesses don't correct for
    // flattops. Hopefully the fits themselves will iron that all out.  
    initialGuess = gsl_vector_alloc(P1_PARAM_COUNT);

    // Set up initial guesses based off the current trace:
    uint16_t max = 0;
    unsigned maxSample = 0;
    for (unsigned i = low; i <= high; i++) {
	if (trace[i] > max) {
	    max = trace[i];
	    maxSample = i;
	}
    }

    double C0 = std::min(trace[low], trace[high]);
    double A10 = max - C0;
    double X10 = maxSample - alignPoint;
  
    gsl_vector_set(initialGuess, P1A1_INDEX, A10);
    gsl_vector_set(initialGuess, P1X1_INDEX, X10);
    gsl_vector_set(initialGuess, P1C_INDEX, C0);
    
    // Initialize the solver using the workspace, function system and initial
    // guess. Can also use gsl_multifit_nlinear_winit if weights are necessary
    // but lets not go down that road at the moment.  
    gsl_multifit_nlinear_init(initialGuess, &function, solver);
    
    // Iterate until there's either convergence or the iteration count is hit.
    // The driver provides a wrapper that combines the iteration and convergence
    // testing into a single function call. Driver returns GSL_SUCCESS for good
    // convergence, GSL_EMAXITER for iteration limit, GSL_ENOPROG when no
    // acceptable step can be taken.
    //
    // GSL params are:
    // - xtol: Test for small step relative to current parameter vector,
    //         ~10^-d for accuracy to d decimal places in the result.
    // - gtol: Test for small gradient indicating a (local) minimum, GSL 
    //         recommended value, see manual.
    // - ftol: Test for residual vector
    // - info: Reason for convergence:
    //            1 - small step,
    //            2 - small gradient,
    //            3 - small residual.  
    int status, info;
    double xtol = 1.0e-8;
    double gtol = pow(GSL_DBL_EPSILON, 1.0/3.0);
    double ftol  = 1.0e-8;

    // Here's the driver for iterating and solving the system.
    // Iteration tracking callback parameters are nullptr.  
    status = gsl_multifit_nlinear_driver(
	SINGLE_MAXITERS, xtol, gtol, ftol, nullptr, nullptr, &info, solver
	);
    
    // Fish the values out of the solvers  
    double A1  = gsl_vector_get(solver->x, P1A1_INDEX);
    double X1  = gsl_vector_get(solver->x, P1X1_INDEX);
    double C   = gsl_vector_get(solver->x, P1C_INDEX);

    double ChiSquare = chiSquare1(A1, X1, C, points, traceTemplate);
    
    // Set the result struct from the fit parameters and the chi square  
    pResult->iterations      = gsl_multifit_nlinear_niter(solver);
    pResult->fitStatus       = status;
    pResult->chiSquare       = ChiSquare;
    pResult->offset          = C;
    pResult->pulse.amplitude = A1;
    pResult->pulse.position  = X1; // Offset from aligned position
    pResult->pulse.steepness = 0;
    pResult->pulse.decayTime = 0;
  
    gsl_multifit_nlinear_free(solver);
    gsl_vector_free(initialGuess);
}

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
gsl_p2Residuals(const gsl_vector* p, void* pData, gsl_vector* r)
{ 
    DDAS::TemplateFit::GslFitParameters* pParams
	= reinterpret_cast<DDAS::TemplateFit::GslFitParameters*>(pData); // Data
  
    // Pull the fit parameterization from p:  
    double A1  = gsl_vector_get(p, P2A1_INDEX); // Pulse 1
    double x1  = gsl_vector_get(p, P2X1_INDEX);  
    double A2  = gsl_vector_get(p, P2A2_INDEX); // Pulse 2
    double x2  = gsl_vector_get(p, P2X2_INDEX);  
    double C   = gsl_vector_get(p, P2C_INDEX);  // Constant baseline
    
    // Convert the raw data into its proper form:  
    const std::vector<std::pair<uint16_t, uint16_t>>&
	points(*pParams->s_pPoints); // Data
    const std::vector<double>& trtmp(*pParams->s_pTraceTemplate); // Template
  
    // Now loop over all the data points, filling in r with the weighted
    // residuals (m - y) where m is the model function.
    for (size_t i = 0; i < points.size(); i++) {
	double x = points[i].first;
	double y = points[i].second;
	double m = DDAS::TemplateFit::doublePulse(A1, x1, A2, x2, C, x, trtmp);
	gsl_vector_set(r, i, (m - y));
    }
    
    return GSL_SUCCESS; // Can't fail this function.
}

/**
 * @details
 * GSL non-linear least-squares fitter with Levenburg-Marquardt method used 
 * to solve the "trust-region problem" of approximating the objective function
 * around its minimum. See 
 * https://www.gnu.org/software/gsl/doc/html/index.html.
 */
void
DDAS::TemplateFit::lmfit2(
    fit2Info* pResult, std::vector<uint16_t>& trace,
    std::vector<double>& traceTemplate,
    unsigned alignPoint,
    const std::pair<unsigned, unsigned>& limits,
    fit1Info* pSinglePulseFit, uint16_t saturation
    )
{
    unsigned low = limits.first;
    unsigned high = limits.second;   
    
    // Now produce a set of x/y points to be fit from the trace,
    // limits and saturation value
    std::vector<std::pair<uint16_t, uint16_t>> points;
    reduceTrace(points, low, high, trace, saturation);
    int npts = points.size(); // Number of points to fit
    
    // Nonlinear least squares fitting in gsl 2.5 is done by approximating
    // the objective function g(x) by some low-order approximation in the
    // vicinity of some point x (a "trust region method"). Aspects of the
    // iteration and the methods for solving the trust region problem are 
    // provided in the gsl_multifit_nlinear_parameters struct. See
    // https://www.gnu.org/software/gsl/doc/html/index.html for details.  
    const gsl_multifit_nlinear_type* method = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_workspace*  solver;
    gsl_multifit_nlinear_fdf         function;
    gsl_multifit_nlinear_parameters  function_params
	= gsl_multifit_nlinear_default_parameters();
    function_params.trs = gsl_multifit_nlinear_trs_lm; // Set the algorithm
    gsl_vector*                      initialGuess;
    
    // Make the solver workspace:  
    solver = gsl_multifit_nlinear_alloc(
	method, &function_params, npts, P2_PARAM_COUNT
	);
    if (!solver) {
	throw std::runtime_error(
	    "lmfit2 Unable to allocate fit solver workspace"
	    );
    }
    
    // Fill in function/data pointers:  
    function.f   = gsl_p2Residuals;
    function.df  = nullptr; // Finite difference method from gsl2.5
    function.n   = npts;
    function.p   = P2_PARAM_COUNT;    
    DDAS::TemplateFit::GslFitParameters params = {&points};
    params.s_pTraceTemplate = &traceTemplate;
    function.params = &params;   

    initialGuess = gsl_vector_alloc(P2_PARAM_COUNT);
    
    // Use Fit with one pulse to get initial guesses:
    // Since often double pulse fits are done after a single pulse fit the user
    // _may_ provide the results of that fit... which still may be nonsense.  
    fit1Info fit1;
    if (!pSinglePulseFit) {
	lmfit1(&fit1, trace, traceTemplate, alignPoint, limits);    
    } else {
	fit1 = *pSinglePulseFit; // Use what's passed in if possbible.
    } 
  
    double C0 = fit1.offset;
    double A10 = fit1.pulse.amplitude;
    double X10 = fit1.pulse.position;

    // If the single pulse fit gives bad values, fall back on the estimation.  
    double Xtrace = X10 + alignPoint; // Position on the actual trace
    if ((A10 < 0) || (Xtrace < 0) || (Xtrace > trace.size())) {
	uint16_t max = 0;
	unsigned maxSample = 0;
	for (unsigned i = low; i <= high; i++) {
	    if (trace[i] > max) {
		max = trace[i];
		maxSample = i;
	    }
	}
	C0 = std::min(trace[low], trace[high]);
	A10 = max - C0;
	X10 = maxSample - alignPoint;    
    }
  
    // Try and estimate the second pulse parameters by subtracting the single
    // pulse. Here we fall back to estimating the amplitude and position from
    // the maximum value on the subtracted trace within the fitting limits and
    // hope that the fit can iron out the rest.
    double max = std::numeric_limits<double>::lowest();
    unsigned maxSample = 0;
    for (unsigned i = low; i <= high; i++) {
	double diff
	    = trace[i] - DDAS::TemplateFit::singlePulse(
		A10, X10, C0, i, traceTemplate
		);
	if(diff > max) {
	    max = diff;
	    maxSample = i;
	}
    }
    double A20 = max;
    double X20 = maxSample - alignPoint;
  
    gsl_vector_set(initialGuess, P2A1_INDEX, A10);
    gsl_vector_set(initialGuess, P2X1_INDEX, X10);
    gsl_vector_set(initialGuess, P2A2_INDEX, A20);
    gsl_vector_set(initialGuess, P2X2_INDEX, X20);
    gsl_vector_set(initialGuess, P2C_INDEX, C0);

    // Initialize the solver using the workspace, function system and initial
    // guess. Can also use gsl_multifit_nlinear_winit if weights are necessary
    // but lets not go down that road at the moment.  
    gsl_multifit_nlinear_init(initialGuess,&function,solver);    
    
    // Iterate until there's either convergence or the iteration count is hit.
    // The driver provides a wrapper that combines the iteration and convergence
    // testing into a single function call. Driver returns GSL_SUCCESS for good
    // convergence, GSL_EMAXITER for iteration limit, GSL_ENOPROG when no
    // acceptable step can be taken.
    //
    // GSL params are:
    // - xtol: Test for small step relative to current parameter vector,
    //         ~10^-d for accuracy to d decimal places in the result.
    // - gtol: Test for small gradient indicating a (local) minimum, GSL 
    //         recommended value, see manual.
    // - ftol: Test for residual vector
    // - info: Reason for convergence:
    //            1 - small step,
    //            2 - small gradient,
    //            3 - small residual.  
    int status = -1;
    double xtol = 1.0e-8; 
    double gtol = pow(GSL_DBL_EPSILON, 1.0/3.0);
    double ftol  = 1.0e-8; 
    int info = 0; 

    // Here's the driver for iterating and solving the system.
    // Iteration tracking callback parameters are NULL.  
    status = gsl_multifit_nlinear_driver(
	DOUBLE_MAXITERS, xtol, gtol, ftol, nullptr, nullptr, &info, solver
	);

    // Fish our results and compute the chi square
    double A1  = gsl_vector_get(solver->x, P2A1_INDEX);
    double X1  = gsl_vector_get(solver->x, P2X1_INDEX);
    double A2  = gsl_vector_get(solver->x, P2A2_INDEX);
    double X2  = gsl_vector_get(solver->x, P2X2_INDEX);
    double C   = gsl_vector_get(solver->x, P2C_INDEX);
    
    double ChiSquare = chiSquare2(A1, X1, A2, X2, C, points, traceTemplate);
  
    pResult->iterations = gsl_multifit_nlinear_niter(solver);
    pResult->fitStatus  = status;
    pResult->chiSquare  = ChiSquare;
    pResult->offset     = C; // Constant
    pResult->pulses[0].amplitude = A1;
    pResult->pulses[0].position  = X1;
    pResult->pulses[0].steepness = 0; // Unused
    pResult->pulses[0].decayTime = 0; // Unused
    pResult->pulses[1].amplitude = A2;
    pResult->pulses[1].position  = X2;
    pResult->pulses[1].steepness = 0; // Unused
    pResult->pulses[1].decayTime = 0; // Unused
    
    gsl_multifit_nlinear_free(solver);    
    gsl_vector_free(initialGuess);
}
