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
 * @file  lmfit_analytic.h
 * @brief Define the analytic fitting functions and data structures for GSL's 
 * Levenburg-Marquardt fitter.
 */

#ifndef LMFIT_ANALYTIC_H
#define LMFIT_ANALYTIC_H

#include <cstdint>
#include <vector>

#include "fit_extensions.h"

/** @namespace ddastoys */
namespace ddastoys {
    /** @namespace ddastoys::analyticfit */
    namespace analyticfit {

	/** 
	 *@ingroup analytic
	 * @{
	 */
	
	/** 
	 * @struct GslFitParameters
	 * @brief Data passed around the fitting subsystem to Jacobian and 
	 * function evaluators 
	 */
	struct GslFitParameters {
	    /** Data points. Pair is (x, y). */
	    const std::vector<std::pair<uint16_t, uint16_t>>* s_pPoints;
	};

	/** @} */

	/** 
	 *@ingroup analytic
	 * @{
	 */
	
	/**
	 * @brief Driver for the GSL LM fitter for single pulses.
	 * @param pResult Struct that will get the results of the fit.
	 * @param trace   Vector of trace points.
	 * @param limits  Limits of the trace over which to conduct the fit.
	 * @param saturation Value at which the ADC is saturated (points at 
	 *   or above this value are removed from the fit.)
	 */	
	void lmfit1(
	    fit1Info* pResult, std::vector<uint16_t>& trace,
	    const std::pair<unsigned, unsigned>& limits,
	    uint16_t saturation = 0xffff
	    );

	/**
	 * @brief Driver for the GSL LM fitter for double pulses.
	 * @param pResult Results will be stored here.
	 * @param trace   References the trace to fit.
	 * @param limits  The limits of the trace that can be fit.
	 * @param pSinglePulseFit The fit for a single pulse, used to seed 
	 *   initial guesses if present. Otherwise a single pulse fit is done 
	 *   for that.
	 * @param saturation  ADC saturation value. Points with values at or 
	 *   above this value are removed from the fit.
	 */
	void lmfit2(
	    fit2Info* pResult, std::vector<uint16_t>& trace,
	    const std::pair<unsigned, unsigned>& limits,
	    fit1Info* pSinglePulseFit = nullptr,
	    uint16_t saturation = 0xffff
	    );

	/**
	 * @brief Driver for the GSL LM fitter for double pulses, constraining
	 * the two timing parameters (rise time and fall time) to be the same 
	 * for both pulses.
	 * @param pResult Results will be stored here.
	 * @param trace   References the trace to fit.
	 * @param limits  The limits of the trace that can be fit.
	 * @param pSinglePulseFit The fit for a single pulse, used to seed 
	 *   initial uesses if present. Otherwise a single pulse fit is done 
	 *   for that.
	 * @param saturation ADC saturation value. Points with values at or 
	 *   above this value are removed from the fit.
	 */
	void lmfit2fixedT(
	    fit2Info* pResult, std::vector<uint16_t>& trace,
	    const std::pair<unsigned, unsigned>& limits,
	    fit1Info* pSinglePulseFit = nullptr,
	    uint16_t saturation = 0xffff
	    );

	/** @} */

    }
};

#endif
