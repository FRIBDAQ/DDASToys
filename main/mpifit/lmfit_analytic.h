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
 * @file   lmfit_analytic.h
 * @brief  Define the analytic fitting functions and data structures for GSL's 
 *         Levenburg-Marquardt fitter.
 */

/** @addtogroup AnalyticFit
 * @{
 */

#ifndef LMFIT_ANALYTIC_H
#define LMFIT_ANALYTIC_H

#include <vector>

#include "fit_extensions.h"

/** @namespace DDAS */
namespace DDAS {
    /** @namespace DDAS::AnalyticFit */
    namespace AnalyticFit {

	/** 
	 * @struct GslFitParameters
	 * @brief Data passed around the fitting subsystem to Jacobian and 
	 * function evaluators 
	 */
	struct GslFitParameters {
	    const std::vector<
		std::pair<std::uint16_t, std::uint16_t>
		>* s_pPoints; /*!< Data points. Pair is x, y. */
	};
	
	/**
	 * @brief Driver for the GSL LM fitter for single pulses.
	 *
	 * @param pResult  Struct that will get the results of the fit.
	 * @param trace    Vector of trace points.
	 * @param limits   Limits of the trace over which to conduct the fit.
	 * @param saturation Value at which the ADC is saturated (points at 
	 *   or above this value are removed from the fit.)
	 */	
	void lmfit1(
	    fit1Info* pResult, std::vector<std::uint16_t>& trace,
	    const std::pair<unsigned, unsigned>& limits,
	    std::uint16_t saturation = 0xffff
	    );

	/**
	 * @brief Driver for the GSL LM fitter for double pulses.
	 *
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
	    fit2Info* pResult, std::vector<std::uint16_t>& trace,
	    const std::pair<unsigned, unsigned>& limits,
	    fit1Info* pSinglePulseFit = nullptr,
	    std::uint16_t saturation = 0xffff
	    );

	/**
	 * @brief Driver for the GSL LM fitter for double pulses, constraining
	 * the two timing parameters (rise time and fall time) to be the same 
	 * for both pulses.
	 *
	 * @param pResult Results will be stored here.
	 * @param trace   References the trace to fit.
	 * @param limits  The limits of the trace that can be fit.
	 * @param pSinglePulseFit The fit for a single pulse, used to seed 
	 *   initial uesses if present. Otherwise a single pulse fit is done 
	 *   for that.
	 * @param saturation  ADC saturation value. Points with values at or 
	 *   above this value are removed from the fit.
	 */
	void lmfit2fixedT(
	    fit2Info* pResult, std::vector<std::uint16_t>& trace,
	    const std::pair<unsigned, unsigned>& limits,
	    fit1Info* pSinglePulseFit = nullptr,
	    std::uint16_t saturation = 0xffff
	    );
    }
};

#endif

/** @} */
