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
 * @file  lmfit_template.h
 * @brief Define the template fitting functions and data structures for GSL's 
 * Levenburg-Marquardt fitter.
 * @note Fit functions are in the DDAS::TemplateFit namespace.
 */

#ifndef LMFIT_TEMPLATE_H
#define LMFIT_TEMPLATE_H

#include <cstdint>
#include <vector>
#include <utility>

#include "fit_extensions.h"

/** @namespace ddastoys */
namespace ddastoys {
    /** @namespace ddastoys::templatefit */
    namespace templatefit {

	/**
	 * @ingroup template
	 * @{
	 */
	
	/** 
	 * @struct GslFitParameters
	 * @brief Data passed around the fitting subsystem to Jacobian and 
	 * function evaluators 
	 */
	struct GslFitParameters {
	    /** Trace data stored as an (x, y) pair. */
	    const std::vector<std::pair<uint16_t, uint16_t>>* s_pPoints;
	    const std::vector<double>* s_pTraceTemplate; //!< Template data.
	};

	/** @} */

	/**
	 * @ingroup template
	 * @{
	 */
	
	/**
	 * @brief Driver for the GSL LM fitter for single pulses.
	 * @param pResult       Struct that will get the results of the fit.
	 * @param trace         References the trace to fit.
	 * @param traceTemplate References the template trace for the fit. 
	 *   The template is assumed to have a baseline of 0 and a maximum 
	 *   value of 1.
	 * @param alignPoint The internal alignment point of the template.
	 * @param limits     Limits of the trace over which to conduct the fit.
	 * @param saturation Value at which the ADC is saturated (points at or 
	 *   above this value are removed from the fit.)
	 */
	void lmfit1(
	    fit1Info* pResult, std::vector<uint16_t>& trace,
	    std::vector<double>& traceTemplate, unsigned alignPoint,
	    const std::pair<unsigned, unsigned>& limits,
	    uint16_t saturation = 0xffff
	    );

	/**
	 * @brief Driver for the GSL LM fitter for double pulses.
	 * @param pResult       Struct that will get the results of the fit.
	 * @param trace         References the trace to fit.
	 * @param traceTemplate References the template trace for the fit. 
	 *   The template is assumed to have a baseline of 0 and a maximum 
	 *   value of 1.
	 * @param alignPoint The internal alignment point of the template.
	 * @param limits     The limits of the trace that can be fit.
	 * @param pSinglePulseFit Pointer to the fit for a single pulse, used 
	 *   to seed initial guesses if present. Otherwise a single pulse fit 
	 *   is done for that.
	 * @param saturation ADC saturation value. Points with values at 
	 *   or above this value are removed from the fit.
	 */
	void lmfit2(
	    fit2Info* pResult, std::vector<uint16_t>& trace,
	    std::vector<double>& traceTemplate, unsigned alignPoint,
	    const std::pair<unsigned, unsigned>& limits,
	    fit1Info* pSinglePulseFit = nullptr,
	    uint16_t saturation = 0xffff
	    );

	/** @} */
    }
};

#endif
