/** 
 * @file   lmfit_template.h
 * @brief  Define the template fitting functions and data structures for GSL's 
 *         Levenburg-Marquardt fitter.
 * @note Fit functions are in the DDAS::TemplateFit namespace.
 */

#ifndef LMFIT_TEMPLATE_H
#define LMFIT_TEMPLATE_H

#include <vector>

#include "fit_extensions.h"

/** @namespace DDAS */
namespace DDAS {
  /** @namespace DDAS::TemplateFit */
  namespace TemplateFit {

    /** 
     * @struct GslFitParameters
     * @brief Data passed around the fitting subsystem to Jacobian and 
     * function evaluators 
     */
    struct GslFitParameters {
      const std::vector<std::pair<std::uint16_t, std::uint16_t> >* s_pPoints; //!< Trace data stored as an (x, y) pair.
      const std::vector<double>* s_pTraceTemplate; //!< Template trace data.
    };

    void lmfit1(
		fit1Info* pResult, std::vector<std::uint16_t>& trace,
		std::vector<double>& traceTemplate, unsigned alignPoint,
		const std::pair<unsigned, unsigned>& limits,
		std::uint16_t saturation = 0xffff
		);
    
    void lmfit2(
		fit2Info* pResult, std::vector<std::uint16_t>& trace,
		std::vector<double>& traceTemplate, unsigned alignPoint,
		const std::pair<unsigned, unsigned>& limits,
		fit1Info* pSinglePulseFit = nullptr,
		std::uint16_t saturation = 0xffff
		);
  }
};

#endif
