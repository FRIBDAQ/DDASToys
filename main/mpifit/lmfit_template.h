/** @file:  lmfit_template.h
 *  @brief: Define the fitting functions and data structures for L-M fits.
 */

#ifndef LMFIT_TEMPLATE_H
#define LMFIT_TEMPLATE_H

#include <vector>

#include "fit_extensions.h"

namespace DDAS {
  namespace TemplateFit {

    // This struct is passed around the fitting subsystem to Jacobian and
    // function evaluators
    struct GslFitParameters {
      const std::vector<std::pair<uint16_t, uint16_t> >* s_pPoints;
      const std::vector<double>* s_pTraceTemplate;
    };

    void lmfit1(
		fit1Info* pResult, std::vector<uint16_t>& trace,
		std::vector<double>& traceTemplate, unsigned alignPoint,
		const std::pair<unsigned, unsigned>& limits,
		uint16_t saturation = 0xffff
		);
    void lmfit2(
		fit2Info* pResult, std::vector<uint16_t>& trace,
		std::vector<double>& traceTemplate, unsigned alignPoint,
		const std::pair<unsigned, unsigned>& limits,
		fit1Info* pSinglePulseFit = nullptr,
		uint16_t saturation = 0xffff
		);

  }
};

#endif
