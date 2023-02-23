/** 
 * @file  functions_template.h
 * @brief Define functions used to fit DDAS pulses using a trace template.
 * @note All functions are in the DDAS::TemplateFit namespace
 */

/** @addtogroup TemplateFit
 * @{
 */

#ifndef FUNCTIONS_TEMPLATE_H
#define FUNCTIONS_TEMPLATE_H

#include <vector>
#include <cstdint>

/** @namespace DDAS */
namespace DDAS {
  /** @namespace DDAS::TemplateFit */
  namespace TemplateFit {
    
    double singlePulse(
		       double S1, double x1, double C, double x,
		       const std::vector<double>& trace_template
		       );

    double doublePulse(
		       double S1, double x1, double S2, double x2,
		       double C, double x,
		       const std::vector<double>& trace_template
		       );
  
    double chiSquare1(
          double S1, double x1, double C,
          const std::vector<std::pair<std::uint16_t, std::uint16_t> >& points,
          const std::vector<double>& trace_template
		      );

    double chiSquare2(
          double S1, double x1, double S2, double x2, double C,
          const std::vector<std::pair<std::uint16_t, std::uint16_t> >& points,
          const std::vector<double>& trace_template
		      );
  }
};
  
#endif

/** @} */
