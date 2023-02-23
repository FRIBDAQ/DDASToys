/** 
 * @file  FitManager.h
 * @brief Defines a class for managing fits including calulating fit function
 * values from fit parameters and fit configuration settings using the 
 * Configuraton class.
 */

/** @addtogroup traceview
 * @{
 */

#ifndef FITMANAGER_H
#define FITMANAGER_H

#include <vector>
#include <cstdint>
#include <string>

class QWidget;

namespace DDAS {
  struct HitExtension;
}
namespace DAQ {
  namespace DDAS {
    class DDASFitHit;
  }
}

class Configuration;

/**
 * @class FitManager
 * @brief Provides an interface for calculating trace fits using HitExtension
 * fit parameters.
 *
 * Wrapper class to call fitting functions from either the analytic fit 
 * or the template fit using the same signature. This is achieved by 
 * encapsulating the template-specfic data (the template itself and its 
 * align point) in this class by accessing the Configuration settings and 
 * switching based on the enum type.
 */

// Overkill probably but:
//   1) I don't want to remember flag values
//   2) Adding more fitting methods by name may be nice
//   3) I felt like making an enum

/** @brief Fit method enum class */
enum fitMethod {
		ANALYTIC, //!< Analytic fit enum value
		TEMPLATE  //!< Template fit enum value
};

class FitManager
{
public:
  FitManager();
  ~FitManager();

  void configure(std::string method); 
  void readConfigFile();
  void readTemplateFile();
  std::vector<double> getSinglePulseFit(const DDAS::HitExtension& ext,
					unsigned low, unsigned high);
  std::vector<double> getDoublePulseFit(const DDAS::HitExtension& ext,
					unsigned low, unsigned high);
  unsigned getLowFitLimit(const DAQ::DDAS::DDASFitHit& hit);
  unsigned getHighFitLimit(const DAQ::DDAS::DDASFitHit& hit);
  void closeWarnings();

  /**
   * @brief Set the fitting method.
   * @param m  fitMethod enum type.
   */
  void setMethod(fitMethod m) {m_method = m;};
  /**
   * @brief Get the fitting method.
   * @return enum  fitMethod enum type.
   */
  enum fitMethod getMethod() {return m_method;};
  
private:
  double singlePulse(double A1, double k1, double k2, double x1,
		     double C, double x);
  double doublePulse(double A1, double k1, double k2, double x1,
		     double A2, double k3, double k4, double x2,
		     double C, double x);
  bool checkParamValue(double param);
  void issueWarning(std::string msg);
  
private:
  Configuration* m_pConfig;
  QWidget* m_pWarningMessage;
  fitMethod m_method;
  bool m_config;
  bool m_templateConfig;
};

#endif

/** @} */
