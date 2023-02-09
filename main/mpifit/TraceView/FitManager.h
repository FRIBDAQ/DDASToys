/** @file: FitManager.h
 *  @brief: Defines a class for managing fits including calulating fit function
 *  values from fit parameters and fit configuration settings using the 
 *  Configuraton class.
 */

#ifndef FITMANAGER_H
#define FITMANAGER_H

#include <vector>
#include <cstdint>
#include <string>

namespace DDAS {
  struct HitExtension;
}
namespace DAQ {
  namespace DDAS {
    class DDASFitHit;
  }
}

class QLabel;

class Configuration;

/**
 * @class FitManager
 *
 *   Wrapper class to call fitting functions from either the analytic fit 
 *   or the template fit using the same signature. This is achieved by 
 *   encapsulating the template-specfic data (the template itself and its 
 *   align point) in this class by accessing the Configuration settings and 
 *   switching based on the enum type.
 */

// Overkill probably but:
//   1) I don't want to remember flag values
//   2) Adding more fitting methods by name may be nice
//   3) I felt like making an enum
enum fitMethod {ANALYTIC, TEMPLATE};

class FitManager
{
public:
  FitManager();
  ~FitManager();

  void configure(std::string method); 
  void readConfigFile();
  void readTemplateFile();
  std::vector<double> getSinglePulseFit(const DDAS::HitExtension& ext, unsigned low, unsigned high);
  std::vector<double> getDoublePulseFit(const DDAS::HitExtension& ext, unsigned low, unsigned high);
  unsigned getLowFitLimit(const DAQ::DDAS::DDASFitHit& hit);
  unsigned getHighFitLimit(const DAQ::DDAS::DDASFitHit& hit);
  void setMethod(fitMethod m) {m_method = m;};
  enum fitMethod getMethod() {return m_method;};
  
private:
  double singlePulse(double A1, double k1, double k2, double x1,
		     double C, double x);
  double doublePulse(double A1, double k1, double k2, double x1,
		     double A2, double k3, double k4, double x2,
		     double C, double x);
  bool validParamValue(double param);
  void issueWarning();
  
private:
  Configuration* m_pConfig;
  QLabel* m_pWarning;
  fitMethod m_method;
  std::string m_warnMessage;
  bool m_config;
  bool m_templateConfig;
  bool m_warned;
};

#endif
