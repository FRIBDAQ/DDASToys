/** @file: FitManager.cpp
 *  @brief: Implement the fit manager class and handle calls to appropriate 
 *  fit functions.
 */

#include "FitManager.h"

#include <iostream>
#include <utility>
#include <cmath>

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>

#include <Configuration.h>
#include <DDASFitHit.h>
#include <functions_analytic.h>
#include <functions_template.h>
#include <fit_extensions.h>

//____________________________________________________________________________
/**
 * Constructor
 */
FitManager::FitManager() :
  m_pConfig(new Configuration), m_pWarning(new QLabel), m_method(ANALYTIC),
  m_config(false), m_templateConfig(false), m_warned(false)
{}

//____________________________________________________________________________
/**
 * Destructor
 */
FitManager::~FitManager()
{
  delete m_pConfig;
}

//____________________________________________________________________________
/**
 * configure
 *   Configure the fit settings based on a text string which should come 
 *   directly from the text of the fit method selection box. We *do* need 
 *   to know what those possible strings are. Will terminate the program 
 *   and issue an error message if the fit method is not identified.
 *
 * @param methdod - string describing the fit method
 */
void
FitManager::configure(std::string method)
{
  // Regardless we want the config file if we haven't already
  
  if (!m_config) {
    readConfigFile();
    m_config = true;
  }
  
  if (method == "Analytic") {
    setMethod(ANALYTIC);
  } else if (method == "Template") {
    setMethod(TEMPLATE);
    
    // Read the template file if we haven't already
    
    if (!m_templateConfig) {
      readTemplateFile();
      m_templateConfig = true;
    }
  } else {
    std::cerr << "ERROR: FitManager cannot configure trace viewer for unknown fit method " << method << std::endl;
    exit(EXIT_FAILURE);
  }
}

//____________________________________________________________________________
/**
 * readConfigFile
 *   Read the configuration file using the Configuration class. Will terminate 
 *   the program and issue an error message if the configuration file cannot 
 *   be read.
 */
void
FitManager::readConfigFile()
{
  try {
    m_pConfig->readConfigFile();
  }
  catch (std::exception& e) {
    std::cerr << "ERROR: failed to configure FitManager -- "
	      << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }  
}

//____________________________________________________________________________
/**
 * readConfigFile
 *   Read the template file using the Configuration class. Will terminate 
 *   the program and issue an error message if the configuration file cannot 
 *   be read.
 */
void
FitManager::readTemplateFile()
{
  try {
    m_pConfig->readTemplateFile();
  }
  catch (std::exception& e) {
    std::cerr << "ERROR: failed to configure FitManager -- "
	      << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

//____________________________________________________________________________
/**
 * getSinglePulseFit
 *   Create and return a vector of fit values for each trace sample in the 
 *   fit range.
 *
 * @param ext  - references the HitExtension containing the fit parameters
 *               for this hit
 * @param low  - low limit of the fit in samples
 * @param high - high limit of the fit in samples
 *
 * @return std::vector<double> - vector of fit values for range [low, high)
 */
std::vector<double>
FitManager::getSinglePulseFit(const DDAS::HitExtension& ext,
			      unsigned low, unsigned high)
{
  std::vector<double> fit;

  double A1 = ext.onePulseFit.pulse.amplitude;
  double x1 = ext.onePulseFit.pulse.position;
  double k1 = ext.onePulseFit.pulse.steepness;
  double k2 = ext.onePulseFit.pulse.decayTime;
  double C = ext.onePulseFit.offset;

  // Checking one parameter ought to be enough to determine if the expected
  // parameter set matches the fit method. For analytic fits the steepness
  // parameter is some number > 0 while for the template fits the steepness
  // is equal to 0 by definition. Note that this warning is issued for the
  // single pulse fits at the moment, and if we have some classifier-steered
  // fitting this may not be sufficient because events will only contain one
  // set of fit data corresponding to the idenfied pulse type.
  
  if (!m_warned && !validParamValue(k1)) {
    issueWarning();
  }
  
  for (unsigned i=low; i<high; i++) {
    fit.push_back(singlePulse(A1, k1, k2, x1, C, i));
  }

  return fit;
}

//____________________________________________________________________________
/**
 * getDoublePulseFit
 *   Create and return a vector of fit values for each trace sample in the 
 *   fit range.
 *
 * @param ext  - references the HitExtension containing the fit parameters 
 *               for this hit
 * @param low  - low limit of the fit in samples
 * @param high - high limit of the fit in samples
 *
 * @return std::vector<double> - vector of fit values for range [low, high)
 */
std::vector<double>
FitManager::getDoublePulseFit(const DDAS::HitExtension& ext, unsigned low, unsigned high)
{
  std::vector<double> fit;

  double A1 = ext.twoPulseFit.pulses[0].amplitude;
  double x1 = ext.twoPulseFit.pulses[0].position;
  double k1 = ext.twoPulseFit.pulses[0].steepness;
  double k2 = ext.twoPulseFit.pulses[0].decayTime;

  double A2 = ext.twoPulseFit.pulses[1].amplitude;
  double x2 = ext.twoPulseFit.pulses[1].position;
  double k3 = ext.twoPulseFit.pulses[1].steepness;
  double k4 = ext.twoPulseFit.pulses[1].decayTime;
  
  double C = ext.twoPulseFit.offset;
 
  for (unsigned i=low; i<high; i++) {
    fit.push_back(doublePulse(A1, k1, k2, x1, A2, k3, k4, x2, C, i));
  }

  return fit;
}

//____________________________________________________________________________
/**
 * getLowFitLimit
 *   Get the lower limit of the fit range mapped in the Configuration class 
 *   for this hit crate/slot/channel. Note that this limit is inclusive.
 *
 * @param hit - references the hit we are currently processing
 *
 * @return unsigned - the lower limit  of the fitting range
 */
unsigned
FitManager::getLowFitLimit(const DAQ::DDAS::DDASFitHit& hit)
{
  unsigned crate = hit.GetCrateID();
  unsigned slot = hit.GetSlotID();
  unsigned channel = hit.GetChannelID();
  auto limits = m_pConfig->getFitLimits(crate, slot, channel);

  return limits.first.first;
}

//____________________________________________________________________________
/**
 * getHighFitLimit
 *   Get the high limit of the fit range mapped in the Configuration class 
 *   for this hit crate/slot/channel. Note that this limit is exclusive.
 *
 * @param hit - references the hit we are currently processing
 *
 * @return unsigned - the high limit  of the fitting range
 */
unsigned
FitManager::getHighFitLimit(const DAQ::DDAS::DDASFitHit& hit)
{
  unsigned crate = hit.GetCrateID();
  unsigned slot = hit.GetSlotID();
  unsigned channel = hit.GetChannelID();
  auto limits = m_pConfig->getFitLimits(crate, slot, channel);

  return limits.first.second;
}

//
// Private methods
//

//____________________________________________________________________________
/**
 * singlePusle
 *   Call the appropriate single pulse fit function to determine the fit value 
 *   for the input fit parameters and data point.
 *
 * @param A1 - amplitude parameter
 * @param k1 - steepnesss parameter (0 for template fits)
 * @param k2 - decay parameter (0 for template fits)
 * @param x1 - position parameter locating the pulse along the trace
 * @param C  - constant offset (baseline)
 * @param x  - data point to determine the fit value
 *
 * @return double - fit value at x
 */
double
FitManager::singlePulse(double A1, double k1, double k2, double x1,
			double C, double x)
{
  switch (m_method) {
  case ANALYTIC:
    return DDAS::AnalyticFit::singlePulse(A1, k1, k2, x1, C, x);
  case TEMPLATE:
    {
      auto traceTemplate = m_pConfig->getTemplate();
      return DDAS::TemplateFit::singlePulse(A1, x1, C, x, traceTemplate);
    }
  default:
    
    // This really is an error, but we'll stuff the fit with zeroes
    
    return 0;
  }
}

//____________________________________________________________________________
/**
 * doublePusle
 *   Call the appropriate double pulse fit function to determine the fit value 
 *   for the input fit parameters and data point.
 *
 * @param A1 - pulse 1 amplitude parameter
 * @param k1 - pulse 1 steepnesss parameter (0 for template fits)
 * @param k2 - pulse 1 decay parameter (0 for template fits)
 * @param x1 - pulse 1 position parameter locating the pulse along the trace
 * @param A2 - pulse 2 amplitude parameter
 * @param k3 - pulse 2 steepnesss parameter (0 for template fits)
 * @param k4 - pulse 2 decay parameter (0 for template fits)
 * @param x2 - pulse 2 position parameter locating the pulse along the trace
 * @param C  - shared constant offset (baseline)
 * @param x  - data point to determine the fit value
 *
 * @return double - fit value at x
 */
double
FitManager::doublePulse(double A1, double k1, double k2, double x1,
			double A2, double k3, double k4, double x2,
			double C, double x)
{
  switch (m_method) {
  case ANALYTIC:
    return DDAS::AnalyticFit::doublePulse(A1, k1, k2, x1,
					  A2, k3, k4, x2,
					  C, x);
  case TEMPLATE:
    {
      auto traceTemplate = m_pConfig->getTemplate();
      return DDAS::TemplateFit::doublePulse(A1, x1, A2, x2,
					    C, x, traceTemplate);
    }
  default:
    
    // This really is an error, but we'll stuff the fit with zeroes
    
    return 0;
  }
}

//____________________________________________________________________________
/**
 * validParamValue
 *   Check if a parameter is valid or not. Useful to check steepness or decay 
 *   parameters which are non-zero in the analytic fit and zero in the template
 *   fit. If there is a mismatch between the selected method and the parameter 
 *   value, set a corresponding warning message.
 *
 * @param p - input parameter to check
 *
 * @return bool - true if the parameter value matches the allowed values 
 *                from the fit method, false otherwise
 */
bool
FitManager::validParamValue(double p)
{
  if (m_method == ANALYTIC && std::fpclassify(p) == FP_ZERO) {
    m_warnMessage = "*****WARNING***** Bad parameter value encountered: expected non-zero and read zero! Are you sure this data was fit using the analytic fit functions?";
    return false;
  } else if (m_method == TEMPLATE && std::fpclassify(p) != FP_ZERO) {
    m_warnMessage = "*****WARNING***** Bad parameter value encountered: expected zero and read a non-zero value! Are you sure this data was fit using the template fit functions?";
    return false;
  } else { 
    return true;
  }
}

//____________________________________________________________________________
/**
 * issueWarning
 *   Create a pop-up window showing a warning message. Issue only once.
 */
void
FitManager::issueWarning()
{
  m_pWarning->setWordWrap(true);
  m_pWarning->setMaximumSize(200, 200);
  m_pWarning->setText(QString::fromStdString(m_warnMessage));
  
  QWidget* w = new QWidget;
  QVBoxLayout* l = new QVBoxLayout;
  l->addWidget(m_pWarning);
  w->setLayout(l);
  w->show();
  
  m_warned = true;
}
