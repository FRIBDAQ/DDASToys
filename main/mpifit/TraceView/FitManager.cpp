/** 
 * @file  FitManager.cpp
 * @brief Implement the fit manager class and handle calls to appropriate 
 * fit functions.
 */

/** @addtogroup traceview
 * @{
 */

#include "FitManager.h"

#include <iostream>
#include <utility>
#include <cmath>

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>

#include <Configuration.h>
#include <DDASFitHit.h>
#include <functions_analytic.h>
#include <functions_template.h>
#include <fit_extensions.h>

//____________________________________________________________________________
/**
 * @brief Constructor.
 */
FitManager::FitManager() :
  m_pConfig(new Configuration), m_pWarningMessage(nullptr),
  m_method(ANALYTIC), m_config(false), m_templateConfig(false)
{}

//____________________________________________________________________________
/**
 * @brief Destructor.
 */
FitManager::~FitManager()
{
  delete m_pConfig;
  delete m_pWarningMessage;
}

//____________________________________________________________________________
/**
 * @brief Configure the fit method settings.
 *
 * Configuration is performed based on a text string which comes directly
 * from the text of the fit method selection box. We *do* need to know what 
 * those possible strings are. Terminate the program and issue an error 
 * message if the fit method is not known.
 *
 * @param method  String describing the fit method.
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
 * @brief Read the configuration file using the Configuration class. 
 *
 * Will terminate the program and issue an error message if the configuration 
 * file cannot be read.
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
 * @brief Read the template file using the Configuration class. 
 *
 * Will terminate the program and issue an error message if the configuration
 * file cannot be read.
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
 * @brief Create and return a vector of fit values for each trace sample in the 
 * fit range.
 *
 * @param ext   References the HitExtension containing the fit parameters
 *              for this hit.
 * @param low   Low limit of the fit in samples.
 * @param high  High limit of the fit in samples.
 *
 * @return std::vector<double>  Vector of fit values for range [low, high].
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
  
  checkParamValue(k1);
  
  for (unsigned i=low; i<=high; i++) {
    fit.push_back(singlePulse(A1, k1, k2, x1, C, i));
  }

  return fit;
}

//____________________________________________________________________________
/**
 * @brief Create and return a vector of fit values for each trace sample in the 
 * fit range.
 *
 * @param ext   References the HitExtension containing the fit parameters 
 *              for this hit.
 * @param low   Low limit of the fit in samples.
 * @param high  High limit of the fit in samples.
 *
 * @return std::vector<double>  Vector of fit values for range [low, high].
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
 
  for (unsigned i=low; i<=high; i++) {
    fit.push_back(doublePulse(A1, k1, k2, x1, A2, k3, k4, x2, C, i));
  }

  return fit;
}

//____________________________________________________________________________
/**
 * @brief Get the lower limit of the fit range.
 *
 * The fitting range is mapped in the Configuration class for hits keyed by a
 * unique identifier derived from their crate/slot/channel. Note that the low
 * fitting limit is inclusive.
 *
 * @param hit  References the hit we are currently processing.
 *
 * @return unsigned  The lower limit of the fitting range.
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
 * @brief Get the upper limit of the fit range.
 *
 * The fitting range is mapped in the Configuration class for hits keyed by a
 * unique identifier derived from their crate/slot/channel. Note that the high
 * fitting limit is inclusive.
 *
 * @param hit  References the hit we are currently processing.
 *
 * @return unsigned  The upper limit of the fitting range.
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

//____________________________________________________________________________
/** 
 * @brief Close the popup warning window.
 * 
 * FitManager does not inherit from QObject and therefore possesses no 
 * closeEvent event handler of its own. This function can be called as part of 
 * an overridden closeEvent function in widget classes which implement the 
 * FitManager to ensure that its warning message windows close when application
 * exits.
 */
void
FitManager::closeWarnings()
{
  if (m_pWarningMessage && m_pWarningMessage->isVisible()) {
    m_pWarningMessage->close();
  }
}

//
// Private methods
//

//____________________________________________________________________________
/**
 * @brief Calculate the value of a single pulse fit at a given data point.
 *
 * Use the fit method combo box value to determine whether to calculate the 
 * fit result using the analytic or template fitting functions.
 *
 * @param A1  Amplitude parameter.
 * @param k1  Steepnesss parameter (0 for template fits).
 * @param k2  Decay parameter (0 for template fits).
 * @param x1  Position parameter locating the pulse along the trace.
 * @param C   Constant offset (baseline).
 * @param x   Data point to determine the fit value.
 *
 * @return double  Fit value at x.
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
    
    // This really is an error, but we'll stuff the fit with zeroes.
    
    return 0;
  }
}

//____________________________________________________________________________
/**
 * @brief Calculate the value of a double pulse fit at a given data point.
 *
 * Use the fit method combo box value to determine whether to calculate the 
 * fit result using the analytic or template fitting functions.
 *
 * @param A1  Pulse 1 amplitude parameter.
 * @param k1  Pulse 1 steepnesss parameter (0 for template fits).
 * @param k2  Pulse 1 decay parameter (0 for template fits).
 * @param x1  Pulse 1 position parameter locating the pulse along the trace.
 * @param A2  Pulse 2 amplitude parameter.
 * @param k3  Pulse 2 steepnesss parameter (0 for template fits).
 * @param k4  Pulse 2 decay parameter (0 for template fits).
 * @param x2  Pulse 2 position parameter locating the pulse along the trace.
 * @param C   Shared constant offset (baseline).
 * @param x   Data point to determine the fit value.
 *
 * @return double  Fit value at x.
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
    
    // This really is an error, but we'll stuff the fit with zeroes.
    
    return 0;
  }
}

//____________________________________________________________________________
/**
 * @brief Check if a parameter is valid and issue a warning if it is not a
 * valid value.
 *
 * Useful to check steepness or decay parameters which are non-zero in the 
 * analytic fit and zero in the template fit. If there is a mismatch between 
 * the selected method and the parameter value, issue a warning message.
 *
 * @param p  Input parameter (steepness or decay) to check.
 *
 * @return bool  True if the parameter value matches the allowed values 
 *               from the fit method, false otherwise.
 */
bool
FitManager::checkParamValue(double p)
{
  if (m_method == ANALYTIC && std::fpclassify(p) == FP_ZERO) {
    std::string msg = "Bad parameter value encountered: expected non-zero and read zero! Are you sure this data was fit using the analytic fit functions?";
    issueWarning(msg);
    return false;
  } else if (m_method == TEMPLATE && std::fpclassify(p) != FP_ZERO) {
    std::string msg = "Bad parameter value encountered: expected zero and read a non-zero value! Are you sure this data was fit using the template fit functions?";
    issueWarning(msg);
    return false;
  } else { 
    return true;
  }
}

//____________________________________________________________________________
/** 
 * @brief Issue a warning message in a popup window.
 *
 * @param msg  The warning message displayed in the popup window.
 */
void
FitManager::issueWarning(std::string msg)
{
  // Create the warning message the first time a warning is issued. Otherwise
  // just reset the label text.
  
  if (!m_pWarningMessage) {
    m_pWarningMessage = new QWidget;
    m_pWarningMessage->setWindowTitle("WARNING");
    m_pWarningMessage->setWindowFlags(Qt::WindowStaysOnTopHint);
    QVBoxLayout* layout = new QVBoxLayout;
    QLabel* label = new QLabel;
    label->setWordWrap(true);
    label->setMaximumSize(600, 200);

    // Since the FitManager does not inherit from QObject, but QLabel does,
    // we need to call the translator for the QLabel object.
    
    label->setText(QLabel::tr(msg.c_str()));
    
    layout->addWidget(label);
    m_pWarningMessage->setLayout(layout);
  } else {
    QLabel* label = m_pWarningMessage->findChild<QLabel*>();
    label->setText(QLabel::tr(msg.c_str()));
  }

  m_pWarningMessage->show();
}

/** @} */

