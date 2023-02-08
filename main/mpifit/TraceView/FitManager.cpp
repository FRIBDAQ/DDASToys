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

FitManager::FitManager() :
  m_pConfig(new Configuration),
  m_pWarning(new QLabel),
  m_method(ANALYTIC),
  m_config(false),
  m_templateConfig(false),
  m_warned(false)
{
  m_pWarning->setWordWrap(true);
  m_pWarning->setMaximumSize(200, 200);
}

FitManager::~FitManager()
{
  delete m_pConfig;
}

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

std::vector<double>
FitManager::getSinglePulseFit(DDAS::HitExtension& ext, unsigned low, unsigned high)
{
  std::vector<double> fit;

  double A1 = ext.onePulseFit.pulse.amplitude;
  double x1 = ext.onePulseFit.pulse.position;
  double k1 = ext.onePulseFit.pulse.steepness;
  double k2 = ext.onePulseFit.pulse.decayTime;
  double C = ext.onePulseFit.offset;

  // Checking one parameter ought to be enough to determine if the expected
  // parameter set matches the fit method
  if (!m_warned && !validParamValue(k1)) {
    issueWarning();
  }
  
  for (unsigned i=low; i<high; i++) {
    fit.push_back(singlePulse(A1, k1, k2, x1, C, i));
  }

  return fit;
}

std::vector<double>
FitManager::getDoublePulseFit(DDAS::HitExtension& ext, unsigned low, unsigned high)
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

unsigned
FitManager::getLowFitLimit(DAQ::DDAS::DDASFitHit& hit)
{
  unsigned crate = hit.GetCrateID();
  unsigned slot = hit.GetSlotID();
  unsigned channel = hit.GetChannelID();
  auto limits = m_pConfig->getFitLimits(crate, slot, channel);

  return limits.first.first;
}

unsigned
FitManager::getHighFitLimit(DAQ::DDAS::DDASFitHit& hit)
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
    // This really is an error
    return 0;
  }
}

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
					    C, x,
					    traceTemplate);
    }
  default:
    // This really is an error
    return 0;
  }
}

bool
FitManager::validParamValue(double param)
{
  if (m_method == ANALYTIC && std::fpclassify(param) == FP_ZERO) {
    m_warnMessage = "*****WARNING***** Bad parameter value encountered: expected non-zero and read zero! Are you sure this data was fit using the analytic fit functions?";
    return false;
  } else if (m_method == TEMPLATE && std::fpclassify(param) != FP_ZERO) {
    m_warnMessage = "*****WARNING***** Bad parameter value encountered: expected zero and read a non-zero value! Are you sure this data was fit using the template fit functions?";
    return false;
  } else { 
    return true;
  }
}

void
FitManager::issueWarning()
{
  QWidget* w = new QWidget;
  m_pWarning->setText(QString::fromStdString(m_warnMessage));
  QVBoxLayout* l = new QVBoxLayout;
  l->addWidget(m_pWarning);
  w->setLayout(l);
  w->show();
  m_warned = true;
}
