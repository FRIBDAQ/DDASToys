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
 * @file  FitManager.cpp
 * @brief Implement the fit manager class and handle calls to appropriate 
 * fit functions.
 */

#include "FitManager.h"

#include <iostream>
#include <cmath>

#include <QMessageBox>

#include <Configuration.h>
#include <DDASFitHit.h>
#include <functions_analytic.h>
#include <functions_template.h>

using namespace ddastoys;

//____________________________________________________________________________
/**
 * @details
 * The FitManager owns a Configuration object used to read settings from 
 * configuration files.
 */
FitManager::FitManager() :
    m_pConfig(nullptr), m_method(ANALYTIC), m_config(false),
    m_templateConfig(false)
{
    m_pConfig = new Configuration;
}

//____________________________________________________________________________
/**
 * @details
 * Delete the Configuration object owned by this class.
 */
FitManager::~FitManager()
{
    delete m_pConfig;
}

//____________________________________________________________________________
/**
 * @details
 * Configuration is performed based on a text string which comes directly
 * from the text of the fit method selection box. We *do* need to know what 
 * those possible strings are. Terminate the program and issue an error 
 * message if the fit method is not known.
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
    } else if (method == "ML_Inference") {
	setMethod(ML_INFERENCE);	
    } else {
	std::cerr << "ERROR: FitManager cannot configure trace viewer"
		  << " for unknown fit method " << method
		  << std::endl;
	exit(EXIT_FAILURE);
    }
}

//____________________________________________________________________________
/**
 * @details
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
std::vector<double>
FitManager::getSinglePulseFit(
    const DDASFitHit& hit, unsigned low, unsigned high
    )
{
    auto ext = hit.getExtension();
    auto crate = hit.getCrateID();
    auto slot = hit.getSlotID();
    auto chan = hit.getChannelID();
    std::vector<double> fit;

    double A1 = ext.onePulseFit.pulse.amplitude;
    double x1 = ext.onePulseFit.pulse.position;
    double k1 = ext.onePulseFit.pulse.steepness;
    double k2 = ext.onePulseFit.pulse.decayTime;
    double C = ext.onePulseFit.offset;

    // Checking one parameter ought to be enough to determine if the expected
    // parameter set matches the fit method. For analytic fits the steepness
    // parameter is some number != 0 while for the template fits the steepness
    // is equal to 0 by definition. Skip the warning for the ML inference using
    // the analyitc fitting functions, as we know that the single-pulse fit
    // does not exist.

    if (m_method == ANALYTIC || m_method == TEMPLATE) {
	checkParamValue(k1);
    }
  
    for (unsigned i = low; i <= high; i++) {
	fit.push_back(singlePulse(crate, slot, chan, A1, k1, k2, x1, C, i));
    }

    return fit;
}

//____________________________________________________________________________
std::vector<double>
FitManager::getDoublePulseFit(const DDASFitHit& hit, unsigned low, unsigned high)
{
    auto ext = hit.getExtension();;
    auto crate = hit.getCrateID();
    auto slot = hit.getSlotID();
    auto chan = hit.getChannelID();
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
 
    for (unsigned i = low; i <= high; i++) {
	fit.push_back(doublePulse(crate, slot, chan, A1, k1, k2, x1, A2, k3, k4, x2, C, i));
    }

    return fit;
}

//____________________________________________________________________________
/**
 * @details
 * The fitting range is mapped in the Configuration class for hits keyed by a
 * unique identifier derived from their crate/slot/channel. Note that the low
 * fitting limit is inclusive.
 */
unsigned
FitManager::getLowFitLimit(const DDASFitHit& hit)
{
    unsigned crate   = hit.getCrateID();
    unsigned slot    = hit.getSlotID();
    unsigned channel = hit.getChannelID();
    auto limits      = m_pConfig->getFitLimits(crate, slot, channel);

    return limits.first;
}

//____________________________________________________________________________
/**
 * @details
 * The fitting range is mapped in the Configuration class for hits keyed by a
 * unique identifier derived from their crate/slot/channel. Note that the high
 * fitting limit is inclusive.
 */
unsigned
FitManager::getHighFitLimit(const DDASFitHit& hit)
{
    unsigned crate   = hit.getCrateID();
    unsigned slot    = hit.getSlotID();
    unsigned channel = hit.getChannelID();
    auto limits      = m_pConfig->getFitLimits(crate, slot, channel);

    return limits.second;
}

///
// Private methods
//

//____________________________________________________________________________
/**
 * @details
 * Use the fit method combo box value to determine whether to calculate the 
 * fit result using the analytic or template fitting functions.
 */
double
FitManager::singlePulse(
    unsigned crate, unsigned slot, unsigned chan,
    double A1, double k1, double k2, double x1, double C, double x
    )
{
    switch (m_method) {
    case ANALYTIC:
	return analyticfit::singlePulse(A1, k1, k2, x1, C, x);
    case TEMPLATE:
    {
	auto traceTemplate = m_pConfig->getTemplate(crate, slot, chan);
	return templatefit::singlePulse(A1, x1, C, x, traceTemplate);
    }
    case ML_INFERENCE:
	return analyticfit::singlePulse(A1, k1, k2, x1, C, x);
    default:
    
	// This really is an error, but we'll stuff the fit with zeroes.
    
	return 0;
    }
}

//____________________________________________________________________________
/**
 * @details
 * Use the fit method combo box value to determine whether to calculate the 
 * fit result using the analytic or template fitting functions.
 */
double
FitManager::doublePulse(
    unsigned crate, unsigned slot, unsigned chan,
    double A1, double k1, double k2, double x1,
    double A2, double k3, double k4, double x2,    
    double C, double x
    )
{
    switch (m_method) {
    case ANALYTIC:
	return analyticfit::doublePulse(A1, k1, k2, x1, A2, k3, k4, x2, C, x);
    case TEMPLATE:
    {
	auto traceTemplate = m_pConfig->getTemplate(crate, slot, chan);
	return templatefit::doublePulse(A1, x1, A2, x2, C, x, traceTemplate);
    }
    case ML_INFERENCE:
	return analyticfit::doublePulse(A1, k1, k2, x1, A2, k3, k4, x2, C, x);
    default:
	
	// This really is an error, but we'll stuff the fit with zeroes:
	
	return 0;
    }
}

//____________________________________________________________________________
/**
 * @details 
 * Useful to check steepness or decay parameters which are non-zero in the 
 * analytic fit and zero in the template fit. If there is a mismatch between 
 * the selected method and the parameter value, issue a warning message.
 */
bool
FitManager::checkParamValue(double p)
{
    if (m_method == ANALYTIC && std::fpclassify(p) == FP_ZERO) {
	std::string msg = "Bad parameter value encountered: expected non-zero "
	    "and read zero! Are you sure this data was fit using the analytic "
	    "fit functions?";
	issueWarning(msg);
	return false;
    } else if (m_method == TEMPLATE && std::fpclassify(p) != FP_ZERO) {
	std::string msg = "Bad parameter value encountered: expected zero "
	    "and read a non-zero value! Are you sure this data was fit using "
	    "the template fit functions?";
	issueWarning(msg);
	return false;
    } else { 
	return true;
    }
}

//____________________________________________________________________________
/** 
 * @details
 * The warning is issued as a modal dialog, blocking until the user closes it.
 */
void
FitManager::issueWarning(std::string msg)
{
    QMessageBox msgBox;
    msgBox.setText(QString::fromStdString(msg));
    msgBox.setIcon(QMessageBox::Warning);
    msgBox.exec();
}

