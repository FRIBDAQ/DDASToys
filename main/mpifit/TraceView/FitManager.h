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
 * @details
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
    ANALYTIC, //!< Analytic fit enum.
    TEMPLATE  //!< Template fit enum.
};

class FitManager
{
public:
    /** @brief Constructor. */
    FitManager();
    /** @brief Destructor. */
    ~FitManager();

    /**
     * @brief Configure the fit method settings.
     * @param method String describing the fit method.
     */
    void configure(std::string method);
    /** @brief Read the configuration file using the Configuration class. */
    void readConfigFile();
    /** @brief Read the template file using the Configuration class. */
    void readTemplateFile();
    /**
     * @brief Create and return a vector of fit values for each trace sample 
     * in the fit range.
     * @param ext  References the HitExtension containing the fit parameters
     *    for this hit.
     * @param low  Low limit of the fit in samples.
     * @param high High limit of the fit in samples.
     * @return Vector of fit values for range [low, high].
     */
    std::vector<double> getSinglePulseFit(
	const DDAS::HitExtension& ext, unsigned low, unsigned high
	);
    /**
     * @brief Create and return a vector of fit values for each trace sample 
     * in the fit range.
     * @param ext  References the HitExtension containing the fit parameters 
     *   for this hit.
     * @param low  Low limit of the fit in samples.
     * @param high High limit of the fit in samples.
     * @return Vector of fit values for range [low, high].
     */   
    std::vector<double> getDoublePulseFit(
	const DDAS::HitExtension& ext, unsigned low, unsigned high
	);
    /**
     * @brief Get the lower limit of the fit range.
     * @param hit References the hit we are currently processing.
     * @return The lower limit of the fitting range.
     */
    unsigned getLowFitLimit(const DAQ::DDAS::DDASFitHit& hit);
    /**
     * @brief Get the upper limit of the fit range.
     * @param hit References the hit we are currently processing.
     * @return The upper limit of the fitting range.
     */
    unsigned getHighFitLimit(const DAQ::DDAS::DDASFitHit& hit);

    /**
     * @brief Set the fitting method.
     * @param m fitMethod enum type.
     */
    void setMethod(fitMethod m) { m_method = m; };
    /**
     * @brief Get the fitting method.
     * @return fitMethod enum type.
     */
    enum fitMethod getMethod() { return m_method; };
  
private:
    /**
     * @brief Calculate the value of a single-pulse fit at a given data point.
     * @param A1 Amplitude parameter.
     * @param k1 Steepnesss parameter (0 for template fits).
     * @param k2 Decay parameter (0 for template fits).
     * @param x1 Position parameter locating the pulse along the trace.
     * @param C  Constant offset (baseline).
     * @param x  Data point to determine the fit value.
     * @return   Fit value at x.
     */
    double singlePulse(
	double A1, double k1, double k2, double x1, double C, double x
	);
    /**
     * @brief Calculate the value of a double-pulse fit at a given data point.
     * @param A1 Pulse 1 amplitude parameter.
     * @param k1 Pulse 1 steepnesss parameter (0 for template fits).
     * @param k2 Pulse 1 decay parameter (0 for template fits).
     * @param x1 Pulse 1 position parameter locating the pulse along the trace.
     * @param A2 Pulse 2 amplitude parameter.
     * @param k3 Pulse 2 steepnesss parameter (0 for template fits).
     * @param k4 Pulse 2 decay parameter (0 for template fits).
     * @param x2 Pulse 2 position parameter locating the pulse along the trace.
     * @param C  Shared constant offset (baseline).
     * @param x  Data point to determine the fit value.
     * @return   Fit value at x.
     */
    double doublePulse(
	double A1, double k1, double k2, double x1,
	double A2, double k3, double k4, double x2,
	double C, double x
	);
    /**
     * @brief Check if a parameter is valid and issue a warning if it is not a
     * valid value.
     * @param p Input parameter (steepness or decay) to check.
     * @return True if the parameter value matches the allowed values from the
     *   fit method, false otherwise.
     */
    bool checkParamValue(double param);
    /** 
     * @brief Issue a warning message in a popup window.
     * @param msg The warning message displayed in the popup window.
     */
    void issueWarning(std::string msg);
  
private:
    Configuration* m_pConfig; //<! Configuration file manager.
    fitMethod m_method;       //!< Fit method selection value.
    bool m_config;            //!< Flag for reading the fit config file.
    bool m_templateConfig;    //!< Flag for reading the template config file.
};

#endif

/** @} */
