/** 
 * @file  Configuration.h
 * @brief Definition of the configuration manager class.
 */

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <string>
#include <map>
#include <vector>
#include <utility>

/**
 * @defgroup analytic libFitEditorAnalytic.so
 * @brief Plugin library for analytic fitting.
 */

/**
 * @defgroup template libFitEditorTemplate.so
 * @brief Plugin library for template fitting.
 */

/**
 * @ingroup analytic template
 * @{
 */

/**
 * @class Configuration
 * @brief Manage fit configuration information.
 *
 * @details
 * This class is a configuration manager for the DDASToys programs. It is 
 * responsible for opening and reading data from configuration files pointed 
 * to by environment variables and managing the configuration data. The class 
 * defines a map of channel information for fitting. The map index is a unique
 * channel identifier and the value is a pair whose first element is itself a 
 * pair consisting of the fit limits [low, high] and whose second value is the
 *  saturation level above which trace data will not be included when fit.
 */

class Configuration
{
public:
    /** @brief Constructor. */
    Configuration() {};
    /** @brief Destructor. */
    ~Configuration() {};

    // Public interface for this class
public:
    /**
     * @brief Read the configuration file. 
     * @throw std::invalid_argument  If there are errors processing the file,
     *   including an inability to open the file.
     */
    void readConfigFile();
    /**
     * @brief Read the formatted tempalate data from a file. 
     * @throw std::length_error If the number of template data points is 
     *   different than what the configuration file expects.
     * @throw std::invalid_arugment If the alignment point of the template is 
     *   not contained in the trace (e.g. align to sample 100 on a 100 sample 
     *   trace [0, 99]).
     * @throw std::invalid_argument If the template data file cannot be opened.
     */
    void readTemplateFile();
    /**
     * @brief Check the map and determine if the channel should be fit.
     * @param crate   The crate ID.
     * @param slot    The slot ID.
     * @param channel The channel ID.
     * @return True if this channel trace should be fit (its in the map), 
     *   false otherwise.
     */
    bool fitChannel(unsigned crate, unsigned slot, unsigned channel);

    // Helpers
public:
    /**
     * @brief Get the fit limits for a single crate/slot/channel combination.
     * @return First is pair of fit limits, second is saturation value.
     */
    std::pair<std::pair<unsigned, unsigned>, unsigned> getFitLimits(
	unsigned crate, unsigned slot, unsigned channel
	);    
    /** 
     * @brief Return the template data.
     * @return std::vector<double>  The template trace data. 
     */
    std::vector<double> getTemplate() { return m_template; };    
    /**
     * @brief Return the template alignment point.
     * @return unsigned  The template trace alignment point. 
     */
    unsigned getTemplateAlignPoint() { return m_alignPoint; };

    // Private methods
private:
    /**
     * @brief Read the name of a configuration file pointed to by an 
     * environment variable.
     * @param envname Environment variable that points to the file.
     * @throw std::invalid_argument If there's no translation.
     * @return Translation of envname.
     */
    std::string getFileNameFromEnv(const char* envname);
    /**
     * @brief Determines if a line is a comment or not.
     * @param line Line to check.
     * @return The trimmed line if the line is not a comment, otherwise an 
     *   empty string.
     */
    std::string isComment(std::string line);
    /**
     * @brief Get global channel index from crate/slot/channel information.
     * @param crate   The crate ID.
     * @param slot    The slot ID.
     * @param channel The channel ID.
     * @return The global channel index.
     */
    unsigned channelIndex(unsigned crate, unsigned slot, unsigned channel);

    // Private data
private:
    /** 
     * Map of channel information for fitting. The map index is a unique channel
     * identifier and the value is a pair whose first element is itself a pair 
     * consisting of the fit limits [low, high] and whose second value is the 
     * saturation level above which trace data will not be included when fit. 
     */
    std::map<unsigned, std::pair<std::pair<unsigned, unsigned>, unsigned> >
    m_fitChannels;
    std::vector<double> m_template; //!< Template trace data.
    unsigned m_alignPoint; //!< Sample no. align point for the template trace.
};

/** @} */

#endif
