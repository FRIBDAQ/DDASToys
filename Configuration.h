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
 * @file  Configuration.h
 * @brief Definition of the configuration manager class.
 */

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <string>
#include <map>
#include <vector>
#include <utility>

/** @namespace ddastoys */
namespace ddastoys {

    /**
     * @defgroup analytic libFitEditorAnalytic.so
     * @brief Plugin library for analytic fitting.
     */

    /**
     * @defgroup template libFitEditorTemplate.so
     * @brief Plugin library for template fitting.
     */

    /**
     * @defgroup mlinference libFitEditorMLInfernence.so
     * @brief Plugin library for machine-learning inference fitting.
     */

    /**
     * @ingroup analytic template mlinference
     * @{
     */

    /**
     * @class Configuration
     * @brief Manage fit configuration information.
     * @details
     * This class is a configuration manager for the DDASToys programs. It is 
     * responsible for opening and reading data from configuration files 
     * pointed to by environment variables and managing the configuration data. 
     * The class defines a map of channel information for fitting. The map 
     * index is a unique channel identifier and the value is a class object 
     * containing the fit limits, ADC satutration value and any additional 
     * info required by the fitters (e.g., the location of a PyTorch model 
     * for ML inference).
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
	 * @throw std::invalid_argument If there are errors processing the 
	 *   file, including an inability to open the file.
	 */
	void readConfigFile();
	/**
	 * @brief Read the formatted tempalate data from a file. 
	 * @throw std::length_error If the number of template data points is 
	 *   different than what the configuration file expects.
	 * @throw std::invalid_arugment If the alignment point of the template 
	 * is not contained in the trace (e.g. align to sample 100 on a 100 
	 * sample trace [0, 99]).
	 * @throw std::invalid_argument If the template data file cannot be 
	 * opened.
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
	 * @brief Get the (inclusive) fit limits for a single 
	 * crate/slot/channel combination.
	 * @param crate   The crate ID.
	 * @param slot    The slot ID.
	 * @param channel The channel ID.
	 * @return Pair of [low, high] fit limits (inclusive).
	 */
	std::pair<unsigned, unsigned> getFitLimits(
	    unsigned crate, unsigned slot, unsigned channel
	    );
	/**
	 * @brief Get the ADC saturation value for a single crate/slot/channel 
	 * combination.
	 * @param crate   The crate ID.
	 * @param slot    The slot ID.
	 * @param channel The channel ID.
	 * @return The saturation value of the trace for this channel.
	 */
	unsigned getSaturationValue(
	    unsigned crate, unsigned slot, unsigned channel
	    );
	/**
	 * @brief Get the ML inference model path for a single 
	 * crate/slot/channel combination.
	 * @param crate   The crate ID.
	 * @param slot    The slot ID.
	 * @param channel The channel ID.
	 * @return Path to the ML inference model.
	 */
	std::string getModelPath(
	    unsigned crate, unsigned slot, unsigned channel
	    );
	/**
	 * @brief Get the list of unique model names specified in the 
	 * configuration file.
	 * @return Vector of unique model paths.
	 */
	std::vector<std::string> getModelList();
	/** 
	 * @brief Return the template data.
	 * @return The template trace data. 
	 */
	std::vector<double> getTemplate() { return m_template; };    
	/**
	 * @brief Return the template alignment point.
	 * @return The template trace alignment point. 
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
	 * @param crate The crate ID.
	 * @param slot The slot ID.
	 * @param channel The channel ID.
	 * @return The global channel index.
	 */
	unsigned channelIndex(unsigned crate, unsigned slot, unsigned channel);

	// Private data
    private:
	/**
	 * @struct ConfigInfo
	 * @brief Configuration information for the fit.
	 */
	struct ConfigInfo {
	    std::pair<unsigned, unsigned> s_limits;
	    unsigned s_saturation;
	    std::string s_modelPath;
	    /** 
	     * @todo (ASC 9/17/24): Rework the template fitting such that a 
	     * per-channel template is supported. Remove the need to read the 
	     * TEMPLATE_CONFIGFILE environment variable. 
	     */
	};
	std::map<unsigned, ConfigInfo> m_fitChannels; //!< Channel map for fits.
	std::vector<double> m_template; //!< Template trace data.
	unsigned m_alignPoint; //!< Alignment point for the template trace.
    };

/** @} */

}

#endif
