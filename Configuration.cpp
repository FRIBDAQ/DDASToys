/** 
 * @file  Configuration.cpp
 * @brief Configuration manager class implementation.
 */

#include "Configuration.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

///
// Local trim functions
//

/** @brief Trim from beginning. */
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

/** @brief Trim from end. */
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

/** @brief Trim from both ends. */
static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}

namespace ddastoys {

/**
 * @details
 * Lines in the configuration file can be empty or have as their first 
 * non-blank character "#" in which case they are ignored. All other lines 
 * specify channels that should be fit and contain six whitespace integers: 
 * crate slot channel low high saturation. The crate, slot, channel identify
 * a channel to fit while low, high are the limits of  the trace to fit (first 
 * sample index, last sample index), and the saturation defines a limit above 
 * which the trace datapoints will not be fit. Most commonly this saturation 
 * value is set to the saturation value of the ADC.
 */
    void
    Configuration::readConfigFile()
    {
	std::string filename = getFileNameFromEnv("FIT_CONFIGFILE");
  
	std::ifstream f(filename);    
	if (f.fail()) {
	    std::string msg("Unable to open the configuration file: ");
	    msg += filename;
	    throw std::invalid_argument(msg);
	}
    
	while (!f.eof()) {
	    std::string originalline("");    
	    std::getline(f, originalline, '\n');   
	    std::string line = isComment(originalline);
    
	    if (line != "") {
		unsigned crate, slot, channel, low, high, saturation;
		std::string modelPath;
		std::stringstream sline(line);
		sline >> crate >> slot >> channel >> low  >> high >> saturation
		      >> modelPath;
	    
		if (sline.fail()) {
		    std::string msg(
			"Error processing line in configuration file '"
			);
		    msg += originalline;
		    msg += "'";
		    throw std::invalid_argument(msg);
		}
	    
		// Compute the channel index and add the channel to the map:
		unsigned index = channelIndex(crate, slot, channel);
		std::pair<unsigned, unsigned> limits(low, high);
		ConfigInfo info = {limits, saturation, modelPath};	    
		m_fitChannels[index] = info;
	    }
	}
    }

/**
 * @details 
 * Lines in the configuration file can be empty or have as their first 
 * non-blank character "#" in which case they are ignored. The first line 
 * consists of two whitespace-separated unsigned integer values which define 
 * the template metadata: the alignment point and the number of points in 
 * the template trace. The remaining lines in the configuration file contain 
 * the floating point template trace data itself.
 */
    void
    Configuration::readTemplateFile()
    {
	std::string filename = getFileNameFromEnv("TEMPLATE_CONFIGFILE");

	std::ifstream f(filename);
	if (f.fail()) {
	    std::string errmsg("Unable to open the template file: ");
	    errmsg += filename;
	    throw std::invalid_argument(errmsg);
	}

	if (!m_template.empty()) m_template.clear();
  
	int nread = 0;
	unsigned npts;
	double val;
	while (!f.eof()) {
	    std::string originalline("");
	    std::getline(f, originalline, '\n');
	    std::string line = isComment(originalline);
	    if (line != "") {
		std::stringstream sline(line);
      
		if (nread == 0) {
		    sline >> m_alignPoint >> npts;
		} else {
		    sline >> val;
		    m_template.push_back(val);
		}
      
		if (sline.fail()) {
		    std::string errmsg("Error processing line in template file '");
		    errmsg += originalline;
		    errmsg += "'";
		    throw std::invalid_argument(errmsg);
		}
      
		nread++;
	    }    
	}

	// The template should know how long it is. If you read in more data
	// points throw an exception.  
	if (m_template.size() != npts) {
	    std::string errmsg("Template configfile thinks the trace is ");
	    errmsg += npts;
	    errmsg += " samples but read in ";
	    errmsg += m_template.size();
	    throw std::length_error(errmsg); // I guess this is the right one?
	}

	// Ensure the alignment point is contained in the trace. Note that because
	// m_alignPoint is an unsigned type it cannot be negative.  
	if (m_alignPoint >= m_template.size()) {
	    std::string errmsg("Invalid template alignment point ");
	    errmsg += m_alignPoint;
	    errmsg += " >= template size ";
	    errmsg += m_template.size();
	    throw std::invalid_argument(errmsg);
	}

	f.close();
    }

/**
 * @details
 * Its up to caller to ensure that there is a map entry for this channel. 
 * Caller should exit with failure message if we attempt to fit an unmapped 
 * channel.
 */
    bool
    Configuration::fitChannel(unsigned crate, unsigned slot, unsigned channel)
    {
	int index = channelIndex(crate, slot, channel);
	return (m_fitChannels.find(index) != m_fitChannels.end());
    }

    std::pair<unsigned, unsigned>
    Configuration::getFitLimits(unsigned crate, unsigned slot, unsigned channel)
    {
	int index = channelIndex(crate, slot, channel);
	return m_fitChannels[index].s_limits;
    }

    unsigned
    Configuration::getSaturationValue(
	unsigned crate, unsigned slot, unsigned channel
	)
    {
	int index = channelIndex(crate, slot, channel);
	return m_fitChannels[index].s_saturation;
    }

    std::string
    Configuration::getModelPath(unsigned crate, unsigned slot, unsigned channel)
    {
	int index = channelIndex(crate, slot, channel);
	return m_fitChannels[index].s_modelPath;
    }

/**
 * @details
 * As a consequence of the sort-and-erase idiom used to uniqueify the model 
 * list vector, the model names are sorted in the returned vector which may 
 * be different than how they appear in the configuration file.
 */
    std::vector<std::string>
    Configuration::getModelList()
    {
	std::vector<std::string> models;
	for (const auto& entry : m_fitChannels) {
	    models.push_back(entry.second.s_modelPath);
	}
	std::sort(models.begin(), models.end());
	models.erase(std::unique(models.begin(), models.end()), models.end());
    
	return models;
    }

///
// Private methods
//

    std::string
    Configuration::getFileNameFromEnv(const char* envname)
    {
	const char* pFilename = getenv(envname);
	if (!pFilename) {
	    std::string msg("No translation for environment variable: ");
	    msg += envname;
	    msg += " Point that to the proper configuration file and re-run";
	    throw std::invalid_argument(msg);
	}
    
	return std::string(pFilename);
    }

    std::string
    Configuration::isComment(std::string line)
    {
	trim(line); // Modifies it
	if (line[0] == '#') return std::string("");

	return line;
    }

    unsigned
    Configuration::channelIndex(unsigned crate, unsigned slot, unsigned channel)
    {
	return (crate << 8) | (slot << 4) | channel;
    }

}
