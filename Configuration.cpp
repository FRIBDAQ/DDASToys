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
 * @file  Configuration.cpp
 * @brief Configuration manager class implementation.
 */

#include "Configuration.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
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


/**
 * @details
 * Lines in the configuration file can be empty or have as their first 
 * non-blank character "#" in which case they are ignored. All other lines 
 * specify channels that should be fit and must contain six whitespace integers
 * and one string: crate slot channel low high saturation model. The crate, 
 * slot, and channel are unsigned integers used to identify a channel to fit. 
 * Low and high specify the inclusive limits of the trace to fit in samples. 
 * The saturation value defines a limit above which the trace data points will 
 * not be fit. Most commonly this saturation value is set to the saturation 
 * value of the ADC. The last parameter is a string specifying a path to a 
 * PyTorch model for the machine-learning inference fitting. Not all parameters
 * are used by each fitting method but default values must be provided.
 * Please refer to the following:
 *
 * @warning While all parameters must be specified in the configuration, 
 * depending on the fitting method, some of them may be ignored in parts of the
 * code. Specifically, the machine-learning inference fitting will ignore the 
 * fit limits, as it requires the input data to be the same shape as the 
 * training data which is assumed to be the full acquired trace. The traceview 
 * plotter will use these low and high limits to draw the fit so in practice it
 * is best to set them to sensible values. Non-ML based fitting methods will 
 * ignore the model parameter. An empty string ("") is a vaild input in the 
 * configuration file. 
*/
void
ddastoys::Configuration::readConfigFile()
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
ddastoys::Configuration::readTemplateFile()
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
    // points throw an exception:
    
    if (m_template.size() != npts) {
	std::string errmsg("Template configfile thinks the trace is ");
	errmsg += npts;
	errmsg += " samples but read in ";
	errmsg += m_template.size();
	throw std::length_error(errmsg); // I guess this is the right one?
    }

    // Ensure the alignment point is contained in the trace. Note that because
    // m_alignPoint is an unsigned type it cannot be negative:
    
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
ddastoys::Configuration::fitChannel(unsigned crate, unsigned slot, unsigned channel)
{
    int index = channelIndex(crate, slot, channel);
    
    return (m_fitChannels.find(index) != m_fitChannels.end());
}

std::pair<unsigned, unsigned>
ddastoys::Configuration::getFitLimits(
    unsigned crate, unsigned slot, unsigned channel
    )
{
    int index = channelIndex(crate, slot, channel);
    
    return m_fitChannels[index].s_limits;
}

unsigned
ddastoys::Configuration::getSaturationValue(
    unsigned crate, unsigned slot, unsigned channel
    )
{
    int index = channelIndex(crate, slot, channel);
    
    return m_fitChannels[index].s_saturation;
}

std::string
ddastoys::Configuration::getModelPath(
    unsigned crate, unsigned slot, unsigned channel
    )
{
    int index = channelIndex(crate, slot, channel);
    
    return m_fitChannels[index].s_modelPath;
}

/**
 * @details
 * As a consequence of the sort-and-erase idiom used to uniquify the model 
 * list, the model paths names are sorted in the returned vector and may be 
 * in a different order than how they appear in the configuration file.
 * It is the responsibilty of the caller to deal with this.
 */
std::vector<std::string>
ddastoys::Configuration::getModelList()
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
ddastoys::Configuration::getFileNameFromEnv(const char* envname)
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
ddastoys::Configuration::isComment(std::string line)
{
    trim(line); // Modifies it
    if (line[0] == '#') return std::string("");

    return line;
}

unsigned
ddastoys::Configuration::channelIndex(
    unsigned crate, unsigned slot, unsigned channel
    )
{
    return (crate << 8) | (slot << 4) | channel;
}
