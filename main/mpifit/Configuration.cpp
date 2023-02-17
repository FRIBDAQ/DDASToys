/** 
 * @file  Configuration.cpp
 * @brief Configuration manager class implementation.
 */

#include "Configuration.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

//
// Local trim functions
//

// Trim from beginning
static inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// Trim from end
static inline std::string &rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

// Trim from both ends
static inline std::string &trim(std::string &s) {
  return ltrim(rtrim(s));
}

/**
 * Constructor.
 */
Configuration::Configuration()
{}

/**
 * Destructor.
 */
Configuration::~Configuration()
{}

/**
 * @brief Read the configuration file. 
 * 
 * Lines in the configuration file can be empty or have as their first 
 * non-blank character "#" in which case they are ignored. All other lines 
 * specify channels that should be fit and contain six whitespace integers: 
 * crate slot channel low high saturation. The crate, slot, channel identify
 * a channel to fit while low, high are the limits of  the trace to fit (first 
 * sample index, last sample index), and the saturation defines a limit above 
 * which the trace datapoints will not be fit. Most commonly this saturation 
 * value is set to the saturation value of the ADC.
 *
 * @throw std::invalid_argument  If there are errors processing the file,
 *                               including an inability to open the file.
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
      std::stringstream sline(line);
      sline >> crate >> slot >> channel >> low  >> high >> saturation;
	    
      if (sline.fail()) {
	std::string msg("Error processing line in configuration file '");
	msg += originalline;
	msg += "'";
	throw std::invalid_argument(msg);
      }
	    
      // Compute the channel index:
            
      unsigned index = channelIndex(crate, slot, channel);
      std::pair<unsigned, unsigned> limits(low, high);
      std::pair<std::pair<unsigned, unsigned>, unsigned>
	value(limits, saturation);     
      m_fitChannels[index] = value;
    }
  }
}

/**
 * @brief Read the formatted tempalate data from a file. 
 *
 * Lines in the configuration file can be empty or have as their first 
 * non-blank character "#" in which case they are ignored. The first line 
 * consists of two whitespace-separated unsigned integer values which define 
 * the template metadata: the alignment point and the number of points in 
 * the template trace. The remaining lines in the configuration file contain 
 * the floating point template trace data itself.
 *
 * @throw std::length_error  If the number of template data points is 
 *                           different than what the configuration file expects.
 * @throw std::invalid_arugment  If the alignment point of the template is 
 *                               not contained in the trace (e.g. align to 
 *                               sample 101 on a 100 sample trace).
 * @throw std::invalid_argument  If the template data file cannot be opened.
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
 * @brief Check the map and determine if the channel should be fit.
 *
 * @param crate   The crate ID.
 * @param slot    The slot ID.
 * @param channel The channel ID.
 *
 * @return bool  True if this channel trace should be fit (its in the map), 
 *               false otherwise.
 */
bool
Configuration::fitChannel(unsigned crate, unsigned slot, unsigned channel)
{
  int index = channelIndex(crate, slot, channel);
  return (m_fitChannels.find(index) != m_fitChannels.end());
}

// \TODO (ASC 2/6/23): Up to caller to ensure that there is a map entry for
// this channel. Should exit with failure message if we attempt to fit an
// unmapped channel.

/**
 * @brief Get the fit limits for a single crate/slot/channel combination.
 *
 * @return std::pair<std::pair<unsigned, unsigned>, unsigned>  First is pair of fit limits, second is saturation value.
 */
std::pair<std::pair<unsigned, unsigned>, unsigned>
Configuration::getFitLimits(unsigned crate, unsigned slot, unsigned channel)
{
  int index = channelIndex(crate, slot, channel);
  return m_fitChannels[index];
}

//
// Private methods
//

/**
 * Read the name of a configuration file pointed to by an environment variable.
 *
 * @param envname  Environment variable that points to the file.
 *
 * @return std::string  Translation of envname.
 *
 * @throw std::invalid_argument  If there's no translation.
 */
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

/**
 * Determines if a line is a comment or not
 *
 * @param line  Line to check.
 *
 * @return std::string The trimmed line if the line is not a comment, 
 *                     otherwise an empty string.
 */
std::string
Configuration::isComment(std::string line)
{
  trim(line); // Modifies it
  if (line[0] == '#') return std::string("");

  return line;
}

/**
 * channelIndex
 *   Get global channel index from crate/slot/channel information
 *
 * @param crate    The crate ID.
 * @param slot     The slot ID.
 * @param channel  The channel ID.
 *
 * @return int  The global channel index.
 */
unsigned
Configuration::channelIndex(unsigned crate, unsigned slot, unsigned channel)
{
  return (crate << 8) | (slot << 4) | channel;
}
