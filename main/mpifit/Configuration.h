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
 * @class Configuration
 * @brief Manage fit configuration information.
 *
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
  Configuration();
  ~Configuration();

  // Public interface for this class
public:
  void readConfigFile();
  void readTemplateFile();  
  bool fitChannel(unsigned crate, unsigned slot, unsigned channel);

  // Helpers
public:
  std::pair<std::pair<unsigned, unsigned>, unsigned> getFitLimits(unsigned crate, unsigned slot, unsigned channel);
  /** 
   * @brief Return the template data.
   * @return std::vector<double>  The template trace data. 
   */
  std::vector<double> getTemplate() {return m_template;};
  /**
   * @brief Return the template alignment point.
   * @return unsigned  The template trace alignment point. 
   */
  unsigned getTemplateAlignPoint() {return m_alignPoint;};

  // Private methods
private:
  std::string getFileNameFromEnv(const char* envname);
  std::string isComment(std::string line);
  unsigned channelIndex(unsigned crate, unsigned slot, unsigned channel);

  // Private data
private:
  /** Map of channel information for fitting. The map index is a unique channel
   *  identifier and the value is a pair whose first element is itself a pair 
   *  consisting of the fit limits [low, high] and whose second value is the 
   *  saturation level above which trace data will not be included when fit. */
  std::map <unsigned, std::pair<std::pair<unsigned, unsigned>, unsigned>> m_fitChannels;
  std::vector<double> m_template; //!< Template trace data.
  unsigned m_alignPoint; //!< Sample no. align point for the template trace.
};

#endif
