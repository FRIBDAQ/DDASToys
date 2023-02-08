/** @file:  Configuration.h
 *  @brief: Definition of abstract base class for reading fit configuration 
 *          information from environment variables. Assumes configuration files
 *          use '#' to prepend comments.
 */

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <string>

#include <map>
#include <vector>
#include <utility>

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
  std::vector<double> getTemplate() {return m_template;};
  unsigned getTemplateAlignPoint() {return m_alignPoint;};

  // Private methods
private:
  std::string getFileNameFromEnv(const char* envname);
  std::string isComment(std::string line);
  unsigned channelIndex(unsigned crate, unsigned slot, unsigned channel);

  // Private data
private:
  // Fit config
  std::map <unsigned, std::pair<std::pair<unsigned, unsigned>, unsigned>> m_fitChannels;
  // Template
  std::vector<double> m_template;
  unsigned m_alignPoint;  
};

#endif
