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
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <typeindex>

// So we're not using magic numbers for the global index:
static constexpr unsigned CHANNEL_BITS =
    5; //!< Number of bits for channel number, [0, 31]
static constexpr unsigned SLOT_BITS =
    4; //!< Number of bits for slot number, [0, 15]
static constexpr unsigned MAX_CHANNEL = 31; //!< Maximum channel number.
static constexpr unsigned MAX_SLOT = 15;    //!< Maximum slot number.

////////////////////////////////////////////////////////////////////////////////
// Local trim functions
//

/** @brief Trim from beginning. */
static inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
  return s;
}

/** @brief Trim from end. */
static inline std::string &rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          s.end());
  return s;
}

/** @brief Trim from both ends. */
static inline std::string &trim(std::string &s) { return ltrim(rtrim(s)); }

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
void ddastoys::Configuration::readConfigFile() {
  std::string filename = getFileNameFromEnv("FIT_CONFIGFILE");

  std::ifstream f(filename);
  if (f.fail()) {
    throw std::invalid_argument("Unable to open the configuration file: " +
                                filename);
  }

  while (!f.eof()) {
    std::string originalline("");
    std::getline(f, originalline, '\n');
    std::string line = isComment(originalline);

    if (line != "") {
      unsigned crate, slot, channel, length, low, high, saturation;
      std::string modelPath, templatePath;
      std::stringstream sline(line);
      sline >> crate >> slot >> channel >> length >> low >> high >>
          saturation >> std::quoted(modelPath) >> std::quoted(templatePath);

      if (sline.fail()) {
        throw std::invalid_argument(
            "Error processing line in configuration file '" + originalline +
            "'");
      }

      if (length == 0) {
        throw std::invalid_argument(
            "Invalid trace length in configuration file '" + originalline +
            "' length = 0");
      }

      if (low >= high) {
        throw std::invalid_argument(
            "Invalid fit limits in configuration file '" + originalline +
            "' low >= high");
      }

      if (high >= length) {
        throw std::invalid_argument(
            "Invalid fit limits in configuration file '" + originalline +
            "' high >= trace length");
      }

      // Compute the channel index, load template data, and add the
      // channel to the map:

      unsigned index = computeGlobalIndex(crate, slot, channel);
      std::pair<unsigned, unsigned> limits(low, high);
      auto tup = readTemplateFile(templatePath, length);
      unsigned align = std::get<0>(tup);
      std::vector<double> templateData = std::get<1>(tup);
      ConfigInfo info = {length,    limits, saturation,
                         modelPath, align,  templateData};
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
std::tuple<unsigned, std::vector<double>>
ddastoys::Configuration::readTemplateFile(std::string path, unsigned npts) {
  if (path == "none" || path.empty()) {
    std::vector<double> empty;
    return std::make_tuple(0, empty);
  }

  std::ifstream f(path);
  if (f.fail()) {
    throw std::invalid_argument("Unable to open the template file: " + path);
  }

  int nread = 0;
  double val;
  unsigned align;
  std::vector<double> data;
  while (!f.eof()) {
    std::string originalline("");
    std::getline(f, originalline, '\n');
    std::string line = isComment(originalline);
    if (line != "") {
      std::stringstream sline(line);

      if (nread == 0) {
        sline >> align;
      } else {
        sline >> val;
        data.push_back(val);
      }

      if (sline.fail()) {
        throw std::invalid_argument("Error processing line in template file '" +
                                    originalline + "'");
      }

      nread++;
    }
  }

  // The template should know how long it is. If you read in more data
  // points throw an exception:

  if (data.size() != npts) {
    throw std::length_error("Template configfile thinks the trace is " +
                            std::to_string(npts) + " samples but read in " +
                            std::to_string(data.size()));
  }

  // Ensure the alignment point is contained in the trace. Note that because
  // m_alignPoint is an unsigned type it cannot be negative:

  if (align >= data.size()) {
    throw std::invalid_argument("Invalid template alignment point " +
                                std::to_string(align) + " >= template size " +
                                std::to_string(data.size()));
  }

  f.close();

  return std::make_tuple(align, data);
}

/**
 * @details
 * Its up to caller to ensure that there is a map entry for this channel.
 * Caller should exit with failure message if we attempt to fit an unmapped
 * channel.
 */
bool ddastoys::Configuration::fitChannel(unsigned crate, unsigned slot,
                                         unsigned channel) {
  auto index = computeGlobalIndex(crate, slot, channel);

  return (m_fitChannels.find(index) != m_fitChannels.end());
}

unsigned ddastoys::Configuration::getTraceLength(unsigned crate, unsigned slot,
                                                 unsigned channel) {
  auto index = computeGlobalIndex(crate, slot, channel);

  return m_fitChannels.at(index).s_length;
}

std::pair<unsigned, unsigned>
ddastoys::Configuration::getFitLimits(unsigned crate, unsigned slot,
                                      unsigned channel) {
  auto index = computeGlobalIndex(crate, slot, channel);

  return m_fitChannels.at(index).s_limits;
}

unsigned ddastoys::Configuration::getSaturationValue(unsigned crate,
                                                     unsigned slot,
                                                     unsigned channel) {
  auto index = computeGlobalIndex(crate, slot, channel);

  return m_fitChannels.at(index).s_saturation;
}

std::string ddastoys::Configuration::getModelPath(unsigned crate, unsigned slot,
                                                  unsigned channel) {
  auto index = computeGlobalIndex(crate, slot, channel);

  return m_fitChannels.at(index).s_modelPath;
}

/**
 * @details
 * As a consequence of the sort-and-erase idiom used to uniquify the model
 * list, the model paths names are sorted in the returned vector and may be
 * in a different order than how they appear in the configuration file.
 * It is the responsibilty of the caller to deal with this.
 */
std::vector<std::string> ddastoys::Configuration::getModelList() {
  std::vector<std::string> models;
  for (const auto &entry : m_fitChannels) {
    models.push_back(entry.second.s_modelPath);
  }
  std::sort(models.begin(), models.end());
  models.erase(std::unique(models.begin(), models.end()), models.end());

  // Remove "none" and empty strings from model list:

  std::vector<std::string>::iterator it;
  for (it = models.begin(); it != models.end();) {
    if (*it == "none" || it->empty()) {
      it = models.erase(it);
    } else {
      ++it;
    }
  }

  return models;
}

/**
 * @details
 * All channels using a particular model are expected to have the same trace
 * length, so we can simply search for the first instance matching the path
 */
unsigned ddastoys::Configuration::getModelShape(std::string path) {
  auto it = std::find_if(
      m_fitChannels.begin(), m_fitChannels.end(),
      [&path](const auto &p) { return p.second.s_modelPath == path; });
  if (it != m_fitChannels.end()) {
    return it->second.s_length;
  } else {
    throw std::invalid_argument("No matching channels for model path '" + path +
                                "'");
  }
}

std::vector<double> ddastoys::Configuration::getTemplate(unsigned crate,
                                                         unsigned slot,
                                                         unsigned channel) {
  auto index = computeGlobalIndex(crate, slot, channel);

  return m_fitChannels.at(index).s_template;
}

unsigned ddastoys::Configuration::getTemplateAlignPoint(unsigned crate,
                                                        unsigned slot,
                                                        unsigned channel) {
  auto index = computeGlobalIndex(crate, slot, channel);

  return m_fitChannels.at(index).s_alignPoint;
}

void ddastoys::Configuration::verifyTemplateData() {
  for (const auto &c : m_fitChannels) {
    if (c.second.s_template.empty()) {
      auto [crate, slot, channel] = decodeGlobalIndex(c.first);
      std::string msg("No template for channel ");
      msg += std::to_string(crate) + "/" + std::to_string(slot) + "/" +
             std::to_string(channel);
      msg += " but template fitting was requested."
             " Provide a template file for this channel in the fit"
             " configuration (its template path is \"none\" or empty).";
      throw std::invalid_argument(msg);
    }
  }
}

///
// Private methods
//

std::string ddastoys::Configuration::getFileNameFromEnv(const char *envname) {
  const char *pFilename = getenv(envname);
  if (!pFilename) {
    throw std::invalid_argument(
        "No translation of environment variable: " + std::string(envname) +
        " Point that to the proper configuration file and "
        "re-run");
  }

  return std::string(pFilename);
}

std::string ddastoys::Configuration::isComment(std::string line) {
  trim(line); // Modifies it
  if (line[0] == '#')
    return std::string("");

  return line;
}

unsigned ddastoys::Configuration::computeGlobalIndex(unsigned crate,
                                                     unsigned slot,
                                                     unsigned channel) {
  if (slot > MAX_SLOT) {
    throw std::invalid_argument("Invalid slot number " + std::to_string(slot) +
                                " > " + std::to_string(MAX_SLOT));
  }
  if (channel > MAX_CHANNEL) {
    throw std::invalid_argument("Invalid channel number " +
                                std::to_string(channel) + " > " +
                                std::to_string(MAX_CHANNEL));
  }
  return (crate << (CHANNEL_BITS + SLOT_BITS)) | (slot << CHANNEL_BITS) |
         channel;
}

/**
 * @details
 * Inverse of computeGlobalIndex.
 */
std::tuple<unsigned, unsigned, unsigned>
ddastoys::Configuration::decodeGlobalIndex(unsigned index) {
  unsigned crate = (index >> (CHANNEL_BITS + SLOT_BITS));
  unsigned slot = (index >> CHANNEL_BITS) & MAX_SLOT;
  unsigned channel = index & MAX_CHANNEL;

  return std::make_tuple(crate, slot, channel);
}