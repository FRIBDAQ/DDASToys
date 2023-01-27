/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     Aaron Chester
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  CFitExtender.cpp
 *  @brief: Provides a fitting extender base class for DDAS Data.
 */

#include "CFitEditor.h"

#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <DDASHit.h>
#include <DDASHitUnpacker.h>

/*
  This file contains code that computes fits of waveforms and, using the 
  Transformer framework provides the fit parameters as extensions to the 
  fragments in each event. An extension is added to each fragmnt. The 
  extension provides a std::uint32_t self inclusive extension size which 
  may be sizeof(std::uint32_t) or, if larger a HitExtension struct (see f
  it_extensions.h)
*/

_FitInfo::_FitInfo() : s_size(sizeof(FitInfo)) {
  memset(&s_extension, 0,sizeof(DDAS::HitExtension));  // Zero fit params.
}

///// Local trim functions /////

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
 * Constructor
 *   Read and parse minimum configuration on construction
 */ 
CFitEditor::CFitEditor()
{
  try {    
    std::string name = getConfigFilename("FIT_CONFIGFILE");
    readConfigFile(name.c_str());
  } catch (std::exception& e) {
    std::cerr << "Error processing fit configuration file: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

/**
 * channelIndex
 *   Get global channel index from crate/slot/channel information
 *
 * @param crate - The crate ID
 * @param slot - The slot ID
 * @param channel - The channel ID
 *
 * @return int - the global channel index
 */
int
CFitEditor::channelIndex(unsigned crate, unsigned slot, unsigned channel)
{
  return (crate << 8) | (slot << 4)  | channel;
}

/**
 * getConfigFilename
 *    Return the name of the configuration file or throw
 *
 * @param envname - environment variable that points to the file
 *
 * @return std::string - translation of envname
 *
 * @throw std::invalid_argument - if there's no translation
 */
std::string
CFitEditor::getConfigFilename(const char* envname)
{
    const char* pFilename = getenv(envname);
    if (!pFilename) {
        std::string msg("No translation for environment variable : ");
        msg += envname;
        msg += " Point that to the fit configuration file and re-run";
        throw std::invalid_argument(msg);
    }
    return std::string(pFilename);
}

/**
 * readConfigFile
 *   Read the configuration file.  Lines in the configuration file can
 *   be empty or have as their first non-blank character "#" in which case
 *   they are ignored. All other lines specify channels that should be fit and
 *   contain six whitespace integers: crate slot channel low high saturation
 *   The crate, slot, channel identify a channel to fit while low, high are
 *   the limits of the trace to fit (first sample index, last sample index),
 *   and saturation is the level at which the digitizer saturates.
 *
 * @param filename - name of the configuration file
 *
 * @throw std::invalid_argument - if there are errors processing the file
 *                                including an inability to open the file
 */
void
CFitEditor::readConfigFile(const char* filename)
{
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
            sline >> crate >> slot >>channel >> low  >> high >> saturation;
	    
            if (sline.fail()) {
                std::string msg("Error processing line in configuration file '");
                msg += originalline;
                msg += "'";
                throw std::invalid_argument(msg);
            }
	    
            // Compute the channel index:
            
            int index = channelIndex(crate, slot, channel);
            std::pair<unsigned, unsigned> limits(low, high);
            std::pair<std::pair<unsigned, unsigned>, unsigned> value(limits, saturation);
            m_fitChannels[index] = value;
        }
    }
}

/**
 * isComment
 *   Determines if a line is a comment or not
 *
 * @param line - line to check
 *
 * @return std::string - if empty this line is comment else it's the 
 *                       trimmed string
*/
std::string
CFitEditor::isComment(std::string line)
{
    trim(line);                      // modifies it.
    if (line[0] == '#') return std::string("");
    return line;
}
