/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  FitExtender.cpp
 *  @brief: Provides a fitting extender for DDAS Data.
 */

#include "FitExtender.h"
#include <CBuiltRingItemExtender.h>
#include <DataFormat.h>
#include <FragmentIndex.h>
#include <DDASHitUnpacker.h>
#include <DDASHit.h>
#include "lmfit.h"
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include <string.h>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <locale>
#include <cctype>

using namespace  DDAS;

/**
 * This file contains code that computes fits of waveforms
 * and, using the Transformer framework provides the fit parameters
 * as extensions to the fragments in each event.
 * An extension is added to each fragmnt.  The extension provides a
 * uint32_t self inclusive extension size which may be sizeof(uint32_t)
 * or, if larger a HitExtension struct (see lmfit):
 */

_FitInfo::_FitInfo() : s_size(sizeof(FitInfo)) {
    memset(&s_extension, 0,sizeof(DDAS::HitExtension));  // Zero the fit params.
}

////////////////////////////// local trim functions   /////////////////////////

static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}

//////////////////////////////////////////////////////////////////////////

static inline int channelIndex(unsigned crate, unsigned slot, unsigned channel)
{
    return (crate << 8) | (slot << 4)  | channel;
}

/**
 * THe extender class definition.:
 */
class CFitExtender : public CBuiltRingItemExtender::CRingItemExtender
{
    std::map <int, std::pair<std::pair<unsigned, unsigned>, unsigned>> m_fitChannels;
public:
    CFitExtender();
    iovec operator()(pRingItem item);
    void free(iovec& e);
private:
    int pulseCount(DAQ::DDAS::DDASHit& hit);
    bool doFit(DAQ::DDAS::DDASHit& hit);
    std::pair<std::pair<unsigned, unsigned>, unsigned> fitLimits(
        DAQ::DDAS::DDASHit& hit
    );
    std::string getConfigFilename(const char* envname);
    void readConfigFile(const char* filename);
    std::string isComment(std::string line);
};
/**
 * cpnstructor
 *   - Figure out the name of our config file (env variable FIT_CONFIGFILE)
 *     points to it.
 *   - Load the config file into m_fitChannels such that each index is the
 *     bottom 12 bits of the first word and values are fit low/high limits.
 *  @note - errors in configuration file processing are flagged via exceptions
 *          which we catch and report, terminating the program.  This termination
 *          will result in abnormal ends if --parallel-strategy is MPI.
 */
CFitExtender::CFitExtender()
{
    try {    
        std::string name = getConfigFilename("FIT_CONFIGFILE");
        readConfigFile(name.c_str());
    } catch (std::exception& e) {
        std::cerr << "Error Processing configuration file: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * operator()
 *   - Parse the fragment into a hit.
 *   - See if the predicate say we should fit it.
 *   - If so ensure there's a trace.
 *   - Get the fit limits.
 *   - Get the number of pulses to fit.
 *   - Fit them
 *   - Create the appropriate extension.
 *
 * @param item - pointer to an event fragment ring item.
 * @return iovec - Describes the extension.
 * @note we use new so free has to use delete.
 */
iovec
CFitExtender::operator()(pRingItem item)
{
    iovec result;
    // Get a pointer to the beginning of the body and
    // parse out the hit:
    
    uint32_t* pBody =
        reinterpret_cast<uint32_t*>(item->s_body.u_hasBodyHeader.s_body);
    DAQ::DDAS::DDASHit hit;
    DAQ::DDAS::DDASHitUnpacker unpacker;
    unpacker.unpack(pBody, nullptr, hit);
    
    if (doFit(hit)) {
        std::vector<uint16_t> trace = hit.GetTrace();
        if (trace.size() > 0) {            // Need a trace to fit.
            std::pair<std::pair<unsigned, unsigned>, unsigned> l
                = fitLimits(hit);
            unsigned low = l.first.first;
            unsigned hi  = l.first.second;
            unsigned sat = l.second;
            if (low != hi) {
                int classification = pulseCount(hit);
                if(classification) {
                    // Now we can do a fit!!!
                    
                    pFitInfo pFit = new FitInfo;
                    
                    // The classification is 'cleverly' bit encoded:
                    // bit  0 - do single pusle fit.
                    // bit  1 - do double pulse fit.
                    
                    if (classification & 1) {
                        DDAS::lmfit1(&(pFit->s_extension.onePulseFit), trace, l.first, sat);
                    }
                    if (classification & 2) {
                        // We need to do the onePulse if asked to do a
                        // for the initial guesses:
                        
                        fit1Info guess;
                        
                        if ((classification & 1) == 0) {
                            DDAS::lmfit1(&guess, trace, l.first, sat);
                        } else {
                            guess = pFit->s_extension.onePulseFit;       // Already got it.
                        }
                        lmfit2(
                            &(pFit->s_extension.twoPulseFit), trace, l.first, &guess, sat
                        );
                    }
                    // Note that classification ==0 leaves us with a fit
                    // full-o-zeroes.
                    
                    result.iov_len = sizeof(FitInfo);
                    result.iov_base= pFit;
                    return result;
                    
                }
            }
        }
    }
    // If we got here we can't do a fit:
    
    pNullExtension ext = new nullExtension;
    result.iov_len = sizeof(nullExtension);
    result.iov_base = ext;
    return result;
}
/**
 * free
 *    Free the storage created by an iovec.
 *    - We use the size to figure out which type of extension it is.
 *
 * @param v - description of the extension we passed to our clients.
 */
void
CFitExtender::free(iovec& v)
{
    if (v.iov_len == sizeof(nullExtension)) {
        pNullExtension pE = static_cast<pNullExtension>(v.iov_base);
        delete pE;
    } else if (v.iov_len == sizeof(FitInfo)) {
        pFitInfo pF = static_cast<pFitInfo>(v.iov_base);
        delete pF;
    } else {
        /// bad bad bad.
        
        std::string msg;
        msg = "CFitExtender asked to free something it never made";
        std::cerr << msg << std::endl;
        throw std::logic_error(msg);
    }
}
//////////////////////////////////////////////////////////////////////////////
// Private  methods.

/**
 * getConfigFilename
 *    Return the name of the configuration file or throw.
 * @param envname - environment variable that points to the file.
 * @return std::string - translation of envname.
 * @throw std::invalid_argument - if there's no translation.
 */
std::string
CFitExtender::getConfigFilename(const char* envname)
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
 *    Read the configuration file.  Lines in the configuration file can
 *    be empty or have as their first non-blank character "#" in which case
 *    they are ignored.  All other lines specify channels that should be fit and
 *    contain six whitespace integers:  crate slot channel  low  high saturation
 *    The crate, slot, channel identify a channel to fit while low, high are
 *    the limits of the trace to fit (first sample index, last sample index),
 *    and saturation is the level at which the digitizer saturates.
 *
 * @param filename - name of the configuration file.
 * @throw std::invalid_argument - if there are errors processing the file
 *                                including an inability to open the file.
 */
void
CFitExtender::readConfigFile(const char* filename)
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
            // compute the channel index:
            
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
 * @param line - line to check.
 * @return std::string - if empty this line is  comment else it's the trimmed string.
*/
std::string
CFitExtender::isComment(std::string line)
{
    trim(line);                      // modifies it.
    if (line[0] == '#') return std::string("");
    return line;
}

/**
 * pulseCount
 *    This is a hook into which to add the ML classifier being developed
 *    by CSE.
 *
 * @param hit - references a hit.
 * @return int
 * @retval 0  - On the basis of the trace no fitting.
 * @retval 1  - Only fit a single trace.
 * @retval 2  - Only fit two traces.
 * @retval 3  - Fit both one and double hit.
 */
int
CFitExtender::pulseCount(DAQ::DDAS::DDASHit& hit)
{
    return 3;                  // in absence of classifier.
}

/*
 * doFit
 *    This is a predicate function:
 *
 * @param crate - crate a hit comes from.
 * @param slot  - Slot a hit comes from.
 * @param channel - Channel within the slot the hit comes from
 * @return bool - if true, the channel is fit.
 * @note This sample predicate requests that all channels be fit.
 */
bool
CFitExtender::doFit(DAQ::DDAS::DDASHit& hit)
{
    int crate = hit.GetCrateID();
    int slot  = hit.GetSlotID();           // In case we are channel dependent.
    int chan  = hit.GetChannelID();
    
    int index = channelIndex(crate, slot, chan);
    return (m_fitChannels.find(index) != m_fitChannels.end());
    
    return true;
}
/**
* fitLimits
*    For each channel we're fitting, we need to know
*    -  The left and right limits of the waveform that will be fittedn
*    -  The digitizer saturation level.
*
*
* @param hit - reference to a single hit.
* @return std::pair<std::pair<unsigned, unsigned>, unsigned>
*       The first element of the outer pair are the left and right
*       fit limits respectively.  The second element of the outer pair
*       is the saturation level.
* @note the caller must have ensured there's a map entry for this channel.
*/
std::pair<std::pair<unsigned, unsigned>, unsigned>
CFitExtender::fitLimits(DAQ::DDAS::DDASHit& hit)
{
    int crate = hit.GetCrateID();
    int slot  = hit.GetSlotID();           // In case we are channel dependent.
    int chan  = hit.GetChannelID();
    
    int index = channelIndex(crate, slot, chan);
    std::pair<std::pair<unsigned, unsigned>, unsigned> result = m_fitChannels[index];
    
    return result;
       
}


/////////////////////////////////////////////////////////////////////////////////
// Factory for our extender.:

extern "C" {
    CFitExtender* createExtender() {
        return new CFitExtender;
    }
}

