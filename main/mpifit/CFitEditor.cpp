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
#include "CFitExtender.h"

/*
  This file contains code that computes fits of waveforms and, using the Transformer framework provides the fit parameters as extensions to the fragments in each event. An extension is added to each fragmnt. The extension provides a uint32_t self inclusive extension size which may be sizeof(uint32_t) or, if larger a HitExtension struct (see fitinfo.h)
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

////////////////////////////////

// Get global channel index from crate/slot/channel
static inline int channelIndex(unsigned crate, unsigned slot, unsigned channel)
{
  return (crate << 8) | (slot << 4)  | channel;
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
    std::cerr << "Error Processing configuration file: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

/**
 * operator()
 *    - Parse the fragment into a hit.
 *    - Produce a IOvec element for the existing hit (without any fit
 *      that migh thave been there).
 *    - See if the predicate says we should fit.
 *    - If so, create the trae.
 *    - Get the fit limits, and saturation.
 *    - Get the number of pulses to fit.
 *    - Do the fits.
 *    - Create an Iovec entry for the extension we created (dynamic).
 *
 * @param pHdr - Pointer to the rin gitem header of the hit.
 * @param bhdr - Pointer to the body header pointer for the hit.
 * @param bodySize - number of _bytes_ in the body.
 * @param pBody    - pointer to the body.
 * @return std::vector<CBuiltRingItemEditor::BodySegment>
 *            final segment descriptors.
 */
std::vector<CBuiltRingItemEditor::BodySegment>
CFitEditor::operator()(
    pRingItemHeader pHdr, pBodyHeader bhdr, size_t bodySize, void* pBody
)
{
  std::vector<CBuiltRingItemEditor::BodySegment> result;
    
  // Regardless we want a segment that includes the hit.  Note
  // that the first uint32_t of the body is the size of the
  // standard hit part in uint16_t words.
    
  uint16_t* pSize = static_cast<uint16_t*>(pBody);
  CBuiltRingItemEditor::BodySegment hitInfo(*pSize*sizeof(uint16_t), pSize, false);
  result.push_back(hitInfo);
    
  // Make the hit:
    
  DAQ::DDAS::DDASHit hit;
  DAQ::DDAS::DDASHitUnpacker unpacker;
  unpacker.unpack(static_cast<uint32_t*>(pBody),
		  static_cast<uint32_t*>(nullptr),
		  hit);
    
  if (doFit(hit)) {
    std::vector<uint16_t> trace = hit.GetTrace();
    
    if (trace.size() > 0) {   /// Need a trace to fit
      auto l = fitLimits(hit);
      unsigned low = l.first.first;   // Left fit limit
      unsigned hi  = l.first.second;  // Right fit limit
      unsigned sat = l.second;        // Saturation value
           
      if (low != hi) {	
	int classification = pulseCount(hit);
	
	if (classification) {
	  // \TODO (ASC 1/18/23): What happens to this memory? Do we need to explicitly free it or wrap it in a smart ptr to automatically delete it when its out of scope?
	  pFitInfo pFit = new FitInfo; // Have an extension tho may be zero

	  
	  // Bit 0 do single fit.
	  // Bit 1 do double fit.
                    
	  if (classification & 1) {
	    fitSinglePulse(pFit->s_extension.onePulseFit, trace,
			   l.first, sat);
	  }
                    
	  if (classification & 2 ) {
	    // Single pulse fit guides initial guess for double pulse. If the single pulse fit does not exist, we do it here.
	    fit1Info guess;                    

	    if ((classification & 1) == 0) {
	      fitSinglePulse(pFit->s_extension.onePulseFit, trace,
			     l.first, sat);
	    } else {
	      guess = pFit->s_extension.onePulseFit;
	    }
	    fitDoublePulse(pFit->s_extension.twoPulseFit, trace,
			   l.first, guess, sat); 
	  }
	  
	}
      }
      
    }
    
    CBuiltRingItemEditor::BodySegment fit(sizeof(FitInfo), pFit, true);
    result.push_back(fit);
    
  } else { // No fit performed
    pNullExtension p = new nullExtension;
    CBuiltRingItemEditor::BodySegment nofit(sizeof(nullExtension), p, true);
    result.push_back(nofit);
  }    
    
  return result; // Return the description
}

/**
 * free
 *   We get handed our fit extension descriptor(s) to free
 *
 * @param info - iovec we need to free
 */
void
CFitEditor::free(iovec& item)
{
  if (item.iov_len == sizeof(FitInfo)) {
    pFitInfo pFit = static_cast<pFitInfo>(item.iov_base);
    delete pFit;
  } else {
    pNullExtension p = static_cast<pNullExtension>(item.iov_base);
    delete p;
  }
}

/**
 * Private methods
 */

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

/**
 * pulseCount
 *   This is a hook into which to add the ML classifier
 *
 * @param hit - references a hit
 *
 * @return int
 * @retval 0  - On the basis of the trace no fitting
 * @retval 1  - Only fit a single trace
 * @retval 2  - Only fit two traces
 * @retval 3  - Fit both one and double hit
 */
int
CFitEditor::pulseCount(DAQ::DDAS::DDASHit& hit)
{
    return 3;                  // in absence of classifier.
}

/*
 * doFit
 *    This is a predicate function:
 *
 * @param crate - crate a hit comes from
 * @param slot  - Slot a hit comes from
 * @param channel - Channel within the slot the hit comes from
 *
 * @return bool - if true, the channel is fit
 *
 * @note This sample predicate requests that all channels be fit
 */
bool
CFitEditor::doFit(DAQ::DDAS::DDASHit& hit)
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
*   For each channel we're fitting, we need to know
*   -  The left and right limits of the waveform that will be fitted
*   -  The digitizer saturation level
*
*
* @param hit - reference to a single hit
*
* @return std::pair<std::pair<unsigned, unsigned>, unsigned>
*   The first element of the outer pair are the left and right
*   fit limits respectively. The second element of the outer pair
*   is the saturation level.
*
* @note the caller must have ensured there's a map entry for this channel.
*/
std::pair<std::pair<unsigned, unsigned>, unsigned>
CFitEditor::fitLimits(DAQ::DDAS::DDASHit& hit)
{
    int crate = hit.GetCrateID();
    int slot  = hit.GetSlotID();           // In case we are channel dependent.
    int chan  = hit.GetChannelID();
    
    int index = channelIndex(crate, slot, chan);
    std::pair<std::pair<unsigned, unsigned>, unsigned> result = m_fitChannels[index];
    
    return result;   
}