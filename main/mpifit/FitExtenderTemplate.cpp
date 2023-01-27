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

/** @file:  FitExtenderTemplate.cpp
 *  @brief: FitExtender class for analytic fitting.
 */

#include "FitExtenderTemplate.h"

#include <iostream>
#include <fstream>

#include <DDASHit.h>
#include <DDASHitUnpacker.h>

#include "lmfit_template.h"

using namespace DDAS::TemplateFit;

/**
 * Construtor
 *   Read the template fit configuration file upon creation
 */
FitExtenderTemplate::FitExtenderTemplate()
{
  // For the template fit we also need to read in the template data. We read
  // a file pointed to by the environment variable TEMPLATE_CONFIGFILE similar
  // to how the fitting configuration is read.
  
  try {
    std::string name = getTemplateFilename("TEMPLATE_CONFIGFILE");
    readTemplateFile(name.c_str());
  }
  catch (std::exception& e) {
    std::cerr << "Error processing template configuration file: "
	      << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

/**
 * Destructor
 */
FitExtenderTemplate::~FitExtenderTemplate()
{}

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
FitExtenderTemplate::operator()(pRingItem item)
{
  iovec result;
  // Get a pointer to the beginning of the body and
  // parse out the hit:
    
  std::uint32_t* pBody =
    reinterpret_cast<std::uint32_t*>(item->s_body.u_hasBodyHeader.s_body);
  DAQ::DDAS::DDASHit hit;
  DAQ::DDAS::DDASHitUnpacker unpacker;
  unpacker.unpack(pBody, nullptr, hit);
    
  if (doFit(hit)) {
    std::vector<std::uint16_t> trace = hit.GetTrace();
    
    if (trace.size() > 0) { // Need a trace to fit.
      std::pair<std::pair<unsigned, unsigned>, unsigned> l = fitLimits(hit);
      unsigned low = l.first.first;
      unsigned hi  = l.first.second;
      unsigned sat = l.second;
      
      if (low != hi) {
	int classification = pulseCount(hit);
	
	if(classification) {
	  
	  // Now we can do a fit

	  // \TODO (ASC 1/27/23): new'd but not deleted does this leak?
	  pFitInfo pFit = new FitInfo;
                    
	  // The classification is 'cleverly' bit encoded:
	  // bit 0 - do single pusle fit.
	  // bit 1 - do double pulse fit.
                    
	  if (classification & 1) {
	    lmfit1(&(pFit->s_extension.onePulseFit), trace, m_template,
		   m_alignPoint, l.first, sat);
	  }
	  
	  if (classification & 2) {
	    // Single pulse fit guides initial guess for double pulse. If the
	    // single pulse fit does not exist, we do it here.
	    DDAS::fit1Info guess;
                        
	    if ((classification & 1) == 0) {
	      lmfit1(&(pFit->s_extension.onePulseFit), trace, m_template,
		     m_alignPoint, l.first, sat);
	    } else {
	      guess = pFit->s_extension.onePulseFit; // Already got it
	    }
	    lmfit2(&(pFit->s_extension.twoPulseFit), trace, m_template,
		   m_alignPoint, l.first, &guess, sat);
	  }
	  
	  // Note that classification == 0 leaves us with a fit full-o-zeroes
	  result.iov_len = sizeof(FitInfo);
	  result.iov_base = pFit;
	  
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
 *   Free the storage created by an iovec. We use the size to figure out 
 *   which type of extension it is.
 *
 * @param v - Description of the extension we passed to our clients
 */
void
FitExtenderTemplate::free(iovec& v)
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
    msg = "FitExtenderTemplate asked to free something it never made";
    std::cerr << msg << std::endl;
    throw std::logic_error(msg);
  }
}

//
// Private methods
//

/**
 * getTemplateFilename
 *   Get template configuratin file name pointed to by an environment variable
 *
 * @param envname - the name of the environment variable pointing to the 
 *                  configuration file
 *
 * @return std::string - the filename
 *
 * @throw std::logic_error - if the envname does not point to a file
 */
std::string
FitExtenderTemplate::getTemplateFilename(const char* envname)
{
    const char* pFilename = getenv(envname);
    if (!pFilename) {
        std::string msg("No translation for environment variable: ");
        msg += envname;
        msg += " Point that to the template configuration file and re-run";
        throw std::invalid_argument(msg);
    }
    
    return std::string(pFilename);
}

/**
 * readTemplateFile
 *   Read the formatted tempalate configuration in formation and template data 
 *   from a file.
 *
 * @param filename - file name to read config info from
 *
 * @throw std::length_error - if the number of template data points is 
 *                            different than what the configuration file expects
 * @throw std::invalid_arugment - if the alignment point of the template is 
 *                                not contained in the trace (eg align to 
 *                                sample 101 on a 100 sample trace)
 * @throw std::invalid_argument - if the template data file cannot be opened
 */
void
FitExtenderTemplate::readTemplateFile(const char* filename)
{
  std::ifstream fileIn;
  fileIn.open(filename, std::ifstream::in);
  if (fileIn.is_open()) {
    unsigned npts;
    double val;

    // \TODO (ASC 1/25/23): What happens when there are fewer than two values on the first line? Should report an error and stop trying to do the fit.
    fileIn >> m_alignPoint >> npts;    
    while (fileIn >> val) {
      m_template.push_back(val);
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
    
  } else {
    std::string errmsg("Cannot open template data file: ");
    errmsg += filename;
    throw std::invalid_argument(errmsg);
  }

  fileIn.close();
}

/**
 * pulseCount
 *   This is a hook into which to add the ML classifier
 *
 * @param hit - references a hit.
 *
 * @return int
 * @retval 0  - On the basis of the trace no fitting.
 * @retval 1  - Only fit a single trace.
 * @retval 2  - Only fit two traces.
 * @retval 3  - Fit both one and double hit.
 */
int
FitExtenderTemplate::pulseCount(DAQ::DDAS::DDASHit& hit)
{
    return 3;                  // in absence of classifier.
}

/*
 * doFit
 *   This is a predicate function:
 *
 * @param crate - crate a hit comes from.
 * @param slot  - slot a hit comes from.
 * @param channel - channel within the slot the hit comes from
 *
 * @return bool - if true, the channel is fit.
 *
 * @note This sample predicate requests that all channels be fit.
 */
bool
FitExtenderTemplate::doFit(DAQ::DDAS::DDASHit& hit)
{
    int crate = hit.GetCrateID();
    int slot  = hit.GetSlotID();
    int chan  = hit.GetChannelID();    
    int index = channelIndex(crate, slot, chan);
    
    return (m_fitChannels.find(index) != m_fitChannels.end());
}

/**
* fitLimits
*   For each channel we're fitting, we need to know
*   - The left and right limits of the waveform that will be fitted
*   - The digitizer saturation level
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
FitExtenderTemplate::fitLimits(DAQ::DDAS::DDASHit& hit)
{
    int crate = hit.GetCrateID();
    int slot  = hit.GetSlotID();
    int chan  = hit.GetChannelID();    
    int index = channelIndex(crate, slot, chan);
    
    std::pair<std::pair<unsigned, unsigned>, unsigned> result = m_fitChannels[index];
    
    return result;       
}

/////////////////////////////////////////////////////////////////////////////
// Factory for our editor:
//
extern "C" {
  FitExtenderTemplate* createExtender() {
    return new FitExtenderTemplate;
  }
}

