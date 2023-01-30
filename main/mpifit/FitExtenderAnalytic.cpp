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

/** @file:  FitExtenderAnalytic.cpp
 *  @brief: FitExtender class for analytic fitting.
 */

#include "FitExtenderAnalytic.h"

#include <iostream>

#include <DDASHit.h>
#include <DDASHitUnpacker.h>

#include "lmfit_analytic.h"

using namespace DDAS::AnalyticFit;

FitExtenderAnalytic::FitExtenderAnalytic()
{}

FitExtenderAnalytic::~FitExtenderAnalytic()
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
FitExtenderAnalytic::operator()(pRingItem item)
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
	    lmfit1(&(pFit->s_extension.onePulseFit), trace, l.first, sat);
	  }
	  
	  if (classification & 2) {
	    // Single pulse fit guides initial guess for double pulse. If the
	    // single pulse fit does not exist, we do it here.
	    DDAS::fit1Info guess;
                        
	    if ((classification & 1) == 0) {
	      lmfit1(&(pFit->s_extension.onePulseFit), trace, l.first,
		     sat);

	    } else {
	      guess = pFit->s_extension.onePulseFit; // Already got it
	    }
	    lmfit2(&(pFit->s_extension.twoPulseFit), trace, l.first,
		   &guess, sat);
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
FitExtenderAnalytic::free(iovec& v)
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
    msg = "FitExtenderAnalytic asked to free something it never made";
    std::cerr << msg << std::endl;
    throw std::logic_error(msg);
  }
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
FitExtenderAnalytic::pulseCount(DAQ::DDAS::DDASHit& hit)
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
FitExtenderAnalytic::doFit(DAQ::DDAS::DDASHit& hit)
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
FitExtenderAnalytic::fitLimits(DAQ::DDAS::DDASHit& hit)
{
    int crate = hit.GetCrateID();
    int slot  = hit.GetSlotID();
    int chan  = hit.GetChannelID();    
    int index = channelIndex(crate, slot, chan);
    
    std::pair<std::pair<unsigned, unsigned>, unsigned> result = m_fitChannels[index];
    
    return result;       
}

/////////////////////////////////////////////////////////////////////////////
// Factory for our extender:
//
extern "C" {
  FitExtenderAnalytic* createExtender() {
    return new FitExtenderAnalytic;
  }
}
