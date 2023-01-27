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

/** @file:  FitEditorAnalytic.cpp
 *  @brief: FitEditor class for analytic fitting.
 */

#include "FitEditorAnalytic.h"

#include <iostream>

#include <DDASHit.h>
#include <DDASHitUnpacker.h>

#include "lmfit_analytic.h"

using namespace DDAS::AnalyticFit;

/*
 * Constructor
 */
FitEditorAnalytic::FitEditorAnalytic()
{}

/*
 * Destructor
 */
FitEditorAnalytic::~FitEditorAnalytic()
{}

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
 * @param pHdr - Pointer to the ring item header of the hit.
 * @param bhdr - Pointer to the body header pointer for the hit.
 * @param bodySize - Number of _bytes_ in the body.
 * @param pBody    - Pointer to the body.
 * @return std::vector<CBuiltRingItemEditor::BodySegment>
 *             - final segment descriptors.
 */
std::vector<CBuiltRingItemEditor::BodySegment>
FitEditorAnalytic::operator()(pRingItemHeader pHdr, pBodyHeader hdr, size_t bodySize, void* pBody)
{
  std::vector<CBuiltRingItemEditor::BodySegment> result;
    
  // Regardless we want a segment that includes the hit. Note that the first
  // std::uint32_t of the body is the size of the standard hit part in
  // std::uint16_t words.
    
  std::uint16_t* pSize = static_cast<std::uint16_t*>(pBody);
  CBuiltRingItemEditor::BodySegment hitInfo(*pSize*sizeof(std::uint16_t),
					    pSize, false);
  result.push_back(hitInfo);
    
  // Make the hit:
    
  DAQ::DDAS::DDASHit hit;
  DAQ::DDAS::DDASHitUnpacker unpacker;
  unpacker.unpack(static_cast<std::uint32_t*>(pBody),
		  static_cast<std::uint32_t*>(nullptr),
		  hit);
    
  if (doFit(hit)) {
    std::vector<std::uint16_t> trace = hit.GetTrace();

    // \TODO (ASC 1/18/23): new'd without a delete, what happens?  
    pFitInfo pFit = new FitInfo; // Have an extension tho may be zero
    
    if (trace.size() > 0) { // Need a trace to fit
      auto l = fitLimits(hit);
      unsigned low = l.first.first;   // Left fit limit
      unsigned hi  = l.first.second;  // Right fit limit
      unsigned sat = l.second;        // Saturation value
           
      if (low != hi) {	
	int classification = pulseCount(hit);
	
	if (classification) {
	  
	  // Bit 0 do single fit.
	  // Bit 1 do double fit.
                    
	  if (classification & 1) {
	    lmfit1(&(pFit->s_extension.onePulseFit), trace, l.first, sat);
	  }
                    
	  if (classification & 2 ) {
	    // Single pulse fit guides initial guess for double pulse. If the single pulse fit does not exist, we do it here.
	    DDAS::fit1Info guess;                    

	    if ((classification & 1) == 0) {
	      lmfit1(&(pFit->s_extension.onePulseFit), trace, l.first, sat);
	    } else {
	      guess = pFit->s_extension.onePulseFit;
	    }
	    lmfit2(&(pFit->s_extension.twoPulseFit), trace, l.first,
		   &guess, sat);
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
 * @param e - iovec we need to free
 */
void
FitEditorAnalytic::free(iovec& e)
{
  if (e.iov_len == sizeof(FitInfo)) {
    pFitInfo pFit = static_cast<pFitInfo>(e.iov_base);
    delete pFit;
  } else {
    pNullExtension p = static_cast<pNullExtension>(e.iov_base);
    delete p;
  }
}

//
// Private methods
//

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
FitEditorAnalytic::pulseCount(DAQ::DDAS::DDASHit& hit)
{
    return 3; // In absence of classifier.
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
FitEditorAnalytic::doFit(DAQ::DDAS::DDASHit& hit)
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
FitEditorAnalytic::fitLimits(DAQ::DDAS::DDASHit& hit)
{
    int crate = hit.GetCrateID();
    int slot  = hit.GetSlotID();           // In case we are channel dependent.
    int chan  = hit.GetChannelID();
    
    int index = channelIndex(crate, slot, chan);
    std::pair<std::pair<unsigned, unsigned>, unsigned> result = m_fitChannels[index];
    
    return result;   
}

/////////////////////////////////////////////////////////////////////////////
// Factory for our editor:
//
extern "C" {
  FitEditorAnalytic* createEditor() {
    return new FitEditorAnalytic;
  }
}
