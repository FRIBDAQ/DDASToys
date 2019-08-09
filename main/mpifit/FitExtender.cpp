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
static bool doFit(DAQ::DDAS::DDASHit& hit)
{
    int crate = hit.GetCrateID();
    int slot  = hit.GetSlotID();           // In case we are channel dependent.
    int chan  = hit.GetChannelID();
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
*/
static std::pair<std::pair<unsigned, unsigned>, unsigned>
fitLimits(DAQ::DDAS::DDASHit& hit)
{
    std::pair<std::pair<unsigned, unsigned>, unsigned> result;
    
    result.second = (1 << 14) -1;          // 14 bits.
    
    // There are sometimes spikes at the edges of the trace,
    // this example excludes the outer channel..
    
    const std::vector<uint16_t>& trace(hit.GetTrace());
    result.first.first = 1;
    result.first.second = trace.size() -1;
    
    return result;
    
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
static int
pulseCount(DAQ::DDAS::DDASHit& hit)
{
    return 3;                  // in absence of classifier.
}

/**
 * THe extender class definition.:
 */
class CFitExtender : public CBuiltRingItemExtender::CRingItemExtender
{
public:
    iovec operator()(pRingItem item);
    void free(iovec& e);
};
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

/////////////////////////////////////////////////////////////////////////////////
// Factory for our extender.:

extern "C" {
    CFitExtender* createExtender() {
        return new CFitExtender;
    }
}

