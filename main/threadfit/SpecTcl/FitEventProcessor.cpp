/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins 
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  FitEventProcessor.cpp
 *  @brief: Implement the FitEventProcessor class.
 */

#include "FitEventProcessor.h"
#include <FragmentIndex.h>
#include <FitHitUnpacker.h>
#include <DDASFitHit.h>
#include <functions.h>
#include <stdexcept>
#include <string>
static const unsigned MAX_HITS(5);  // Max hits per event.

/**
 *  Constructor
 *     Creates the tree parameters.  We expect external forces to
 *     register these.  If the event processor is constructed
 *     at file level -- that happens automatically.
 */
FitEventProcessor::FitEventProcessor() :
    m_f1ChiSquare("fit1.Chisquare", 8000, 0.0, 400.0, "", 5, 0),
    m_f1A0("fit1.Amplitude", 16384, 0.0, 8192.0, "", 5, 0),
    m_f1Iterations("fit1.iterations", 50, 0, 50, "Iterations", 5, 0),
    
    m_f2ChiSquare("fit2.ChiSquare", 8000, 0.0, 400.0, "", 5, 0),
    m_f2A0("fit2.Amplitude1", 16834, 0.0, 8192.0, "", 5, 0),
    m_f2A1("fit2.Amplitude2", 16384, 0.0, 8192.0, "", 5, 0),
    m_f2Dt("fit2.dt", 512, 0.0, 256.0, "Sample", 5, 0),
    m_f2Iterations("fit2.iterations", 50, 0, 50, "Iterations", 5, 0),
    m_energy("Energy", 16384, 0.0, 16384.0, ""),
    m_chiratio(std::string("ChiSq1overChisq2"), 1000 , 0.0, 100.0,std::string( ""))
{}


/**
 * operator()
 *    Process events;
 *    - break the event up into fragments.
 *    - For each fragment, decode the fitted hit.
 *    - Stuff the parameters.  Note if needed, the pulses are flipped so that
 *      A0, is always the left most pulse and Dt is always positive.
 *
 * @param pEvent - pointer to the event body -- and that's all we care about.
 * @return kfTRUE.
 */
Bool_t
FitEventProcessor::operator()(
    const Address_t pEvent, CEvent& rEvent,
    CAnalyzer& rAnalyzer, CBufferDecoder& rDecoder
)
{
    try {
        
        FragmentIndex frags(reinterpret_cast<uint16_t*>(pEvent));
        DAQ::DDAS::FitHitUnpacker unpacker;
        unsigned h = 0;
        unsigned fragno = 0;
        for (auto p = frags.begin(); p != frags.end(); p++) {
            
            DAQ::DDAS::DDASFitHit hit;
            hit.Reset();
            unpacker.decode(p->s_itemhdr, hit);
	    
            if (hit.hasExtension() && (h < MAX_HITS)) {
                m_energy = hit.GetEnergy();
                if (h > 0) {
                        std::cerr << "That's odd more than one hit with a fit in an event\n";
                        std::cerr << "crate " << hit.GetCrateID()
                            << " slot " << hit.GetSlotID()
                            << " chan " << hit.GetChannelID() << std::endl;
                }
                const DDAS::HitExtension& extension(hit.getExtension());
                
                m_f1ChiSquare[h] = extension.onePulseFit.chiSquare;
                m_f1A0[h]        = extension.onePulseFit.pulse.amplitude;
                m_f1Iterations[h]     = extension.onePulseFit.iterations;
                
                m_f2ChiSquare[h] = extension.twoPulseFit.chiSquare;
                m_f2Iterations[h]= extension.twoPulseFit.iterations;
                
                // Figure out which is left (a0) and which is right (a1).
                
                int left  = 0;
                int right = 1;
                if(
                   extension.twoPulseFit.pulses[0].position >
                   extension.twoPulseFit.pulses[1].position
                ) {
                    left = 1;
                    right = 0;
                }
                m_f2A0[h] = extension.twoPulseFit.pulses[left].amplitude;
                m_f2A1[h] = extension.twoPulseFit.pulses[right].amplitude;
                m_f2Dt[h] =
                    extension.twoPulseFit.pulses[right].position -
                    extension.twoPulseFit.pulses[left].position;
                
                
                h++;
		if (extension.twoPulseFit.chiSquare > 0.0) {
		  m_chiratio = extension.onePulseFit.chiSquare / extension.twoPulseFit.chiSquare;
		}
            }
	    fragno++;
        }
	// std::cerr << fragno << " Fragments\n";
        
        return kfTRUE;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return kfFALSE;
    } catch (...) {
        std::cerr << "Caught an unanticipated exception type\n";
        throw;
    }
}
