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
 * @file TraceViewProcessor.cpp
 * @brief Implemenation of event processor class for DDAS events.
 */

#include "TraceViewProcessor.h"

#include <iostream>1
#include <ctime>

#include <FragmentIndex.h> // From UnifiedFormat
#include <CRingItem.h>
#include <CRingStateChangeItem.h>
#include <CPhysicsEventItem.h>
#include <CDataFormatItem.h>

#include <DDASFitHit.h>
#include <DDASFitHitUnpacker.h>

using namespace ufmt;
using namespace ddastoys;
    
//____________________________________________________________________________
/**
 * @details
 * Also constructs a DDASFitHitUnpacker object.
 */
TraceViewProcessor::TraceViewProcessor() :
    m_pUnpacker(new DDASFitHitUnpacker)
{}

//____________________________________________________________________________
/**
 * @details
 * Processor owns the unpacker, so delete on destruction.
 */
TraceViewProcessor::~TraceViewProcessor()
{
    delete m_pUnpacker;
}

//____________________________________________________________________________
/**
 * @details
 * Break PHYSICS_EVENTs into fragments, convert the fragments into DDASFitHits 
 * and append them to the event (just a vector of DDASFitHits).
 */
void
TraceViewProcessor::processEvent(CPhysicsEventItem& item)
{ 
    // Clear event vector before processing the hit.
  
    m_hits.clear();

    // Bust the ring item up into event builder fragments.
  
    FragmentIndex frags(
	reinterpret_cast<std::uint16_t*>(item.getBodyPointer())
	);

    // Decode the DDAS hit in each fragment and add it to the event.
  
    DDASFitHit hit;
    for (unsigned i = 0; i < frags.getNumberFragments(); i++) {
	hit.Reset();
	m_pUnpacker->decode(frags.getFragment(i).s_itemhdr, hit);
	m_hits.push_back(hit);
    }  
}
