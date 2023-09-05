/** 
 * @file  DDASRingItemProcessor.cpp
 * @brief Implemenation of event processor class for DDAS events.
 */

/** @addtogroup traceview
 * @{
 */

#include "DDASRingItemProcessor.h"

#include <iostream>
#include <ctime>

#include <FragmentIndex.h>
#include <CRingItem.h>
#include <CRingStateChangeItem.h>
#include <CPhysicsEventItem.h>
#include <CDataFormatItem.h>

#include <DDASFitHit.h>
#include <DDASFitHitUnpacker.h>

//____________________________________________________________________________
/**
 * @details
 * Also constructs a DDASFitHitUnpacker object.
 */
DDASRingItemProcessor::DDASRingItemProcessor() :
    m_pUnpacker(new DAQ::DDAS::DDASFitHitUnpacker)
{}

//____________________________________________________________________________
/**
 * @details
 * Processor owns the unpacker, so delete on destruction.
 */
DDASRingItemProcessor::~DDASRingItemProcessor()
{
    delete m_pUnpacker;
}

//____________________________________________________________________________
/**
 * @details
 * Do a partial dump of the event data:
 *   - BEGIN/END run we'll give the timestamp, source id, run number, 
 *     title and and time offset into the run.
 *   - PAUSE/RESUME we'll just give the time and time into the run.
 */
void
DDASRingItemProcessor::processStateChangeItem(CRingStateChangeItem& item)
{
    time_t tm = item.getTimestamp();
    std::cout << item.typeName() << " item recorded for run "
	      << item.getRunNumber() << " source ID "
	      << item.getSourceId() << std::endl;
    std::cout << "Title: " << item.getTitle() << std::endl;
    std::cout << "Occured at: " << std::ctime(&tm)
	      << " " << item.getElapsedTime() << " sec. into the run\n";
}

//____________________________________________________________________________
void
DDASRingItemProcessor::processEvent(CPhysicsEventItem& item)
{ 
    // Clear event vector before processing the hit.
  
    m_hits.clear();

    // Bust the ring item up into event builder fragments.
  
    FragmentIndex frags(
	reinterpret_cast<std::uint16_t*>(item.getBodyPointer())
	);

    // Decode the DDAS hit in each fragment and add it to the event.
  
    DAQ::DDAS::DDASFitHit hit;
    for (unsigned i = 0; i < frags.getNumberFragments(); i++) {
	hit.Reset();
	m_pUnpacker->decode(frags.getFragment(i).s_itemhdr, hit);
	m_hits.push_back(hit);
    }  
}

//____________________________________________________________________________
/**
 * @details 
 * 11.x and beyond runs have, as their first record an event format record
 * that indicates the data format.
 */
void
DDASRingItemProcessor::processFormat(CDataFormatItem& item)
{
    std::cout << " Data format is for: " << item.getMajor() << "."
	      << item.getMinor() << std::endl;
}

//____________________________________________________________________________
/**
 * @details
 * This can happen if we're seeing a ring item that we've not specified a 
 * handler for (unlikely) or the item types have expanded but the data format
 * is the same (possible) or the user has defined and is using their own ring 
 * item type. We'll just dump the item.
 */
void
DDASRingItemProcessor::processUnknownItemType(CRingItem& item)
{
    std::cout << item.toString() << std::endl;
}

/** @} */