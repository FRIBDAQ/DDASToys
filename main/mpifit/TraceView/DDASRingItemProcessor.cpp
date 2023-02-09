/** @file: DDASRingItemProcessor.cpp
 *  @brief: Implemenation of event processor class for DDAS events.
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
 * Constructor
 */
DDASRingItemProcessor::DDASRingItemProcessor() :
  m_pUnpacker(new DAQ::DDAS::DDASFitHitUnpacker)
{}

//____________________________________________________________________________
/**
 * Destructor
 */
DDASRingItemProcessor::~DDASRingItemProcessor()
{
  delete m_pUnpacker;
}

//____________________________________________________________________________
/**
 * processStateChangeItem.
 *   Processes a run state change item. Again we're just going to do a partial 
 *   dump:
 *    - BEGIN/END run we'll give the timestamp, run number and title, and time
 *      offset into the run.
 *    - PAUSE/RESUME we'll just give the time and time into the run.
 *
 * @param item - reference to the state change item
 */
void
DDASRingItemProcessor::processStateChangeItem(CRingStateChangeItem& item)
{
  time_t tm = item.getTimestamp();
  std::cout << item.typeName() << " item recorded for run "
	    << item.getRunNumber() << std::endl;
  std::cout << "Title: " << item.getTitle() << std::endl;
  std::cout << "Occured at: " << std::ctime(&tm)
	    << " " << item.getElapsedTime() << " sec. into the run\n";
}

//____________________________________________________________________________
/**
 * processEvent
 *   Process physics events. Unpack the event into a vector of DDASFitHits.
 *
 *  @param item - references the physics event item that we are 'analyzing'
 */
void
DDASRingItemProcessor::processEvent(CPhysicsEventItem& item)
{ 
  // Clear event vector before processing the hit
  
  m_hits.clear();

  // Bust the ring item up into event builder fragments
  
  FragmentIndex frags(reinterpret_cast<std::uint16_t*>(item.getBodyPointer()));

  // Decode the DDAS hit in each fragment and add it to the event
  
  DAQ::DDAS::DDASFitHit hit;
  for (unsigned i=0; i<frags.getNumberFragments(); i++) {
    hit.Reset();
    m_pUnpacker->decode(frags.getFragment(i).s_itemhdr, hit);
    m_hits.push_back(hit);
  }  
}

//____________________________________________________________________________
/**
 * processFormat
 *   11.x runs have, as their first record an event format record that
 *   indicates that the data format is for 11.0.
 *
 * @param item - references the format item
 */
void
DDASRingItemProcessor::processFormat(CDataFormatItem& item)
{
  std::cout << " Data format is for: "
	    << item.getMajor() << "." << item.getMinor() << std::endl;
}

//____________________________________________________________________________
/**
 * processUnknownItemType
 *   Process a ring item with an unknown item type. This can happen if we're 
 *   seeing a ring item that we've not specified a handler for (unlikely) or 
 *   the item types have expanded but the data format is the same (possible) 
 *   or the user has defined and is using their own ring item type. We'll just
 *    dump the item.
 *
 * @param item - references the generic ring item
 */
void
DDASRingItemProcessor::processUnknownItemType(CRingItem& item)
{
  std::cout << item.toString() << std::endl;
}
