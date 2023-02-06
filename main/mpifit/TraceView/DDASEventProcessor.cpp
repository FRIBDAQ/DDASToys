#include "DDASEventProcessor.h"

#include <iostream>

#include <FragmentIndex.h>
#include <CPhysicsEventItem.h>

#include <DDASFitHit.h>
#include <DDASFitHitUnpacker.h>

DDASEventProcessor::DDASEventProcessor() :
  m_pUnpacker(new DAQ::DDAS::DDASFitHitUnpacker)
{}

DDASEventProcessor::~DDASEventProcessor()
  
{
  delete m_pUnpacker;
}

void
DDASEventProcessor::processEvent(CPhysicsEventItem& item)
{
  std::cout << item.toString() << std::endl;
  
  // Clear event vector before processing the hit
  
  m_hits.clear();

  // Bust the ring item up into event builder fragments
    
  FragmentIndex frags(reinterpret_cast<std::uint16_t*>(item.getBodyPointer()));

  // Decode the DDAS hit in each fragment and add it to the event. Note that
  // AddHit does a copy construction of the hit into new storage.
  
  DAQ::DDAS::DDASFitHit hit;
  for (unsigned i=0; i<frags.getNumberFragments(); i++) {
    hit.Reset();
    m_pUnpacker->decode(frags.getFragment(i).s_itemhdr, hit);
    m_hits.push_back(hit);
  }  

}
