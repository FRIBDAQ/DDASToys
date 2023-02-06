#ifndef DDASEVENTPROCESSOR_H
#define DDASEVENTPROCESSOR_H

#include "CRingItemProcessor.h"

#include <vector>

class CPhysicsEventItem;

class MyFitHitUnpacker;
namespace DAQ {
  namespace DDAS {
    class DDASFitHit;
    class MyFitHitUnpacker;
  }
}

class DDASEventProcessor : public CRingItemProcessor
{
 public:
  DDASEventProcessor();
  virtual ~DDASEventProcessor();

  // Interface from CRingItemProcessor
public:
  virtual void processEvent(CPhysicsEventItem& item);

public:
  std::vector<DAQ::DDAS::DDASFitHit> getUnpackedHits() {return m_hits;};

  // Unique to this class
private:
  DAQ::DDAS::MyFitHitUnpacker* m_pUnpacker;
  std::vector<DAQ::DDAS::DDASFitHit> m_hits; 
};

#endif
