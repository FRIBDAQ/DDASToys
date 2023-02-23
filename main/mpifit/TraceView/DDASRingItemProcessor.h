/** 
 * @file  DDASRingItemProcessor.h
 * @brief Defines an event processor class for handing DDAS events.
 */

/** @addtogroup traceview
 * @{
 */

#ifndef DDASRINGITEMPROCESSOR_H
#define DDASRINGITEMPROCESSOR_H

#include <vector>

class CRingScalerItem;
class CRingStateChangeItem;
class CRingTextItem;
class CPhysicsEventItem;
class CRingPhysicsEventCountItem;
class CDataFormatItem;
class CGlomParameters;
class CRingItem;
namespace DAQ {
  namespace DDAS {
    class DDASFitHit;
    class DDASFitHitUnpacker;
  }
}

/**
 * @class DDASRingItemProcessor
 * @brief A basic ring item processor.
 *
 * A ring item processor class for a small subset of relavent ring items. See 
 * latest $DAQROOT/share/recipes/process/processor.h/cpp for a more general
 * example.
 */

class DDASRingItemProcessor
{
public:
  DDASRingItemProcessor();
  ~DDASRingItemProcessor();

public:
  
  // Implemented
  
  void processStateChangeItem(CRingStateChangeItem& item);
  void processEvent(CPhysicsEventItem& item);
  void processFormat(CDataFormatItem& item);
  void processUnknownItemType(CRingItem& item);
  
  // Ignored item types
  
  /** @brief Scaler ring items are ignored. */
  void processScalerItem(CRingScalerItem&) {return;};
  /** @brief Text ring items are ignored. */
  void processTextItem(CRingTextItem&) {return;};
  /** @brief PhysicsEventCount ring items are ignored. */
  void processEventCount(CRingPhysicsEventCountItem&) {return;};
  /** @brief GlomParameters ring items are ignored. */
  void processGlomParams(CGlomParameters&) {return;};

  /**
   * @brief Return the unpacked event data.
   * @return std::vector<DAQ::DDAS::DDASFitHit>  Vector containing the 
   *                                             event data.
   */ 
  std::vector<DAQ::DDAS::DDASFitHit> getUnpackedHits() {return m_hits;};

private:
  DAQ::DDAS::DDASFitHitUnpacker* m_pUnpacker;
  std::vector<DAQ::DDAS::DDASFitHit> m_hits;
};

#endif

/** @} */
