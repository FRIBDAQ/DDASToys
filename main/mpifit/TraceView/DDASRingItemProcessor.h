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
 * @details
 * A ring item processor class for a small subset of relavent ring items. See 
 * latest $DAQROOT/share/recipes/process/processor.h/cpp for a more general
 * example.
 */

class DDASRingItemProcessor
{
public:
    /** @brief Constructor. */
    DDASRingItemProcessor();
    /** @brief Destructor. */
    ~DDASRingItemProcessor();

    // Implemented item types
    
    /**
     * @brief Processes a run state change item. 
     * @param item Reference to the state change item.
     */
    void processStateChangeItem(CRingStateChangeItem& item);
    /**
     * @brief Process physics events. Unpack the event into a vector of
     * DDASFitHits.
     * @param item Reference to the physics event item.
     */
    void processEvent(CPhysicsEventItem& item);
    /**
     * @brief Process data format ring items. 
     * @param item Reference to the format item.
     */
    void processFormat(CDataFormatItem& item);
    /**
     * @brief Process a ring item with an unknown item type. 
     * @param item Reference to the generic item.
     */
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
     * @return Vector containing the event data.
     */ 
    std::vector<DAQ::DDAS::DDASFitHit> getUnpackedHits() {return m_hits;};

private:
    DAQ::DDAS::DDASFitHitUnpacker* m_pUnpacker; //!< Unpacker for DDASFitHits.
    std::vector<DAQ::DDAS::DDASFitHit> m_hits;  //!< Event data.
};

#endif

/** @} */
