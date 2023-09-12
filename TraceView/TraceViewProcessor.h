/** 
 * @file TraceViewProcessor.h
 * @brief Defines an event processor class for handing DDAS events.
 */

#ifndef TRACEVIEWPROCESSOR_H
#define TRACEVIEWPROCESSOR_H

#include <CRingItemProcessor.h>

#include <vector>

namespace DAQ {
    namespace DDAS {
	class DDASFitHit;
	class DDASFitHitUnpacker;
    }
}

/**
 * @class TraceViewProcessor
 * @brief A basic ring item processor.
 *
 * @details
 * A ring item processor class for a small subset of relavent ring items. See 
 * latest $DAQROOT/share/recipes/process/processor.h/cpp for a more general
 * example. This processer:
 * - overrides base class functions to process events, and
 * - ignores scalers, text items, event counts and glom params.
 */

class TraceViewProcessor : public CRingItemProcessor
{
public:
    /** @brief Constructor. */
    TraceViewProcessor();
    /** @brief Destructor. */
    virtual ~TraceViewProcessor();

    // Implemented item types:
    
     /**
     * @brief Process physics events.
     * @param item Reference to the physics event item.
     */
    virtual void processEvent(CPhysicsEventItem& item);

    // Ignored item types:
    
    /** @brief Scaler ring items are ignored. */
    virtual void processScalerItem(CRingScalerItem&) { return; };
    /** @brief Text ring items are ignored. */
    virtual void processTextItem(CRingTextItem&) { return; };
    /** @brief PhysicsEventCount ring items are ignored. */
    virtual void processEventCount(CRingPhysicsEventCountItem&) { return; };
    /** @brief GlomParameters ring items are ignored. */
    virtual void processGlomParams(CGlomParameters&) { return; };

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
