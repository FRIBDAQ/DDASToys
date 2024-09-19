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
 * @file TraceViewProcessor.h
 * @brief Defines an event processor class for handing DDAS events.
 */

#ifndef TRACEVIEWPROCESSOR_H
#define TRACEVIEWPROCESSOR_H

#include <CRingItemProcessor.h>

#include <vector>

class CRingScalerItem;
class CRingStateChangeItem;
class CRingTextItem;
class CPhysicsEventItem;
class CRingPhysicsEventCountItem;
class CDataFormatItem;
class CGlomParameters;
class CRingItem;

namespace ddastoys {
    class DDASFitHit;
    class DDASFitHitUnpacker;
}

/**
 * @class TraceViewProcessor
 * @brief A basic ring item processor.
 *
 * @details
 * A ring item processor class for a small subset of relavent ring items. 
 * See latest $DAQROOT/share/recipes/process/processor.h/cpp for a more 
 * general example. This processer:
 * - implements mandatory interface to process events,
 * - ignores scalers, text items, event counts and glom params,
 * - inherits the rest of its behavior from the base class.
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
    std::vector<ddastoys::DDASFitHit> getUnpackedHits() {return m_hits;};

private:
    ddastoys::DDASFitHitUnpacker* m_pUnpacker; //!< Unpacker.
    std::vector<ddastoys::DDASFitHit> m_hits;  //!< Event data.
};

#endif
