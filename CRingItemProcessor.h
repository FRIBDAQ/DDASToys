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

/** 
 * @file CRingItemProcessor.h
 * @brief Defines an abstract base class for ring item processing.
 */

#ifndef CRINGITEMPROCESSOR_H
#define CRINGITEMPROCESSOR_H

class CRingScalerItem;
class CRingStateChangeItem;
class CRingTextItem;
class CPhysicsEventItem;
class CRingPhysicsEventCountItem;
class CDataFormatItem;
class CGlomParameters;
class CRingItem;

/**
 * @class CRingItemProcessor
 * @brief Base class to support type-independent ring item processing.
 * 
 * @details
 * A base class for constructing ring item processors. The concept of this 
 * class is really simple. A virtual method for each ring item type that we 
 * differentiate between. Similarly a virtual method for ring item types that 
 * we don't break out. Derived classes can implement their own processing 
 * methods by overriding the virutal methods defined here. The default 
 * behavior is to dump everything to stdout using each ring item's toString() 
 * method.
 */

class CRingItemProcessor
{
public:
    /** @brief Constructor. */
    CRingItemProcessor();
    /** @brief Destructor. */
    virtual ~CRingItemProcessor();
  
public:
    /**
     * @brief Output an abbreviated scaler dump to stdout.
     * @param item Reference to the scaler ring item to process.
     */
    virtual void processScalerItem(CRingScalerItem& item);
    /**
     * @brief Output a state change item to stdout.
     * @param item Reference to the state change item.
     */
    virtual void processStateChangeItem(CRingStateChangeItem& item);
    /**
     * @brief Output a text item to stdout. 
     * @param item Refereinces the CRingTextItem we got.
     */
    virtual void processTextItem(CRingTextItem& item);
    /**
     * @brief Output a physics event ring item to stdout.
     * @param item Reference the physics event item.
     */
    virtual void processEvent(CPhysicsEventItem& item);
    /**
     * @brief Output an event count item to stdout.
     * @param item References the CPhysicsEventCountItem being dumped.
     */
    virtual void processEventCount(CRingPhysicsEventCountItem& item);
    /**
     * @brief Output the ring item format to stdout.
     * @param item References the format item.
     */
    virtual void processFormat(CDataFormatItem& item);
    /**
     * @brief Output a glom parameters item to stdout. 
     *  @param item References the glom parameter record. 
     */    
    virtual void processGlomParams(CGlomParameters& item);
    /**
     * @brief Output a ring item with unknown type to stdout.
     * @param item References the ring item for the event.
     */
    virtual void processUnknownItemType(CRingItem& item);
};

#endif
