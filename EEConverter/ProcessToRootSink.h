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
 * @file  ProcessToRootSink.h
 * @brief Define a ring item processes which processes data into a ROOT 
 * data file sink
 */

#ifndef PROCESSTOROOTSINK_H
#define PROCESSTOROOTSINK_H

#include <CRingItemProcessor.h>

#include <string>

class RootFileDataSink;
class CPhysicsEventItem;

/**
 * @class ProcessToRootSink
 * @brief A ring item processeor concrete class that overrides the 
 * CRingItemProcessor base class method to put PHYSICS_EVENTs into a 
 * ROOT file data sink.
 */

class ProcessToRootSink : public CRingItemProcessor
{
public:
    /** @brief Constructor. */
    ProcessToRootSink(std::string sink);
    /** @brief Destructor. */
    virtual ~ProcessToRootSink();

    // Mandatory interface from CRingItemProcessor:
  
public:
    /**
     * @brief Process physics events. 
     * @param item  References the physics event item that we are 'analyzing'.
     */
    virtual void processEvent(CPhysicsEventItem& item);
    
    // Private data:
  
private:
    RootFileDataSink* m_pSink; //!< ROOT file data sink used to write.
};

#endif
