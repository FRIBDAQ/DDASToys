/** 
 * @file  RootSinkProcessor.h
 * @brief Define a ring item processes which processes data into a ROOT 
 * data file sink
 */

#ifndef ROOTSINKPROCESSOR_H
#define ROOTSINKPROCESSOR_H

#include <CRingItemProcessor.h>

#include <string>

class RootFileDataSink;

/**
 * @class RootSinkProcessor
 * @brief Process data to a ROOT file sink.
 * @details
 * Overrides CRingItemProcessor processEvent() method to handle physics events.
 */

class RootSinkProcessor : public CRingItemProcessor
{
public:
    /** @brief Constructor. */
    RootSinkProcessor(std::string sink);
    /** @brief Destructor. */
    virtual ~RootSinkProcessor();

    // Mandatory interface from CRingItemProcessor:
  
public:
    /**
     * @brief Process physics events to the ROOT sink. 
     * @param item  References the physics event item that we are 'analyzing'.
     */
    virtual void processEvent(CPhysicsEventItem& item);
    
    // Private data:
  
private:
    RootFileDataSink* m_pSink; //!< ROOT file data sink used to write.
};

#endif
