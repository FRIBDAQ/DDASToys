/** 
 * @file  RootSinkProcessor.cpp
 * @brief Implement mandatory interface from CRingItemProcessor to process 
 * data to a ROOT file sink.
 */

#include "RootSinkProcessor.h"

#include <memory>
#include <iostream>

#include <CPhysicsEventItem.h>

#include "RootFileDataSink.h"

/**
 * @brief Constructor.
 */
RootSinkProcessor::RootSinkProcessor(std::string sink) :
    m_pSink(new RootFileDataSink(sink.c_str()))
{}

/**
 * @brief Destructor.
 */
RootSinkProcessor::~RootSinkProcessor()
{
    delete m_pSink;
}

///
// Mandatory interface
//

/**
 * @details
 * Derived class decides what to do with PHYSICS_EVENT ring items. In this 
 * case, we just pass the data to the ROOT file sink and let it handle the rest.
 */
void
RootSinkProcessor::processEvent(CPhysicsEventItem& item)
{
    std::cout << item.toString() << std::endl;
    // m_pSink->putItem(item);
}
