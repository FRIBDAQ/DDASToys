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
 * @file  ProcessToRootSink.cpp
 * @brief Implement mandatory interface from CRingItemProcessor to process 
 * data to a ROOT file sink.
 */

#include "ProcessToRootSink.h"

#include <CRingItemFactory.h>
#include <CPhysicsEventItem.h>

#include "RootFileDataSink.h"

/**
 * @brief Constructor.
 */
ProcessToRootSink::ProcessToRootSink(std::string sink) :
    m_pSink(new RootFileDataSink(sink.c_str()))
{}

/**
 * @brief Destructor.
 */
ProcessToRootSink::~ProcessToRootSink()
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
ProcessToRootSink::processEvent(CPhysicsEventItem& item)
{
    m_pSink->putItem(item);
}
