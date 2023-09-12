/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** 
 * @file  DataSource.cpp
 * @brief Implementation of the non pure virtual methods of DataSource.
 */

#include "DataSource.h"
#include <RingItemFactoryBase.h>

/**
 * @details 
 * Just saves the factory pointer - note that we gain ownershp of the factory 
 * and, therefore, it's deleted on our destruction.
 */
DataSource::DataSource(RingItemFactoryBase* pFactory) :
    m_pFactory(pFactory)
{}

/**
 * @details
 * Destroys the factory.
 */
DataSource::~DataSource()
{
    delete m_pFactory;
}

/**
 * @details
 * - Delete the current factory.
 * - Set a new factory (this is done if the format changes).
 */
void
DataSource::setFactory(RingItemFactoryBase* pFactory)
{
    delete m_pFactory;
    m_pFactory = pFactory;
}
