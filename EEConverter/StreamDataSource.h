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
#ifndef STREAMDATASOURCE_H
#define STREAMDATASOURCE_H

/** 
 * @file  StreamDataSource.h
 * @brief Defines a class that gets ring items from a stream.
 */

#include "DataSource.h"

#include <istream>

/**
 * @class StreamDataSource
 * @brief Concrete class to give ring items from a stream.
 */

class StreamDataSource : public DataSource
{
private:
    std::istream& m_str; //!< Stream to read from.
public:
    /**
     * @brief Constructor.
     * @param pFactory Factory for ring items.
     * @param str References stream from which to get ring items.
     */    
    StreamDataSource(RingItemFactoryBase* pFactory, std::istream& str);
    /** Destructor. */
    virtual ~StreamDataSource();
    /**
     * @brief Implementation of getItem method for this concrete class.
     * @return Pointer to the next ring item from the stream. nullptr if none.
     */
    virtual CRingItem* getItem();
};


#endif
