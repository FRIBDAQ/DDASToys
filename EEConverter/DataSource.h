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
#ifndef DATASOURCE_H
#define DATASOURCE_H

/**
 * @file  DataSource.h
 * @brief Works with factories to provide a data source for undifferntiated 
 * ring items.
 * @note Abstract base class for FdDataSource, StreamDataSource and 
 *   RingDataSource
 */

class CRingItem;
class RingItemFactoryBase;

/**
 * @class DataSource
 * @brief Pure abstract data source which uses a factory's ring item getters to
 * provide ring item from a data source. 
 * 
 * @details
 * Since the factory provides this, we'll need concrete classes:
 * - FdDataSource - give data from a file descriptor.
 * - StreamDataSource - give data from a stream.
 * - RingDataSource - give data from a ringbuffer.
 *
 * @note (ASC 9/11/23): For EEConverter only FdDataSource is implemented.
 */
class DataSource {
protected:
    RingItemFactoryBase* m_pFactory; //!< Base class for abstract factory.
public:
    /** @brief Constructor. */
    DataSource(RingItemFactoryBase* pFactory);
    /** @brief Destructor. */
    virtual ~DataSource();
    /** @brief Pure virtual method to get ring items from a source. */
    virtual CRingItem* getItem() = 0;
    /**
     * @brief Set the factory for the data source.
     * @param pFactory New factory to set.
     */
    void setFactory(RingItemFactoryBase* pFactory);
};


#endif
