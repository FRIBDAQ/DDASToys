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

/** @file:  CSortingOutputter.h
 *  @brief: Output class that can be  used when data needs re-ordering.
 */
#ifndef CSORTINGOUTPUTTER_H
#define CSORTINGOUTPUTTER_H

#include  <map>
#include  <vector>
#include <list>
#include <stdint.h>


class CDataSink;
class CRingItem;
/**
 * @class CSortingOutputter
 *     This class is a data ordering outputter.  It maintains queues that,
 *     are themselves time ordered internally but not with respect to each other.
 *     The data are output in a sorted manner.  The assumption is that the
 *     data are roughly uniformly distributed across the data sources.
 *     For a parallelized processing system you really hope this is the case.
 *
 *     This assumption is inherent in the sorting 'algorithm' which is just to
 *     do a selection sort when there are items in all queues.
 *
 *     The client of this class has a few responsibilities:
 *
 *     -  Data sources must be registered so that all queues are known in advance.
 *     -  Data sources must be shut down when they won't provide more data.
 *     -  Data must, of course be sent to the class.
 *     -  A data sink must be provided to which the sorte data are put.
 */
class CSortingOutputter
{
public:
    
    // What a queue item looks like:

    struct QueueItem {
        uint64_t    s_timestamp;           //  sorting key actually.
        CRingItem*  s_item;
    };
private:
    
    

    // Queue data structure:
    
    typedef std::list<QueueItem> DataQueue;
    typedef std::map<int, int>   QueueMap;
    
    // The queues and mapping between nodes and queues:
    
    std::vector<DataQueue>    m_queues;
    QueueMap                  m_SourceToQueue;
    
    CDataSink&                m_sink;    // Data goes here.
    
    // the queue merge sort done in outputOrdered, needs a functor to
    // compare items by timestamp:
    
    class Qcompare {
    public:
        bool operator() (
            CSortingOutputter::QueueItem& f, CSortingOutputter::QueueItem& s
        )
        {
            return f.s_timestamp < s.s_timestamp;
        }
    };
    
public:
    CSortingOutputter(CDataSink& sink);
    
    void addSource(int sourceId);
    void shutdownSource(int soureId);
    void queueItem(int sourceId, uint64_t timestamp, CRingItem* pItem);
private:
    void flushItems();
    std::pair<DataQueue*, uint64_t>  canOutput();
    void outputOrdered(unsigned maxPerQueue);
    void reapQueues();
    void outputRun(DataQueue& q, uint64_t tsLimit);
   
};

#endif