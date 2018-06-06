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

/** @file:  CSortingOutputter.cpp
 *  @brief: Implement the sorting outputter.
 */

#include "CSortingOutputter.h"

#include <CDataSink.h>
#include <CRingItem.h>
#include <algorithm>
#include <limits.h>
#include <iostream>

/**
 * constructor
 *    Just store a reference to the data sink.
 *
 * @param sink - data sink to which data will be written.
 */

CSortingOutputter::CSortingOutputter(CDataSink& sink) :
    m_sink(sink)
{}

/**
 * addSource:
 *    - appends a new queue to the m_queues member.
 *    - adds a map entry that maps the id of the node to the queue.
 *
 *  @note depending on actual performance, it may be better to use a potentially
 *        sparse vector and set of used items...though I'd guess iterating/searching
 *        the set is about as expensive as iterating/searching the map.
 *
 *  @param sourceId - id  of the new source.
 */
void
CSortingOutputter::addSource(int sourceId)
{
    int index = m_queues.size();          // Will be the index of the new queue.
    DataQueue  newQueue;                  // Need one to copy into the quuees.
    m_queues.push_back(newQueue);
    m_SourceToQueue[sourceId] = index; 
    
}
/**
 * shutdownSource
 *    Indicates a data source won't  be providing any more data.
 *    This just places a queue item at the end of the sources data queue
 *    with a timestamp of all ones and a null item pointer.
 *    When that migrates to the front of the queue the queue is removed from the
 *    map.
 *
 *  @param sourceId - id of the source that won't send any more data.
 */
void
CSortingOutputter::shutdownSource(int sourceId)
{
    QueueItem endOfData = {0xffffffffffffffff, nullptr};
    m_queues[m_SourceToQueue[sourceId]].push_back(endOfData);
    flushItems();              // Output what we can.
}

/**
 * queueItem
 *    Queue an item to a source data queue:
 *
 *  @param sourceId  - id of the data sourcea that's sending the data.
 *  @param timestamp - timestamp to put on the item.
 *  @param pItem     - Ring item to queue.
 */
void
CSortingOutputter::queueItem(int sourceId, uint64_t timestamp, CRingItem* pItem)
{
    QueueItem item = {timestamp, pItem};
    m_queues[m_SourceToQueue[sourceId]].push_back(item);
    
    flushItems();            // May make output possible.
}
/*-----------------------------------------------------------------------------
 * Utility methods.
 */
/**
 * flushItems
 *     All items that can be output will be output.
 */
void
CSortingOutputter::flushItems()
{
    reapQueues();                 // Get rid of finished queues.
    std::pair<DataQueue*, uint64_t> info = canOutput();
    while(info.first) {
        outputRun(*info.first, info.second);
        info = canOutput();
    }


}
/**
 * canOutput
 *    Data can be output if all queues have data.  Data can be output from the
 *    queue who's head has the smallest timestamp until that queue's head is
 *    larger than the queue with the next larger timestamp.
 *    This method determines:
 *    -  If all queues have data.
 *    -  Which queue has the smallest timestamp.
 *    -  The value of the timestamp at the front of the queue with the next
 *       largest timestamp.  Note that if there's only one active queue,
 *       the  value of that timestamp will be 0xffffffffffffffff.
 *
 *   @return std::pair<DataQueue*, uint64_t> First is the queue that can
 *        be output, nullptr if none can.
 *        Second is the timestamp at the front of the queue with the second oldest
 *        front stamp.
 *   
 */ 
std::pair<CSortingOutputter::DataQueue*, uint64_t>
CSortingOutputter::canOutput()
{
    uint64_t smallest = 0xffffffffffffffff;
    uint64_t nextsmallest = 0xffffffffffffffff;
    DataQueue* pQueue = nullptr;
    
    for (auto p = m_SourceToQueue.begin(); p != m_SourceToQueue.end(); p++) {
        DataQueue& q(m_queues[p->second]);
        if (q.empty()) {
            return std::pair<DataQueue*, uint64_t>(nullptr, 0xffffffffffffffff);     // Can't deq.
        }
        uint64_t ts = q.front().s_timestamp;
        if (ts < smallest) {
            pQueue = &q;
            nextsmallest = smallest;
            smallest = ts;
        } else if (ts < nextsmallest) {
            nextsmallest = ts;
        }
    }
    return std::pair<DataQueue*, uint64_t>(pQueue, nextsmallest);
}

/*
 * reapQueues
 *    Removes all elements from the m_SourceToQueue map for
 *    queues that are at end.  The queues themselves are untouched.
 */
void
CSortingOutputter::reapQueues()
{
    // First save the reapable queue indices  done this way because
    // erase can invalidate our iterator
    std::vector<int> reapable;
    for(auto p = m_SourceToQueue.begin(); p != m_SourceToQueue.end(); p++) {
      if(!m_queues[p->second].empty()) {
        QueueItem front = m_queues[p->second].front();
        if (!front.s_item) {                 // Null item is end marker.
	  reapable.push_back(p->first);
        }
      }
    }
    // Now remove the reapable keys from the map, making these invisible.
    
    for (int i =0; i < reapable.size(); i++) {
        m_SourceToQueue.erase(reapable[i]);
    }
}
/**
 * outputRun
 *    Output a run of data from a data queue until one of these conditions
 *    is true:
 *    -  the queue is empty.
 *    -  and end  of data marker is found (in which case reapQueues is called).
 *    -  the timestamp at the front of the queue is larger than a limtinig
 *    timestamp.
 *
 *  @param q   - Reference to a queue to output from.
 *  @param tsLimit - largest timestamp that can be output from the queue.
 */
void
CSortingOutputter::outputRun(CSortingOutputter::DataQueue& q, uint64_t tsLimit)
{

  // Debugging:

  uint64_t smallest = 0xffffffffffffffff;
  for (auto p = m_SourceToQueue.begin(); p!= m_SourceToQueue.end(); p++) {
    DataQueue& aq(m_queues[p->second]);
    if (&aq != &q) {
      uint64_t ts = aq.front().s_timestamp;
      if (ts  < smallest) smallest = ts;
    }
  }
  if (smallest != tsLimit) {
    std::cerr << "Error incorrect tslimit!!!! " <<  smallest << " " << tsLimit << "\n";
  }
  //
  
    while (! q.empty()) {
        QueueItem& item(q.front());
        if (item.s_item) {           // Actual item.
            if (item.s_timestamp <= tsLimit) {
                m_sink.putItem(*(item.s_item));
                delete item.s_item;
                q.pop_front();
            } else {
                return;               // Can't output any more.
            }
        } else {                      // End marker reap all end item queues.
            reapQueues();
            return;
        }
    }
}
