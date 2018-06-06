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

/** @file:  CZMQRingOutputter.cpp
 *  @brief: Implements CZMQRingOutputter
 *
 */

#include "CZMQRingOutputter.h"
#include "DDASFitHit.h"
#include "FitHitUnpacker.h"
#include "CDDASAnalyzer.h"
#include <FragmentIndex.h>
#include "zmqwritethread.h"

#include <CRingItem.h>
#include <DataFormat.h>
#include <fragment.h>

#include <string.h>
#include <vector>
#include <iostream>

/**
 * constructor
 *   @param sock - reference to the ZMQ socket on which to output the data.
 */


CZMQRingOutputter::CZMQRingOutputter(zmq::socket_t& sock) :
    m_socket(sock)
{   
}

/**
 * end
 *    Send an end message:
 *
 *  @param id - node id ignored.
 */
void
CZMQRingOutputter::end(int id) {
    sendEnd(m_socket, id);
}

/**
 * outputItem
 *     Outputs a result item.
 *
 *  @param id -- thread id (we plug this in as the data source in the body header).
 *  @param pItem - Actually a  pointer to CDDASAnalyzer::outData
 *                 the payload part of this is a pointer to an std::vector<FragmentInfo>
 *                 Each fragment is either an unmodified DDAS hit without a trace or
 *                 a DDAS hit with a trace to which a HitExtension has been appended.
 *  @note Hits that have traces and a hit extension must be free'd as they were
 *        originally malloced.
 *
 *
 *  What we write to file is a ring item that looks like it came from the
 *  event builder.  id is our sourceid, the timestamp comes from pItem,
 *  The body has the usual total byte count and then the fragments.
 */
void
CZMQRingOutputter::outputItem(int id, void* pData)
{
    // First the pData looks like a DDASAnalyzer::outData struct:
    
    CDDASAnalyzer::outData* pRawEvent =
        reinterpret_cast<CDDASAnalyzer::outData*>(pData);
    
    // The payload is a pointer to a vector of FragmentInfo.  Note that
    // frag.s_itemhdr and frag.s_itembody will, if there's a waveform,
    // be dynamic and frag.s_itemhdr will have to be free(3)ed when we're done
    // with it.
    
    std::vector<FragmentInfo>& fragments(
        *reinterpret_cast<std::vector<FragmentInfo>* >(pRawEvent->payload)
    );
    // Size the fragment bodies.  From that we figure out the size
    // of the ring item....with slop because I'm cautious.
    // This initial size represents a ring item with a body header and that
    // total size longword.
    //
    size_t totalSize =
        sizeof(uint32_t) + sizeof(RingItemHeader) + sizeof(BodyHeader);
    uint32_t bodySize(sizeof(uint32_t));
    for (int i = 0; i < fragments.size(); i++) {
        uint32_t itemSize = fragments[i].s_size;
        totalSize += itemSize;;           // I think this is all inclusive.
        totalSize += sizeof(EVB::FragmentHeader);
        bodySize += itemSize  + sizeof(EVB::FragmentHeader);
    }
    
    totalSize += 100;                             // Slop.
    
    // This rigmarole is because the data are sent by pointer to the writer.
    // and deleted by it.
    
    CRingItem* pOutputItem =
        new CRingItem(PHYSICS_EVENT, pRawEvent->timestamp, id, 0, totalSize);
    CRingItem& outputItem(*pOutputItem);
    
    void* p = outputItem.getBodyCursor();   // Allows for body header...
    
    // Put in the total item size .. this is bodySize
    
    memcpy(p, &bodySize, sizeof(uint32_t));
    p = reinterpret_cast<void*>(reinterpret_cast<uint32_t*>(p) + 1);
    
    // put each fragment into the ring. I _think_ s_size of the fragment info
    // includes the fragment header and ring item and all that shit.
    // Fragments that have waveforms, need to have their storage freed as they
    // were dynamically allocated.  We'll do that if the fragment throws an
    // exception when parsed by the DDASHitUnpacker (since it'll think the length)
    // is inconsistent.
    //
    for (int i =0; i < fragments.size(); i++) {
        // Fragment header
        
        EVB::FragmentHeader h;
        h.s_timestamp = fragments[i].s_timestamp;
        h.s_sourceId = fragments[i].s_sourceId;
        h.s_barrier  = fragments[i].s_barrier;
        h.s_size     = fragments[i].s_size;
        
        memcpy(p, &h, sizeof(EVB::FragmentHeader));
        p  = reinterpret_cast<void*>(
            reinterpret_cast<uint8_t*>(p) + sizeof(EVB::FragmentHeader)
        );
        
        // Fragment ring item:
        memcpy(p, fragments[i].s_itemhdr, fragments[i].s_size); // I think that's right.
        p = reinterpret_cast<void*>(
            reinterpret_cast<uint8_t*>(p) + fragments[i].s_size
        );
        deleteIfNeeded(fragments[i]);
    }
    // At this point the ring item just needs to have its pointers updated
    // which updates the header too:
    
    
    outputItem.setBodyCursor(p);     
    outputItem.updateSize();         // Get the size right
    sendRingItem(m_socket, id, pOutputItem);
}
/**
 * deleteIfNeeded
 *    If the ring item contained a fit we need to free it as it was dynamically
 *    allocated.  This is done in a bit of a dirty way;   If we added an
 *    extension to this, the the DDASHitUnpacker will think this item has
 *    a bad size and throw an std::runtime_error -- we'll catch that and
 *    free the dynamic storage in the cathc handler.
 * @param item - FragmentInfo for the item.
 */
void
CZMQRingOutputter::deleteIfNeeded(FragmentInfo& info)
{
   
    DAQ::DDAS::FitHitUnpacker unpacker;
    DAQ::DDAS::DDASFitHit     hit;
    // Figure out the body start and end+1 pointers:
    
    uint32_t* pStart = reinterpret_cast<uint32_t*>(info.s_itemhdr);

    try {
        unpacker.decode(pStart, hit);
        if(hit.hasExtension()) free(info.s_itemhdr);
    }
    catch (...) {
        std::cerr << "Fileoutputter caught exception\n";
    }
}