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

/** @file:  DDASFileOutputter.cpp
 *  @brief: Output data to file from ddas analyzer.
 */

#include "DDASFileOutputter.h"
#include "CDDASAnalyzer.h"           // Need access to output data structure.
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <FragmentIndex.h>
#include <fragment.h>

/**
 * constructor
 *    Open the file and create a new buffer.  Errors are, by necessity
 *    via exception throws.
 *
 *    @param filename   - name of the file to open.
 *    @param bufferSize - size of largest write to perform.
 */
DDASFileOutputter::DDASFileOutputter(const char* filename, size_t bufferSize) :
    m_nFd(-1), m_pBuffer(nullptr), m_pBufferCursor(nullptr),
    m_nBufferSize(bufferSize), m_nBytesUsed(0)
{
    // Try to open the file.
    
    m_nFd = open(filename, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP);
    if (m_nFd < 0) {
        throwErrno("Unable to open the output file");
    }
    
    // Create the initial buffer.
    
    newBuffer();
}
/**
 * destructor
 *   - If there's data in the buffer flush it.
 *   - If the file is open close it...
 *   - delete the buffer.
 */
DDASFileOutputter::~DDASFileOutputter()
{
    
    // If the file is open flush any data and close the file.
    
    if (m_nFd > 0) {                           
        if(m_nBytesUsed) flushBuffer();
        close(m_nFd);
    }
    // Free storage
    
    free(m_pBuffer);

}

/**
 * outputItem
 *    Output a new item to buffer/file.
 *
 *  @param id source of the item (e.g. thread that produced it).
 *  @param pItem - pointer to the item.
 *
 *  Errors are reported via exceptions.
 */
void
DDASFileOutputter::outputItem(int id, void* pItem)
{
    // Only do anything if the file is still open (end not yet called).
    
    if (m_nFd > 0) {
        size_t dataSize = eventSize(pItem);
        if (!fits(dataSize) ) {
            flushBuffer();
            newBuffer();
        }
        emplaceEvent(id, dataSize, pItem);
    } else {
        throw std::logic_error("Attempting to output and item after end of data");
    }
    
}

/**
 * end
 *    Called when there won't be any more data:
 *    - Flush the buffer.
 *    - Reset the buffer jut for good form's sake.
 *    - Close the file.
 *
 * @param id - source of the data (unused by this implementation)
 */
void
DDASFileOutputter::end(int id)
{
    if (m_nBytesUsed) flushBuffer();
    newBuffer();
    
    close(m_nFd);
    m_nFd = -1;            // ensure errors even if that fd is reused elsewhere.
}

/*------------------------------------------------------------------------------
 *  Private methods.
 */
/**
 * newBuffer
 *   Called when a new buffer is required.  Note that this is a misnomer
 *   since the buffer will only be allocated if there isn't one.
 *   Otherwise the cursor and bytes used are just reset.
 */
void
DDASFileOutputter::newBuffer()
{
    // If necessary allocate the buffer.
    
    if (!m_pBuffer) {
        m_pBuffer = malloc(m_nBufferSize);
        if (!m_pBuffer) {
            throwErrno("Unable to allocate the output buffer");
        }
    }
    // Reset the various pointers.
   
    m_pBufferCursor = reinterpret_cast<uint8_t*>(m_pBuffer);
    m_nBytesUsed = 0;
    
}
/**
 * flushBuffer
 *    Write the used part of the output buffer to file.  Note that
 *    -  m_nBytesUsed bytes are written and nothing is written if that's 0.
 *    -  It's possible multiple write(2) calls will be needed, if so that's done.
 *    -  There are recoverable write errors (EINTR, EAGAIN, EWOULDBLOCK).
 *       those are appropriately retried.
 *    -  Errors are signalled via an exception.
 *    -  Normally the caller should follow this with a call to newBuffer().
 */
void
DDASFileOutputter::flushBuffer()
{
    size_t bytesLeft = m_nBytesUsed;
    uint8_t*       p = reinterpret_cast<uint8_t*>(m_pBuffer);
    
    while(bytesLeft) {
        ssize_t bytesWritten = write(m_nFd, p, bytesLeft);
        
        // Handle errors -- including retryable ones:
        
        if (bytesWritten < 0) {
            if ((errno != EINTR) && (errno != EAGAIN) && (errno != EWOULDBLOCK)) {
                throwErrno("Failed to write data to file");
            }
        } else {
            bytesLeft -= bytesWritten;
            p         += bytesWritten;
        }
    }
}
/**
 *  emplaceEvent
 *     Put an event in the buffer at m_pBufferCursor
 *     -  The caller must ensure the event will fit.
 *     -  m_pBuferCursor and m_nBytesUsed are updated after the emplacement.
 *     -  See the header file for the format of the event.
 *
 * @param id - the source of this item - this is written into the event.
 * @param nBytes - number of byte in the event.
 * @param pItem - pointer to the item.. Actually a pointer to a
 *              DDASAnalyzer::outData
 */
void
DDASFileOutputter::emplaceEvent(int id, size_t nBytes, const void* pItem)
{
    size_t initialBytesUsed = m_nBytesUsed;   // for sanity check:
    
    const CDDASAnalyzer::outData* pData =
        reinterpret_cast<const CDDASAnalyzer::outData*>(pItem);
 
    // emplace the event header:
    
    emplaceItem(static_cast<uint32_t>(nBytes));   // probably warns about truncation.
    emplaceItem(static_cast<uint32_t>(id));       // May warn depending on sizeof(int)
    emplaceItem(pData->timestamp);
    
    // The payload member of the output data is actually a pointer to a vector
    // of FragmentInfo items.
    
    const std::vector<FragmentInfo>& frags(
        *(reinterpret_cast<std::vector<FragmentInfo>* >(pData->payload))
    );
    
    for (int i =0; i < frags.size(); i++) {
        emplaceFragment(frags[i]);
    }
    
    // Check that we emplaced as many bytes as we claimed we would:
    
    if ((m_nBytesUsed - initialBytesUsed) != nBytes) {
        throw std::logic_error("emplaceEvent - size consistency check failed!!!");
    }
}
/**
 * template method to emplace a fixed size chunk of data.
 *
 *  @param item the item to emplace.
 *  @note - the cursor and bytes used members are incremented appropriately.
 */
template<typename T>
void
DDASFileOutputter::emplaceItem(const T& item)
{
    size_t n = sizeof(T);
    memcpy(m_pBufferCursor, &item, n);
    
    m_pBufferCursor += n;
    m_nBytesUsed    += n;
}
/**
 * emplaceFragment
 *    Put a fragment of data into the buffer.
 *    We put in a flat fragment as per 7.4.1 of
 *  http://docs.nscl.msu.edu/daq/newsite/nscldaq-11.2/x4509.html
 *
 *  @param pFragment - points to a FragmentInfo struct that describes the fragment.
 *  @note as you might expect, m_nBytesUsed and m_pBufferCursor are adjusted
 *        for the total number of bytes used.
 */
void
DDASFileOutputter::emplaceFragment(const FragmentInfo& pFragment)
{
    // Save a slot for the  bodysize:
    
    size_t initialBytes = m_nBytesUsed;
    uint32_t* pBodySize = reinterpret_cast<uint32_t*>(m_pBufferCursor);
    m_pBufferCursor += sizeof(uint32_t);
    m_nBytesUsed   += sizeof(uint32_t);
    
    // Event header first:
    
    emplaceItem(pFragment.s_timestamp);
    emplaceItem(pFragment.s_sourceId);
    emplaceItem(pFragment.s_size);
    emplaceItem(pFragment.s_barrier);
    
    // Now the item itself:
    
    size_t itemSize = pFragment.s_size;
    memcpy(m_pBufferCursor, pFragment.s_itemhdr, itemSize);
    
    m_pBufferCursor += itemSize;
    m_nBytesUsed    += itemSize;
    
    // Fill in the body size:
    
    *pBodySize = m_nBytesUsed - initialBytes;
}
/**
 * eventSize
 *    Figure out how big an event is..
 *
 *  @param pItem - pointer to the item to emplace.
 *  @return size_t - number of bytes required to serialize the event.
 */
size_t
DDASFileOutputter::eventSize(const void* pItem) const
{
    //  Event header is fixed size, size, timestamp and source id,
    //  and the number of fragment bytes.
    
    size_t result = 3* sizeof(uint32_t) + sizeof(uint64_t);
    
    // Now we need to size the fragments themselves:
    
    const CDDASAnalyzer::outData* pOutData =
        reinterpret_cast<const CDDASAnalyzer::outData*>(pItem);
    const std::vector<FragmentInfo>& fragments =
        *reinterpret_cast<const std::vector<FragmentInfo>* >(pOutData->payload);
        
    // Now we can size each fragment
    
    for (int i =0; i < fragments.size(); i++) {
        // Each fragment has a header consisting of a timestamp, sourceid,
        // payload size, and barrier type.  All are uint32 exceapt the timestamp
        // which is uint64:
        
        result += 3*sizeof(uint32_t) + sizeof(uint64_t);
        
        // the size of the ring item payload is in the fragment size field:
        
        result += fragments[i].s_size;
    }
    
    return result;
}

/**
 * fits
 *    Returns true if we have at least the specified number of bytes
 *    free in the buffer.
 *
 * @param itemSize - size of the item to check.
 * @return bool
 */
bool
DDASFileOutputter::fits(size_t itemSize) const
{
    return (m_nBytesUsed + itemSize) <= m_nBufferSize;
}

/**
 * throwErrno
 *    Throws a runtime_error due to an error that set errno:
 *    This is a lot like perror but the result is an exception
 *    thrown with the error text instead.
 *
 *  @param prefix - prefix text for the exception.
 */
void
DDASFileOutputter::throwErrno(const char* prefix) const
{
    const char* errorString = strerror(errno);
    
    std::string msg(prefix);
    msg += ":";
    msg += errorString;
    
    throw std::runtime_error(msg);
}