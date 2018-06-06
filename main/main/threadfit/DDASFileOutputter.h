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

/** @file:  DDASFileOutputter.h
 *  @brief: This outputter is used to output data items
 */
#ifndef DDASFILEOUTPUTTER_H
#define DDASFILEOUTPUTTER_H

#include "Outputter.h"            // Base class.
#include <stdint.h>
#include <stddef.h>

struct FragmentInfo;

/**
 * @class DDASFileOutputter
 *    This class is responsible for outputting data from the CDDASAnalyer to file.
 *    This is used both for testing the individual classes and for
 *    doing the final output from the farmer.
 *
 * The output file is a sequence of events (only physics events).  The
 * format of each event is:
 *
 *  +-----------------------------------------+
 *  | Total event size  (uint32_t)            |
 *  +-----------------------------------------+
 *  | source id (uint32_t)                    |
 *  +-----------------------------------------+
 *  |  Event timestamp (uint64_t)             |
 *  +-----------------------------------------+
 *  |  total size of fragment data (uint32_t) |
 *  +-----------------------------------------+
 *  |  frag1                                  |
 *      ....
 *
 *   frag n have the format shown in e.g.
 *     http://docs.nscl.msu.edu/daq/newsite/nscldaq-11.2/x4509.html
 *   section 7.4.1
 *
 *   This allows FragmentIndex to unravel the fragments when
 *   pointed at the total size of fragment data.
 *
 *  This looks suspiciously like a ring item with no body header whose
 *  type is the source id.
 *
 *  Blocked I/O is used for performance however each block is 'shortened'
 *  so that there's no dead space and >our< blocks don't do any spanning.
 */
class DDASFileOutputter  : public Outputter
{
private:
    int       m_nFd;
    void*     m_pBuffer;
    uint8_t*  m_pBufferCursor;
    size_t    m_nBufferSize;
    size_t    m_nBytesUsed;

public:
    DDASFileOutputter(const char* filename, size_t bufferSize);
    virtual ~DDASFileOutputter();
    
    // Required interface for a concrete class:
    
    virtual void outputItem(int id, void* pItem); // DDASAnalyzer::outData*.
    virtual void end(int id);
private:
    void newBuffer();
    void flushBuffer();
    void emplaceEvent(int id, size_t nBytes, const void* pItem);
    template<typename T> void emplaceItem(const T& item);
    void emplaceFragment(const FragmentInfo& pFragment);
    size_t eventSize(const void* pItem) const;
    
    bool fits(size_t itemSize) const;
    
    void throwErrno(const char* prefix) const;
    
};

#endif
