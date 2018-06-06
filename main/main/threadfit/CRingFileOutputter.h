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

/** @file:  CRingFileOutputter.h
 *  @brief: Outputter for CDDASAnalyzer that writes ring items.
 */
#ifndef CRINGFILEOUTPUTTER_H
#define CRINGFILEOUTPUTTER_H
#include "Outputter.h"            // Base class.
#include <CFileDataSink.h>
#include <FragmentIndex.h>

/**
 *  @class CRingFileOutputter
 *     This outputter writes ring items produced by CDDASAnalyzer to file.
 *     it's not terribly efficient as it's not blocked; it uses a CFileDataSink
 *     rather than buffered/blocked data.
 *
 *     To improve efficiency we could produce a new type of data sink that
 *     is blocked and substitute that -- that would be pretty easy you'd think.
 *
 *  @note - this and all ddas analyzer outputters must deal with the fact that
 *          the bodies of items that have fits must be deleted.  See
 *          the implementation of deleteIfNeeded
 *
 *  @todo deleteIfNeeded could/should be hosted into a protected base class method.
 *        .. or event a utility package.
 */

class CRingFileOutputter : public Outputter
{
private:
    CFileDataSink  m_sink;
    
public:
    CRingFileOutputter(const char* filename);
    virtual void outputItem(int id, void* pItem);
    virtual void end(int id);
private:
    void deleteIfNeeded(FragmentInfo& frag);
};


#endif