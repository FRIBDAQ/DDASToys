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

/** @file:  CRootFileDataSink.h
 *  @brief: Define a CDataSink that writes out result ring items as Root files.
 */
#ifndef CROOTFILEDATASINK_H
#include <CDataSink.h>                 // base class

class CRingItem;
class DDASRootFitEvent;               // Holds the decoded event for output.
class TTree;                          // We write to a root tree 
class TFile;                          // Located in this file.

/**
 * @CRootFileDataSink
 *    This class knows how to write root files from the ring items created by
 *    the fitting program.   Since it's a data sink, it can just be
 *    used as the data sink for the SortingOutputter the output thread uses.
 *
 *  @note - put is not intended to be used by this file.  If it's used,
 *          a warning will be output to stderr.  pData will then be treated
 *          as a raw ring item, turned into a CRingItem and putItem will be called.
 *          from then on.
 */
class CRootFileDataSink : public CDataSink
{
private:
    DDASRootFitEvent*    m_TreeEvent;
    TTree*               m_tree;
    TFile*               m_file;
    bool                 m_warnedPutUsed;
public:
    CRootFileDataSink(const char* filename, const char* treename="DDASFit");
    virtual ~CRootFileDataSink();
public:
    virtual void putItem(const CRingItem& item);
    virtual void put(const void* pData, size_t nBytes);
};



#endif