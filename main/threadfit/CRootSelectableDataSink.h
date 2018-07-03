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

/** @file:  CRootSelectableDataSink.h
 *  @brief: Support for root output format that can do selectors.
 */

#ifndef CROOTSELECTABLEDATASINK_H
#define CROOTSELECTABLEDATASINK_H
#include <CDataSink.h>
#include "rootTreeClasses.h"

class CRingItemn;
class DDASRootFitEvent;
class TTree;
class TFile;

/**
 * @class CRootSelectableDataSink
 *
 *     The love hate relationship with CERN ROOT continues.  CRootFileDataSink
 *     produces what ought to be perfectly good root trees.  These trees
 *     can be 'played back' and iterated over just fine.  Unfortunately,
 *     ROOT balks at making selectors for those trees.
 *     After much trial and error, Giordano and I found that the root
 *     branch format this class will write works for that.
 *
 *     Therefore this class provides the option to write rootselectable
 *     trees.
 *
 *     If I was smarter, I oculd probably write this in a way that
 *     factors out much of the common code between this class anbd
 *     CRootFileDataSink -- but I'm not so phui.
 */
class CRootSelectableDataSink : public CDataSink
{
private:
    DDASRootFitEvent*    m_RawEvent;    // Buffer for raw event branch.
    fit*                 m_fits;        // Fits for the hits.
    
    TTree*               m_tree;        // We write this tree.
    TFile*               m_file;        // We write to this file.
    
    // The put method cannot be used in this class.  If it it we warn
    // exactly once and do try to interpret pData as a raw ring item.
    
    bool                m_warnedPutUsed;
    
    // Canonicals:
public:
    CRootSelectableDataSink(const char* filename, const char* treename="DDASFit");
    virtual ~CRootSelectableDatSink();
    
    // CDataSink mandatory interface:
    
public:
    virtual void putItem(const CRingItem& item);
    virtual void put(const void* pData size_t nBytes);
};


#endif