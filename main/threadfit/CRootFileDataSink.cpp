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

/** @file:  CRootFileDataSink.cpp
 *  @brief: Implement the root file data sink for DDAS fit events.
 */
#include "CRootFileDataSink.h"
#include "DDASRootFitEvent.h"
#include "DDASRootFitHit.h"
#include <TTree.h>
#include <TFile.h>
#include <TDirectory.h>
#include <CRingItem.h>
#include <CRingItemFactory.h>
#include <FragmentIndex.h>

#include <iostream>

/**
 * constructor
 *    @param filename - Root file to open. 
 *    @param treename - Name of the tree to create in the root file.
 *                      The tree name defaults to "DDASFit" if not provided.
 *
 *  @note  We're going to make this sink so it can be used in other program.s
 *         That implies presrving root's concept of a current working directory
 *         across our operation.
 *  @note  If the file already exists, we're going to open for update so our
 *         changes will get merged into that existing file.
 */
CRootFileDataSink::CRootFileDataSink(const char* filename, const char* treename) :
    m_TreeEvent(0),
    m_tree(0),
    m_file(0),
    m_warnedPutUsed(false)
{
    std::string oldDir = gDirectory->GetPath();
    gDirectory->Cd("/");              // tack this file onto root.
    try {
        m_file = new TFile(filename, "UPDATE");   // Sets the default dir.
        m_TreeEvent = new DDASRootFitEvent();
        m_tree      = new TTree(treename, treename);
        m_tree->Branch("DDASHits", m_TreeEvent);
        
        gDirectory->Cd(oldDir.c_str());           // Restor the directory.
        
    } catch (...) {
        delete m_TreeEvent;             // Clean up and throw.
        delete m_tree;
        delete m_file;
        gDirectory->Cd(oldDir.c_str());  // Back to original cd.
        throw;                           // propagate the error.
    }
}
/**
 * destructor
 *    Flush the stuff to file and delete all the dynamic components.
 */
CRootFileDataSink::~CRootFileDataSink()
{
    m_file->Write();                     // Flush.
    delete m_tree;
    delete m_file;
    delete m_TreeEvent;
}

/**
 * putItem
 *    Put a ring item to file.  The ring item is assumed to consist of a set
 *    of fragments.  Each fragment contains a hit.  The hits are decoded
 *    and added to the tree event.  Once that's done we can fill the tree
 *    delete any dynamic storage we got.
 *
 * @param item - reference to a ring item object.
 */
void
CRootFileDataSink::putItem(const CRingItem& item)
{
    m_TreeEvent->Reset();                    // Free dynamic hist from last event.
    
    // Bust the ring item up into event builder fragments.
    
    FragmentIndex frags(reinterpret_cast<uint16_t*>(item.getBodyPointer()));
    
    // Decode the DDAS hit in each fragment and add it to the event.
    // Note that AddHit does a copy construction of the hit into new storage.
    
    for(int i = 0; i < frags.getNumberFragments(); i++) {
        DDASRootFitHit hit;
        hit.UnpackChannelData(frags.getFragment(i).s_itemhdr);
        m_TreeEvent->AddHit(hit);
    }
    // Fill the tree now that we have all the hits marshalled:
    
    m_tree->Fill();    
}
/**
 * put
 *    Called to put arbitrary data to the file.  We really don't know how to do
 *    this so:
 *    - First time we're called we'll emit a warning that users shouldn't really
 *      do this.
 *    - We'll treat the data pointer as a pointer to a raw ring item, turn it
 *      into a CRingItem and call putItem.
 *
 * @param pData  - pointer to the data.
 * @param nBytes - number of bytes of data to put; actually ignored.
 */
void
CRootFileDataSink::put(const void* pData, size_t nBytes)
{
    if (!m_warnedPutUsed) {
        m_warnedPutUsed = true;
        std::cerr << "***WARNING*** CRootFileDataSink::put was called\n";
        std::cerr << "  Normally you should use PutItem to translate and put ring items\n";
        std::cerr << "containing DDAS hits that potentially have fits.\n";
        std::cerr << "We'll treat this as an attempt to output a raw ring item\n";
        std::cerr << "If that's not the case this can fail spectacularly\n";
        std::cerr << " YOU HAVE BEEN WARNED - be sure your code is right\n";
    }
    
    void* p = const_cast<void*>(pData);
    CRingItem* pItem =  CRingItemFactory::createRingItem(p);
    putItem(*pItem);
    delete pItem;
}