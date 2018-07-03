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

/** @file:  CRootSelectableDataSink.cpp
 *  @brief: Implementation of a tree sink that root can make selectors for.
 */

#include "CRootSelectableDataSink.h"
#include <CRingItem.h>
#include <CRingItemFactory.h>
#include "DDASRootFitEvent.h"
#include "DDASRootFitHit.h"
#include <TTree.h>
#include <TFile.h>
#include <TDirectory.h>
#include <FragmentIndex.h>
#include "FitHitUnpacker.h"

#include <iostream>
/**
 * constructor
 *   Create the TFile, the TTree, the buffers link them all together
 *   and set the m_warnedPutUsed flag to false.
 *
 * @param filename - name of the file the data will be written to.
 * @param treename - Optional name of the root tree we'll write in that
 *                   file.  This defaults to "DDASFit".
 * @note - the Root default directory will be unchanged after this
 *         constructor completes normally.
 * @note - the tree title will be the same as its name.
 */
CRootSelectableDataSink::CRootSelectableDataSink(
    const char* filename, const char* treename
) : m_RawEvent(0), m_fits(0), m_tree(0), m_file(0), m_warnedPutUsed(false)
    
{
    std::string oldDir = gDirectory->GetPath();    // Save old dir.
    gDirectory->Cd("/");
    try {
        m_file = new TFile(filename, "RECREATE");
        m_RawEvent = new DDASRootFitEvent;
        m_fits      = new fit;
        
        m_tree = new TTree(treename, treename);
        m_tree->Branch("RawHits", m_RawEvent);
        m_tree->Branch("HitFits", m_fits);
    }
    catch(...) {
        // Put it all back the way it was but re-throw the exception
        
        delete m_RawEvent;
        delete m_fits;
        delete m_tree;
        delete m_file;
        
        gDirectory->Cd(oldDir.c_str());
        
        throw;
    }
}
/** Destructor
 *     Delete all the crap we made.
 */
CRootSelectableDataSink::~CRootSelectableDataSink()
{
    m_file->Write();            // Flush buffers.

    delete m_tree;
    delete m_file;
    delete m_RawEvent;
    delete m_fits;
}
/**
 * putItem
 *    Put a ring item into the output file as tree leaves.
 *    The raw event is put in the RawHits branch while the fit
 *    information is put in the m_fits branch.
 *
 * @param item - reference to the ring itemt to output.
 */
void
CRootSelectableDataSink::putItem(const CRingItem& item)
{
    m_RawEvent->Reset();              // Clear its vectors etc.
    m_fits->clear();                  // Same.
    
    // Iterate over the fragments in the event.  Assume each fragment is
    // a DDAS::DDAS::DDASFitHit
    
    DDASRootFitHit        rootHit;
    DAQ::DDAS::DDASFitHit fitHit;
    
    FragmentIndex frags(reinterpret_cast<uint16_t*>(item.getBodyPointer()));
    DAQ::DDAS::FitHitUnpacker unpacker;
    size_t nFrags = frags.getNumberFragments();
    
    for (int i = 0; i < nFrags; i++) {
        rootHit.Reset();
        fitHit.Reset();
        
        unpacker.decode(frags.getFragment(i).s_itemhdr, fitHit);
        rootHit = fitHit;
        
        m_RawEvent->AddHit(rootHit);
        
        // Add the extension/fit:
        
        RootHitExtension ext;
        if (fitHit.hasExtension()) {
            ext = fitHit.getExtension();
        }
        m_fits->addFit(ext);
    }
    // Now that both branch buffers have been stocked, fill the tree.
    
    m_tree->Fill();
    
}
/**
 * put
 *    Called to put arbitrary data to file.  We really only know how to put
 *    ring items to file.  Therefore, the first time we warn this is not
 *    a recommended method.  We then try to construct a ring item
 *    and pass that to putItem.
 *
 * @param pData  -  Pointer to the data.
 * @param nBytes - Number of bytes of data (ignored).
 */
void
CRootSelectableDataSink::put(const void* pData, size_t nBytes)
{
    if (!m_warnedPutUsed) {
        m_warnedPutUsed = true;
        std::cerr << "***WARNING*** CRootSelectableDataSink::put was called\n";
        std::cerr << "  Normally you should use PutItem to translate and put ring items\n";
        std::cerr << "containing DDAS hits that potentially have fits.\n";
        std::cerr << "We'll treat this as an attempt to output a raw ring item\n";
        std::cerr << "If that's not the case this can fail spectacularly\n";
        std::cerr << " YOU HAVE BEEN WARNED - be sure your code is right\n";
    }
    
    // See if pData can be put as a ring item.
    
    void* p = const_cast<void*>(pData);    // Factory does not expect const.
    CRingItem* pItem = CRingItemFactory::createRingItem(p);
    putItem(*pItem);
    delete pItem;
}