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

/** 
 * @file  RootFileDataSink.cpp
 * @brief Implement the ROOT file data sink for DDAS fit events.
 */

#include "RootFileDataSink.h"

#include <iostream>

#include <CRingItem.h>
#include <CRingItemFactory.h>
#include <FragmentIndex.h>
#include <TTree.h>
#include <TFile.h>
#include <TDirectory.h>
 
#include "DDASRootFitEvent.h"
#include "DDASRootFitHit.h"
#include "DDASFitHitUnpacker.h"
#include "RootExtensions.h"
#include "DDASFitHit.h"

static const Int_t BUFFERSIZE(1024*1024); // 1 MB

/**
 * @details
 * We're going to make this sink so it can be used in other programs. That 
 * implies presrving ROOT's concept of a current working directory across 
 * our operation.
 */
RootFileDataSink::RootFileDataSink(const char* filename, const char* treename) :
    m_pUnpacker(new DAQ::DDAS::DDASFitHitUnpacker), m_pTreeEvent(nullptr),
    m_pTree(nullptr), m_pFile(nullptr), m_warnedPutUsed(false)
{  
    const char* oldDir = gDirectory->GetPath();
    gDirectory->Cd("/"); // Have to start somewhere
    
    try {
	m_pFile = new TFile(filename, "RECREATE"); // Sets the default directory
	m_pTree = new TTree(treename, treename);
	m_pTreeEvent = new DDASRootFitEvent();        
	m_pTree->Branch("RawHits", m_pTreeEvent, BUFFERSIZE);
	//m_pTree->Branch("HitFits", &m_extensions, BUFFERSIZE);
	gDirectory->Cd(oldDir); // Restore the directory
        
    } catch (...) {
	delete m_pTreeEvent; // Clean up and throw
	delete m_pTree;
	delete m_pFile;
	gDirectory->Cd(oldDir); // Back to original directory
	throw; // Propagate the error
    }
}

/**
 * @details
 * Flush the stuff to file and delete all the dynamic components.
 */
RootFileDataSink::~RootFileDataSink()
{
    m_pFile->Write();
    delete m_pUnpacker;  
    delete m_pTreeEvent;
    delete m_pTree;
    delete m_pFile; // Deleting the object saves and closes the file
}

/**
 * @details
 * The ring item is assumed to consist of a set of fragments. Each fragment
 * contains a hit. The hits are decoded and added to the tree event. Once 
 * that's done we can fill the tree and delete any dynamic storage we got.
 */
void
RootFileDataSink::putItem(const CRingItem& item)
{
    m_pTreeEvent->Reset(); // Free dynamic hist from last event
    //m_extensions.clear();
    
    // Bust the ring item up into event builder fragments:
    
    FragmentIndex frags(
	reinterpret_cast<std::uint16_t*>(item.getBodyPointer())
	);
    
    // Decode the DDAS hit in each fragment and add it to the event. Note that
    // AddHit does a copy construction of the hit into new storage.

    DAQ::DDAS::DDASFitHit fitHit;
    DDASRootFitHit rootFitHit;
    for (unsigned i = 0; i < frags.getNumberFragments(); i++) {
	fitHit.Reset();
	rootFitHit.Reset();
	m_pUnpacker->decode(frags.getFragment(i).s_itemhdr, fitHit);
	rootFitHit = fitHit; // The base part
	m_pTreeEvent->AddHit(rootFitHit);

	// Check if the hit has an extension (initialized to false). If so,
	// fill it from the fit and set has extension to true.
	
	// RootHitExtension ext;
	// if (fitHit.hasExtension()) {
	//     ext = fitHit.getExtension(); 
	// }
	// m_extensions.push_back(ext); // Add to fit branch
    }
    
    // Fill the tree now that we have all the hits marshalled:
    
    m_pTree->Fill();
}

/**
 * @details
 * We really don't know how to do this so:
 *   - First time we're called we'll emit a warning that users shouldn't really
 *     do this.
 *   - We'll treat the data pointer as a pointer to a raw ring item, turn it
 *     into a CRingItem and call putItem.
 */
void
RootFileDataSink::put(const void* pData, size_t nBytes)
{
    if (!m_warnedPutUsed) {
	m_warnedPutUsed = true;
	std::cerr << "***WARNING*** RootFileDataSink::put was called.\n";
	std::cerr << "You should use putItem to translate and put ring items\n";
	std::cerr << "containing DDAS hits that potentially have fits.\n";
	std::cerr << "We'll treat this as an attempt to output a raw ring item\n";
	std::cerr << "If that's not the case this can fail spectacularly.\n";
	std::cerr << "YOU HAVE BEEN WARNED: be sure your code is right!\n";
    }
  
    void* p = const_cast<void*>(pData);
    CRingItem* pItem =  CRingItemFactory::createRingItem(p);
    putItem(*pItem);
    delete pItem;  
}
