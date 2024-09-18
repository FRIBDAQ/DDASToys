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
 * @file  DDASRootFitEvent.cpp
 * @brief Implement the methods of DDASRootFitHit and signal the 
 * implementation to ROOT.
 */

#include "DDASRootFitEvent.h"

#include "DDASRootFitHit.h"

/**
 * @details
 * Pretty much null but ROOT requires an implementation.
 */
ddastoys::DDASRootFitEvent::DDASRootFitEvent() : TObject()
{}

ddastoys::DDASRootFitEvent::DDASRootFitEvent(const DDASRootFitEvent& rhs) : TObject()
{
    *this = rhs;
}

/**
 * @details
 * Destroy the hits. Note that the event vector destroys itself.
 */
ddastoys::DDASRootFitEvent::~DDASRootFitEvent()
{
    Reset();
}

/**
 * @details
 * Assignment will:
 *   - Kill off our vector of hits.
 *   - For each hit on the right hand side, dynamically copy construct a new
 *     one and insert it into our vector.
 */
ddastoys::DDASRootFitEvent&
ddastoys::DDASRootFitEvent::operator=(const ddastoys::DDASRootFitEvent& rhs)
{
    if (this != &rhs) {
	Reset(); // Kill off any old stuff.
	TObject::operator=(rhs); // Base class elements.
	for(unsigned i=0; i<rhs.m_hits.size(); i++) {
	    AddHit(*rhs.m_hits[i]);            
	}
    }
    return *this;
}

std::vector<ddastoys::DDASRootFitHit*>
ddastoys::DDASRootFitEvent::GetData()
{
    return m_hits;
}

Double_t
ddastoys::DDASRootFitEvent::GetFirstTime() const
{
    if (!m_hits.empty()) {
	return m_hits.front()->getTime();
    } else {
	return 0.0;
    }
}

Double_t
ddastoys::DDASRootFitEvent::GetLastTime() const
{
    if (!m_hits.empty()) {
	return m_hits.back()->getTime();
    } else {
	return 0.0;
    }
}

Double_t
ddastoys::DDASRootFitEvent::GetTimeWidth() const
{
    return GetLastTime() - GetFirstTime();
}

void
ddastoys::DDASRootFitEvent::AddHit(const DDASRootFitHit& hit)
{
    DDASRootFitHit* pHit = new DDASRootFitHit(hit);
    m_hits.push_back(pHit);
}

void
ddastoys::DDASRootFitEvent::Reset()
{
    for (unsigned i=0; i<m_hits.size(); i++) {
	m_hits[i]->Reset();
	delete m_hits[i];
    }  
    m_hits.clear();
}
