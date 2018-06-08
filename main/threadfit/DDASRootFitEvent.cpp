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

/** @file:  DDASRootFitEvent.cpp
 *  @brief: Implement the methods of DDASRootFitHit and signal the impl to root.
 */
#include "DDASRootFitEvent.h"
#include "DDASRootFitHit.h"

// Signal the implementation:


ClassImp(DDASRootFitEvent);

/*-------------------------------------------------------------------------
 * Canonicals implementations
 */

/**
 *  Default constructor Pretty much null but Root requires an implementation.
 */
DDASRootFitEvent::DDASRootFitEvent() : TObject()
{}

/**
 * destructor
 *    Destroy the hits.  Note that the vector destroys itself
 */
DDASRootFitEvent::~DDASRootFitEvent()
{
    Reset();
}

/**
 *  Copy Constructor
 *      @param rhs - the object we're being copied from.
 */
DDASRootFitEvent::DDASRootFitEvent(const DDASRootFitEvent& rhs) : TObject()
{
    *this = rhs;                    // Assigment and copying are about the same.
}
/**
 * operator=
 *    Assignment operator
 *    -  Kill off our vector of hits.
 *    -  For each hit on the right hand side, dynamically copy construct a new
 *       one and insert it into our vector.
 *       
 *  @param rhs - the object we're assigning to *this.
 *  @return DDASRootFitHit& - referencing *this.
 */
DDASRootFitEvent&
DDASRootFitEvent::operator=(const DDASRootFitEvent& rhs)
{
    if (this != &rhs) {
        Reset();                       // Kill off any old stuff.
        TObject::operator=(rhs);       // Base class elements.
        for(int i =0; i < rhs.m_hits.size(); i++) {
            AddHit(*rhs.m_hits[i]);
            
        }
    }
    return *this;
}

/*---------------------------------------------------------------------------
 * Selector implementations
 */

/**
 * GetData
 *    @return std::vector<DDASRootFitHit*> - Returns a reference to the vector of hits.
 */
std::vector<DDASRootFitHit*>
DDASRootFitEvent::GetData()
{
    return m_hits;
}
/**
 * GetFirstTime
 *    @return Double_t return the time from the first hit of data.
 *    @retval 0.0 is returned if there are no hits.
 */
Double_t
DDASRootFitEvent::GetFirstTime() const
{
    if (!m_hits.empty()) {
        return m_hits.front()->GetTime();
    } else {
        return 0.0;
    }
}
/**
 * GetLastTime
 * @return Double_t return the time from the last hit of data.
 * @retval 0.0 is returned if there are no hits.
 */
Double_t
DDASRootFitEvent::GetLastTime() const
{
    if (!m_hits.empty()) {
        return m_hits.back()->GetTime();
    } else {
        return 0.0;
    }
}
/**
 * GetTimeWidth
 *   @return Double_t the range of time covered by the hits in the event.
 */
Double_t
DDASRootFitEvent::GetTimeWidth() const
{
    return GetLastTime() - GetFirstTime();
}

/*---------------------------------------------------------------------------
 *  Operations implementation
 */
/**
 * AddHit
 *    Add a hit to the event.
 *
 *    @param hit - References the hit
 *    @note - the hit is copied into a dynamically allocated hit to ensure
 *            all hit objects in the vector are dynamic.
*/
void
DDASRootFitEvent::AddHit(const DDASRootFitHit& hit)
{
    DDASRootFitHit* pHit = new DDASRootFitHit(hit);
    m_hits.push_back(pHit);
}

/**
 * Reset
 *    Delete the hits and empty the array of hits.
 */
void
DDASRootFitEvent::Reset()
{
    for (int i = 0; i < m_hits.size(); i++) {
        m_hits[i]->Reset();             // Shouldn't need to.
        delete m_hits[i];
    }
    m_hits.clear();
}