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

/** @file:  DDASRootFitEvent.h
 *  @brief: Header for events that are writable to root trees.
 */

#ifndef DDASROOTFITEVENT_H
#define DDASROOTFITEVENT_H

#include <TOBject.h>                // Root base class
#include <vector>

/**
 * @class DDASRootFithit
 *     This is the object that's put in a Root Tree for each event.
 *     An event, however is just a sequencde (vector) of hits.  Hits
 *     are stored here as pointers to DDASRootFitHit objects which are
 *     hits that are also derived from TObject.
 *
 *  @note - The code assumes that the pointers in m_hits point to dynamically
 *          allocated data.  Thus the destructor will destroy the hits as well.
 *  @note - Copy construction and assignment are implemented as deep operations
 *          as demanded by Root.
 */
class DDASRootFitHit;

class DDASRootFitEvent : public TObject
{
private:
    std::vector<DDASRootFitHit*> m_hits;     // An event is a vector of hits.
    
    // canonical methods:
public:
    DDASRootFitEvent();
    DDASRootFitEvent(const DDASRootFitEvent& rhs);
    ~DDASRootFitEvent();
    
    DDASRootFitEvent& operator=(const DDASRootFitEvent& rhs);
    
    // Selectors; Some of these are provided  because maybe root analysis
    // needs them?
    
public:
    std::vector<DDASRootFitHit*> GetData();
    Double_t GetFirstTime() const;
    Double_t GetLastTime() const;
    Double_t GetTimeWidth() const;
    
    //Operations:
    
public:
    void AddHit(const DDASRootFitHit& hit);   // Allocates copies and appends.
    void Reset();                             // Clear the vector, freeing the hits.
    
    
    // Root needs this macro:
    
    ClassDef(DDASRootFitEvent, 1);
};


#endif