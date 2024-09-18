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
 * @file  DDASRootFitEvent.h
 * @brief Header for events that are writable to ROOT trees.
 */

#ifndef DDASROOTFITEVENT_H
#define DDASROOTFITEVENT_H

#include <TObject.h> // ROOT base class

#include <vector>

/** @namespace ddastoys */
namespace ddastoys {
    
    /**
     * @ingroup ddasrootfit
     * @{
     */

    /**
     * @class DDASRootFitEvent
     * @brief Defines the object that's put in a ROOT TTree for each event. 
     *
     * @details
     * An event is just a sequence (vector) of hits. Hits are stored here as 
     * pointers to DDASRootFitHit objects which are hits that are also derived 
     * from TObject.
     *
     * @note The code assumes that the pointers in m_hits point to dynamically
     * allocated data. Thus the destructor will destroy the hits as well.
     * @note Copy construction and assignment are implemented as deep operations
     * as demanded by ROOT.
     */
    class DDASRootFitHit;

    class DDASRootFitEvent : public TObject
    {
    public:
	std::vector<DDASRootFitHit*> m_hits; //!< An event is a vector of hits.
    
	// Canonical methods:
  
    public:
	/** @brief Constructor. */
	DDASRootFitEvent();
	/**
	 * @brief Copy constructor.
	 * @param rhs The object we're being copied from.
	 */
	DDASRootFitEvent(const DDASRootFitEvent& rhs);
	/** @brief Destructor. */
	~DDASRootFitEvent();

	/**
	 * @brief Assignment operator.
	 * @param rhs The object we're assigning to *this.
	 * @return Referencing *this.
	 */
	DDASRootFitEvent& operator=(const DDASRootFitEvent& rhs);
    
	// Selectors; Some of these are provided because maybe ROOT analysis
	// needs them?
  
    public:
	/**
	 * @brief Return the event vector.
	 * @return Vector of references to hit objects.
	 */
	std::vector<DDASRootFitHit*> GetData();
	/**
	 * @brief Get the time from the first hit in the event.
	 * @return Double_t The time from the first hit of data.
	 * @retval 0.0 Returned if there are no hits.
	 */
	Double_t GetFirstTime() const;
	/**
	 * @brief Get the time from the last hit in the event.
	 * @return Double_t The time from the last hit of data.
	 * @retval 0.0 Returned if there are no hits.
	 */
	Double_t GetLastTime() const;
	/**
	 * @brief Get the time from the last hit in the event.
	 * @return Double_t The time from the last hit of data.
	 * @retval 0.0 Returned if there are no hits.
	 */
	Double_t GetTimeWidth() const;
    
	// Operations:
  
    public:
	/**
	 * @brief Add a hit to the event.
	 * @param hit References the hit to add.
	 * @note The hit is copied into a dynamically allocated hit to ensure 
	 *   all hit objects in the vector are dynamic.
	 */    
	void AddHit(const DDASRootFitHit& hit);
	/** @brief Delete the hits and empty the array of hits. */
	void Reset();
  
	// ROOT needs this macro:
    
	ClassDef(DDASRootFitEvent, 1)
    };

    /** @} */
}

#endif
