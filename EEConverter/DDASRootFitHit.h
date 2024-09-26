/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins
	     Aaron Chester
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** 
 * @file  DDASRootFitHit.h
 * @brief Define a hit with fitting data that can serialized by ROOT.
 */

#ifndef DDASROOTFITHIT_H
#define DDASROOTFITHIT_H

#include <DDASFitHit.h>
#include <TObject.h>

/** @namespace ddastoys */
namespace ddastoys {

    /**
     * @defgroup ddasrootfitformat libDDASRootFitFormat.so
     * @brief ROOT data format and classes. Basically a ROOT-ized DDASFitHit 
     * and fit_extensions.h.
     */

    /**
     * @ingroup ddasrootfitformat
     * @{
     */

    /**
     * @class DDASRootFitHit
     * @brief This class is a ROOT serializable DDAS hit with possible fit data.
     * @details
     * This is a very, very simple class. It inherits from ddastoys::DDASFitHit
     * and ROOT's TObject class and adds a ROOT `ClassDef()` macro.
     */
    
    class DDASRootFitHit : public DDASFitHit, public TObject
    {
    public:
	/** @brief Constructor. */
	DDASRootFitHit();
	/**
	 * @brief Copy constructor.
	 * @param rhs Reference the hit to copy-construct.
	 */
	DDASRootFitHit(const DDASRootFitHit& rhs);
    
	/**
	 * @brief Assignment from another DDASRootFitHit.
	 * @param rhs The hit we are assigning from.
	 * @return *this.
	 */
	DDASRootFitHit& operator=(const DDASRootFitHit& rhs);
	/**
	 * @brief Assignment from a DAQ::DDAS::DDASFitHit.
	 * @param rhs The hit we're assigning from.
	 * @return *this.
	 * @note This is the "normal" way we'll get data. Default construction 
	 *   and then assignment from a decoded DDASFitHit.
	 */
	DDASRootFitHit& operator=(const DDASFitHit& rhs);

	/**
	 * @brief Reset the object to empty. This is really just a matter of 
	 * assigning a reset hit to ourselves.
	 */
	void Reset();
  
	// Tell ROOT we're implementing the class:
  
	ClassDef(DDASRootFitHit, 1)
    };

/** @} */
}

#endif
