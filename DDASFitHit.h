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
 * @file  DDASFitHit.h
 * @brief Extends DDASHit to include fit data that's been tacked on to the
 * end of a hit with traces.
 */

#ifndef DDASFITHIT_H
#define DDASFITHIT_H

#include <DDASHit.h> // Base class

#include <stdexcept>

#include "fit_extensions.h" // Defines HitExtension.

namespace DAQ {
    namespace DDAS {       

	/**
	 * @class DDASFitHit
	 * @brief Encapsulates data for DDAS hits that may have fitted traces.
	 *
	 * @details
	 * This is produced by FitHitUnpacker::decode. This is basically just 
	 * a DDASHit with extra fields.     
	 */

	class DDASFitHit : public DDASHit
	{
	private:
	    bool m_haveExtension; //!< True iff has extension data.
	    ::DDAS::HitExtension m_extension; //!< The extension data.
      
	public:
	    DDASFitHit() { Reset(); } //!< Constructor.
	    virtual ~DDASFitHit() {} //!< Destructor.

	    /** @brief Reset the hit information. */
	    void Reset() {
		m_haveExtension = false;
		DDASHit::Reset(); // Reset base class membrers.
	    }
	    /** 
	     * @brief Set the hit extension information for this hit. 
	     * @param extension Reference to the extension for this hit.
	     */
	    void setExtension(const ::DDAS::HitExtension& extension) {
		m_extension = extension;
		m_haveExtension = true;
	    }
	    /**
	     * @brief Check whether hit has a fit extension.
	     * @return True if the hit contains an extension, false otherwise.
	     */
	    bool hasExtension() const {return m_haveExtension;}
	    /** 
	     * @brief Get the extension data from the current hit.
	     * @throw std::logic_error If the hit does not contain an extension.
	     * @return Reference to the extension of the current hit.
	     */      
	    const ::DDAS::HitExtension& getExtension() const {
		if (m_haveExtension) {
		    return m_extension;
		} else {
		    throw std::logic_error(
			"Asked for extension for event with none"
			);
		}
	    }   
	};
    } // DDAS
} // DAQ

#endif
