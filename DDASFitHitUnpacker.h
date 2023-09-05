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
 * @file  DDASFitHitUnpacker.h
 * @brief Unpack DDAS data where the traces may have a HitExtension
 * that contains one and two pulse fits.
 */

#ifndef DDASFITHITUNPACKER_H
#define DDASFITHITUNPACKER_H

#include <DDASHitUnpacker.h>

// Let's put this sob in the same namespace as the DDASHitUnpacker method.

namespace DAQ {
    namespace DDAS {
	class DDASFitHit;
    
	/**
	 * @class DDASFitHitUnpacker
	 * @brief Unpack raw hit data from DDAS event files.
	 *
	 * @details
	 * DAQ::DDAS::DDASHitUnpacker is capable of unpacking raw hits from 
	 * DDAS files. Typical trace analysis may involve fitting traces to 
	 * one or two pulses This class extends the DDASHitUnpacker class to 
	 * support access to the results of the fit which have been tacked on 
	 * the back end of a hit by some hit extender.
	 */    
	class DDASFitHitUnpacker : public DDASHitUnpacker
	{
	public:
	    /**
	     * @brief Decode the current event and unpack it into a DDASFitHit.
	     * @param p Pointer to the ring item to decode. For an event built
	     *   fragment, this is normally the FragmentInfo's s_itemhdr 
	     *   pointer. Note the difference from DDASHitUnpacker which 
	     *   expects a pointer to the body.
	     * @param hit Hit item that we will unpack data into.
	     * @throw std::length_error An unexpected hit or extension size is 
	     *   encountered.
	     * @return A pointer just after the ring item.
	     */
	    const void* decode(const void* p, DDASFitHit& hit);
	};    
    }
}

#endif
