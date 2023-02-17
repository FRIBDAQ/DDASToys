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
 * @file  DDASFitHitUnpacker.cpp
 * @brief Implements DDASFitHitUnpacker.
 */

#include "DDASFitHitUnpacker.h"

#include <string>
#include <stdexcept>
#include <iostream>

#include <DataFormat.h>

#include "DDASFitHit.h"

/**
 * @brief Decode the current event and unpack it into a DDASFitHit.
 * 
 * The decode function:
 * - Determines the limits of the old style hit.
 * - Determines where, or if, there's an extension block.
 * - Unpacks the original hit using DDASHitUnpacker::unpack.
 * - Sets the extension if there is one.
 * 
 * @param p  Pointer to the ring item to decode. For an event built fragment, 
 *           this is normally the FragmentInfo's s_itemhdr pointer. Note the 
 *           difference from DDASHitUnpacker which expects a pointer to the 
 *           body.
 * @param hit  Hit item that we will unpack data into.
 *
 * @throw std::length_error  An unexpected hit or extension size is 
 *                           encountered.
 *
 * @return void*  A pointer just after the ring item.
 */
const void*
DAQ::DDAS::DDASFitHitUnpacker::decode(const void* p, DDASFitHit& hit)
{  
  // Find the ring item body:
    
  const RingItem* pItem = reinterpret_cast<const RingItem*>(p);
  const std::uint8_t*  pBody;
  std::uint32_t bodyHeaderSize; // Number of bytes in the body header.
  if (pItem->s_body.u_noBodyHeader.s_mbz) {
    pBody = reinterpret_cast<const std::uint8_t*>(pItem->s_body.u_hasBodyHeader.s_body);
    bodyHeaderSize = pItem->s_body.u_hasBodyHeader.s_bodyHeader.s_size;
  } else {
    pBody = reinterpret_cast<const std::uint8_t*>(pItem->s_body.u_noBodyHeader.s_body);
    bodyHeaderSize = sizeof(std::uint32_t);
  }
  
  // Generate a pointer to off the end of the body.
    
  std::uint32_t bodySize = pItem->s_header.s_size - bodyHeaderSize - sizeof(RingItemHeader);
  const std::uint8_t* pEnd = pBody + bodySize;
  
  /*
   * The first 32 bits of the body is the number of 16 bit words of
   * DDAS hit.
   * - If this works out equivalent to bodySize - there's no extension.
   * - If this is larger, to accommodate data from ringblockdealer or the 
   *   editor fitting framework, we have the following cases:
   *     1) The extra data is the size of HitExtension - the extra data is a 
   *        HitExtension from ringblockdealer.
   *     2) The extra data is sizeof(std::uint32_t) - must contain 
   *        sizeof(std::uint32_t) and is a null extension from the editor 
   *        fitting framework.
   *     3) The extra data is sizeof(FitInfo) from FitExtender - the extra data
   *        is FitExtender fit information
   *     4) Anything else - we don't know how to do with and fail with an error 
   *        message.
   */
    
  const std::uint32_t* pHitSize = reinterpret_cast<const std::uint32_t*>(pBody);
  std::uint32_t bodyWords = *pHitSize;
  std::uint32_t bodyBytes = bodyWords*sizeof(std::uint16_t);    
 
  if (bodyBytes == bodySize) {
    
    // This is just an ordinary hit
    
    unpack(
	   reinterpret_cast<const std::uint32_t*>(pBody),
	   reinterpret_cast<const std::uint32_t*>(pEnd),
	   hit
	   );
        
  } else if ((bodyBytes + sizeof(::DDAS::HitExtension)) == bodySize) {
    
    // Hit with fits
    
    pEnd -= sizeof(::DDAS::HitExtension);  // also points to extension.
    unpack(
	   reinterpret_cast<const std::uint32_t*>(pBody),
	   reinterpret_cast<const std::uint32_t*>(pEnd),
	   hit
	   );
    hit.setExtension(*(reinterpret_cast<const ::DDAS::HitExtension*>(pEnd)));
    
  } else if (bodyBytes + sizeof(std::uint32_t) == bodySize) {
        
    // There's no hit extension actually -- it's a FitExtender null extension
        
    pEnd -= sizeof(std::uint32_t);
    unpack(
	   reinterpret_cast<const std::uint32_t*>(pBody),
	   reinterpret_cast<const std::uint32_t*>(pEnd),
	   hit
	   );
                
  } else if (bodyBytes + sizeof(FitInfo) == bodySize) {
    
    // FitExtender fit information
    
    pEnd -= sizeof(FitInfo);
    unpack(
	   reinterpret_cast<const std::uint32_t*>(pBody),
	   reinterpret_cast<const std::uint32_t*>(pEnd),
	   hit
	   );
    const FitInfo* pExtension = reinterpret_cast<const FitInfo*>(pEnd);
    hit.setExtension(pExtension->s_extension);
    
  } else {
    throw std::length_error("Inconsistent event size for DDASHit or extended hit");
  }
    
  return nullptr; // Should not get here.    
}
