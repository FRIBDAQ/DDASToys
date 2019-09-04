/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  StripTrace.cpp
 *  @brief: Shared library code for event editor that strips traces.
 */

#include <CBuiltRingItemEditor.h>
#include <stdexcept>

class StripTrace : public CBuiltRingItemEditor::BodyEditor
{
    virtual std::vector<CBuiltRingItemEditor::BodySegment> operator()(
            pRingItemHeader pHdr, pBodyHeader hdr, size_t bodySize, void* pBody
        );
    virtual void free(iovec& item);    
};
/**
 * operator()
 *    Strips any traces off a DDAS hit if it has an extension(fit).
 *    Note that if there are
 *    fits at the end of the event they are retained:
 *    - Gets the length of the trace.
 *    - Sets the hit trace length to zero.
 *    - Subtracts trace length/2 from the event length.
 *    - Creates a vector with one or two descriptors:
 *       * One descriptor to the hit with trace deleted if no fit.
 *       * An additional descriptor for any post hit data (e.g. fits).
 *
 * @param pHdr - Ring item header of the hit.
 * @param bhdr  - Body header pointer for the hit.
 * @param bodySize - Number of _bytes_ in the event body.
 * @param pBody - Pointer to the body.
*  @return std::vector<CBuiltRingItemEditor::BodySegment>
*        The segment descriptors.  In this case none are dynamic.
*/
std::vector<CBuiltRingItemEditor::BodySegment>
StripTrace::operator()(
    pRingItemHeader pHdr, pBodyHeader bhdr, size_t bodySize, void* pBody
)
{
    std::vector<CBuiltRingItemEditor::BodySegment> result;
    
    // The body is a set of uint32_t's.  Note that it has a
    // size longword and a digitizer type longword in front of the
    // hit.
    
    uint32_t* pB = static_cast<uint32_t*>(pBody);
    uint32_t traceLen16 = ((pB[5] >> 16) & 0x3fff); // # 32 bit trace uint16's
    uint32_t evtlen     = (pB[2] >> 17) & 0x3fff;   // Initial eventlen.
    evtlen -= traceLen16/2;                        // 2 samples/long
    pB[2] = (pB[2] & 0x8001ffff) | (evtlen << 17); // Update event len.
    pB[5] = (pB[2] & 0x8000ffff);                  // Zero out the trace len.
    
    *pB  -= traceLen16;                            // update word count.
    
    //  If there's an extension, we can kill off the trace and keep the
    // extension instead.  Otherwise, keep the entire ring item.
    
    
    pB += 2 + evtlen  + traceLen16/2;      // Point past trace.
    int extSize =
        bodySize - (2 + evtlen + traceLen16/sizeof(uint16_t))*sizeof(uint32_t); // left over bytes:
    
    if (extSize) {                    // There's an extension
        CBuiltRingItemEditor::BodySegment hit(   // Wave form removed 
            (evtlen+2)*sizeof(uint32_t), pBody
        );
        result.push_back(hit);                   // fit extension.
            CBuiltRingItemEditor::BodySegment extension(extSize, pB);
            result.push_back(extension);
    } else {                    // Keep the whole ring item.
        // One descriptor for the entire body:
        CBuiltRingItemEditor::BodySegment body(bodySize, pBody);
        result.push_back(body);
        
    }
       
    return result;
}
/**
 * free
 *    Nothing is dynamic so throw std::logic_error
 *
 * @para item - item descriptor.
 */
void
StripTrace::free(iovec& item)
{
    throw std::logic_error("SkipTrace being asked to free storage!!");
}

/**
 *  Now the factory so that the framework can make StripTrace
 *  instances.
 */

extern "C" {
    CBuiltRingItemEditor::BodyEditor* createEditor()
    {
        return new StripTrace;
    }
}