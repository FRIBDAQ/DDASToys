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

/** 
 * @file  FitEditorAnalytic.h
 * @brief Definition of the FitEditor class for analytic fitting.
 */

#ifndef FITEDITORANALYTIC_H
#define FITEDITORANALYTIC_H

#include <CBuiltRingItemEditor.h>

#include <vector>

namespace DAQ {
    namespace DDAS {
	class DDASHit;
    }
}

class Configuration;

/**
 * @class FitEditorAnalytic
 * @brief Fit trace data using the analytic fitting functions and extend hits.
 *
 * @details
 * Extending the hit with this editor overwrites any existing extension. This 
 * class is intended for use with the EventEditor framework providing a 
 * complete description of the new event body. Resides in the DAQ::DDAS
 * namespace.
 */

class FitEditorAnalytic : public CBuiltRingItemEditor::BodyEditor
{
public:
    FitEditorAnalytic(); //!< Constructor.
    virtual ~FitEditorAnalytic(); //!< Destructor.

    // Mandatory interface from CBuiltRingItemEditor::BodyEditor
public:
    /**
     * @brief Perform the fit and create a fit extension for a single fragment. 
     * @param pHdr     Pointer to the ring item header of the hit.
     * @param pBHdr    Pointer to the body header pointer for the hit.
     * @param bodySize Number of bytes in the body.
     * @param pBody    Pointer to the body.
     * @return         Final segment descriptors.
     */
    virtual std::vector<CBuiltRingItemEditor::BodySegment> operator()(
	pRingItemHeader pHdr, pBodyHeader pBHdr, size_t bodySize, void* pBody
	);
    /**
     * @brief Free the dynamic fit extension descriptor(s).
     * @param e IOvec we need to free.
     */
    virtual void free(iovec& e);

    // Additional functionality for this class
private:
    /**
     * @brief This is a hook into which to add the ML classifier *
     * @param hit - references a hit *
     * @return int
     * @retval 0  On the basis of the trace no fitting.
     * @retval 1  Only fit a single trace.
     * @retval 2  Only fit two traces.
     * @retval 3  Fit both one and double hit.
     */
    int pulseCount(DAQ::DDAS::DDASHit& hit);
  
    // Private member data
private:
    Configuration* m_pConfig; //!< Configuration file parser.
};

#endif
