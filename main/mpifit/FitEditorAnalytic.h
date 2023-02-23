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

/** @addtogroup AnalyticFit
 * @brief Analytic FitEditor and fitting subroutines.
 * @{
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
 * Extending the hit with this editor overwrites any existing extension. This 
 * class is intended for use with the EventEditor framework providing a 
 * complete description of the new event body.
 */

class FitEditorAnalytic : public CBuiltRingItemEditor::BodyEditor
{
public:
  FitEditorAnalytic();
  virtual ~FitEditorAnalytic();

  // Mandatory interface from CBuiltRingItemEditor::BodyEditor
public:
  virtual std::vector<CBuiltRingItemEditor::BodySegment> operator()(pRingItemHeader pHdr, pBodyHeader pBHdr, size_t bodySize, void* pBody);
  virtual void free(iovec& e);

  // Additional functionality for this class
private:
  int pulseCount(DAQ::DDAS::DDASHit& hit);
  
  // Private member data
private:
  Configuration* m_pConfig; //!< Configuration file parser.
};

#endif

/** @} */
