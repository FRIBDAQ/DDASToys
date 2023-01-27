/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     Aaron Chester
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  FitEditorAnalytic.h
 *  @brief: FitEditor class for analytic fitting
 */

#ifndef FITEDITORANALYTIC_H
#define FITEDITORANALYTIC_H

#include "CFitEditor.h"

#include <vector>

class FitEditorAnalytic : public CFitEditor
{
  // Canonicals
public:
  FitEditorAnalytic();
  ~FitEditorAnalytic();
 
  // Mandatory interface from CFitEditor
public:
  virtual std::vector<CBuiltRingItemEditor::BodySegment> operator()(pRingItemHeader pHdr, pBodyHeader hdr, size_t bodySize, void* pBody);
  virtual void free(iovec& e);

  // Utilities
private:
  int pulseCount(DAQ::DDAS::DDASHit& hit);
  bool doFit(DAQ::DDAS::DDASHit& hit);
  std::pair<std::pair<unsigned, unsigned>, unsigned> fitLimits(DAQ::DDAS::DDASHit& hit);
};

#endif
