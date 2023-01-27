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

/** @file:  FitEditorTemplate.h
 *  @brief: FitEditor class for analytic fitting
 */

#ifndef FITEDITORTEMPLATE_H
#define FITEDITORTEMPLATE_H

#include "CFitEditor.h"

#include <vector>

class FitEditorTemplate : public CFitEditor
{
 public:
  FitEditorTemplate();
  ~FitEditorTemplate();

  // Mandatory interface from CFitEditor
public:
  virtual std::vector<CBuiltRingItemEditor::BodySegment> operator()(pRingItemHeader pHdr, pBodyHeader hdr, size_t bodySize, void* pBody);
  virtual void free(iovec& e);

  // Additional functionality for this class
private:
  std::vector<double> m_template;
  unsigned m_alignPoint;
  std::string getTemplateFilename(const char* envname);
  void readTemplateFile(const char* filename);
  int pulseCount(DAQ::DDAS::DDASHit& hit);
  bool doFit(DAQ::DDAS::DDASHit& hit);
  std::pair<std::pair<unsigned, unsigned>, unsigned> fitLimits(DAQ::DDAS::DDASHit& hit);
};

#endif
