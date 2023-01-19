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

/** @file:  CFitEditor.h
 *  @brief: Similar to FitExtender.cpp, pastes a fit to select traces.
 */
#ifndef CFITEDITOR_H
#define CFITEDITOR_H

#include <vector>
#include <map>

#include <CBuiltRingItemEditor.h>

#include "fit_extension.h"

namespace DAQ {
  namespace DDAS {
    class DDASHit;
  }
}

/**
 * @class CFitEditor
 *   This is similar to CFitExtender however, where that class is intended 
 *   to work with the Transformer program to extend hits with a fit, and 
 *   multiple applications will result in multiple extensions. This class 
 *   extends the hit, overwriting any existing extension. It's intended for 
 *   use with the EventEditor framework providing a complete description of 
 *   the new event body.
 */
class CFitEditor : public CBuiltRingItemEditor::BodyEditor
{  
public:
  CFitEditor();
  virtual ~CFitEditor() {};

  // Mandatory interface from CBuiltRingItemEditor
  virtual std::vector<CBuiltRingItemEditor::BodySegment> operator()(pRingItemHeader pHdr, pBodyHeader hdr, size_t bodySize, void* pBody);
  virtual void free(iovec& item);

  // Data accessible to derived classes
protected:
  std::map <int, std::pair<std::pair<unsigned, unsigned>, unsigned>> m_fitChannels;
  
  // Pure virtual methods that need to be implemented in derived classes
private:
  virtual void fitSinglePulse(DDAS::fit1Info& result, std::vector<uint16_t>& trace, const std::pair<unsigned, unsigned>& limits, uint16_t saturation) = 0;
  virtual void fitDoublePulse(DDAS::fit2Info& result, std::vector<uint16_t>& trace, const std::pair<unsigned, unsigned>& limits, DDAS::fit1Info& singlePulseFit, uint16_t saturation) = 0;

  // Private methods
private:
  int pulseCount(DAQ::DDAS::DDASHit& hit);
  bool doFit(DAQ::DDAS::DDASHit& hit);
  std::pair<std::pair<unsigned, unsigned>, unsigned> fitLimits(DAQ::DDAS::DDASHit& hit);
  std::string getConfigFilename(const char* envname);
  void readConfigFile(const char* filename);
  std::string isComment(std::string line);
};

#endif
