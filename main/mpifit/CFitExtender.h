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

/** @file:  CFitExtender.h
 *  @brief: Public base class exports of data structures from FitExtender.
 */
#ifndef CFITEXTENDER_H
#define CFITEXTENDER_H

#include <map>
#include <vector>

#include <CBuiltRingItemExtender.h>

#include "fit_extensions.h"

// \TODO (ASC 1/18/23): These additional structs are used by the unpacker, editor, etc. Can they either bypassed somehow or moved into the fit_extension header so that we don't need to include this (or CFitEditor) header in the unpacker?

namespace DAQ {
  namespace DDAS {
    class DDASHit;
  }
}

// Here are the hit extensions, Constructors fill in the hit extension sizes

// typedef struct _nullExtension {
//     uint32_t s_size;
//     _nullExtension() : s_size(sizeof(uint32_t)) {}
// } nullExtension, *pNullExtension;

// typedef struct _FitInfo {
//     uint32_t  s_size;
//     DDAS::HitExtension s_extension;
//     _FitInfo();
// } FitInfo, *pFitInfo;

/**
 * The extender base class definition, derived from CRingItemExtender
 */
class CFitExtender : public CBuiltRingItemExtender::CRingItemExtender
{    
public:
  CFitExtender();
  virtual ~CFitExtender() {};
  
  // Mandatory interface from CRingItemExtender
  virtual iovec operator()(pRingItem item);
  virtual void free(iovec& e);

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
