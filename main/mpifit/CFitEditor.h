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

/** @file:  CFitEditor.h
 *  @brief: Similar to FitExtender.cpp, pastes a fit to select traces.
 */

#ifndef CFITEDITOR_H
#define CFITEDITOR_H

#include <CBuiltRingItemEditor.h>

#include <map>

#include "fit_extensions.h"

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
  
  // Overridden virtual methods from base class
  virtual std::vector<CBuiltRingItemEditor::BodySegment> operator()(pRingItemHeader pHdr, pBodyHeader hdr, size_t bodySize, void* pBody) = 0;
  virtual void free(iovec& e) = 0;

protected:
  int channelIndex(unsigned crate, unsigned slot, unsigned channel);
  
  // Virtual functions related to reading configuration file which may be overridden in derived classes as well as a predicate function to decide whether to fit a channel and a metho
protected:
  virtual std::string getConfigFilename(const char* envname);
  virtual void readConfigFile(const char* filename);
  virtual std::string isComment(std::string line);

  // Data accessible to derived classes
protected:
  std::map <int, std::pair<std::pair<unsigned, unsigned>, unsigned>> m_fitChannels;
};

#endif
