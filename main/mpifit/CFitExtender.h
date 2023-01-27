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

#include <CBuiltRingItemExtender.h>

#include <map>

#include "fit_extensions.h"

namespace DAQ {
  namespace DDAS {
    class DDASHit;
  }
}

/**
 * The extender abstract base class definition, derived from CRingItemExtender.
 * A pure virutal method to fit traces in a DDASHit is provided which must be 
 * implemented in derived classes. Some default virtual methods for reading 
 * configuration files from the FIT_CONFIGFILE environment variable are 
 * provided but can be overridden in the derived classes.
 */

class CFitExtender : public CBuiltRingItemExtender::CRingItemExtender
{    
public:
  CFitExtender();
  virtual ~CFitExtender() {};
  
  // Mandatory interface from CRingItemExtender... which is again pure virtual and implemented by specific fitters
public:
  virtual iovec operator()(pRingItem item) = 0;
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
