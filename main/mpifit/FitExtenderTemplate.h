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

/** @file:  FitExtenderTemplate.h
 *  @brief: FitExtender class for analytic fitting
 */

#ifndef FITEXTENDERTEMPLATE_H
#define FITEXTENDERTEMPLATE_H

#include "CFitExtender.h"

#include <vector>

class FitExtenderTemplate: public CFitExtender
{
public:
  FitExtenderTemplate();
  virtual ~FitExtenderTemplate();

  // Mandatory interface from CFitExtender
public:
  virtual iovec operator()(pRingItem item);
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
