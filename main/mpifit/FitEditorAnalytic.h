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

class FitEditorAnalytic : public CFitEditor
{
 public:
  FitEditorAnalytic();
  ~FitEditorAnalytic();

 private:
  virtual void fitSinglePulse(DDAS::fit1Info& result, std::vector<uint16_t>& trace, const std::pair<unsigned, unsigned>& limits, uint16_t saturation);
  virtual void fitDoublePulse(DDAS::fit2Info& result, std::vector<uint16_t>& trace, const std::pair<unsigned, unsigned>& limits, DDAS::fit1Info& singlePulseFit, uint16_t saturation);
};

#endif
