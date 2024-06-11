/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins 
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** 
 * @file  DDASRootFitHit.cpp
 * @brief Implement the DDASRootFitHit class.
 */

#include "DDASRootFitHit.h"

#include <iostream>

#include "DDASFitHit.h"
#include "RootExtensions.h"

DDASRootFitHit::DDASRootFitHit() : DAQ::DDAS::DDASFitHit(), TObject()
{}

DDASRootFitHit::DDASRootFitHit(const DDASRootFitHit& rhs) :
    DAQ::DDAS::DDASFitHit(rhs), TObject(rhs)
{
    *this = rhs;
}

DDASRootFitHit&
DDASRootFitHit::operator=(const DDASRootFitHit& rhs)
{
    if (this != &rhs) {
	DAQ::DDAS::DDASFitHit::operator=(rhs);
	TObject::operator=(rhs);
    }

    return *this;
}

DDASRootFitHit&
DDASRootFitHit::operator=(const DAQ::DDAS::DDASFitHit& rhs)
{
    if (this != &rhs) {
	DAQ::DDAS::DDASFitHit::operator=(rhs);
	TObject::Clear(); // ??
    }

    return *this;
}

void
DDASRootFitHit::Reset()
{
    DAQ::DDAS::DDASFitHit hit;
    hit.Reset();
    *this = hit;
}
