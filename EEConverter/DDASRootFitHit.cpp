/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins
	     Aaron Chester
	     FRIB
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

using namespace ddastoys;

ddastoys::DDASRootFitHit::DDASRootFitHit() : DDASFitHit(), TObject()
{}

ddastoys::DDASRootFitHit::DDASRootFitHit(const DDASRootFitHit& rhs) :
    DDASFitHit(rhs), TObject(rhs)
{
    *this = rhs;
}

DDASRootFitHit&
ddastoys::DDASRootFitHit::operator=(const DDASRootFitHit& rhs)
{
    if (this != &rhs) {
	DDASFitHit::operator=(rhs);
	TObject::operator=(rhs);
    }

    return *this;
}

DDASRootFitHit&
ddastoys::DDASRootFitHit::operator=(const DDASFitHit& rhs)
{
    if (this != &rhs) {
	DDASFitHit::operator=(rhs);
	TObject::Clear(); // ??
    }

    return *this;
}

void
ddastoys::DDASRootFitHit::Reset()
{
    DDASFitHit hit;
    hit.Reset();
    *this = hit;
}
