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
 * @file   LinkDef.h
 * @brief  Defines the linkages to supply to root.
 */

#ifdef __CLING__

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedefs;

#pragma link C++ namespace ddastoys;
#pragma link C++ defined_in ddastoys;
#pragma link C++ namespace ddasfmt;
#pragma link C++ defined_in ddasfmt;

#pragma link C++ class ddasfmt::DDASHit+;
#pragma link C++ class ddastoys::DDASFitHit+;
#pragma link C++ class ddastoys::HitExtension+;
#pragma link C++ class ddastoys::fit1Info+;
#pragma link C++ class ddastoys::fit2Info+;
#pragma link C++ class ddastoys::PulseDescription+;
#pragma link C++ class ddastoys::DDASRootFitEvent+;
#pragma link C++ class ddastoys::DDASRootFitHit+;

#endif
