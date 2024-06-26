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
 * @file   LinkDef.h
 * @brief  Defines the linkages to supply to root.
 */

#ifdef __CLING__

// Turn off everything by default.

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedefs;

#pragma link C++ class DAQ::DDAS::DDASFitHit+;
#pragma link C++ class DAQ::DDAS::DDASHit+;
#pragma link C++ class DDAS::HitExtension+;
#pragma link C++ class DDAS::fit1Info+;
#pragma link C++ class DDAS::fit2Info+;
#pragma link C++ class DDAS::PulseDescription+;
#pragma link C++ class DDASRootFitEvent+;
#pragma link C++ class DDASRootFitHit+;
#pragma link C++ class RootFit1Info+;
#pragma link C++ class RootFit2Info+;
#pragma link C++ class RootHitExtension+;
#pragma link C++ class RootPulseDescription+;
#pragma link C++ class std::vector<RootHitExtension>!;

#endif
