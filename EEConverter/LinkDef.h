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

#ifdef __CINT__

// Turn off everything by default.

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class DDASRootFitEvent+;
#pragma link C++ class DDASRootFitHit+;
#pragma link C++ class RootFit1Info+;
#pragma link C++ class RootFit2Info+;
#pragma link C++ class RootHitExtension+;
#pragma link C++ class RootPulseDescription+;
#pragma link C++ class std::vector<RootHitExtension>!;

#endif
