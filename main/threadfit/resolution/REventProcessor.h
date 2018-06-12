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

/** @file:  REventProcessor.h
 *  @brief: Resolution event processor definition
 */

#ifndef REVENTPROCESSOR_H
#define REVENTPROCESSOR_H
#include <config.h>
#include <EventProcessor.h>
#include <TreeParameter.h>
#include "functions.h"
#include "DtRange.h"

/**
 * @class REventProcessor
 *    This SpecTcl event processor analyzes data produced by DtRange.  It
 *    unpackes the events it writes allowing us to explore the time/energy
 *    resolution of the fitting software.
 *    We'll create/use a set of tree parameters that closely match those
 *    in the Event struct in DtRange.h
 */

class REventProcessor : public CEventProcessor
{
private:
    // Data Types.. setting all this up makes unpacking trivial.
    // The first set of structs map the functions.h HitExtension
    // struct and its components into TreeParameter based parallel structures.
    
    struct TreePulseDescription {
        CTreeParameter   s_position;
        CTreeParameter   s_amplitude;
        CTreeParameter   s_steepness;
        CTreeParameter   s_decayTime;
        
        // Methods:        
        void Initialize(const char* baseName);
        TreePulseDescription& operator=(const DDAS::PulseDescription& rhs);
    };
    
    struct TreeFit1Info {
        CTreeParameter s_iterations;
        CTreeParameter s_fitStatus;
        CTreeParameter s_chiSquare;
        TreePulseDescription s_fit;
        CTreeParameter s_offset;
        
        // Methods;
        
        void Initialize(const char* baseName);
        TreeFit1Info& operator=(const DDAS::fit1Info& rhs);
    };
    struct TreeFit2Info {
        CTreeParameter s_iterations;
        CTreeParameter s_fitStatus;
        CTreeParameter s_chiSquare;
        TreePulseDescription s_fits[2];
        CTreeParameter s_offset;
        
        // Methods
        
        void Initialize(const char* baseName);
        TreeFit2Info& operator=(const DDAS::fit2Info& rhs);
    };
    struct TreeHitExtension {
        TreeFit1Info   s_onePulseFit;
        TreeFit2Info   s_twoPulseFit;
        
        void Initialize(const char* baseName);
        TreeHitExtension& operator=(const DDAS::HitExtension& rhs);
    };
    // The next set of structs  map the Event struct and its components
    // onto a struct of tree parameters.  The key to all of these is the
    // assignment operator that allows assignment to the struct
    // from the raw event.  That makes unpacking a one-liner.
    
    struct TreeEvent {
        CTreeParameter    s_isDouble;
        TreeHitExtension  s_fitInfo;
        CTreeParameter    s_actualOffset;
        TreePulseDescription s_pulses[2];      // Must have largest number.
        
        void Initialize(const char* baseName);
        TreeEvent& operator=(const Event& rhs);
    };
    // Data members:
private:
    TreeEvent   m_event;           // 'raw' parameters.
    
    // Computed parameters
    
    // Differences between fits and actuals.
    
    CTreeParameter fit1AmplitudeDiff;
    CTreeParameter fit2Amp1Diff;
    CTreeParameter fit2Amp2Diff;
    
    CTreeParameter fit1XposDiff;
    CTreeParameter fit2X1PosDiff;
    CTreeParameter fit2X2PosDiff;
    
    // Fitted/actual time differences
    
    CTreeParameter actualDt;
    CTreeParameter fittedDt;
    CTreeParameter DtDifference;
    
    CTreeParameter ChiSquareRatio;
    
    // methdods of the event processor:
    
public:
    REventProcessor(const char* basename);
    
    Bool_t operator()(
        const Address_t pEvent, CEvent& rEvent,
        CAnalyzer& rA, CBufferDecoder& d
    );
    
};

#endif