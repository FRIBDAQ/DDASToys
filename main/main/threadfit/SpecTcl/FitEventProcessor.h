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

/** @file:  FitEventProcessor.h
 *  @brief: Define the event processor for fitted DDAS data.
 */
#ifndef FITEVENTPROCESSOR_H
#define FITEVENTPROCESSOR_H
#include <config.h>
#include <EventProcessor.h>
#include <TreeParameter.h>

class FitEventProcessor : public CEventProcessor
{
private:
    // Tree parameter definitions for single pulse fits:
    
    CTreeParameterArray m_f1ChiSquare;
    CTreeParameterArray m_f1A0;
    CTreeParameterArray m_f1Iterations;
    
    // Tree parameter definitions for double pulse fit:
    
    CTreeParameterArray m_f2ChiSquare;
    CTreeParameterArray m_f2A0;
    CTreeParameterArray m_f2A1;
    CTreeParameterArray m_f2Dt;
    CTreeParameterArray m_f2Iterations;

    CTreeParameter      m_energy;
    CTreeParameter      m_chiratio;
    
public:
    FitEventProcessor();
    virtual Bool_t operator()(
        const Address_t pEvent, CEvent& rEvent, CAnalyzer& rAnalyzer,
        CBufferDecoder& rDecoder
    );
};


#endif
