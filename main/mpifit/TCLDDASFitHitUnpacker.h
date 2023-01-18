/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  TCLDDASFitHitUnpacker.h
 *  @brief: Define DDAS Fit Hit unpacker for DDAS.
 */
#ifndef TCLDDASFITHITUNPACKER_H
#define TCLDDASFITHITUNPACKER_H

#include <TCLObjectProcessor.h>
#include <map>
#include <string>
#include <DDASFitHit.h>

class CDataSource;
class CRingItem;

/**
 * @class CTCLDDASFitHitUnpacker.
 *
 *    Unpack ddas event built events (with fit extensions) into hits for Tcl processing.
 *    This is mostly intended for fit/trace visualization but is complete.
 *    The package provides a tcl commande:
 *
 *    ddasunpack
 *
 *    This is an ensemble with the following subcommands:
 *
 *    *  use - provides the URL of a data source. returns a handle
 *    *  next - Unpackes the next hit from the data source takes a handle
 *    *  close- takes a handle.
 *
 * The next subcommand produces an empty result if there's no more events or
 * a list of dicts.  Each dict represents one hit in the event.  Dicts have
 * the following keys:
 *    * crate
 *    * slot
 *    * channel
 *    * energy
 *    * time
 *    * trace  (if there's a trace) list of trace data points.
 *    * fits   (if there's a fit extension).
 *
 *  The fits dictionary value is itself a dict containing the keys fit1  and
 *  fit2. fit1 is a dict that contains
 *     * position
 *     * amplitude
 *     * steepness
 *     * decayTime
 *     * iterations
 *     * fitstatus
 *     * chisquare
 *     * offset
 *
 *   fit2 is a dict that contains the same keys but position, amplitude,steepness,
 *   and decyatime are two element lists for the first and second pulse in that
 *   order.
 */
class CTCLDDASFitHitUnpacker : public CTCLObjectProcessor
{
private:
    std::map<std::string, CDataSource*> m_activeSources;
    static int                              m_openIndex;
public:
    CTCLDDASFitHitUnpacker(CTCLInterpreter& interp);
    virtual ~CTCLDDASFitHitUnpacker();
    
    int operator()(CTCLInterpreter& interp, std::vector<CTCLObject>& objv);
protected:
    void use(CTCLInterpreter& interp, std::vector<CTCLObject>& objv);
    void next(CTCLInterpreter& interp, std::vector<CTCLObject>& objv);
    void close(CTCLInterpreter& interp, std::vector<CTCLObject>& objv);
    
private:
    std::string makeHandle();
    void makeHitDict(
        CTCLInterpreter& interp, CTCLObject& result, DAQ::DDAS::DDASFitHit& hit
    );

    CRingItem*       nextPhysicsItem(CDataSource* pSource);
    
    // Utilties to help format my dicts:
    
    void addKeyValue(
        CTCLInterpreter& interp, CTCLObject& dict, const char* key, int value
    );
    void addKeyValue(
        CTCLInterpreter& interp, CTCLObject& dict, const char* key, double value
    );
    void addKeyValue(
        CTCLInterpreter& interp, CTCLObject& dict, const char*  key,
        std::vector<uint16_t>& value
    );
    void addKeyValue(
        CTCLInterpreter& interp, CTCLObject& dict, const char* key,
        CTCLObject& value
    );
    void makeFitsDict(
        CTCLInterpreter& interp, CTCLObject& result, DAQ::DDAS::DDASFitHit& hit
    );
    void makeFit1Dict(
        CTCLInterpreter& interp, CTCLObject& result, DAQ::DDAS::DDASFitHit& hit
    );
    void makeFit2Dict(
        CTCLInterpreter& interp, CTCLObject& result, DAQ::DDAS::DDASFitHit& hit
    );
    
    void addKeyValues(
        CTCLInterpreter& interp, CTCLObject& result, const char* key,
        double val1, double val2
    );
};


#endif