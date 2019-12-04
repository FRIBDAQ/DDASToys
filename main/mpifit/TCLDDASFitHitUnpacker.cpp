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

/** @file:  TCLDDASFitHitUnpacker.cpp
 *  @brief: Unpacker for ddas hit data in Tcl.
 */
#include "TCLDDASFitHitUnpacker.h"
#include <CDataSource.h>
#include <CRingItem.h>
#include <CDataSourceFactory.h>
#include <Exception.h>
#include <TCLInterpreter.h>
#include <TCLObject.h>
#include <FragmentIndex.h>
#include <DataFormat.h>
#include "DDASFitHit.h"
#include <stdexcept>
#include <FragmentIndex.h>
#include "FitHitUnpacker.h"


#include <sstream>

int CTCLDDASFitHitUnpacker::m_openIndex(0);       // Used to create handles.

/**
 * constructor
 *    Create and register the command.
 * @param interp - interpreter on whidh the command is registered.
 */
CTCLDDASFitHitUnpacker::CTCLDDASFitHitUnpacker(CTCLInterpreter& interp) :
    CTCLObjectProcessor(interp, "ddasunpack", true)
{}
/**
 * destructor
 *    Destroys any open data sources.  The map takes care of itself
 */
CTCLDDASFitHitUnpacker::~CTCLDDASFitHitUnpacker()
{
    for (auto p : m_activeSources) {
        delete p.second;
    }
}
/**
 * operator()
 *     Called in response to the command.
 *     - sets up exception based error handling.
 *     - Binds all command line objects to the interpreter.
 *     - checks that we have at least a subcommand.
 *     - Dispatches to the appropriate subcommand handler.
 *
 *  @param interp - interpreter reference.
 *  @param objv   - Command line words.
 *  @return int   - TCL_OK on success.
 */
int
CTCLDDASFitHitUnpacker::operator()(
    CTCLInterpreter& interp, std::vector<CTCLObject>& objv
)
{
    try {
        bindAll(interp, objv);
        requireAtLeast(objv, 2);
        std::string sc = objv[1];           // Extract the subcommand.
        
        if(sc == "use") {
            use(interp, objv);
        } else if (sc == "next") {
            next(interp, objv);
        } else if (sc == "close") {
            close(interp, objv);
        } else {
            throw std::string("Invalid subcommand to ddasunpack");
        }
    }
    catch (CException &e) {
        interp.setResult(e.ReasonText());
        return TCL_ERROR;
    }   
    catch (std::exception& e) {
        interp.setResult(e.what());
        return TCL_ERROR;
    }
    catch (std::string msg) {
        interp.setResult(msg);
        return TCL_ERROR;
    }
    catch (char* msg) {
        interp.setResult(msg);
        return TCL_ERROR;
    }
    catch (...) {
        interp.setResult("Unanticipated exception type caught by ddasunpack");
        return TCL_ERROR;
    }
    
    return TCL_OK;
}

/**
 * use
 *    Open a new data source.  The data source is assigned a handle and
 *    entered in the m_activeSources map indexed by that handle.
 *    The handle is the command result.
 */
void
CTCLDDASFitHitUnpacker::use(CTCLInterpreter& interp, std::vector<CTCLObject>& objv)
{
    requireExactly(objv, 3);
    std::string source = objv[2];
    std::vector<uint16_t> empty;
    
    CDataSource* pSource =
        CDataSourceFactory::makeSource(source, empty, empty);
    if (pSource) {
        std::string handle = makeHandle();
        m_activeSources[handle] = pSource;
        interp.setResult(handle);
    } else {
        throw std::invalid_argument("Unable to open event source");
    }
}
/**
 * close
 *    Close the indicated handle.
 *    - Find the handle.
 *    - If it exists, delete the source and remove it from the map.
 */
void
CTCLDDASFitHitUnpacker::close(CTCLInterpreter& interp, std::vector<CTCLObject>& objv)
{
    requireExactly(objv, 3);
    std::string handle = objv[2];
    
    auto p = m_activeSources.find(handle);
    if (!(p == m_activeSources.end())) {
        delete p->second;
        m_activeSources.erase(p);
    } else {
        std::string msg = "No such data source handle: ";
        msg += handle;
        throw std::invalid_argument(msg);
    }
}
/**
 * next
 *    - Get the next physics event.
 *    - For each hit in the event (fragment in the event built data),
 *      make a dict describing that hit and append it to the result.
 * @note If we hit the end of file, we return an empty string.
 */
void
CTCLDDASFitHitUnpacker::next(CTCLInterpreter& interp, std::vector<CTCLObject>& objv)
{
    
    requireExactly(objv, 3);
    std::string handle = objv[2];
    auto p = m_activeSources.find(handle);
    if (!(p == m_activeSources.end())) {
        CDataSource* pSource = p->second;
        CRingItem*   pItem   = nextPhysicsItem(pSource);
        if (pItem) {
            DAQ::DDAS::FitHitUnpacker unpacker;
            CTCLObject result;
            result.Bind(interp);
            uint16_t* pBody = static_cast<uint16_t*>(pItem->getBodyPointer());
            FragmentIndex frags(pBody);
            for (int i =0; i < frags.getNumberFragments(); i++) {
                FragmentInfo frag = frags.getFragment(i);
                
                DAQ::DDAS::DDASFitHit hit;
                unpacker.decode(frag.s_itemhdr, hit);

                
                CTCLObject hitDict;
                hitDict.Bind(interp);

		// If the hit's body header is longer than sizeof(BodyHeader)
		// by sizeof(uint32_t), the extension is assumed to be classification
		// probabilities and we'll add an entry in the hit dict for the
		// two clasification probabilities.

		pRingItem prItem = reinterpret_cast<pRingItem>(frag.s_itemhdr);
		pBodyHeader pBH= &(prItem->s_body.u_hasBodyHeader.s_bodyHeader);
		if (pBH->s_size == (sizeof(BodyHeader) + sizeof(uint32_t))) {
		  uint32_t* pClass = reinterpret_cast<uint32_t*>(pBH+1);  // Just past the 'standard' body header.
		  uint32_t scaledClass = *pClass;
		  double pSingle = scaledClass & 0xffff;
		  pSingle /= 10000;                  // Now it's a probability.

		  double pDouble = scaledClass >> 16;
		  pDouble /= 10000;

		  addKeyValue(interp, hitDict, "singlePulseProbability", pSingle);
		  addKeyValue(interp, hitDict, "doublePulseProbability", pDouble);
		}
		
                makeHitDict(interp, hitDict, hit);
                result += hitDict;
                
            }
            
            delete pItem;
            interp.setResult(result);
        } 
    } else {
        std::string msg = "No such data source handle: ";
        msg += handle;
        throw std::invalid_argument(msg);
    }
}
/**
 * makeHandle
 *    Create a unique ring handle (think of this like the file descriptor
 *    [open] creates but for ring buffer files/rings)
 *
 * @return std::string
 */
std::string
CTCLDDASFitHitUnpacker::makeHandle()
{
    m_openIndex++;                     // Uniquifier.
    std::stringstream shandle;
    shandle << "ring" << m_openIndex;
    
    return shandle.str();
}
/**
 * makeHitDict
 *   Given a  hit, produce a dict that describes the hit.  See the
 *   header for the dict specification.  Note that since the CTCLObject
 *   interface does not (yet?) support the dict interface, we use a bit
 *   of shimmer magic.  The text representation of a dict is the same as
 *   a list of key1 value1 key2 value2 ...  That's what we'll actually
 *   construct.  Nested keys look like:
 *   key1 [list nested key1 value ...].
 *
 * @param interp - interpreter to Bind any CTCLObjects we make.
 * @param result - CTCLObject reference into which we build the result.
 * @param hit    - References the hit we're describing.
 */
void
CTCLDDASFitHitUnpacker::makeHitDict(
    CTCLInterpreter& interp, CTCLObject& result, DAQ::DDAS::DDASFitHit& hit
)
{
    // Add the individual simple keys:
    
    addKeyValue(interp, result, "crate", (int)hit.GetCrateID());
    addKeyValue(interp, result, "slot",  (int)hit.GetSlotID());
    addKeyValue(interp, result, "channel", (int)hit.GetChannelID());
    addKeyValue(interp, result, "energy", (int)hit.GetEnergy());
    addKeyValue(interp, result, "time",   hit.GetTime());
    
    // Note that there might not be a trace:
    
    if (hit.GetTraceLength() > 0)  {
        std::vector<uint16_t> t = hit.GetTrace();
        addKeyValue(interp, result, "trace", t);
    }
    if (hit.hasExtension()) {
      CTCLObject extensionDict;
      extensionDict.Bind(interp);
      makeFitsDict(interp, extensionDict, hit);
      addKeyValue(interp, result, "fits", extensionDict);
    }
}
/**
 * nextPhysicsItem
 *    Return the next ring item from a data source that is a physics
 *    item.
 * @param pSource - pointer to the data source.
 * @return CRingItem* - null pointer if no more physics items in the file..
 */
CRingItem*
CTCLDDASFitHitUnpacker::nextPhysicsItem(CDataSource* pSource)
{
    CRingItem* pResult;
    while (true) {
        pResult = pSource->getItem();            // Null on end.
        if (!pResult) break;
        
        if (pResult->type() == PHYSICS_EVENT) break;
        delete pResult;                    // Keep looking.
    }
    return pResult;
}

////////////////////////////////////////////////////////////////////////
// Dictionary handling methods.
//  Overloads for addKeyValue for various value types:
//
/**
 * addKeyValue
 *  @param  interp[in] - references the interpreter.
 *  @param  dict[inout]   - The dict we're adding the key/value to.
 *  @param  key[in]    - The text key.
 *  @param  value[in]  - The value - varying data type.
 */

            // Integer value:
void
CTCLDDASFitHitUnpacker::addKeyValue(
    CTCLInterpreter& interp, CTCLObject& dict, const char* key, int value
)
{
    dict += key;
    dict += value;
}
            // Double value:
void
CTCLDDASFitHitUnpacker::addKeyValue(
    CTCLInterpreter& interp, CTCLObject& dict, const char* key, double value
)
{
    dict += key;
    dict += value;
}
        // Vector of uint16_t (tracxe) value
void
CTCLDDASFitHitUnpacker::addKeyValue(
    CTCLInterpreter& interp, CTCLObject& dict, const char*  key,
    std::vector<uint16_t>& value
)
{
    CTCLObject v;
    v.Bind(interp);
    for (int i =0; i < value.size(); i++) {
        v += value[i];
    }
    addKeyValue(interp, dict, key, v);
}
    // CTCLObject as value
void
CTCLDDASFitHitUnpacker::addKeyValue(
    CTCLInterpreter& interp, CTCLObject& dict, const char* key,
    CTCLObject& value
)
{
    dict += key;
    dict += value;
}
////////////////////////////////////////////////////////////////////////////////
// Making the fits dict:

/**
 * makeFitsDict
 *    Create the fits dictionary value.
 * @param interp - interpreter to Bind temp CTCLObjects to.
 * @param result  - References the object to create.
 * @param hit    - References the hit from which to create it.
*/
void
CTCLDDASFitHitUnpacker::makeFitsDict(
    CTCLInterpreter& interp, CTCLObject& result, DAQ::DDAS::DDASFitHit& hit
)
{
    CTCLObject fit1;
    fit1.Bind(interp);
    makeFit1Dict(interp, fit1, hit);
    
    CTCLObject fit2;
    fit2.Bind(interp);
    makeFit2Dict(interp, fit2, hit);
    
    addKeyValue(interp, result, "fit1", fit1);
    addKeyValue(interp, result, "fit2", fit2);
}
/**
 * makeFit1Dict
 *    Creates the dictionary that describes the single pulse fit.
 *
 *  @param interp - interpreter to bind temporary CTCLObjects to.
 *  @param result - Referencs the object into which the dict is built.
 *  @param hit    - The hit - must have been verified to have an extension.
 */
void
CTCLDDASFitHitUnpacker::makeFit1Dict(
    CTCLInterpreter& interp, CTCLObject& result, DAQ::DDAS::DDASFitHit& hit
)
{
    const DDAS::HitExtension& ext(hit.getExtension());
    DDAS::fit1Info  f = ext.onePulseFit;
    
    addKeyValue(interp, result, "position", f.pulse.position);
    addKeyValue(interp, result, "amplitude", f.pulse.amplitude);
    addKeyValue(interp, result, "steepness", f.pulse.steepness);
    addKeyValue(interp, result, "decaytime", f.pulse.decayTime);
    addKeyValue(interp, result, "fitstatus", (int)f.fitStatus);
    addKeyValue(interp, result, "chisquare", f.chiSquare);
    addKeyValue(interp, result, "iterations", (int)f.iterations);
    addKeyValue(interp, result, "offset",    f.offset);
    
}
/**
 * makeFit2Dict
 *    Make the dict that describes the 2 pulse fit.
 *
 * @param interp - inteprreter used to bind temporary objects.
 * @param result - The dict we're formatting.
 * @param hit    - Hit, must contain an extension.
 */
void
CTCLDDASFitHitUnpacker::makeFit2Dict(
    CTCLInterpreter& interp, CTCLObject& result, DAQ::DDAS::DDASFitHit& hit
)
{
    const DDAS::HitExtension& ext(hit.getExtension());
    DDAS::fit2Info  f = ext.twoPulseFit;
    
    addKeyValues(
        interp, result, "position",
        f.pulses[0].position, f.pulses[1].position
    );
    addKeyValues(
        interp, result,  "amplitude",
        f.pulses[0].amplitude, f.pulses[1].amplitude
    );
    addKeyValues(
        interp, result,  "steepness",
        f.pulses[0].steepness, f.pulses[1].steepness
    );
    addKeyValues(
        interp, result,  "decaytime",
        f.pulses[0].decayTime, f.pulses[1].decayTime
    );
    addKeyValue(interp, result, "fitstatus", (int)f.fitStatus);
    addKeyValue(interp, result, "chisquare", f.chiSquare);
    addKeyValue(interp, result, "offset",    f.offset);
    addKeyValue(interp, result, "iterations", (int)(f.iterations));
    
}
/**
 * addKeyValues
 *    Adds a key whose value is a two element list to a dict.
 *
 * @param interp - interpreter used to do any temp objectbindings.
 * @param result - references the dict we're adding to.
 * @param key    - key string.
 * @param val1, val2 - The two values to turn into the value list.
 */
void
CTCLDDASFitHitUnpacker::addKeyValues(
    CTCLInterpreter& interp, CTCLObject& result, const char* key,
    double val1, double val2
)
{
    CTCLObject value;
    value.Bind(interp);
    value += val1;
    value += val2;
    addKeyValue(interp, result, key, value);
}

/////////////////////////////////////////////////////////////////////////////
// Package initialization called by package require:

extern "C" {
    int Tclunpacker_Init(Tcl_Interp* pRaw)
    {
        Tcl_PkgProvide(pRaw, "ddasunpack", "1.0");
        
        CTCLInterpreter* pInterp = new CTCLInterpreter(pRaw);
        new CTCLDDASFitHitUnpacker(*pInterp);
        
        return TCL_OK;
    }
}
