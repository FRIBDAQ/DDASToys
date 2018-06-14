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

/** @file:  REventProcessor.cpp
 *  @brief: Implementation of the REventProcessor class.
 */

#include "REventProcessor.h"
#include <stdlib.h>
/**
 * Note that the bulk of this implementation is the implementation of the
 * methods of the various private structs that are used to organize the data.
 * Specifically, unpacking is hidden in the assignment operators of those
 * structs.
 */

/**----------------------------------------------------------------------------
 *   TreePulseDescription
 *       This struct describes a pulse (without the offset).  Parallel to
 *       DDAS::PulseDescription
 */

/**
 * Initialize
 *    Provides names to the members.
 * @param baseName - the elements will be prefixed with 'baseName'.
 */
void
REventProcessor::TreePulseDescription::Initialize(const char* baseName)
{
    std::string base(baseName);
    
    
    s_position.Initialize(base + ".position");
    s_amplitude.Initialize(base + ".amplitude");
    s_steepness.Initialize(base + ".steepness");
    s_decayTime.Initialize(base + ".decayTime");
}
/**
 * operator=
 *    Assign from a reference to a DDAS::PulseDescription.
 * @param rhs - the pulse description we're assigning from
 * @return *this
 */
REventProcessor::TreePulseDescription&
REventProcessor::TreePulseDescription::operator=(
    const DDAS::PulseDescription& rhs
)
{
    s_position  = rhs.position;
    s_amplitude = rhs.amplitude;
    s_steepness = rhs.steepness;
    s_decayTime = rhs.decayTime;
    
    return *this;
}

/*----------------------------------------------------------------------------
 *  TreeFit1Info
 *     This struct captures information about the fit for a single pulse
 *     in tree parameter form.  It's parallel to DDAS::fit1Info.
 */

/**
 * Initialize
 *    Give tree parameters their names.
 *
 *  @param baseName - base name for the tree parameters.
 */
void
REventProcessor::TreeFit1Info::Initialize(const char* baseName)
{
    std::string base = baseName;
    s_iterations.Initialize(base + ".iterations");
    s_fitStatus.Initialize(base + ".fitStatus");
    s_chiSquare.Initialize(base + ".chiSquare");
    s_offset.Initialize(base + ".offset");
    
    base += ".fit";
    s_fit.Initialize(base.c_str());
}
/**
 * operator=
 *   Assignment from a DDAS::fit1Info struct.
 * @param rhs - fit1info struct reference to assign fromm.
 * @return *this.
 */
REventProcessor::TreeFit1Info&
REventProcessor::TreeFit1Info::operator=(const DDAS::fit1Info& rhs)
{
    s_iterations = rhs.iterations;
    s_fitStatus  = rhs.fitStatus;
    s_chiSquare  = rhs.chiSquare;
    s_fit        = rhs.pulse;
    s_offset     = rhs.offset;
    
    return *this;
}
/*---------------------------------------------------------------------------
 * TreeFit2Info
 *    Parallel to DDAS::fit2Info.
 */

/**
 * Initialize - assign names to the tree parameters and structured members.
 *
 *    @param baseName - base name to assign to these.
 */
void
REventProcessor::TreeFit2Info::Initialize(const char* baseName)
{
    std::string base(baseName);
    
    s_iterations.Initialize(base + ".iterations");
    s_fitStatus.Initialize(base + ".fitStatus");
    s_chiSquare.Initialize(base + ".chiSquare");
    s_offset.Initialize(base + ".offset");
    
    std::string base1 = base;
    base1 += ".pulse1";
    s_fits[0].Initialize(base1.c_str());
    std::string base2 = base;
    base2 += ".pulse2";
    s_fits[1].Initialize(base2.c_str());
}
/**
 * operator=
 *    Assign from a ::DDAS::fit2Info
 */
REventProcessor::TreeFit2Info&
REventProcessor::TreeFit2Info::operator=(const DDAS::fit2Info& rhs)
{
    s_iterations = rhs.iterations;
    s_fitStatus  = rhs.fitStatus;
    s_chiSquare  = rhs.chiSquare;
    s_fits[0]    = rhs.pulses[0];           // 2 isn't worth a loop.
    s_fits[1]    = rhs.pulses[1];
    s_offset     = rhs.offset;
    
    return *this;
}
/*---------------------------------------------------------------------------
 *  TreeHitExtension - parallel to DDAS::HitExtension.
 */
/**
 * Initialize
 *    Assign names to the tree parameters and other fields.
 *
 * @param baseName -- base name to assign.
 */
void
REventProcessor::TreeHitExtension::Initialize(const char* baseName)
{
    std::string base(baseName);
    
    s_onePulseFit.Initialize((base + ".onepulsefit").c_str());
    s_twoPulseFit.Initialize((base + ".twopulsefit").c_str());
}
/**
 * operator=
 *    Assign from DDAS::HitExtension
 *
 * @param rhs - the hit extension to assign from.
 * @return *this
 */
REventProcessor::TreeHitExtension&
REventProcessor::TreeHitExtension::operator=(const DDAS::HitExtension& rhs)
{
    s_onePulseFit = rhs.onePulseFit;
    s_twoPulseFit = rhs.twoPulseFit;
    
    return *this;
}

/*---------------------------------------------------------------------
 *  TreeEvent - this is parallel to the Event struct in DtRange.h
 */

/**
 * Initialize
 *    Assign names to the tree parameters and other members.
 *
 * @param baseName - the base name that will be elaborated on.
 */
void
REventProcessor::TreeEvent::Initialize(const char* baseName)
{
    std::string base(baseName);
    
    s_isDouble.Initialize(base + ".isdouble");
    s_actualOffset.Initialize(base + ".actualOffset");
    
    std::string fitBase(base + ".fits");
    s_fitInfo.Initialize(fitBase.c_str());
    
    std::string actualBase(base + ".actual");
    std::string actualBaseL(actualBase + ".left");
    std::string actualBaseR(actualBase + ".right");
    
    s_pulses[0].Initialize(actualBaseL.c_str());
    s_pulses[1].Initialize(actualBaseR.c_str());
}
/**
 * operator=
 *    Assign from an Event.  This is a bit tricky (not too much).
 *    -  We assign s_isDouble to be 100 or 200 for false/true to give good
 *       separation for gating.
 *    - We assign s_pulses[1] only if s_isDouble is true.
 *
 * @param rhs -References the Event from which we assign.
 * @return *this (as Usual).
 */
REventProcessor::TreeEvent&
REventProcessor::TreeEvent::operator=(const Event& rhs)
{
    s_isDouble = rhs.s_isDouble ? 200 : 100;
    s_fitInfo  = rhs.s_fitinfo;
    s_actualOffset = rhs.s_actualOffset;
    
    s_pulses[0] = rhs.s_pulses[0];
    if(rhs.s_isDouble) {
        s_pulses[1] = rhs.s_pulses[1];
    }
    
    
    return *this;
}

/*----------------------------------------------------------------------------
 *  Implementation of the REventProcessor class.  With all the work
 *  we did defining the structs and their operator='s above, this shouild
 *  be pretty trivial.
 */

/**
 * constructor
 *    @param basename - base name for the entire event.
 */
REventProcessor::REventProcessor(const char* basename)
{
   
    // Differences between fit and actual amplitudes:
    // For fit 1, the difference between the only and the left pulse amplitude.
    // For fit 2, if the pulse is single, the a2diff is set to 5000.0 arbitrarily.
    
    fit1AmplitudeDiff.Initialize("event.fit1.AmplitudeDiff");
    fit2Amp1Diff.Initialize("event.fit2.a1diff");
    fit2Amp2Diff.Initialize("event.fit2.a2diff");
    
    // Differences between the actual peak positions and fitted ones. This is
    // done in a manner similar to the above.
    
    fit1XposDiff.Initialize("event.fit1.Tdiff");
    fit2X1PosDiff.Initialize("event.fit2.T0Diff");
    fit2X2PosDiff.Initialize("event.fit2.T1Diff");
    
    // Time difference actual, fitted and difference between them:
    actualDt.Initialize("event.dt");
    fittedDt.Initialize("event.fittedDt");
    DtDifference.Initialize("event.dtdiff");
    
    ChiSquareRatio.Initialize("event.chiratio2over1");
 
     m_event.Initialize(basename);
    
}
/**
 * operator()
 *    Unpacks the event.
 * @param pEvent - pointer to the raw event body.  Actually a Event*
 *                 The remainder of the parameters we just don't care about.
 * @return kfTRUE - I don't know how this can fail.
 */
Bool_t
REventProcessor::operator()(
    const Address_t pEvent, CEvent& e, CAnalyzer& a, CBufferDecoder& d
)
{
    // Are you read?  Drum roll please:
    
    const Event* p = reinterpret_cast<Event*>(pEvent);
    m_event = *p;                 //  That's really all there is to it.
    
    
    // How we compute the differences between fit and actual depends on whether
    // the fit is right for the raw event.  We use 'huge' differences for
    // the 'wrong' fit type.
    
    // To do this we need to be sure the left pulse is the left pulse in the
    // fitted data as this can be swapped.. otherwise we're comparing left to
    // right in fitted vs. actual.  We'll put things in the right order into fit
    // below:
    
    DDAS::fit2Info fit;
    fit = p->s_fitinfo.twoPulseFit;
    if (
        p->s_fitinfo.twoPulseFit.pulses[0].position >
        p->s_fitinfo.twoPulseFit.pulses[1].position) {   // they're flopped:
        
        fit.pulses[0] = p->s_fitinfo.twoPulseFit.pulses[1]; // So swap 
        fit.pulses[1] = p->s_fitinfo.twoPulseFit.pulses[0]; // the fitted pulses.
        
    }
    
    
    
    // Amplitudes.
    
    if (p->s_isDouble) {
        fit1AmplitudeDiff = 5000;             // It's actually a double pulse.
        fit2Amp1Diff =
            fit.pulses[0].amplitude - p->s_pulses[0].amplitude;
        fit2Amp2Diff =
            fit.pulses[1].amplitude - p->s_pulses[1].amplitude;
    } else {
        fit1AmplitudeDiff =
            p->s_fitinfo.onePulseFit.pulse.amplitude - p->s_pulses[0].amplitude;
        fit2Amp1Diff  = 5000;                // It's a single pulse after all.
        fit2Amp2Diff  = 5000;
    }
    // Positions (time).  Note that the dt's are also compute for doubles.
    
    if (p->s_isDouble) {
        fit1XposDiff = 5000;               // Double pulse after all.
        fit2X1PosDiff =
            fit.pulses[0].position - p->s_pulses[0].position;
        fit2X2PosDiff =
            fit.pulses[1].position - p->s_pulses[1].position;
        double adt = abs(p->s_pulses[1].position - p->s_pulses[0].position);
        actualDt = adt;
        double fdt = fit.pulses[1].position - fit.pulses[0].position;
        fittedDt = fdt;
        double dtdiff = fdt - adt;
        DtDifference = dtdiff;
    } else {
        
        fit1XposDiff =
            p->s_fitinfo.onePulseFit.pulse.position - p->s_pulses[0].position;
        fit2X1PosDiff = 5000;
        fit2X2PosDiff = 5000;
        fittedDt      = 5000;
        DtDifference  = 5000;
    }
    // Ratio of fit chisquares:
    
    ChiSquareRatio =
        p->s_fitinfo.twoPulseFit.chiSquare / p->s_fitinfo.onePulseFit.chiSquare;
    
    return kfTRUE;
}
