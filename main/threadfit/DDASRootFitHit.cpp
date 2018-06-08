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

/** @file:  DDASRootFitHit.cpp
 *  @brief: Implement the DDASRootFitHit class.
 */

#include "DDASRootFitHit.h"
#include "FitHitUnpacker.h"

ClassImp(DDASRootFitHit);

/*-----------------------------------------------------------------------------
 * Canonicals:
 */
/**
 * constructor
 */
DDASRootFitHit::DDASRootFitHit() :
    TObject()
{
    Reset();
}
/**
 * copy constructor
 */
DDASRootFitHit::DDASRootFitHit(const DDASRootFitHit& rhs) : TObject(rhs)
{
    *this = rhs;
}
/**
 * Destructor
 */
DDASRootFitHit::~DDASRootFitHit()
{
    Reset();
}
/**
 * operator=
 *    Assignment from another DDASRootFitHit:
 *  @param rhs - assigned from this.
 *  @return *this
 */
DDASRootFitHit&
DDASRootFitHit::operator=(const DDASRootFitHit& hit)
{
    if (this != &hit) {
        TObject::operator=(hit);
        Reset();                       // Probably not needed.
        time                = hit.GetTime();
        coarsetime          = hit.GetCoarseTime();
        cfd                 = hit.cfd;
        energy              = hit.GetEnergy();
        timehigh            = hit.GetTimeHigh();
        timelow             = hit.GetTimeLow();
        timecfd             = hit.timecfd;
        channelnum          = hit.GetChannelID();
        finishcode          = hit.GetFinishCode();
        channellength       = hit.GetChannelLength();
        channelheaderlength = hit.GetChannelLengthHeader();
        overflowcode        = hit.GetOverflowCode();
        chanid              = hit.GetChannelID();
        slotid              = hit.GetSlotID();
        crateid             = hit.GetCrateID();
        id                  = hit.id;
        cfdtrigsourcebit    = hit.cfdtrigsourcebit;
        cfdfailbit          = hit.cfdfailbit;
        tracelength         = hit.tracelength;
        ModMSPS             = hit.GetModMSPS();
        energySums          = hit.energySums;
        qdcSums             = hit.qdcSums;
        trace               = hit.GetTrace();
        externalTimestamp   = hit.externalTimestamp;
        m_adcResolution     = hit.GetADCResolution();
        m_hdwrRevision      = hit.GetHardwareRevision();
        m_adcOverUnderflow  = hit.GetADCOverflowUnderflow();

    }
    return *this;
}
/**
 * Assignment from a DAQ::DDAS::DDASFitHit.
 *
 * @param   rhs  - hit we're being assigned from.
 * @return *this.
 * @note This is the normal way we'll get data.  Default construction and then
 *       assignment from a decoded DDASFitHit.
 *
 */
DDASRootFitHit&
DDASRootFitHit::operator=(const ::DAQ::DDAS::DDASFitHit& hit)
{
    
    time                = hit.GetTime();
    coarsetime          = hit.GetCoarseTime();
    cfd                 = 0; // not used
    energy              = hit.GetEnergy();
    timehigh            = hit.GetTimeHigh();
    timelow             = hit.GetTimeLow();
    timecfd             = hit.GetTimeCFD();
    channelnum          = hit.GetChannelID();
    finishcode          = hit.GetFinishCode();
    channellength       = hit.GetChannelLength();
    channelheaderlength = hit.GetChannelLengthHeader();
    overflowcode        = hit.GetOverflowCode();
    chanid              = hit.GetChannelID();
    slotid              = hit.GetSlotID();
    crateid             = hit.GetCrateID();
    id                  = 0;
    cfdtrigsourcebit    = hit.GetCFDTrigSource();
    cfdfailbit          = hit.GetCFDFailBit();
    tracelength         = hit.GetTraceLength();
    ModMSPS             = hit.GetModMSPS();
    energySums          = hit.GetEnergySums();
    qdcSums             = hit.GetQDCSums();
    trace               = hit.GetTrace();
    externalTimestamp   = hit.GetExternalTimestamp();
    m_adcResolution     = hit.GetADCResolution();
    m_hdwrRevision      = hit.GetHardwareRevision();
    m_adcOverUnderflow  = hit.GetADCOverflowUnderflow();

    return *this;
}
/*------------------------------------------------------------------------------
 * Operations;
 */


/**
 * Reset
 *    Reset the object to empty.  This is really just a matter of 
 *    Assigning a resret hit to us.
 */
void
DDASRootFitHit::Reset()
{
    DAQ::DDAS::DDASFitHit hit;
    hit.Reset();
    *this = hit;
}
/*-----------------------------------------------------------------------------
 *  Selector implementation.
 */

