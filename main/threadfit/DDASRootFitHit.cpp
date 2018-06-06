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
DDASRootFitHit(const DDASRootFitHit& rhs) : TObject()
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
    if (this != &rhs) {
        TObject::operator=(rhs);
        Reset();                       // Probably not needed.
        time                = hit.GetTime();
        coarsetime          = hit.GetCoarseTime();
        cfd                 = hit.cfd;
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
        id                  = hit.id;
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

        m_haveExtensions    = hit.m_haveExtension;
        m_extension         = hit.m_extension;
        
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
DDASRootFitHit::operator=(const ::DAQ::DDAS::DDASFitHit& rhs)
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
    if (hit.hasExtension()) {
      m_haveExension = true;
      m_extension    = hit.getExtension;
    }
    return *this;
}
/*------------------------------------------------------------------------------
 * Operations;
 */

/**
 * UnpackChannelData
 *    This must unpacks a DDASFitHit using FitHitUnpacker. The resulting
 *    hit is assigned to us.
 *
 * @param p - pointer to the data to decode.  This should be a
 *            pointer to a raw ring item which contains only the data of one hit.
 *            That can be a ring item  from a fragment of the event builder.
 */
void 
DDASRootFitHit::UnpackChannelData(void* p)
{
    DAQ::DDAS::DDASFitHit     hit;
    DAQ::DDAS::FitHitUnpacker decoder;
    decoder.decode(p, hit);
    *this = hit;                    // (Easy button).
}
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

/**
 * hasExtension
 *    @return Bool_t - true if ther's an extension.
 */
Bool_t
DDASRootFitHit::hasExtension() const
{
    return m_haveExtension;
}
/**
 * getExtension
 *   Returns the extension.  This will be full of zeroes if m_haveExtension is
 *   false.
 *
 *  @return DDASHitExtension&
 */
DDAS::HitExtension&
DDASRootFitHit::getExtension() const
{
    return m_extension();
}