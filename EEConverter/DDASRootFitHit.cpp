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
 * @file  DDASRootFitHit.cpp
 * @brief Implement the DDASRootFitHit class.
 */

#include "DDASRootFitHit.h"

#include "DDASFitHit.h"
#include "RootExtensions.h"

/**
 * @details
 * Resets the hit vector.
 */
DDASRootFitHit::DDASRootFitHit() : TObject()
{
    Reset();
}

DDASRootFitHit::DDASRootFitHit(const DDASRootFitHit& rhs) : TObject(rhs)
{
    *this = rhs;
}

/**
 * @details
 * Resets the hit vector.
 */
DDASRootFitHit::~DDASRootFitHit()
{
    Reset();
}

DDASRootFitHit&
DDASRootFitHit::operator=(const DDASRootFitHit& rhs)
{
    if (this != &rhs) {
	TObject::operator=(rhs);
	Reset(); // Probably not needed
	time                = rhs.GetTime();
	coarsetime          = rhs.GetCoarseTime();
	cfd                 = rhs.cfd;
	energy              = rhs.GetEnergy();
	timehigh            = rhs.GetTimeHigh();
	timelow             = rhs.GetTimeLow();
	timecfd             = rhs.timecfd;
	channelnum          = rhs.GetChannelID();
	finishcode          = rhs.GetFinishCode();
	channellength       = rhs.GetChannelLength();
	channelheaderlength = rhs.GetChannelLengthHeader();
	overflowcode        = rhs.GetOverflowCode();
	chanid              = rhs.GetChannelID();
	slotid              = rhs.GetSlotID();
	crateid             = rhs.GetCrateID();
	id                  = rhs.id;
	cfdtrigsourcebit    = rhs.cfdtrigsourcebit;
	cfdfailbit          = rhs.cfdfailbit;
	tracelength         = rhs.tracelength;
	ModMSPS             = rhs.GetModMSPS();
	energySums          = rhs.energySums;
	qdcSums             = rhs.qdcSums;
	trace               = rhs.GetTrace();
	externalTimestamp   = rhs.externalTimestamp;
	m_adcResolution     = rhs.GetADCResolution();
	m_hdwrRevision      = rhs.GetHardwareRevision();
	m_adcOverUnderflow  = rhs.GetADCOverflowUnderflow();

    }
  
    return *this;
}

DDASRootFitHit&
DDASRootFitHit::operator=(const DAQ::DDAS::DDASFitHit& rhs)
{
    time                = rhs.GetTime();
    coarsetime          = rhs.GetCoarseTime();
    cfd                 = 0; // not used
    energy              = rhs.GetEnergy();
    timehigh            = rhs.GetTimeHigh();
    timelow             = rhs.GetTimeLow();
    timecfd             = rhs.GetTimeCFD();
    channelnum          = rhs.GetChannelID();
    finishcode          = rhs.GetFinishCode();
    channellength       = rhs.GetChannelLength();
    channelheaderlength = rhs.GetChannelLengthHeader();
    overflowcode        = rhs.GetOverflowCode();
    chanid              = rhs.GetChannelID();
    slotid              = rhs.GetSlotID();
    crateid             = rhs.GetCrateID();
    id                  = 0;
    cfdtrigsourcebit    = rhs.GetCFDTrigSource();
    cfdfailbit          = rhs.GetCFDFailBit();
    tracelength         = rhs.GetTraceLength();
    ModMSPS             = rhs.GetModMSPS();
    energySums          = rhs.GetEnergySums();
    qdcSums             = rhs.GetQDCSums();
    trace               = rhs.GetTrace();
    externalTimestamp   = rhs.GetExternalTimestamp();
    m_adcResolution     = rhs.GetADCResolution();
    m_hdwrRevision      = rhs.GetHardwareRevision();
    m_adcOverUnderflow  = rhs.GetADCOverflowUnderflow();

    return *this;
}

void
DDASRootFitHit::Reset()
{
    DAQ::DDAS::DDASFitHit hit;
    hit.Reset();
    *this = hit;
}

