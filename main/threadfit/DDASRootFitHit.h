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

/** @file:  DDASRootFitHit.h
 *  @brief: Define a Hit with fitting data that can serialized by CERN Root.
 */
#ifndef DDASROOTFITHIT_H
#define DDASROOTFITHIT_H

#include <TObject.h>
#include <functions.h>
#include "DDASFitHit.h"         // We're really a rootized one of these.


class CRingItme;

/**
 * @class DDASRootFitHit
 *    This class is a Root serializable DDAS hit with possible fit data.
 *    It has all the member data and functions of ddaschannel along with
 *    the Fit flags and fit extension data of the DDASFitHit class.
 *
 *    Note that this could certainly be derived from ddaschannel but the whole
 *    nonesense of Root dictionaries makes that tougher to port (I think).
 *    Therefore we're entirely 'new' code.
 */
class DDASRootFitHit : public TObject
{
private:
    
  // These are stolen from ddaschannel -- along with the comment that's violated
    
  /********** Variables **********/

  // ordering is important with regards to access and file size.  Should
  // always try to maintain order from largest to smallest data type
  // Double_t, Int_t, Short_t, Bool_t, pointers

  /* Channel events always have the following info. */
  Double_t time;              ///< assembled time including cfd
  Double_t coarsetime;        ///< assembled time without cfd
  Double_t cfd;               ///< cfd time only \deprecated

  UInt_t energy;              ///< energy of event
  UInt_t timehigh;            ///< bits 32-47 of timestamp
  UInt_t timelow;             ///< bits 0-31 of timestamp
  UInt_t timecfd;             ///< raw cfd time

  Int_t channelnum;           ///< \deprecated
  Int_t finishcode;           ///< indicates whether pile-up occurred
  Int_t channellength;        ///< number of 32-bit words of raw data
  Int_t channelheaderlength;  ///< length of header
  Int_t overflowcode;         ///< 1 = overflow
  Int_t chanid;               ///< channel index
  Int_t slotid;               ///< slot index
  Int_t crateid;              ///< crate index
  Int_t id;                   ///< \deprecated

  Int_t cfdtrigsourcebit;     ///< value of trigger source bit(s) for 250 MSPS and 500 MSPS
  Int_t cfdfailbit;           ///< indicates whether the cfd algo failed

  Int_t tracelength;          ///< length of stored trace

  Int_t ModMSPS;              ///< Sampling rate of the module (MSPS)
  Int_t m_adcResolution;      ///< adc resolution (i.e. bit depth)
  Int_t m_hdwrRevision;       ///< hardware revision
  Bool_t m_adcOverUnderflow;  ///< whether adc overflowed or underflowed

  /* A channel may have extra information... */
  std::vector<UInt_t> energySums;  ///< Energy sum data
  std::vector<UInt_t> qdcSums;   ///< QDC sum data
  
  /* A waveform (trace) may be stored too. */
  std::vector<UShort_t> trace;     ///< Trace data

  Double_t externalTimestamp;  ///< External clock

  // Additional data we need.  Hopefully the fact that the hit extension is
  // implemented in terms of native types rather than e.g. Double_t is not
  // going to cause problems.
  
  Bool_t m_haveExtension;
  ::DDAS::HitExtension m_extension;
  
    // Canonicals:
    
public:
    DDasRootFitHit();
    DDASRootFitHit(const DDASRootFitHit& rhs);
    ~DDASRootFitHit();
    
    DDASRootFitHit& operator=(const DDASRootFitHit& rhs);
    DDASRootFitHit& operator=(const DDASFitHit& rhs);
    
    // Operations:
    
    void UnpackChannelData(const void* p);
    void Reset();

    
    // Selectors:

  UInt_t GetEnergy() const {return energy;}
  UInt_t GetTimeHigh() const {return timehigh;}
  UInt_t GetTimeLow() const {return timelow;}
  UInt_t GetCFDTime() const {return timecfd;}
  Double_t GetTime() const {return time;}
  Double_t GetCoarseTime() const {return coarsetime;}
  Double_t GetCFD() const {return cfd;}
  UInt_t GetEnergySums(Int_t i) const {return energySums[i];}
  Int_t GetChannelNum() const {return channelnum;}
  Int_t GetFinishCode() const {return finishcode;}
  Int_t GetChannelLength() const {return channellength;}
  Int_t GetChannelLengthHeader() const {return channelheaderlength;}
  Int_t GetOverflowCode() const {return overflowcode;}
  Int_t GetSlotID() const {return slotid;}
  Int_t GetCrateID() const {return crateid;}
  Int_t GetChannelID() const {return chanid;}
  Int_t GetID() const {return id;}
  Int_t GetModMSPS() const {return ModMSPS;}
  std::vector<UShort_t> GetTrace() const {return trace;}
  Int_t GetADCResolution() const { return m_adcResolution; }
  Int_t GetHardwareRevision() const { return m_hdwrRevision; }
  Bool_t GetADCOverflowUnderflow() const { return m_adcOverUnderflow; }
  uint32_t GetCfdTrigSource() const { return cfdtrigsourcebit; }; // Per S.L. request.
  
  // New ones for this class
  
  Bool_t hasExtension() const;
  const DDAS::HitExtension& getExtension() const;

  // Tell root we're implementing the class.
  
    ClassDef(DDASRootFitHit, 1)
};

#endif
