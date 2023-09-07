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
 * @file  DDASRootFitHit.h
 * @brief Define a Hit with fitting data that can serialized by ROOT.
 */

#ifndef DDASROOTFITHIT_H
#define DDASROOTFITHIT_H

#include <TObject.h>

namespace DAQ {
    namespace DDAS {
	class DDASFitHit;
    }
}

/**
 * @todo (ASC 3/10/23): Does repacking the data actually save any disk space?
 */

/**
 * @class DDASRootFitHit
 * @brief This class is a ROOT serializable DDAS hit with possible fit data.
 *
 * @details
 * It has all the member data and functions of ddaschannel along with the fit 
 * flags and fit extension data of the DDASFitHit class.
 *
 * @note That this could certainly be derived from ddaschannel but the whole 
 * nonesense of ROOT dictionaries makes that tougher to port (I think).
 * Therefore we're entirely 'new' code.
 */
class DDASRootFitHit : public TObject
{
public:

    // These are stolen from ddaschannel -- along with this violated comment:
  
    // Ordering is important with regards to access and file size. Should always
    // try to maintain order from largest to smallest data type Double_t, Int_t,
    // Short_t, Bool_t, pointers
  
    std::vector<UInt_t> energySums; //!< Energy sum data.
    std::vector<UInt_t> qdcSums;    //!< QDC sum data.  
    std::vector<UShort_t> trace;    //!< Trace data.
  
    Double_t time;       //!< Assembled time including CFD.
    Double_t coarsetime; //!< Assembled time without CFD.
    Double_t cfd;        //!< @deprecated CFD time correction in nanoseconds.
    Double_t externalTimestamp; //!< External clock.

    UInt_t energy;              //!< Energy of event.
    UInt_t timehigh;            //!< Bits 32-47 of timestamp.
    UInt_t timelow;             //!< Bits 0-31 of timestamp.
    UInt_t timecfd;             //!< Raw CFD time.

    Int_t channelnum;           //!< @deprecated Channel number. Use crateid, slotid, and chanid to identify the source of the hit.
    Int_t finishcode;           //!< Indicates whether pile-up occurred.
    Int_t channellength;        //!< Number of 32-bit words of raw data.
    Int_t channelheaderlength;  //!< Length of header in 32-bit words.
    Int_t overflowcode;         //!< 0: no overflow, 1: overflow.
    Int_t chanid;               //!< Channel index.
    Int_t slotid;               //!< Slot index.
    Int_t crateid;              //!< Crate index.
    Int_t id;                   //!< @deprecated Unknown what this ID was meant to. Could be source ID, global channel ID, ??? (ASC 3/10/23).
    Int_t cfdtrigsourcebit;     //!< Value of trigger source bit(s) for 250 MSPS and 500 MSPS.
    Int_t cfdfailbit;           //!< Indicates whether the cfd algo failed.
    Int_t tracelength;          //!< Length of stored trace.
    Int_t ModMSPS;              //!< Sampling rate of the module (MSPS).
    Int_t m_adcResolution;      //!< ADC resolution (i.e. bit depth).
    Int_t m_hdwrRevision;       //!< Hardware revision.
  
    Bool_t m_adcOverUnderflow;  //!< True if ADC overflow or underflow.
  
    // Canonicals:
  
public:
    /**
     * @brief Constructor.
     */
    DDASRootFitHit();
    /**
     * @brief Copy constructor.
     * @param rhs Reference the hit to copy-construct.
     */
    DDASRootFitHit(const DDASRootFitHit& rhs);
    /**
     * @brief Destructor.
     */
    ~DDASRootFitHit();

    /**
     * @brief Assignment from another DDASRootFitHit.
     * @param rhs The hit we are assigning from.
     * @return *this.
     */
    DDASRootFitHit& operator=(const DDASRootFitHit& rhs);
    /**
     * @brief Assignment from a DAQ::DDAS::DDASFitHit.
     * @param rhs The hit we're assigning from.
     * @return *this.
     * @note This is the "normal" way we'll get data. Default construction and 
     * then assignment from a decoded DDASFitHit.
     */
    DDASRootFitHit& operator=(const DAQ::DDAS::DDASFitHit& rhs);
    
    // Operations:

    /**
     * @brief Reset the object to empty. This is really just a matter of 
     * assigning a reset hit to ourselves.
     */
    void Reset();
    
    // Selectors:

    /**
     * @brief Get the hit energy.
     * @return UInt_t  Energy in ADC units.
     */
    UInt_t GetEnergy() const {return energy;}
    /**
     * @brief Get the upper 16 bits of the timestamp in clock ticks.
     * @return UInt_t  The upper 16 bits of the timestamp in clock ticks.
     */
    UInt_t GetTimeHigh() const {return timehigh;}
    /**
     * @brief Get the lower 16 bits of the timestamp in clock ticks.
     * @return UInt_t  The lower 16 bits of the timestamp in clock ticks.
     */
    UInt_t GetTimeLow() const {return timelow;}
    /**
     * @brief Get the CFD time from the hit. See Pixie-16 User's Guide for more 
     * information on how the CFD is packed into the data.
     * @return UInt_t  The CFD time.
     */
    UInt_t GetCFDTime() const {return timecfd;}
    /**
     * @brief Get the hit time in nanoseconds.
     * @return Double_t  The hit time in nanoseconds.
     */
    Double_t GetTime() const {return time;}
    /**
     * @brief Get the coarse hit time in nanoseconds (no CFD correction).
     * @return Double_t  The coarse hit time in nanoseconds.
     */
    Double_t GetCoarseTime() const {return coarsetime;}
    /**
     * @brief Get the CFD correction in nanoseconds.
     * @return Double_t  The CFD correction in nanoseconds..
     */
    Double_t GetCFD() const {return cfd;}
    /**
     * @brief Get the ith energy sum from the hit.
     * @param i  The index of the energy sum to retrieve.
     * @return UInt_t  The ith energy sum.
     */
    UInt_t GetEnergySums(Int_t i) const {return energySums[i];}
    /**
     * @deprecated Deprecated method to return the channel number.
     */
    Int_t GetChannelNum() const {return channelnum;}
    /**
     * @brief Get the hit finish code.
     * @return Int_t  The hit finish code.
     */
    Int_t GetFinishCode() const {return finishcode;}
    /**
     * @brief Get the size of the hit in 32-bit data words.
     * @return Int_t  The number of 32-bit data words in the event.
     */
    Int_t GetChannelLength() const {return channellength;}
    /**
     * @brief Get the size of the hit header in 32-bit data words.
     * @return Int_t  The size of the hit header in 32-bit data words.
     */
    Int_t GetChannelLengthHeader() const {return channelheaderlength;}
    /**
     * @brief Get the hit overflow code.
     * @return Int_t  The overflow code.
     */
    Int_t GetOverflowCode() const {return overflowcode;}
    /**
     * @brief Get the hit slot ID.
     * @return Int_t  The hit slot ID.
     */
    Int_t GetSlotID() const {return slotid;}
    /**
     * @brief Get the hit crate ID.
     * @return Int_t  The hit crate ID.
     */
    Int_t GetCrateID() const {return crateid;}
    /**
     * @brief Get the hit channel ID.
     * @return Int_t  The hit channel ID.
     */
    Int_t GetChannelID() const {return chanid;}
    /**
     * @deprecated Deprecated method to return the id.
     */
    Int_t GetID() const {return id;}
    /**
     * @brief Get the sampling speed of the module in megasamples per 
     * second (MSPS).
     * @return Int_t  Module sampling speed in MSPS.
     */
    Int_t GetModMSPS() const {return ModMSPS;}
    /**
     * @brief Get the hit QDC sums.
     * @return std::vector<UInt_t>  Vector containing the QDC sum data.
     */
    std::vector<UInt_t> GetQDCSums() const {return qdcSums;}
    /**
     * @brief Get the hit trace data.
     * @return std::vector<UInt_t>  The trace data.
     */
    std::vector<UShort_t> GetTrace() const {return trace;}
    /**
     * @brief Get the ADC resolution for this module.
     * @return Int_t  Module ADC resolution.
     */
    Int_t GetADCResolution() const {return m_adcResolution;}
    /**
     * @brief Get the hardware revision of the module.
     * @return Int_t  Module hardware revision.
     */
    Int_t GetHardwareRevision() const {return m_hdwrRevision;}
    /**
     * @brief Get the hit overflow/underflow code.
     * @return Bool_t
     * @retval True   If the hit overflows or underflows the ADC.
     * @retval False  Otherwise.
     */
    Bool_t GetADCOverflowUnderflow() const {return m_adcOverUnderflow;}
    /**
     * @brief Get the trigger source bit for the CFD trigger.
     * @brief The CFD trigger source bit.
     */
    uint32_t GetCfdTrigSource() const {return cfdtrigsourcebit;}; 
  
    // Tell ROOT we're implementing the class:
  
    ClassDef(DDASRootFitHit, 1)
};

#endif
