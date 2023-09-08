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
 * @file  RootExtensions.h
 * @brief Provide streamable versions of the event extensions defined in
 * fit_extensions.h
 *
 * @details
 * These structs mirror those having to do with the fit extensions to DDAS 
 * data, but they support ROOT serialization and, therefore, can be put into
 * ROOT files/tree leaves.
 */

#ifndef ROOTEXTENSIONS_H
#define ROOTEXTENSIONS_H

#include <TObject.h>

namespace DDAS {
    struct PulseDescription;
    struct fit1Info;
    struct fit2Info;
    struct HitExtension;
}

/**
 * @ingroup ddasrootfit
 * @{
 */

/**
 * @struct RootPulseDescription
 * @brief Describes a single pulse without an offset.
 */
struct RootPulseDescription : public TObject
{
    // Data members:
   
    Double_t position;  //!< Where the pulse is.
    Double_t amplitude; //!< Pulse amplitude.
    Double_t steepness; //!< Logistic steepness factor.
    Double_t decayTime; //!< Decay time constant.
   
    // Methods:

    /** @brief Constructor. */
    RootPulseDescription();
    /**
     * @brief Assignment from a DDAS::PulseDescription. This is how this struct
     * is intended to get is values.
     * @param rhs Reference to a DDAS::PulseDescription that will provide 
     *   our values.
     * @return RootPulseDescription&  To support assignment chaining.
     */
    RootPulseDescription& operator=(const DDAS::PulseDescription& rhs);
  
    /** @brief Required for inheritence from TObject. */  
    ClassDef(RootPulseDescription, 1);
};

/** @} */


/**
 * @ingroup ddasrootfit
 * @{
 */

/**
 * @struct RootFit1Info
 * @brief Full fitting information for the single pulse.
 */
struct RootFit1Info : public TObject
{
    // Data members:
  
    RootPulseDescription pulse; //!< Description of the pulse parameters.
    Double_t chiSquare;         //!< Chi-square value of the fit.
    Double_t offset;            //!< Constant offset.
    UInt_t iterations;          //!< Iterations for fit to converge
    UInt_t fitStatus;           //!< Fit status from GSL.
   
    // Methods:

    /** @brief Constructor. */
    RootFit1Info();
    /**
     * @brief Assignment from a DDAS::fit1Info
     * @param rhs References the DDAS::fitInfo from which we'll get our values.
     * @return *this
     */
    RootFit1Info& operator=(const DDAS::fit1Info& rhs);

    /** @brief Required for inheritence from TObject. */
    ClassDef(RootFit1Info, 1);
};

/** @} */

/**
 * @ingroup ddasrootfit
 * @{
 */

/**
 * @struct RootFit2Info
 * @brief Full fitting information for the double pulse.
 */
struct RootFit2Info : public TObject
{
    // Data members:
  
    RootPulseDescription pulses[2]; //!< The two pulses.
    Double_t chiSquare;             //!< Chi-square value of the fit.
    Double_t offset;                //!< Offset on which they sit.
    UInt_t iterations;              //!< Iterations needed to converge.
    UInt_t fitStatus;               //!< Fit status from GSL.
   
    // Methods:
    
    /** @brief Constructor. */    
    RootFit2Info();    
    /**
     * @brief Assignment from a DDAS::fit2Info.
     * @param rhs References the DDAS::fit2Info from which we get our values.
     * @return *this
     */
    RootFit2Info& operator=(const DDAS::fit2Info& rhs);

    /** @brief Required for inheritence from TObject. */
    ClassDef(RootFit2Info, 1);
};

/** @} */

/**
 * @ingroup ddasrootfit
 * @{
 */

/**
 * @struct RootHitExtension
 * @brief The data structure containing the full information about the fits.
 */
struct RootHitExtension : public TObject
{
    RootFit1Info onePulseFit; //!< Single pulse fit information.
    RootFit2Info twoPulseFit; //!< Double pulse fit information.
    Bool_t haveExtension;     //!< True if there is fit information for the hit.

    /** @brief Constructor.*/
    RootHitExtension();
    /**
     * @brief Assignment from a DDAS::HitExtension.
     * @param rhs References the DDAS::HitExtension that gives us our values.
     * @return *this
     */
    RootHitExtension& operator=(const DDAS::HitExtension& rhs);

    /** @brief Required for inheritence from TObject. */
    ClassDef(RootHitExtension, 1);
};

/** @} */

#endif
