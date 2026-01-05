/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins
	     Aaron Chester
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** 
 * @file  fit_extensions.h
 * @brief Define structs used by fitting functions and to extend DDAS hits.
 */

#ifndef FIT_EXTENSIONS_H
#define FIT_EXTENSIONS_H

#include <cstdint>
#include <cstring>

/** @namespace ddastoys */
namespace ddastoys {

    /**
     * @struct PulseDescription
     * @brief Describes a single pulse without an offset. Zero-initialized.
     */
    struct PulseDescription {  
	double position  = 0; //!< Where the pulse is.
	double amplitude = 0; //!< Pulse amplitude.
	double steepness = 0; //!< Logistic steepness factor.
	double decayTime = 0; //!< Exponential decay constant.
    };

    /**
     * @struct fit1Info
     * @brief Full fitting information for the single pulse.
     */
    struct fit1Info { // Info from single pulse fit:
	PulseDescription pulse;  //!< Description of the pulse parameters.
	double   chiSquare  = 0; //!< Chi-square value of the fit.
	double   offset     = 0; //!< Constant offset.
	unsigned iterations = 0; //!< Iterations for fit to converge.
	unsigned fitStatus  = 0; //!< Fit status from GSL.
    };
    
    /**
     * @struct fit2Info
     * @brief Full fitting information for the double pulse.
     */
    struct fit2Info { // Info from double pulse fit:
	PulseDescription pulses[2]; //!< The two pulses.
	double   chiSquare  = 0;    //!< Chi-square value of the fit.
	double   offset     = 0;    //!< Shared constant offset.
	unsigned iterations = 0;    //!< Iterations needed to converge.
	unsigned fitStatus  = 0;    //!< Fit status from GSL.
    };

    /**
     * @struct HitExtensionLegacy
     * @brief Legacy data structure appended to each fit hit. This is the hit 
     * extension struct for DDASToys pre-6.0-000.
     */
    struct HitExtensionLegacy { // Data added to hits with traces:
	fit1Info onePulseFit; //!< Single-pulse fit information.
	fit2Info twoPulseFit; //!< Double-pulse fit information.
    };

    /**
     * @struct HitExtension
     * @brief The data structure appended to each fit hit.
     */
    struct HitExtension { // Data added to hits with traces:
	fit1Info onePulseFit;  //!< Single-pulse fit information.
	fit2Info twoPulseFit;  //!< Double-pulse fit information.
	double singleProb = 0; //!< Probability of single pulse
	double doubleProb = 0; //!< Probability of double pulse.
	
	/**
	 * @brief Default constructor - Needed because we have a custom
	 * constructor from the legacy class and we don't get the default
	 * one by, uh, default :(
	 */
	HitExtension() = default;	    
	/** 
	 * @brief Construct from legacy extension.
	 * @param leg Legacy extension to construct from.
	 * @note Single- and double-pulse probabilities are set to 0 since the 
	 * old-style extension does not contain any classificaton data.
	 */
	HitExtension(const HitExtensionLegacy& leg) :
	    onePulseFit(leg.onePulseFit), twoPulseFit(leg.twoPulseFit),
	    singleProb(0), doubleProb(0) {}
    };
    
    /**
     * @struct nullExtension
     * @brief A null fit extension is a single 32-bit word.
     */
    struct nullExtension {
	uint32_t s_size = sizeof(uint32_t);
    };

    /**
     * @struct FitInfoLegacy
     * @brief Legacy fit extension that knows its size. This is the fit info 
     * struct for DDASToys pre-6.0-000.
     */ 
    struct FitInfoLegacy {
	HitExtensionLegacy s_extension;          //!< The hit extension data.
	uint32_t s_size = sizeof(FitInfoLegacy); //!< sizeof(FitInfoLegacy)
    };
    
    /**
     * @struct FitInfo
     * @brief A fit extension that knows its size.
     */ 
    struct FitInfo {
	HitExtension s_extension;          //!< The hit extension data.
	uint32_t s_size = sizeof(FitInfo); //!< sizeof(FitInfo)
    };

} // namespace ddastoys

#endif
