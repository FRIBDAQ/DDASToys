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
 * @file  fit_extensions_legacy.h
 * @brief Legacy data structs used by fitting functions and to extend DDAS 
 * hits prior to the incorporation of the ML inference model.
 */

#ifndef FIT_EXTENSIONS_LEGACY_H
#define FIT_EXTENSIONS_LEGACY_H

#include <cstdint>
#include <cstring>

namespace DDAS {

    /**
     * @struct PulseDescriptionLegacy
     * @brief Legacy description of a single pulse without an offset.
     */
    struct PulseDescriptionLegacy {  
	double position;  //!< Where the pulse is.
	double amplitude; //!< Pulse amplitude.
	double steepness; //!< Logistic steepness factor.
	double decayTime; //!< Decay time constant.    
    };

    /**
     * @struct fit1InfoLegacy
     * @brief Legacy description of full fitting information for a single pulse.
     */
    struct fit1InfoLegacy { // Info from single pulse fit:
	PulseDescriptionLegacy pulse; //!< Description of the pulse parameters.
	double   chiSquare;     //!< Chi-square value of the fit.
	double   offset;        //!< Constant offset.
	unsigned iterations;    //!< Iterations for fit to converge.
	unsigned fitStatus;     //!< Fit status from GSL.

    };
    
    /**
     * @struct fit2InfoLegacy
     * @brief Legacy description of full fitting information for a double pulse.
     */
    struct fit2InfoLegacy { // Info from double pulse fit:
	PulseDescriptionLegacy pulses[2]; //!< The two pulses.
	double   chiSquare;         //!< Chi-square value of the fit.
	double   offset;            //!< Shared constant offset.
	unsigned iterations;        //!< Iterations needed to converge.
	unsigned fitStatus;         //!< Fit status from GSL.
    };

    /**
     * @struct HitExtensionLegacy
     * @brief Legacy data structure appended to each fit hit.
     */
    struct HitExtensionLegacy { // Data added to hits with traces:
	fit1InfoLegacy onePulseFit; //!< Single pulse fit information.
	fit2InfoLegacy twoPulseFit; //!< Double pulse fit information.
    };  
}

/**
 * @struct nullExtensionLegacy
 * @brief Legacy null fit extension is a single 32-bit word.
 */
struct nullExtensionLegacy {
    std::uint32_t s_size; //!< sizeof(std::uint32_t)
    /** @brief Creates a nullExtension and sets its size. */
    nullExtensionLegacy() : s_size(sizeof(std::uint32_t)) {}
};

/**
 * @struct FitInfoLegacy
 * @brief Legacy fit extension that knows its size.
 */ 
struct FitInfoLegacy {
    DDAS::HitExtensionLegacy s_extension; //!< The hit extension data.
    std::uint32_t      s_size;      //!< sizeof(DDAS::HitExtension)
    /** @brief Creates FitInfo, set its size, and zeroes fit parameters. */
    FitInfoLegacy() : s_size(sizeof(FitInfoLegacy)) {
	memset(&s_extension, 0, sizeof(DDAS::HitExtensionLegacy));
    }
};

#endif
