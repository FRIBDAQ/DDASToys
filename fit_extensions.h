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
 * @brief Define structs used by fitting functions and to extend DDAS hits
 */

#ifndef FIT_EXTENSIONS_H
#define FIT_EXTENSIONS_H

#include <cstdint>
#include <cstring>

/** @namespace ddastoys */
namespace ddastoys {

    /**
     * @struct PulseDescription
     * @brief Describes a single pulse without an offset.
     */
    struct PulseDescription {  
	double position;  //!< Where the pulse is.
	double amplitude; //!< Pulse amplitude.
	double steepness; //!< Logistic steepness factor.
	double decayTime; //!< Decay time constant.    
    };

    /**
     * @struct fit1Info
     * @brief Full fitting information for the single pulse.
     */
    struct fit1Info { // Info from single pulse fit:
	PulseDescription pulse; //!< Description of the pulse parameters.
	double   chiSquare;     //!< Chi-square value of the fit.
	double   offset;        //!< Constant offset.
	unsigned iterations;    //!< Iterations for fit to converge.
	unsigned fitStatus;     //!< Fit status from GSL.
    };
    
    /**
     * @struct fit2Info
     * @brief Full fitting information for the double pulse.
     */
    struct fit2Info { // Info from double pulse fit:
	PulseDescription pulses[2]; //!< The two pulses.
	double   chiSquare;         //!< Chi-square value of the fit.
	double   offset;            //!< Shared constant offset.
	unsigned iterations;        //!< Iterations needed to converge.
	unsigned fitStatus;         //!< Fit status from GSL.
    };

    /**
     * @struct HitExtension
     * @brief The data structure appended to each fit hit.
     */
    struct HitExtension { // Data added to hits with traces:
	fit1Info onePulseFit; //!< Single pulse fit information.
	fit2Info twoPulseFit; //!< Double pulse fit information.
	double doublePulseProb; //!< Probability the trace is a double pulse.
    };  


/**
 * @struct nullExtension
 * @brief A null fit extension is a single 32-bit word.
 */
    struct nullExtension {
	std::uint32_t s_size; //!< sizeof(std::uint32_t)
	/** @brief Creates a nullExtension and sets its size. */
	nullExtension() : s_size(sizeof(std::uint32_t)) {}
    };

/**
 * @struct FitInfo
 * @brief A fit extension that knows its size.
 */ 
    struct FitInfo {
	HitExtension s_extension; //!< The hit extension data.
	std::uint32_t s_size;      //!< sizeof(ddastoys::HitExtension)
	/** @brief Creates FitInfo, set its size, and zeroes fit parameters. */
	FitInfo() : s_size(sizeof(FitInfo)) {
	    memset(&s_extension, 0, sizeof(HitExtension));
	}
    };
    
}

#endif
