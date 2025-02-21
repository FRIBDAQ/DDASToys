/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Aaron Chester
	     Bashir Sadeghi
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/**
 * @file mlinference.h
 * @brief Function definitions for the machine-learning inference editor.
 */

#ifndef MLINFERENCE_H
#define MLINFERENCE_H

#include <cstdint>
#include <vector>

#include <torch/script.h> // Can forward-declare?

/** @namespace ddastoys */
namespace ddastoys {

    struct FitInfo;
    
    /** @namespace ddastoys::mlinference */
    namespace mlinference {
	
	/** 
	 *@ingroup mlinference
	 * @{
	 */

	/**
	 * @brief Perform ML inference to determine the pulse parameters.
	 * @param[in, out] pResult Pointer to the fit results.
	 * @param[in] trace References the trace we're processing.
	 * @param[in] saturation ADC saturation value. Only samples below the 
	 *   saturation threshold are used to extract the pulse parameters.
	 * @param[in] module References the inference model for this channel.
	 */
	void performInference(
	    FitInfo* pResult,
	    std::vector<uint16_t>& trace, unsigned saturation,
	    torch::jit::script::Module& module
	    );
	
	/** @} */	
    }
}

#endif
