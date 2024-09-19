/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     Aaron Chester
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** 
 * @file  CFitEngine.h
 * @brief Define the CFitEngine abstract base class.
 */

#ifndef CFITENGINE_H
#define CFITENGINE_H

#include <cstdint>
#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

/** @namespace ddastoys */
namespace ddastoys {

    /**
     * @class CFitEngine
     * @brief Abstract base class for marshalling data to the fitting 
     * subsystems to calculate Jacobian elements and residuals.
     */
    class CFitEngine {
    protected:
	std::vector<std::uint16_t> x; //!< Trace x coords.
	std::vector<std::uint16_t> y; //!< Trace y coords.
    public:
	/** @brief Constructor. */
	CFitEngine(
	    std::vector<std::pair<std::uint16_t, std::uint16_t>>& data
	    );
	/** @brief Destructor. */
	virtual ~CFitEngine() {}
	/** @brief Virtual method for calculating the Jacobian matrix */
	virtual void jacobian(const gsl_vector* p,  gsl_matrix *J) = 0;
	/** @brief Virtual method to calculating the residual. */
	virtual void residuals(const gsl_vector* p, gsl_vector* r)  = 0;
    };
		
}

#endif
