/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  CFitEngine.h
 *  @brief: Define base class.
 */

#ifndef CFITENGINE_H
#define CFITENGINE_H

#include <cstdint>
#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

class CFitEngine {
 protected:
  std::vector<std::uint16_t> x;           // Trace x coords
  std::vector<std::uint16_t> y;           // Trace y coords
 public:
  CFitEngine(std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data);
  virtual ~CFitEngine(){}
  virtual void jacobian(const gsl_vector* p,  gsl_matrix *J) = 0;
  virtual void residuals(const gsl_vector*p, gsl_vector* r)  = 0;
};

#endif
