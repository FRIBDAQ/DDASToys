/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Girodano Cerizza
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/**
 * @file  jacobian_analytic.h
 * @brief Provides class definitions for families of engines to support 
 *        analytic trace fitting using GSL.
 */

/** @addtogroup AnalyticFit
 * @{
 */

#ifndef JACOBIAN_ANALYTIC_H
#define JACOBIAN_ANALYTIC_H

#include "CFitEngine.h"

// Concrete classes

/**
 * @class SerialFitEngine1
 * @brief Fit engine for analytic single pulse fits.
 *
 *   The concept is that each GSL lmfitter can supply a pair of methods:
 *   One that computes a vector of residuals and one that computes a Jacobian
 *   matrix of partial derivatives. At the implementation level we have two 
 *   types of fits we need done: Single pulse fits and double pulse fits (the 
 *   engines with names ending in 1 or 2). For each fit type we have two fit 
 *   engines:
 *      1) Serial computation (the engines with names starting with Serial)
 *      2) GPU accelerated (the engines with names starting with Cuda).
 *
 *    Finally a fit factory can generate the appropriate fit engine as desired
 *    by the actual fit.
 */
class SerialFitEngine1 : public CFitEngine {
public:
  SerialFitEngine1(std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data); //!< Constructor
  ~SerialFitEngine1() {} //!< Destructor
  virtual void jacobian(const gsl_vector* p,  gsl_matrix *J);
  virtual void residuals(const gsl_vector*p, gsl_vector* r);
};

/**
 * @class CudaFitEngine1
 * @brief CUDA-aware fit engine for analytic single pulse fits.
 *
 *   The concept is that each GSL lmfitter can supply a pair of methods:
 *   One that computes a vector of residuals and one that computes a Jacobian
 *   matrix of partial derivatives. At the implementation level we have two 
 *   types of fits we need done: Single pulse fits and double pulse fits (the 
 *   engines with names ending in 1 or 2). For each fit type we have two fit 
 *   engines:
 *      1) Serial computation (the engines with names starting with Serial)
 *      2) GPU accelerated (the engines with names starting with Cuda).
 *
 *    Finally a fit factory can generate the appropriate fit engine as desired
 *    by the actual fit.
 */
class CudaFitEngine1 : public CFitEngine {
private:
    void* m_dXtrace;          // Device ptr to trace x. [in]
    void* m_dYtrace;          // device ptr to trace y. [in]
    void* m_dResiduals;       // device ptr to residuals. [out]
    void* m_dJacobian;        // device ptr to jacobian [out]
    unsigned m_npts;          // # points in the trace.
    
public:
  CudaFitEngine1(std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data); //!< Constructor.
  ~CudaFitEngine1(); //!< Destructor. Deallocate dev resources.
  virtual void jacobian(const gsl_vector* p,  gsl_matrix *J);
  virtual void residuals(const gsl_vector*p, gsl_vector* r);
private:
  void throwCudaError(const char* msg);
};

/**
 * @class SerialFitEngine2
 * @brief Fit engine for analytic double pulse fits.
 *
 *   The concept is that each GSL lmfitter can supply a pair of methods:
 *   One that computes a vector of residuals and one that computes a Jacobian
 *   matrix of partial derivatives. At the implementation level we have two 
 *   types of fits we need done: Single pulse fits and double pulse fits (the 
 *   engines with names ending in 1 or 2). For each fit type we have two fit 
 *   engines:
 *      1) Serial computation (the engines with names starting with Serial)
 *      2) GPU accelerated (the engines with names starting with Cuda).
 *
 *    Finally a fit factory can generate the appropriate fit engine as desired
 *    by the actual fit.
 */
class SerialFitEngine2 : public CFitEngine {
public:
  SerialFitEngine2(std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data); //!< Constructor
  virtual void jacobian(const gsl_vector* p,  gsl_matrix *J);
  virtual void residuals(const gsl_vector*p, gsl_vector* r);
};

/**
 * @class CudaFitEngine2
 * @brief CUDA-aware fit engine for analytic double pulse fits.
 *
 *   The concept is that each GSL lmfitter can supply a pair of methods:
 *   One that computes a vector of residuals and one that computes a Jacobian
 *   matrix of partial derivatives. At the implementation level we have two 
 *   types of fits we need done: Single pulse fits and double pulse fits (the 
 *   engines with names ending in 1 or 2). For each fit type we have two fit 
 *   engines:
 *      1) Serial computation (the engines with names starting with Serial)
 *      2) GPU accelerated (the engines with names starting with Cuda).
 *
 *    Finally a fit factory can generate the appropriate fit engine as desired
 *    by the actual fit.
 */
class CudaFitEngine2 : public CFitEngine {
private:
    void* m_dXtrace;          // Device ptr to trace x. [in]
    void* m_dYtrace;          // device ptr to trace y. [in]
    void* m_dResiduals;       // device ptr to residuals. [out]
    void* m_dJacobian;        // device ptr to jacobian [out]
    unsigned m_npts;          // # points in the trace.
public: 
  CudaFitEngine2(std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data); //!< Constructor
  ~CudaFitEngine2(); //!< Destructor. Deallocate dev resources.
  virtual void jacobian(const gsl_vector* p,  gsl_matrix *J);
  virtual void residuals(const gsl_vector*p, gsl_vector* r);
private:
  void throwCudaError(const char* msg); //!<
};

/**
 * @class FitEngineFactory
 * @brief Factory for creating fit serial and Cuda-enabled fit engines.
 */
class FitEngineFactory {
public:
  /** 
   * @brief Create a single pulse Cuda-enabled fit engine. 
   * @param data  (x, y) values of the data to be fit.
   * @return CFitEngine*  Pointer to the created fit engine object. 
   */
  CFitEngine* createSerialFitEngine1(std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data);
  /** 
   * @brief Create a single pulse Cuda-enabled fit engine. 
   * @param data  (x, y) values of the data to be fit.
   * @return CFitEngine*  Pointer to the created fit engine object. 
   */
  CFitEngine* createCudaFitEngine1(std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data);
  /** 
   * @brief Create a double pulse serial fit engine. 
   * @param data  (x, y) values of the data to be fit.
   * @return CFitEngine*  Pointer to the created fit engine object. 
   */
  CFitEngine* createSerialFitEngine2(std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data);
  /** 
   * @brief Create a double pulse Cuda-enabled fit engine. 
   * @param data  (x, y) values of the data to be fit.
   * @return CFitEngine*  Pointer to the created fit engine object. 
   */
  CFitEngine* createCudaFitEngine2(std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data);

};

#endif

/** @} */
