/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Girodano Cerizza
	     Aaron Chester
	     FRIB
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/**
 * @file  jacobian_analytic.h
 * @brief Provides class definitions for families of engines to support 
 *        analytic trace fitting using GSL.
 */

#ifndef JACOBIAN_ANALYTIC_H
#define JACOBIAN_ANALYTIC_H

#include "CFitEngine.h"

/** @namespace ddastoys */
namespace ddastoys {
    /** @namespace ddastoys::analyticfit */
    namespace analyticfit {

	// Concrete classes

	/**
	 * @class SerialFitEngine1
	 * @brief Fit engine for analytic single pulse fits.
	 *
	 * @details
	 * The concept is that each GSL lmfitter can supply a pair of methods:
	 * One that computes a vector of residuals and one that computes a 
	 * Jacobian matrix of partial derivatives. At the implementation level 
	 * we have two  types of fits we need done: Single pulse fits and 
	 * double pulse fits (the engines with names ending in 1 or 2). For 
	 * each fit type we have two fit  engines:
	 *   1) Serial computation (the engines with names starting with Serial)
	 *   2) GPU accelerated (the engines with names starting with Cuda).
	 * Finally a fit factory can generate the appropriate fit engine as 
	 * desired by the actual fit.
	 */
	class SerialFitEngine1 : public CFitEngine {
	public:
	    /** 
	     * @brief Constructor. 
	     * @param data The trace data.
	     */
	    SerialFitEngine1(
		std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data
		);
	    /** Destructor. */
	    ~SerialFitEngine1() {}
	    /**
	     * @brief Compute the Jacobian matrix.
	     * @param[in] p  Current fit parameterization.
	     * @param[out] J The Jacobian for this iteration of the fit.
	     */
	    virtual void jacobian(const gsl_vector* p,  gsl_matrix *J);
	    /**
	     * @brief Compute the vector of residuals.
	     * @param[in] p  Current fit parameters.
	     * @param[out] r Vector of residual values.
	     */
	    virtual void residuals(const gsl_vector*p, gsl_vector* r);
	};

	/**
	 * @class CudaFitEngine1
	 * @brief CUDA-aware fit engine for analytic single pulse fits.
	 *
	 * The concept is that each GSL lmfitter can supply a pair of methods:
	 * One that computes a vector of residuals and one that computes a 
	 * Jacobian matrix of partial derivatives. At the implementation level 
	 * we have two types of fits we need done: Single pulse fits and double
	 * pulse fits (the engines with names ending in 1 or 2). For each fit 
	 * type we have two fit engines:
	 *   1) Serial computation (the engines with names starting with Serial)
	 *   2) GPU accelerated (the engines with names starting with Cuda).
	 *
	 * Finally a fit factory can generate the appropriate fit engine as 
	 * desired by the actual fit.
	 *
	 * @todo (ASC 10/30/23): Manages some device pointers. Not copyable, 
	 * and most likely is not ever copied. But we can make it safe. 
	 */
	class CudaFitEngine1 : public CFitEngine {
	private:
	    void* m_dXtrace;    //!< [in]  Device ptr to trace x data.
	    void* m_dYtrace;    //!< [in]  Device ptr to trace y data.
	    void* m_dResiduals; //!< [out] Device ptr to residuals.
	    void* m_dJacobian;  //!< [out] Device ptr to Jacobian.
	    unsigned m_npts;    //!< Number of points in the trace.
    
	public:
	    /**
	     * @brief Constructor.
	     */
	    CudaFitEngine1(
		std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data
		);
	    /**
	     * @brief Destructor.
	     */
	    ~CudaFitEngine1();
	    /**
	     * @brief Invoke the kernal to do the pointwise parallel Jacobian 
	     * computation.
	     * @param p Parameter vector.
	     * @param J Jacobian matrix.
	     */
	    virtual void jacobian(const gsl_vector* p, gsl_matrix *J);
	    /**
	     * @brief Triggers a pointwise parallel residual kernel in the 
	     * device and impedance matches that with GSL's requirements.
	     * @param p Parameter vector.
	     * @param r Residual vector.
	     */
	    virtual void residuals(const gsl_vector*p, gsl_vector* r);
    
	private:
	    void throwCudaError(const char* msg);
	};

	/**
	 * @class SerialFitEngine2
	 * @brief Fit engine for analytic double pulse fits.
	 *
	 * @details
	 * The concept is that each GSL lmfitter can supply a pair of methods:
	 * One that computes a vector of residuals and one that computes a 
	 * Jacobian matrix of partial derivatives. At the implementation level 
	 * we have two types of fits we need done: Single pulse fits and double
	 * pulse fits (the engines with names ending in 1 or 2). For each fit
	 * type we have two fit engines:
	 *   1) Serial computation (the engines with names starting with Serial)
	 *   2) GPU accelerated (the engines with names starting with Cuda).
	 * Finally a fit factory can generate the appropriate fit engine as
	 * desired by the actual fit.
	 */
	class SerialFitEngine2 : public CFitEngine {
	public:
	    /**
	     * @brief Constructor.
	     * @param data The trace.
	     */
	    SerialFitEngine2(
		std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data
		);
	    /**
	     * @brief Compute the Jacobian matrix.
	     * @param[in] p  The current parameters.
	     * @param[out] J The Jacobian matrix to fill in.
	     */
	    virtual void jacobian(const gsl_vector* p,  gsl_matrix *J);
	    /**
	     * @brief Compute the residual vector.
	     * @param[in] p  Current parameter set.
	     * @param[out] r Pointwise residuals.
	     */
	    virtual void residuals(const gsl_vector*p, gsl_vector* r);
	};

	/**
	 * @class CudaFitEngine2
	 * @brief CUDA-aware fit engine for analytic double pulse fits.
	 *
	 * The concept is that each GSL lmfitter can supply a pair of methods:
	 * One that computes a vector of residuals and one that computes a 
	 * Jacobian matrix of partial derivatives. At the implementation level 
	 * we have two types of fits we need done: Single pulse fits and double
	 * pulse fits (the  engines with names ending in 1 or 2). For each fit 
	 * type we have two fit  engines:
	 *   1) Serial computation (the engines with names starting with Serial)
	 *   2) GPU accelerated (the engines with names starting with Cuda).
	 *
	 * Finally a fit factory can generate the appropriate fit engine as 
	 * desired by the actual fit.
	 *
	 * @todo (ASC 10/30/23): Manages some device pointers. Not copyable, 
	 * and most likely is not ever copied. But we can make it safe. 
	 */
	class CudaFitEngine2 : public CFitEngine {
	private:
	    void* m_dXtrace;    //!< Device ptr to trace x. [in]
	    void* m_dYtrace;    //!< Device ptr to trace y. [in]
	    void* m_dResiduals; //!< Device ptr to residuals. [out]
	    void* m_dJacobian;  //!< Device ptr to jacobian [out]
	    unsigned m_npts;    //!< # points in the trace.
    
	public:
	    /**
	     * @brief Constructor.
	     * @param data The trace data in x/y pairs.
	     */
	    CudaFitEngine2(
		std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data
		);
	    /** @brief Destructor. */
	    ~CudaFitEngine2();
	    /**
	     * @brief Marshall the parameter and call the jacobian2 kernel. 
	     * Then pull the Jacobian matrix out of the GPU and marshall it 
	     * back into the GSL Jacobian matrix.
	     * @param p Parameter vector from GSL.
	     * @param J Jacobian matrix to output.
	     */
	    virtual void jacobian(const gsl_vector* p,  gsl_matrix *J);
	    /**
	     * @brief Fire off the kernel to compute the pointwise residuals.
	     * @param p Fit parameters.
	     * @param r Residual vector.
	     */
	    virtual void residuals(const gsl_vector*p, gsl_vector* r);
    
	private:
	    void throwCudaError(const char* msg);
	};

	/**
	 * @class FitEngineFactory
	 * @brief Factory for creating fit serial and Cuda-enabled fit engines.
	 */
	class FitEngineFactory {
	public:
	    /** 
	     * @brief Create a single pulse Cuda-enabled fit engine. 
	     * @param data (x, y) values of the data to be fit.
	     * @return Pointer to the created fit engine object. 
	     */
	    CFitEngine* createSerialFitEngine1(
		std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data
		);
	    /** 
	     * @brief Create a single pulse Cuda-enabled fit engine. 
	     * @param data (x, y) values of the data to be fit.
	     * @return Pointer to the created fit engine object. 
	     */
	    CFitEngine* createCudaFitEngine1(
		std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data
		);
	    /** 
	     * @brief Create a double pulse serial fit engine. 
	     * @param data (x, y) values of the data to be fit.
	     * @return Pointer to the created fit engine object. 
	     */
	    CFitEngine* createSerialFitEngine2(
		std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data
		);
	    /** 
	     * @brief Create a double pulse Cuda-enabled fit engine. 
	     * @param data (x, y) values of the data to be fit.
	     * @return Pointer to the created fit engine object. 
	     */
	    CFitEngine* createCudaFitEngine2(
		std::vector<std::pair<std::uint16_t, std::uint16_t>>&  data
		);
	};

	
    }
};

#endif
