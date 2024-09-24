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

/** 
 * @file  CudaFitEngineAnalytic.cu
 * @brief Provide CUDA fit engines for single- and double-pulse fits.
 * @note  This requires that the CUDA compiler be used.
 * @note  Experimentally the Jacobian for double pulses needs to be double 
 * precision so we've got functions named XXXX which are float and identical 
 * functions named XXXXd which are double.
 */

#include <stdexcept>
#include <math.h>

#include "jacobian_analytic.h"

using namespace ddastoys;
using namespace ddastoys::analyticfit;

// Single pulse fit parameter indices:

static const int P1A_INDEX(0);
static const int P1K1_INDEX(1);
static const int P1K2_INDEX(2);
static const int P1X1_INDEX(3);
static const int P1C_INDEX(4);

// Double pulse fit with all parameters free:

static const int P2A1_INDEX(0);
static const int P2K1_INDEX(1);
static const int P2K2_INDEX(2);
static const int P2X1_INDEX(3);
 
static const int P2A2_INDEX(4);
static const int P2K3_INDEX(5);
static const int P2K4_INDEX(6);
static const int P2X2_INDEX(7);

static const int P2C_INDEX(8);

// From functions_analytic.cpp -> device:

/**
 *
 * @brief Evaluate a logistic function for the specified parameters and point.
 *
 * @details
 * A logistic function is a function with a sigmoidal shape.  We use it
 * to fit the rising edge of signals DDAS digitizes from detectors.
 * See e.g. https://en.wikipedia.org/wiki/Logistic_function for
 * a discussion of this function.
 *
 * @param A  Amplitude of the signal.
 * @param k1 Steepness of the signal (related to the rise time).
 * @param x1 Mid point of the rise of the sigmoid.
 * @param x  Location at which to evaluate the function.
 *
 * @return Logistic function evaluated at x.
 */
__device__
static float
logistic(float A, float k, float x1, float x)
{
    return A/(1+expf(-k*(x-x1)));
}

/**
 * @brief Signals from detectors usually have a falling shape that approximates
 * an exponential. This function evaluates this decay at some point.
 *
 * @param A1 Amplitude of the signal
 * @param k1 Decay time factor f the signal.
 * @param x1 Position of the pulse.
 * @param x  Where to evaluate the signal.
 *
 * @return Value of the exponential decay at x.
 */
__device__
static float
decay(float A, float k, float x1, float x)
{
    return A*(expf(-k*(x-x1)));
}

/**
 * @brief Evaluate the value of a single pulse in accordance with our
 * canonical functional form.  
 * 
 * @details 
 * The form is a sigmoid rise with an exponential decay that sits on top of 
 * a constant offset. The exponential decay is turned on with switchOn() 
 * above when x > the rise point of the sigmoid.
 *
 * @param A1 Pulse amplitiude.
 * @parm  k1 Sigmoid rise steepness.
 * @param k2 Exponential decay time constant.
 * @param x1 Sigmoid position.
 * @param C  Constant offset.
 * @param x  Position at which to evaluat this function
 *
 * @return Single pulse evaluated at x.
 */
__device__
static float
singlePulse(
    float A1, float k1, float k2, float x1, float C, float x
    )
{
    return (logistic(A1, k1, x1, x)  * decay(1.0, k2, x1, x)) + C;
}

/**
 * @brief Evaluate the canonical form of a double pulse.
 *
 * @details 
 * This is done by summing two single pulses. The constant term is thrown 
 * into the first pulse. The second pulse gets a constant term of 0.
 *
 * @param A1 Amplitude of the first pulse.
 * @param k1 Steepness of first pulse rise.
 * @param k2 Decay time of the first pulse.
 * @param x1 Position of the first pulse.
 * @param A2 Amplitude of the second pulse.
 * @param k3 Steepness of second pulse rise.
 * @param k4 Decay time of second pulse.
 * @param x2 Position of second pulse.
 * @param C  Constant offset the pulses sit on.
 * @param x  Position at which to evaluate the pulse.
 *
 * @return Double pulse evaluated at x.
 */
__device__
static float
doublePulse(
    float A1, float k1, float k2, float x1,
    float A2, float k3, float k4, float x2,
    float C, float x    
    )
{
    float p1 = singlePulse(A1, k1, k2, x1, C, x);
    float p2 = singlePulse(A2, k3, k4, x2, 0.0, x);
    
    return p1 + p2;
}

///
// Support functions that are in the device.
//

/**
 * @brief Returns the partial derivative of a single pulse with respect to the
 * amplitude evaluated at a point
 *
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x at which to evaluate all this.
 * @param w  Weight for the point.
 *
 * @return Value of (dP1/dA)(x)/w
 */
__device__
static float
dp1dA(float k1, float k2, float x1, float x, float w,
      float erise, float efall)
{
    float d = efall;                      // decay(1.0, k2, x1, x);
    float l = 1.0/(1.0 + erise);          // logistic(1.0, k1, x1, x);
    
    return d*l/w;
}
/**
 * @brief Returns the partial derivative of a single pulse with respect to the
 * amplitude evaluated at a point
 *
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x at which to evaluate all this.
 * @param w  Weight for the point.
 *
 * @return Value of (dP1/dA)(x)/w
 */
__device__
static double
dp1dAd(double k1, double k2, double x1, double x, double w,
       double erise, double efall)
{
    double d = efall;                      // decay(1.0, k2, x1, x);
    double l = 1.0/(1.0 + erise);          // logistic(1.0, k1, x1, x);
    
    return d*l/w;
}

/**
 * @brief Partial of single pulse with respect to the rise time constant k1.
 *
 * @param A  Current guess at amplitude.
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x at which to evaluate all this.
 * @param w  Weight for the point.
 *
 * @return Value of (dP1/dk1)(x)/w
 */
__device__
static float
dp1dk1(float A, float k1, float k2, float x1, float x, float w,
       float erise, float efall)
{
    float d1 =   A*efall;               // decay(A, k2, x1, x);  
    float d2 =   erise; //              // decay(1.0, k1, x1,  x);
    float num = d1*d2*(x - x1);
    float l   =  1.0/(1.0 + erise);     //  logistic(1.0, k1, x1, x);   
    
    return (num*l*l)/w;
}
/**
 * @brief Partial of single pulse with respect to the rise time constant k1.
 *
 * @param A  Current guess at amplitude.
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x at which to evaluate all this.
 * @param w  Weight for the point.
 *
 * @return Value of (dP1/dk1)(x)/w
 */
__device__
static double
dp1dk1d(double A, double k1, double k2, double x1, double x, double w,
	double erise, double efall)
{
    double d1 =   A*efall;               // decay(A, k2, x1, x);  
    double d2 =   erise; //              // decay(1.0, k1, x1,  x);
    double num = d1*d2*(x - x1);
    double l   =  1.0/(1.0 + erise);     //  logistic(1.0, k1, x1, x);   
    
    return (num*l*l)/w;
}

/**
 * @brief Partial of a single pulse with respect to the decay time constant.

 * @param A  Current guess at amplitude.
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x at which to evaluate all this.
 * @param w  Weight for the point .
 *
 * @return   Value of (dP1/dk2)(x)/w.
 */
__device__
static float
dp1dk2(float A, float k1, float k2, float x1, float x, float w,
       float erise, float efall)
{
    float d1 = A*efall;                   // decay(A, k2, x1, x);
    float num = d1*(x1 - x);
    float l = 1.0/(1.0 + erise);          // logistic(1.0, k1, x1, x);
    
    return (num*l)/w;
}
/**
 * @brief Partial of a single pulse with respect to the decay time constant.

 * @param A  Current guess at amplitude.
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x at which to evaluate all this.
 * @param w  Weight for the point .
 *
 * @return   Value of (dP1/dk2)(x)/w.
 */
__device__
static double
dp1dk2d(double A, double k1, double k2, double x1, double x, double w,
	double erise, double efall)
{
    double d1 = A*efall;                   // decay(A, k2, x1, x);
    double num = d1*(x1 - x);
    double l = 1.0/(1.0 + erise);          // logistic(1.0, k1, x1, x);
    
    return (num*l)/w;
}

/**
 * @brief Partial of a single pulse with respect to the time at the middle
 * of the pulse's rise.
 *
 * @param A  Current guess at amplitude.
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x at which to evaluate all this.
 * @param w  Weight for the point.
 * 
 * @return Value of (dP1/dk2)(x)/w.
 */
__device__
static float
dp1dx1(float A, float k1, float k2, float x1, float x, float w,
       float erise, float efall)
{
    float dk1 = erise;                   // decay(1.0, k1, x1, x);
    float dk2 = efall;                   // decay(1.0, k2, x1, x);
    float l   = 1.0/(1.0 + erise);       // logistic(1.0, k1, x1, x);
    
    float left = A*k2*dk2*l;
    float right = A*k1*dk1*dk2*l*l;
    
    return (left - right)/w;
}
/**
 * @brief Partial of a single pulse with respect to the time at the middle
 * of the pulse's rise.
 *
 * @param A  Current guess at amplitude.
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x at which to evaluate all this.
 * @param w  Weight for the point.
 * 
 * @return Value of (dP1/dk2)(x)/w.
 */
__device__
static double
dp1dx1d(double A, double k1, double k2, double x1, double x, double w,
	double erise, double efall)
{
    double dk1 = erise;                   // decay(1.0, k1, x1, x);
    double dk2 = efall;                   // decay(1.0, k2, x1, x);
    double l   = 1.0/(1.0 + erise);       // logistic(1.0, k1, x1, x);
    
    double left = A*k2*dk2*l;
    double right = A*k1*dk1*dk2*l*l;
    
    return (left - right)/w;
}

/**
 * @brief Partial derivative of single pulse with respect to the constant term
 * evaluated at a point.
 *
 * @param A  Current guess at amplitude.
 * @param k1 Current guess at rise steepness param (log(81)/risetime90).
 * @param k2 Current guess at the decay time constant.
 * @param x1 Current guess at pulse position.
 * @param x  x at which to evaluate all this.
 * @param w  Weight for the point.
 *
 * @return Value of (dP1/dC)(x)/w
 */
__device__
static float
dp1dC(float A, float k1, float k2, float x1, float x, float w)
{
    return 1.0/w;
}


///
// The residual and Jacobian computations are pointwise parallel in the GPU
//

/**
 * @brief Compute the residual for a point in the trace with a single pulse fit.
 *
 * @param tx  Pointer to trace x values.
 * @param ty  Pointer to trace y values.
 * @param res Pointer to residual values.
 * @param len Number of trace elements.
 * @param C   Constant baseline.
 * @param A   Scale factor.
 * @param k1  Rise steepeness.
 * @param k2  Decay time.
 * @param x1  Position.
 */
__global__
void residual1(
    void* tx, void* ty, void* res, unsigned len,
    float C, float A, float k1, float k2, float x1)
{
    // Figure out our index... we just don't do anything if it's
    // bigger than len:    
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < len) {
	unsigned short* tracex = static_cast<unsigned short*>(tx);
	unsigned short* tracey = static_cast<unsigned short*>(ty);
	float* resid  = static_cast<float*>(res);
        float x = tracex[i];
        float y = tracey[i];
        
        // Compute the function value.
	// ___device__ function.
        float value = singlePulse(A, k1, k2, x1, C, x);  
        
        // Compute and store the difference:        
        resid[i] = (value - y);        
    }  
}

/**
 * @brief Compute the Jacobian at a single point of the trace for a 
 * single pulse fit.
 *
 * @param tx  Pointer to the trace x coords.
 * @param ty  Pointer to the trace y coords.
 * @param jac Pointer to the Jacobian matrix (len*5 elements)
 * @param len Trace length.
 * @param A   Scale factor.
 * @param k1  Risetime steepeness.
 * @param k2  Decay time.
 * @param x1  Position.
 */
__global__
void jacobian1(
    void* tx, void* ty, void* jac, unsigned len,
    float A, float k1, float k2, float x1
    )
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < len) {
	unsigned short* tracex = static_cast<unsigned short*>(tx);

	float* j = static_cast<float*>(jac);
        float x = tracex[i];
        
        // Common sub-expression elimination:        
        float erise = expf(-k1*(x - x1));
        float efall = expf(-k2*(x - x1));
        
        float dA = dp1dA(k1, k2, x1, x, 1.0, erise, efall);
        float dk1= dp1dk1(A, k1, k2, x1, x, 1.0, erise, efall);
        float dk2= dp1dk2(A, k1, k2, x1, x, 1.0, erise, efall);
        float dx = dp1dx1(A, k1, k2, x1, x, 1.0, erise, efall);
        float dC = dp1dC(A, k1, k2, x1, x, 1.0);
        
        // Put these results in the appropriate Jacobian element:        
        int n = i;
        j[n] = dA;   n += len;
        j[n] = dk1;  n += len;
        j[n] = dk2;  n += len;
        j[n] = dx;   n += len;
        j[n] = dC;
    }
}

///
//  The class implementationon is in the host (CPU).
//

/**
 * @details
 * Allocate the device vectors/matrices. Push the trace x/y points into the 
 * GPU where they stay until we're destroyed.
 */
ddastoys::CudaFitEngine1::CudaFitEngine1(
    std::vector<std::pair<uint16_t, uint16_t>>& data
    ) :
    FitEngine(data)
{
    // Mashall the trace into x/y arrays... this lets them be CUDA memcpied
    // to the GPU    
    m_npts = data.size();    
    unsigned short x[m_npts];
    unsigned short y[m_npts];
    for (int i = 0; i < m_npts; i++) {
        x[i] = data[i].first;
        y[i] = data[i].second;
    }
    
    // The trace:    
    if (cudaMalloc(&m_dXtrace, m_npts*sizeof(unsigned short)) != cudaSuccess) {
        throwCudaError("Failed to allocated X trace points");
    }
    if (cudaMalloc(&m_dYtrace, m_npts*sizeof(unsigned short)) != cudaSuccess) {
        throwCudaError("Failed to allocatee Y trace points");
    }
    if (cudaMemcpy(
            m_dXtrace, x, m_npts*sizeof(unsigned short), cudaMemcpyHostToDevice
	    ) != cudaSuccess) {
        throwCudaError("Failed to move trace x coords -> gpu");
    }
    if (cudaMemcpy(
            m_dYtrace, y, m_npts*sizeof(unsigned short), cudaMemcpyHostToDevice
	    ) != cudaSuccess) {
        throwCudaError("Failed to move trace y coords -> gpu");
    }
    
    // The residual and jacobians need to be allocated but are filled in by
    // the GPU kernels:    
    if(cudaMalloc(&m_dResiduals, m_npts*sizeof(float)) != cudaSuccess) {
        throwCudaError("Failed to allocate residual vector");
    }
    if (cudaMalloc(&m_dJacobian, m_npts*5*sizeof(float)) != cudaSuccess) {
        throwCudaError("Failed to allocated Jacobian");
    }
}

/**
 * @details
 * Just deallocate the GPU resources.
 */
ddastoys::CudaFitEngine1::~CudaFitEngine1()
{
    // Not much point in error checking as we're not going to be able to
    // do anything about errors here anyway.    
    cudaFree(m_dXtrace);
    cudaFree(m_dYtrace);
    cudaFree(m_dResiduals);
    cudaFree(m_dJacobian);
}

/**
 * @details
 * We use a Y size of 32 and x size of npts+31/32. That is one warp wide.
 */
void
ddastoys::CudaFitEngine1::jacobian(const gsl_vector* p, gsl_matrix* J)
{
    float A   = gsl_vector_get(p, P1A_INDEX);
    float k1  = gsl_vector_get(p, P1K1_INDEX);
    float k2  = gsl_vector_get(p, P1K2_INDEX);
    float x1  = gsl_vector_get(p, P1X1_INDEX);
    float C   = gsl_vector_get(p, P1C_INDEX);
    
    jacobian1<<<(m_npts+31)/32, 32>>>(
        m_dXtrace, m_dYtrace, m_dJacobian, m_npts, A, k1, k2, x1
	);
    
    if(cudaDeviceSynchronize() != cudaSuccess) {
	throwCudaError("Synchronizing kernel"); // Block until kernel done.
    }
    
    // Now we need to pull the Jacobian out of the device:    
    float Jac[m_npts*5]; // We'll do it flat
    if(
	cudaMemcpy(
	    Jac, m_dJacobian, m_npts*5*sizeof(float), cudaMemcpyDeviceToHost
	    ) != cudaSuccess
	) {
        throwCudaError("failed to copy Jacobian from device");
    }
    
    // Finally, we have to put the jacobian into the GSL J matrix.    
    for (int i = 0; i < m_npts; i++) {
        gsl_matrix_set(J, i, 0, Jac[i]);
        gsl_matrix_set(J, i, 1, Jac[i+m_npts]);
        gsl_matrix_set(J, i, 2, Jac[i+(2*m_npts)]);
        gsl_matrix_set(J, i, 3, Jac[i+(3*m_npts)]);
        gsl_matrix_set(J, i, 4, Jac[i+(4*m_npts)]);
    }
}

void
ddastoys::CudaFitEngine1::residuals(const gsl_vector* p, gsl_vector* r)
{
    float A   = gsl_vector_get(p, P1A_INDEX);
    float k1  = gsl_vector_get(p, P1K1_INDEX);
    float k2  = gsl_vector_get(p, P1K2_INDEX);
    float x1  = gsl_vector_get(p, P1X1_INDEX);
    float C   = gsl_vector_get(p, P1C_INDEX);

    residual1<<<(m_npts+31)/32, 32>>>(
        m_dXtrace, m_dYtrace, m_dResiduals, m_npts, C, A, k1, k2, x1
	);
    
    if(cudaDeviceSynchronize() != cudaSuccess) {
	throwCudaError("Synchronizing kernel");	// Block for kernel completion.
    }
    
    // Fetch out the residuals and push the into the r vector:    
    float resids[m_npts];
    if (
	cudaMemcpy(
	    resids, m_dResiduals, m_npts*sizeof(float), cudaMemcpyDeviceToHost
	    ) != cudaSuccess
	) {
        throwCudaError("Failed to pull residuals from GPU");
    }
    
    // Push the results into r:    
    for (int i =0; i < m_npts; i++) {
        gsl_vector_set(r, i, resids[i]);
    }
}

/** 
 * @breif Throw a CUDA error as std::runtime_error.
 *
 * @details
 * - Find the last CUDA error.
 * - Make a string out of the message we're passed and the CUDA error.
 * - Throw this all as a runtime_error.
 *
 *  @param msg Context message.
 */
void
ddastoys::CudaFitEngine1::throwCudaError(const char* msg)
{
    std::string e="Error: ";
    e += msg;
    e += " : ";
    
    cudaError_t status = cudaGetLastError();
    e += cudaGetErrorString(status);
    
    throw std::runtime_error(e);
}

///////////////////////////////////////////////////////////////////////////
// CudaFitEngine2 implementation - double pulse fits.
//

// Device (GPU) kernels needed:

/**
 * @brief Computes the double-pulse residual pointwise parallel.
 *
 * @param xtc  x-coordinates of trace.
 * @param ytc  y-coordinates of trace.
 * @param res  Residuals to compute.
 * @param npts Number of trace points.
 * @param C    Constant offset fit parameter.
 * @param A1   Scale factor for pulse1.
 * @param k11  k1 for pulse 1.
 * @param k12  k2 for pulse 1.
 * @param x1   Position of pulse 1
 * @param A2   Scale factof for pulse 2.
 * @param k21  k1 for pulse 2.
 * @param k22  k2 for pulse 2.
 * @param x2   Position of pulse 2.
 */
__global__
void residual2(
    void* xtc, void* ytc, void* res, unsigned npts,
    float C,
    float A1, float k11, float k12, float x1,
    float A2, float k21, float k22, float x2
    )
{
    // Compute our index and only do anything if its < npts:    
    int i  = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < npts) {
	unsigned short* xc = static_cast<unsigned short*>(xtc);
	unsigned short* yc = static_cast<unsigned short*>(ytc);
	float* r = static_cast<float*>(res);
	float x = xc[i];
        float y = yc[i];
        float fit = doublePulse(A1, k11, k12, x1, A2, k21, k22, x2, C, x);
        r[i] = fit - y;
    }
}

/**
 * @brief Compute the double-pulse Jacobian on a point of the pulse. 
 * The Jacobian matrix is an npts x 9 matrix.
 *
 * @param xtc  x-coordinates of the trace.
 * @param jac  Jacobian matrix.
 * @param npts Number of points in the fit.
 * @param A1, k1, k2, x1 Fit parameters for first pulse.
 * @param A2, k3, k4, x2 Fit parameters for the second pulse.
 * @param C    Constant term of the fit.
 */
__global__
void jacobian2(
    void* xtc,  void* jac, unsigned npts,
    double A1, double k1, double k2, double x1,
    double A2, double k3, double k4, double x2,
    double C
    )
{
    // Figure out which point we're doing and compute if it's in the range
    // of the trace:    
    int i  = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < npts) {
	unsigned short* xc = static_cast<unsigned short*>(xtc);
	double* j = static_cast<double*>(jac);
      
        // Now the Jacobian elements:        
        int k = i; // We'll increment this by npts for each j element
        double x = xc[i];
        
        // Common subexpression elmiination between functions:        
        double erise1 = exp(-k1*(x - x1));
        double efall1 = exp(-k2*(x - x1));
        
        double erise2 = exp(-k3*(x - x2));
        double efall2 = exp(-k4*(x - x2));
        
        // Pulse 1 elements:       
        j[k] = dp1dAd(k1, k2, x1, x, 1.0, erise1, efall1);      k += npts;
        j[k] = dp1dk1d(A1, k1, k2, x1, x, 1.0, erise1, efall1); k += npts;
        j[k] = dp1dk2d(A1, k1, k2, x1, x, 1.0, erise1, efall1); k += npts;
        j[k] = dp1dx1d(A1, k1, k2, x1, x, 1.0, erise1, efall1); k += npts;
        
        // Pulse 2 elements:     
        j[k] = dp1dAd(k3, k4,x2,x, 1.0, erise2, efall2);        k += npts;
        j[k] = dp1dk1d(A2, k3, k4, x2, x, 1.0, erise2, efall2); k += npts;
        j[k] = dp1dk2d(A2, k3, k4, x2, x, 1.0, erise2, efall2); k += npts;
        j[k] = dp1dx1d(A2, k3, k4, x2, x, 1.0, erise2, efall2); k += npts;
        
        // Constant element:       
        j[k] = 1.0;
    }
}

////////////////////
// Host class implementation:
//

/**
 * @details
 * Allocate the GPU resources:
 * * Trace x array
 * * Trace y array.
 * * Residual array.
 * * Jacobian vector (m_npts * 9)
 * * Move the trace into the GPU where it stays for all iterations of the fit.
 */
ddastoys::CudaFitEngine2::CudaFitEngine2(
    std::vector<std::pair<uint16_t, uint16_t>>&  data
    ) :
    FitEngine(data)
{
    // Make separate x/y arrays from the data:    
    m_npts = data.size();
    unsigned short x[m_npts];
    unsigned short y[m_npts];
    for (int i =0; i < m_npts; i++) {
	x[i] = data[i].first;
	y[i] = data[i].second;
    }
    
    // Allocate the trace arrays and move the trace in:    
    if (cudaMalloc(&m_dXtrace, m_npts*sizeof(unsigned short)) != cudaSuccess) {
        throwCudaError("Unable to allocate GPU x trace array");
    }
    if (cudaMalloc(&m_dYtrace, m_npts*sizeof(unsigned short)) != cudaSuccess) {
        throwCudaError("Unable to allocate CPU y trace array");
    }
    
    if(
        cudaMemcpy(
            m_dXtrace, x, m_npts*sizeof(unsigned short), cudaMemcpyHostToDevice
	    ) != cudaSuccess
	) {
        throwCudaError("Unable to move x coords of trace -> GPU");
    }
    if(cudaMemcpy(
	   m_dYtrace, y, m_npts*sizeof(unsigned short), cudaMemcpyHostToDevice
	   ) != cudaSuccess ) {
        throwCudaError("Unable to move y coords of trace -> GPU");
    }
    
    // Allocate the residuals and Jacobian:     
    if(cudaMalloc(&m_dResiduals, m_npts*sizeof(float)) != cudaSuccess) {
        throwCudaError("Unable to allocate residual array in GPU");
    }
    if (cudaMalloc(&m_dJacobian, m_npts*9*sizeof(double)) != cudaSuccess) {
        throwCudaError("Unable to allocated jacobian matrix in GPU");
    }
}

/**
 * @details
 * Just frees the device blocks.
 */
ddastoys::CudaFitEngine2::~CudaFitEngine2()
{
    // No point in looking for errors since we don't know how to recover:    
    cudaFree(m_dXtrace);
    cudaFree(m_dYtrace);
    cudaFree(m_dResiduals);
    cudaFree(m_dJacobian);
}

/**
 * @note We organize the computing into 32 thread blocks because there are 
 * 32 thread per warp.
 */
void
ddastoys::CudaFitEngine2::jacobian(const gsl_vector* p, gsl_matrix* J)
{
    double A1    = gsl_vector_get(p, P2A1_INDEX);   // Pulse 1.
    double k1    = gsl_vector_get(p, P2K1_INDEX);
    double k2    = gsl_vector_get(p, P2K2_INDEX);
    double x1    = gsl_vector_get(p, P2X1_INDEX);
    
    double A2    = gsl_vector_get(p, P2A2_INDEX);   // Pulse 2.
    double k3    = gsl_vector_get(p, P2K3_INDEX);
    double k4    = gsl_vector_get(p, P2K4_INDEX);
    double x2    = gsl_vector_get(p, P2X2_INDEX);
    
    double C     = gsl_vector_get(p, P2C_INDEX);    // constant.
    
    jacobian2<<<(m_npts + 31)/32, 32>>>(
        m_dXtrace, m_dJacobian, m_npts,
        A1, k1, k2, x1,
        A2, k3, k4, x2,
        C
	);
    
    if(cudaDeviceSynchronize() != cudaSuccess)
	throwCudaError("Failed kernel synchronization");
    
    // Fetch the jacobian and marshall it into j:    
    double jac[m_npts*9];
    if (
	cudaMemcpy(
	    jac, m_dJacobian, m_npts*9*sizeof(double), cudaMemcpyDeviceToHost
	    ) != cudaSuccess
	) {
        throwCudaError("Failed to fetch double-pulse Jacobian from GPU");
    }
    
    for (int i =0; i < m_npts; i++) {
        int k = i;
        gsl_matrix_set(j, i, 0, jac[k]);  k += m_npts;
        gsl_matrix_set(j, i, 1, jac[k]);  k += m_npts;
        gsl_matrix_set(j, i, 2, jac[k]);  k += m_npts;
        gsl_matrix_set(j, i, 3, jac[k]);  k += m_npts;
        gsl_matrix_set(j, i, 4, jac[k]);  k += m_npts;
        gsl_matrix_set(j, i, 5, jac[k]);  k += m_npts;
        gsl_matrix_set(j, i, 6, jac[k]);  k += m_npts;
        gsl_matrix_set(j, i, 7, jac[k]); k += m_npts;
        gsl_matrix_set(j, i, 8, jac[k]); k += m_npts;    
    }
}

void
ddastoys::CudaFitEngine2::residuals(const gsl_vector* p, gsl_vector* r)
{
    // Pull out the current fit parameters:    
    float A1    = gsl_vector_get(p, P2A1_INDEX);   // Pulse 1.
    float k1    = gsl_vector_get(p, P2K1_INDEX);
    float k2    = gsl_vector_get(p, P2K2_INDEX);
    float x1    = gsl_vector_get(p, P2X1_INDEX);
    
    float A2    = gsl_vector_get(p, P2A2_INDEX);   // Pulse 2.
    float k3    = gsl_vector_get(p, P2K3_INDEX);
    float k4    = gsl_vector_get(p, P2K4_INDEX);
    float x2    = gsl_vector_get(p, P2X2_INDEX);
    
    float C     = gsl_vector_get(p, P2C_INDEX);    // constant.
 
    // Fire off the kernel to do all this in pointwise parallel:    
    residual2<<<(m_npts+31)/32,  32>>>(
        m_dXtrace, m_dYtrace, m_dResiduals, m_npts,
        C, 
	A1, k1, k2, x1, 
	A2, k3, k4, x2
	);
    
    if(cudaDeviceSynchronize() != cudaSuccess)
	throwCudaError("Failed to synchronize kernel");
    
    // Now we pull out the residuals vector and put it into r:    
    float residuals[m_npts];
    if (
	cudaMemcpy(
	    residuals, m_dResiduals, m_npts*sizeof(float),
	    cudaMemcpyDeviceToHost
	    ) != cudaSuccess
	) {
        throwCudaError("Unable to fetch residuals from GPU");
    }
    
    for (int i =0; i < m_npts; i++) {
        gsl_vector_set(r, i, residuals[i]);
    }
}
/**
 * @brief See this method in CudaFitEngine1.
 * 
 * @details
 * Here's a source for factorization into a base class... along with the 
 * allocation of the trace and residual as well as the push of the trace 
 * into the GPU.
 * 
 * @param msg - message used to construct the exception messgae.
 */
void
ddastoys::CudaFitEngine2::throwCudaError(const char* msg)
{
    std::string e="Error: ";
    e += msg;
    e += " : ";
    
    cudaError_t status = cudaGetLastError();
    e += cudaGetErrorString(status);
    
    throw std::runtime_error(e);    
}

