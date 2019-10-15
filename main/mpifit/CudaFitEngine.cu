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

/** @file:  CudaFitEngine.cpp
 *  @brief: Provide CUDA fit engines for 1-2 pulse fits.
 *          Note this requires that the cuda compiler be used.
 */


#include "jacobian.h"
#include <stdexcept>

// From functions.cpp -> device:




/**
 * logistic
 *    Evaluate a logistic function for the specified parameters and point.
 *    A logistic function is a function with a sigmoidal shape.  We use it
 *    to fit the rising edge of signals DDAS digitizes from detectors.
 *    See e.g. https://en.wikipedia.org/wiki/Logistic_function for
 *    a discussion of this function.
 *
 * @param A  - Amplitude of the signal.
 * @param k1 - steepness of the signal (related to the rise time).
 * @param x1 - Mid point of the rise of the sigmoid.
 * @param x  = Location at which to evaluate the function.
 * @return double
 */
__device__
static float
logistic(float A, float k, float x1, float x)
{
    return A/(1+exp(-k*(x-x1)));
}

/**
 * decay
 *    Signals from detectors usually have a falling shape that approximates
 *    an exponential.  This function evaluates this decay at some point.
 *
 *  @param A1 - amplitude of the signal
 *  @param k1 - Decay time factor f the signal.
 *  @param x1 - Position of the pulse.
 *  @param x  - Where to evaluate the signal.
 *  @return double
 */
__device__
static float
decay(float A, float k, float x1, float x)
{
    return A*(exp(-k*(x-x1)));
}

/**
 * singlePulse
 *    Evaluate the value of a single pulse in accordance with our
 *    canonical functional form.  The form is a sigmoid rise with an
 *    exponential decay that sits on top of a constant offset.
 *    The exponential decay is turned on with switchOn() above when
 *    x > the rise point of the sigmoid.
 *
 * @param A1  - pulse amplitiude
 * @parm  k1  - sigmoid rise steepness.
 * @param k2  - exponential decay time constant.
 * @param x1  - sigmoid position.
 * @param C   - Constant offset.
 * @param x   - Position at which to evaluat this function
 * @return double
 */
__device__
static float
singlePulse(
    float A1, float k1, float k2, float x1, float C, float x
)
{
    return (logistic(A1, k1, x1, x)  * decay(1.0, k2, x1, x)) // decay term
        + C;                                        // constant.
}

/**
 * doublePulse
 *    Evaluate the canonical form of a double pulse.  This is done
 *    by summing two single pulses.  The constant term is thrown into the
 *    first pulse.  The second pulse gets a constant term of 0.
 *
 * @param A1   - Amplitude of the first pulse.
 * @param k1   - Steepness of first pulse rise.
 * @param k2   - Decay time of the first pulse.
 * @param x1   - position of the first pulse.
 *
 * @param A2   - Amplitude of the second pulse.
 * @param k3   - Steepness of second pulse rise.
 * @param k4   - Decay time of second pulse.
 * @param x2   - position of second pulse.
 *
 * @param C    - Constant offset the pulses sit on.
 * @param x    - position at which to evaluate the pulse.
 * @return double.
 * 
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

// Support functions that are in the device:

/**
 * dp1dA
 *    Returns the partial derivative of a single pulse with respect to the
 *    amplitude evaluated at a point
 *
 * @param k1 - current guess at rise steepness param (log(81)/risetime90).
 * @param k2 - current guess at the decay time constant.
 * @param x1 - Current guess at pulse position.
 * @param x  - X at which to evaluate all this.
 * @param w  - weight for the point 
 * @return double - Value of (dP1/dA)(x)/w
*/
__device__
static float
dp1dA(float k1, float k2, float x1, float x, float w,
      float erise, float efall)
{
    float d = efall;                      // decay(1.0, k2, x1, x);
    float l = 1.0/(1.0 + erise);              // logistic(1.0, k1, x1, x);
    return d*l / w;
}
/**
 * dp1dk1
 *    Partial of single pulse with respect to the rise time constant k1.
 *
 * @param A - current guess at amplitude.
 * @param k1 - current guess at rise steepness param (log(81)/risetime90).
 * @param k2 - current guess at the decay time constant.
 * @param x1 - Current guess at pulse position.
 * @param x  - X at which to evaluate all this.
 * @param w  - weight for the point 
 * @return double - Value of (dP1/dk1)(x)/w
 */
__device__
static float
dp1dk1(float A, float k1, float k2, float x1, float x, float w,
       float erise, float efall)
{
    float d1 =   A*efall;               // decay(A, k2, x1, x);  
    float d2 =   erise; //              // decay(1.0, k1, x1,  x);   // part of logistic deriv.
    float num = d1*d2*(x - x1);
    float l   =  1.0/(1.0 + erise);     //  logistic(1.0, k1, x1, x);   
    
    
    return (num*l*l)/w;
}
/**
 * dp1dk2
 *    Partial of a single pulse with respect to the decay time constant.
 * @param A - current guess at amplitude.
 * @param k1 - current guess at rise steepness param (log(81)/risetime90).
 * @param k2 - current guess at the decay time constant.
 * @param x1 - Current guess at pulse position.
 * @param x  - X at which to evaluate all this.
 * @param w  - weight for the point 
 * @return double - Value of (dP1/dk2)(x)/w
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
 * dp1dx1
 *    Partial of a single pulse with respect to the time at the middle
 *    of the pulse's rise.
 *
 * @param A - current guess at amplitude.
 * @param k1 - current guess at rise steepness param (log(81)/risetime90).
 * @param k2 - current guess at the decay time constant.
 * @param x1 - Current guess at pulse position.
 * @param x  - X at which to evaluate all this.
 * @param w  - weight for the point 
 * @return double - Value of (dP1/dk2)(x)/w
 */
__device__
static float
dp1dx1(float A, float k1, float k2, float x1, float x, float w,
       float erise, float efall)
{
    float dk1 = erise;                   // = decay(1.0, k1, x1, x);
    float dk2 = efall;                   // decay (1.0, k2, x1, x);
    float l   = 1.0/(1.0 + erise);       // logistic(1.0, k1, x1, x);
    
    float left = A*k2*dk2*l;
    float right = A*k1*dk1*dk2*l*l;
    
    return (left - right)/w;
}
/**
 * dp1dC
 *    Partial derivative of single pulse with respect to the constant term
 *    evaluated at a point.
 *
 * @param A - current guess at amplitude.
 * @param k1 - current guess at rise steepness param (log(81)/risetime90).
 * @param k2 - current guess at the decay time constant.
 * @param x1 - Current guess at pulse position.
 * @param x  - X at which to evaluate all this.
 * @param w  - weight for the point 
 * @return double - Value of (dP1/dC)(x)/w
 */
__device__
static float
dp1dC(float A, float k1, float k2, float x1, float x, float w)
{
    
    return 1.0/w;
}


/**
 * The residual and jacobian copmutations are pointwise parallel in the device
 *  (GPU)
 */

/**
 * residual1
 *    Compute the residual for a point in the trace with a single pulse fit.
 *
 * @param tracex  - Pointer to trace x values.
 * @param tracey  - Pointer to trace y values.
 * @param resid   - Pointer to residual values.
 * @param len     - Number of trace elements
 * @param C       - Constant.
 * @param A       - Scale factor.
 * @param k1      - rise-steepeness.
 * @param k2      - decay time.
 * @param x1      - position.
 */
__global__
void residual1(
    unsigned short* tracex, unsigned short* tracey, float* resid, unsigned len,
    float C, float A, float k1, float k2, float x1)
{
    // Figure out our index... we just don't do anything if it's
    // bigger than len:
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < len) {
        float x = tracex[i];
        float y = tracey[i];
        
        // Compute the function value.
        
        float value = singlePulse(A, k1, k2, x1, C, x);  // ___device__ function.
        
        // Compute and store the difference:
        
        resid[i] = (value - y);
        
    }  
}
/**
 * jacobian1
 *    Compute the jacobian at a single point of the trace for a single pulse fit.
 *
 * @param tracex - pointer to the trace x coords.
 * @param tracey - pointer to the trace y coords.
 * @param j      - pointer to the jacobian matrix (len*5 elements)
 * @param len    - trace length.
 * @param A       - Scale factor.
 * @param k1      - rise-steepeness.
 * @param k2      - decay time.
 * @param x1      - position.
 */
__global__
void jacobian1(
    unsigned short* tracex, unsigned short* tracey, float* j, unsigned len,
    float A, float k1, float k2, x1
)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < len) {
        float x = tracex[i];
        
        // Common sub-expression elimination:
        
        float erise = exp(-k1*(xi - x1));
        float efall = exp(-k2*(xi - x1));
        
        float dA = dp1dA(k1, k2, x1, x, 1.0, erise, efall);
        float dk1= dp1k1(A, k1, k2, x1, x, 1.0, erise, efall);
        float dk2= dp1k2(A, k1, k2, x1m xm 1,9m erise, efall);
        float dx = dp1dx1(A, k1, k2, x1, x, 1.0, erise, efall);
        float dC = dp1dC(a, k1, k2, x1, x, 1.0);
        
        // Put these results in the appropriate Jacobian element:
        
        int n = i;
        j[n] = dA;   n += len;
        j[n] = dk1;  n += len;
        j[n] = dk2;  n += len;
        j[n] = dx;   n += len;
        j[n] = dC;
    }
}

/**
 *  The class implementationon is in the host (CPU).
 */

/**
 * constructor
 *   - Allocate the device vectors/matrices.
 *   - push the trace x/y points into the GPU where they stay until we're destroyed.
 */
CudaFitEngine1::CudaFitEngine1(std::vector<std::pair<uint16_t, uint16_t>>& data)
{
    // Mashall the trace into x/y arrays.. this lets them be cuda memcpied to the
    // GPU
    
    unsigned m_npts = data.size();
    
    unsigned short x[m_npts];
    unsigned short y[m_npts];
    for (int i =0; i < m_npts; i++) {
        x[i] = data[i].first;
        y[i] = data[i].second;
    }
    
    // The trace:
    
    if (cudaMalloc(&m_dxTrace, m_npts*sizeof(unsigned short)) != cudaSuccess) {
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
 * destructor just deallocateds the GPU resources.
 */
CudaFitEngine1::~CudaFitEngine1()
{
    // Not much point in error checking as we're not going to be able to
    // do anything about errors here anyway.
    
    CudaFree(m_dXtrace);
    CudaFree(m_dYtrace);
    CudaFree(m_dResiduals);
    CudaFree(m_dJacobian);
}
/**
 * jacobian
 *    Invoke the kernal to do the pointwise parallel jacobian computation.
 *    We use a Y size of 32 and x size of npts+31/32.  That is one warp wide.
 *
 * @param p - Parameter vector.
 * @param j - Jacobian matrix.
 */
void
CudaFitEngine1::jacobian(const gsl_vector* p, gsl_matrix* J)
{
    float A   = gsl_vector_get(p, P1A_INDEX);
    float k1  = gsl_vector_get(p, P1K1_INDEX);
    float k2  = gsl_vector_get(p, P1K2_INDEX);
    float x1  = gsl_vector_get(p, P1X1_INDEX);
    float C   = gsl_vector_get(p, P1C_INDEX);
    
    jacobian1<<<(m_npts+31)/32 32>>>(
        m_dXtrace, m_dYtrace, m_dJacobian, m_npts,
        A, k1, k2, x1
    );
    // Now we need to pull the jacobian out of the device:
    
    float Jac[npts*5];       // we'll do it flat:
    if(
        cudaMemcpy(Jac, m_dJacobian, npts*5*sizeof(float), cudaMemcpyDeviceToHost)
        != cudaSuccess
    ) {
        throwCudaError("failed to copy Jacobian from device");
    }
    
    // finally, we have to put the jacobian into the gsl J matrix.
    
    for (int i = 0; i < m_npts; i++) {
        gsl_matrix_set(J, i, 0, Jac[i]);
        gsl_matrix_set(J, i, 1, Jac[i+m_npts]);
        gsl_matrix_set(J, i, 2, Jac[i+(2*m_npts)]);
        gsl_matrix_set(J, i, 3, Jac[i+(3*m_npts)]);
        gsl_matrix_set(J, i, 4, Jac[i+(4*m_npts)]);
    }
}
/**
 * residuals
 *    Triggers a pointwise parallel residual kernel in the
 *    Device and impedance matches that with gsl's requirements.
 *
 *  @param p  - parameter vector.
 *  @param r  - Residual vector.
 */
void
CudaFitEngine1::residuals(const gsl_vector* p, gsl_vector* r)
{
    float A   = gsl_vector_get(p, P1A_INDEX);
    float k1  = gsl_vector_get(p, P1K1_INDEX);
    float k2  = gsl_vector_get(p, P1K2_INDEX);
    float x1  = gsl_vector_get(p, P1X1_INDEX);
    float C   = gsl_vector_get(p, P1C_INDEX);

    residual1<<<(m_npts+31)/32, 32>>>(
        m_dXtrace, m_dYtrace, m_dResiduals, m_npts,
        C, A, k1, k2, x1
    );
    // Fetch out the residuals and push the minto the r vector:
    
    float resids[m_npts];
    if (cudaMemcpy(
        resids, m_dResiduals, m_npts*sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        throwCudaError("Failed to pull residuals from GPU");
    }
    // Push the results into r:
    
    for (int i =0; i < m_npts; i++) {
        gsl_vector_set(r, i, resids[i]);
    }
}
/**
 * throwCudaError
 *    - Find the last cuda error
 *    - Make a string out of the message we're passed and the cuda error.
 *    - throw this all as a runtime_error
 *
 *  @param msg - context message.
 */
void
CudaFitEngine1::throwCudaError(const char* msg)
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