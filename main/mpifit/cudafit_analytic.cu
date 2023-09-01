/**
 * @author Ron Fox<fox@nscl.msu.edu>
 * @file cudafit_analytic.cu
 * @brief Provide trace fitting using the libucdafit library.
 * @note  We provide call compatible interfaces with lmfit1 and lmfit2.
 * @note  This fit will not thread due to libcudaoptimize's need for us to 
 * have global data for the device pointers to the trace.
 */

#include <limits>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <float.h>

#include <DE_Optimizer.h>   // Differntial evolution Optimizer beast.
#include <PSO_Optimizer.h>  // Particle swarm

#include "fit_extensions.h" // For the fit extension formats.
#include "functions_analytic.h"
#include "reductions.cu"

// Define the parameter numbers for the fits:

static const unsigned A1 = 0;
static const unsigned K1 = 1; // Rise steepness
static const unsigned K2 = 2; // Exponential decay
static const unsigned X1 = 3;
static const unsigned C  = 4;

static const unsigned P1_NPARAMS = 5;

static const unsigned A2 = 5;
static const unsigned K3 = 6;
static const unsigned K4 = 7;
static const unsigned X2 = 8;

static const unsigned P2_NPARAMS = 9;

/**
 * Here's why we can't have good things (threadable). The libcudaoptimizer 
 * does not let me (to my knowledge) pass a parameter to my fitness function 
 * so I don't know how to get this information to it other than making it file 
 * scoped which is inherently thread-unsafe.
 */

static unsigned short* d_xCoords;     // Trace x-coordinates.
static unsigned short* d_yCoords;     // Trace y-coordinates.
static std::vector<uint16_t> xcoords; // The trace locally.
static std::vector<uint16_t> ycoords; // For debugging.
static unsigned        n_tracePoints; // Number of points in the trace.
static float*          h_pWeights(0); // Host weights pointer.
static float*          d_pWeights(0); // Device weights pointer.

/**
 * @brief Report the most recent Cuda error as an std::runtime_error.
 *
 * @param context Describes the error context.
 */
static void
reportCudaError(const char* context)
{
    std::string msg("Error: ");
    msg += context;
    msg += " : ";
    cudaError_t status = cudaGetLastError();
    msg += cudaGetErrorString(status);
    throw std::runtime_error(msg);

}

/**
 * @brief Use the limits and saturation values to suppress some trace points. 
 * Generates the x/y coordinates of the tracea that's left.
 *
 * @param trace      Raw trace.
 * @param limits     Left/right limits of the trace.
 * @param saturation Saturation values for the trace (values >= to this are 
 *   eliminated).
 * 
 * @return Final number of points to fit.
 */
static unsigned traceToGPU(
    std::vector<uint16_t> trace, std::pair<unsigned, unsigned> limits,
    uint16_t saturation
    )
{
    xcoords.clear();
    ycoords.clear(); 

    int result(0);
    for (int i = limits.first; i < limits.second; i++) {
	if (trace[i] < saturation) {
	    xcoords.push_back(i);
	    ycoords.push_back(trace[i]);
	    result++;
	}
    }
    
    // Allocate a pair of unsigned short device arrays: d_xCoords and d_yCoords
    // and move the data from xcoords and ycoords into them:
    if (
	cudaMalloc(
	    &d_xCoords, xcoords.size()*sizeof(unsigned short)) != cudaSuccess
	) {
	reportCudaError("Allocating GPU memory for trace x-coordinates");
    }
    if (
	cudaMalloc(
	    &d_yCoords, ycoords.size()*sizeof(unsigned short)
	    ) != cudaSuccess
	) {
	reportCudaError("Allocating GPU memory for trace y-coordinates");
    }

    if (cudaMemcpy(
	    d_xCoords, xcoords.data(), xcoords.size()*sizeof(unsigned short),
	    cudaMemcpyHostToDevice
	    ) != cudaSuccess) {
	reportCudaError("Moving trace x coordinates into the GPU");
    }
    if (cudaMemcpy(
	    d_yCoords, ycoords.data(), ycoords.size()*sizeof(unsigned short),
	    cudaMemcpyHostToDevice
	    ) != cudaSuccess) {
	reportCudaError("Moving trace y coordinates into the GPU");
    }
  
    // We'll use weights of 1.0; this can be modified here:
    h_pWeights = static_cast<float*>(malloc(result * sizeof(float)));
    for (int i =0; i < result; i++) {
	h_pWeights[i] = 1.0;
    }
  
    if(cudaMalloc(&d_pWeights, result*sizeof(float)) != cudaSuccess) {
	reportCudaError("Failed to allocates device weights array");
    }

    if (cudaMemcpy(
	    d_pWeights, h_pWeights, result*sizeof(float),
	    cudaMemcpyHostToDevice
	    ) != cudaSuccess) {
	reportCudaError("Failed to copy wieghts into the device");
    }

    n_tracePoints = result;
    return result;
}
/**
 * @brief Release the GPU memory associated with the trace:
 */
static void
freeTrace()
{
    cudaFree(d_xCoords);
    cudaFree(d_yCoords);
    cudaFree(d_pWeights);
    free(h_pWeights);
}
/**
 * @brief Evaluate a logistic function for the specified parameters and point.
 *
 * @details
 * A logistic function is a function with a sigmoidal shape. We use it
 * to fit the rising edge of signals DDAS digitizes from detectors.
 * See e.g. https://en.wikipedia.org/wiki/Logistic_function for
 * a discussion of this function.
 *
 * @param A  Amplitude of the signal.
 * @param k  Steepness of the signal (related to the rise time).
 * @param x1 Mid point of the rise of the sigmoid.
 * @param x  Location at which to evaluate the function.
 * @return Logistic function evaluated at x.
 */
__host__ __device__ float
logistic(float A, float  k, float x1, float x)
{
    return A/(1+expf(-k*(x-x1)));
}

/**
 * @brief Exponential decay function.
 *
 * @details
 * Signals from detectors usually have a falling shape that approximates
 * an exponential. This function evaluates this decay at some point.
 *
 * @param A  Amplitude of the signal
 * @param k  Decay time factor f the signal.
 * @param x1 Position of the pulse.
 * @param x  Where to evaluate the signal.
 * @return Exponential decay evaluated at x.
 */
__host__ __device__ float
decay(float A, float k, float  x1, float x)
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
 * @param A1 Pulse amplitiude
 * @param  k1 Sigmoid rise steepness.
 * @param k2 Exponential decay time constant.
 * @param x1 Sigmoid position.
 * @param C  Constant offset.
 * @param x  Position at which to evaluate this function.
 * @return Single pulse evaluated at x.
 */
__host__ __device__ float
singlePulse(
    float A1, float  k1, float  k2, float x1, float  C, float  x
    )
{
    return (logistic(A1, k1, x1, x) * decay(1.0, k2, x1, x)) + C;
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
 * @return Value of the double-pulse function evaluated at x.
 * 
 */
__host__ __device__ float
doublePulse(
    float A1, float k1, float k2, float x1,
    float A2, float k3, float  k4, float  x2,
    float  C, float  x    
    )
{
    float  p1 = singlePulse(A1, k1, k2, x1, C, x);
    float  p2 = singlePulse(A2, k3, k4, x2, 0.0, x);
    return p1 + p2;
}

/**
 * @brief Computes chi-square fitness for one point in one solution 
 * given that d_fitness has pulled out what we need. This fitness is for a 
 * single-pulse fit.
 * 
 * @details
 * The choice of weight determines whether the returned value is Neyman's 
 * (weight by variance estimated from data wt = 1/y) or Pearson's (weight by 
 * variance estimated from fit wt = 1/fit).
 *
 * @param pParams Pointer to this solutions parameters.
 * @param x       x-coordinate.
 * @param y       y-coordinate.
 * @param wt      Weight for this coordinate. 
 *
 * @return Chi-square between fit function and trace data.
 */
__host__ __device__
float chiFitness1(const float* pParams, float x, float y, float wt)
{
    // Get the parameters from the fit:
    float a  = pParams[A1];
    float k1 = pParams[K1];
    float k2 = pParams[K2];
    float x1 = pParams[X1];
    float c = pParams[C];

    float fit = singlePulse(a, k1, k2, x1, c, x);
    float d   = (y - fit);
    
    return (d*d*wt); 
}

/**
 * @brief Computes chi-square fitness contribution for one point in
 * one solution given that our caller has pulled out what we need.
 *
 * @details
 * The choice of weight determines whether the returned value is Neyman's 
 * (weight by variance estimated from data wt = 1/y) or Pearson's (weight by 
 * variance estimated from fit wt = 1/fit).
 *
 * @param pParams Pointer to this solutions parameters.
 * @param x       x-coordinate.
 * @param y       y-coordinate.
 * @param wt      Weight for this coordinate.
 *
 * @return Chi-square between fit function and trace data.
 */
__host__ __device__
float chiFitness2(const float* pParams, float x, float y, float wt)
{
    // Get the parameters from the fit:
    float a1 = pParams[A1];
    float a2 = pParams[A2];
    float k1 = pParams[K1];
    float k3 = pParams[K3];
    float k2 = pParams[K2];
    float k4 = pParams[K4];
    float x1 = pParams[X1];
    float x2 = pParams[X2];
    float c  = pParams[C];

    float fit = doublePulse(a1, k1, k2, x1, a2, k3, k4, x2, c, x);
    float d   = y - fit;
  
    return (d*d*wt);
}

/**
 * @brief Return the single-pulse chi-square goodness of fit over the range.
 *
 * @param params The fit parameters.
 *
 * @return The chi-square evaluated over the fit range.
 *
 * @note Returns FLT_MAX if the chi-square value is not finite.
 */
static float h_fit1(float* params)
{
    float result = 0;
    int npts = xcoords.size();
  
    for (int i = 0; i < npts; i++) {
	float x = xcoords[i];
	float y = ycoords[i];

	// Ensure point has a valid weight for Neyman chisq
	if (y != 0.0) {
	    result += chiFitness1(params, x, y, 1.0/y); // Unweighted
	}
    }
  
    if (!isfinite(result)) result = FLT_MAX;
    
    return result;
}

/**
 * @brief This function lives in the GPU and:
 *  * Computes the chi-square contribution for a single point for a single 
 *    solution in the swarm for a single pulse with an offset.
 *  * Uses reduceToSum to sum the chisquare contributions over the entire trace.
 * The result is put into the fitness value for our solution.

 * @param pSolutions Pointer to solutions array in the GPU.
 * @param pFitnesses Pointer to the array of fitnesse for all solutions in 
 *   the swarm.
 * @param nParams    Number of parameters in the fit (should be 5).
 * @param nSol       Number of solutions in the swarm.
 * @param pXcoords   Trace x-coordinates array.
 * @param pYcoords   Trace y-coordinates array.
 * @param pWeights   y weights to apply.
 * @param nPoints    Number of points in the trace.
 */
__global__
void d_fitness1(
    const float* pSolutions, float* pFitnesses, int nParams,
    int nSol, unsigned short* pXcoords, unsigned short* pYcoords,
    float* pWeights, int nPoints
    )
{
    extern __shared__ float sqdiff[];  // Locate the chisqr contribs in shmem.

    // Figure out which solution and point we're working on. This is based 
    // on our place in the computation's geometry:
    int swarm = blockIdx.x;
    int solno = blockIdx.y + swarm*nSol; // Our solution.
    int ptno  = threadIdx.x;	       // Our point.

    if ((solno <  nSol*gridDim.x)) {
	if (ptno < nPoints) {
	    int ipt = ptno + swarm*nPoints;
	    float x = pXcoords[ipt];
	    float y = pYcoords[ipt];

	    // Ensure that the Neyman chisq weight is valid
	    if (y != 0.0) {
		sqdiff[ptno]  = chiFitness1(
		    pSolutions + (solno*nParams), x, y, 1.0/y
		    ); // Unreduced
	    } else { // No contribution total chisq if no data.
		sqdiff[ptno] = 0.0;
	    }
	} else {
	    sqdiff[ptno] = 0.0; // So it won't contribute to the chisquare sum.
	}
	
	// Reduce threads won't work for us evidently.    
	__syncthreads();
	
	// Serial sum - if we are ptno 0 the we sum all npoints of sqdiff
	// into the solution

	if (ptno == 0)  {
	    pFitnesses[solno] = 0;
	    for (int i = 0; i < nPoints; i++) {
		pFitnesses[solno] += sqdiff[i];
	    }
	    if (!isfinite(pFitnesses[solno])) pFitnesses[solno] = FLT_MAX;
	}
    }  
}

/**
 * @brief This function lives in the GPU and:
 *  * Computes the chi-square contribution for a single point for a single 
 *    solution in the swarm for a single pulse with an offset.
 *  * Uses reduceToSum to sum the chisquare contributions over the entire trace.
 * The result is put into the fitness value for our solution.
 *
 * @param pSolutions Pointer to solutions array in the GPU.
 * @param pFitnesses Pointer to the array of fitnesse for all solutions 
 *   in the swarm.
 * @param nParams    Number of parameters in the fit (should be 5).
 * @param nSol       Number of solutions in the swarm.
 * @param pXcoords   Trace x-coordinates array.
 * @param pYcoords   Trace y-coordinates array.
 * @param pWeights   y weights to apply.
 * @param nPoints    Number of points in the trace.
 */
__global__
void d_fitness2(
    const float* pSolutions, float* pFitnesses, int nParams,
    int nSol, unsigned short* pXcoords, unsigned short* pYcoords,
    float* pWeights, int nPoints
    )
{
    extern __shared__ float sqdiff[];  // Locate the chisqr contribs in shmem.

    // Figure out which solution and point we're working on. This is based 
    // on our place in the computation's geometry:
    int swarm = blockIdx.x;
    int solno = blockIdx.y + swarm*nSol; // Our solution.
    int ptno  = threadIdx.x;	         // Our point.

    if (solno <  nSol*gridDim.x) {
	if (ptno < nPoints) {
	    int ipt = ptno + swarm*nPoints;
	    float x = pXcoords[ipt];
	    float y = pYcoords[ipt];

	    // Validate weight for Neyman chisq
	    if (y != 0.0) {
	    sqdiff[ptno]  = chiFitness2(
		pSolutions + (solno*nParams), x, y, 1.0/y
		); // Unreduced
	    } else {
		sqdiff[ptno] = 0.0;
	    }
	}  else {
	    sqdiff[ptno] = 0.0; // So it won't contribute to the chisquare sum.
	}
	
	__syncthreads(); //Ensure all elements of sqdiff are in.

	// Serial computation of total chisq
	if (ptno == 0) {
	    pFitnesses[solno] = 0;
	    for (int i =0; i < nPoints; i++) {
		pFitnesses[solno] += sqdiff[i];
	    }
	    if(!isfinite(pFitnesses[solno])) pFitnesses[solno] = FLT_MAX;
	}
    }
}

/**
 * @brief Invokes the kernel that produces the fitness measure. The fitness 
 * is computed in the GPU and is the chi square.
 *
 * @param solutions Pointer to the Cuda solution set.
 * @param fitnesses Pointer to the current fitness set
 * @param grid      Computational grid being used.
 * @param block     Shapes of blocks within the grid.
 */
void
h_fitSingle(
    const CudaOptimize::SolutionSet* solutions,
    CudaOptimize::FitnessSet* fitnesses, dim3 grid, dim3 block
    )
{
    static int  calls = 0;
    // d_solutions are current solutions
    // d_fitnesses are where fitnesses go
    const float* d_solutions = solutions->getDevicePositionsConst(); 
    float* d_fitnesses = fitnesses->get();
    calls++;

    std::cerr << " fitness 1 " << calls << std::endl;
  
    // Figure out how many warps the fitnesses require:
    //int nParams = solutions->getProblemDimension();
    //nParamBlocks     = (nParams + 31)/32;
    //nParamBlocks     = nParamBlocks*32;
  
    // Number of solutions we're floating around.  
    int nsol = solutions->getSolutionNumber();

    // Figure out the bocksize of the computation:
    dim3 myBlockSize(n_tracePoints, 1, 1); 

    d_fitness1<<< grid, myBlockSize, n_tracePoints*sizeof(float) >>>(
        d_solutions, d_fitnesses, P1_NPARAMS, nsol, d_xCoords, d_yCoords,
	d_pWeights, n_tracePoints
	);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
	reportCudaError("Failed to run single pulse fitness kernel");
    }

    // Fetch solutions and fitnesses and compare them with host computed values:
#define DEBUG1
#ifdef DEBUG1
    float computedF[nsol];
    float computedP[nsol*P1_NPARAMS];

    cudaMemcpy(
	computedF, d_fitnesses, nsol*sizeof(float), cudaMemcpyDeviceToHost
	);
    cudaMemcpy(
	computedP, d_solutions, nsol*P1_NPARAMS*sizeof(float),
	cudaMemcpyDeviceToHost
	);

    for (int i =0; i < nsol; i++) {
	float* pSol = computedP + i*P1_NPARAMS;
	float fit = h_fit1(pSol);
	if (fit != computedF[i]) {
	    std::cerr << "Pass : " << calls
		      << " Solution: " << i << " mismatch "
		      << "host: " << fit << " gpu " << computedF[i]
		      << std::endl;
	}
    }
#endif
}

/**
 * @brief Host part to setup computation of the fitnesses across the swarm 
 * for our fits for a double pulse. Really this just sets up the kernel call 
 * for fitness2 which does the rest.
 * 
 * @param solutions  Pointer to the current Solution set.
 * @param fitnesses  Pointer to the current fitness set
 * @param grid       Computaional grid geometry.
 * @param block      Shapes of the blocks within the grid.
 */
void
h_fitDouble(
    const CudaOptimize::SolutionSet* solutions,
    CudaOptimize::FitnessSet* fitnesses,
    dim3 grid, dim3 block
    )
{
    // d_solutions are current solutions
    // d_fitnesses are where fitnesses go    
    const float*   d_solutions = solutions->getDevicePositionsConst();
    float*         d_fitnesses = fitnesses->get();
  
    // How big is the swarm?
    int nsol = solutions->getSolutionNumber();

    // Figure out the bocksize of the computation:
    dim3 myBlockSize(MAXPOINTS, 1, 1);
    d_fitness2<<< grid, myBlockSize, MAXPOINTS*sizeof(float) >>>(
	d_solutions, d_fitnesses, P2_NPARAMS, nsol, d_xCoords, d_yCoords,
	d_pWeights, n_tracePoints
	);
  
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
	reportCudaError("Failed to run single pulsse fitness kernel");
    }  
}

void
cudafit1(
    DDAS::fit1Info* pResult, const std::vector<uint16_t>& trace,
    const std::pair<unsigned, unsigned>& limits,
    uint16_t saturation, bool freeTraceWhenDone
    )
{
    size_t nPoints = traceToGPU(trace, limits, saturation);

    // Create and setup the optimizer - fitness function will be done in
    // the device. Last parameter the swarm size?:
    CudaOptimize::DE_Optimizer opt(&h_fitSingle, P1_NPARAMS, 1,  200);
   
    opt.setTerminationFlags(
	(CudaOptimize::TERMINATION_FLAGS)(
	    CudaOptimize::TERMINATE_GENS | CudaOptimize::TERMINATE_FIT
	    )
	);
    opt.setGenerations(10000);
    opt.setStoppingFitness(10.0);
    opt.setMutation(CudaOptimize::DE_RANDOM);
    opt.setCrossover(CudaOptimize::DE_BINOMIAL);
    opt.setHostFitnessEvaluation(false);

    // Set constraints on the parameters.
    // Let the positions go a bit before/past the trace.
    // Baseline <= 25% full scale offset should be generous.
    opt.setBounds(0, A1, make_float2(saturation*10, 0.0));
    opt.setBounds(0, K1, make_float2(2, 0.0));
    opt.setBounds(0, K2, make_float2(0.1, 0.0));
    opt.setBounds(0, X1, make_float2(nPoints, 0)); 
    opt.setBounds(0, C,  make_float2(saturation/4.0, 0.0));
    
    opt.optimize();

    if (freeTraceWhenDone) freeTrace();

    // Pull out the fit values into the pResult:
    // opt.getFunctionEvals() is as close to an iteration count as we have.
    pResult->fitStatus =  0;
    pResult->iterations = opt.getFunctionEvals(); 
    float* pParams      = opt.getBestSolution(0);
    pResult->offset     = pParams[C];
    pResult->pulse.position = pParams[X1];
    pResult->pulse.amplitude= pParams[A1];
    pResult->pulse.steepness= pParams[K1];
    pResult->pulse.decayTime = pParams[K2];

    pResult->chiSquare =  DDAS::AnalyticFit::chiSquare1(
	pParams[A1], pParams[K1], pParams[K2], pParams[X1], pParams[C],
	trace, limits.first, limits.second
	);
}

void
cudafit2(
    DDAS::fit2Info* pResult, const std::vector<uint16_t>& trace,
    const std::pair<unsigned, unsigned>& limits,
    uint16_t saturation = 0xffff, bool traceIsLoaded = false
    )
{
    // If needed get the trace into the GPU:
    size_t nPoints;
    if (traceIsLoaded) {
	nPoints = n_tracePoints; // From prior load.
    } else {
	nPoints = traceToGPU(trace, limits, saturation);
    }

    // Set up the optimizer with the fitness done in the GPU:
    CudaOptimize::DE_Optimizer opt(&h_fitDouble, P2_NPARAMS, 1, 200);
    opt.setTerminationFlags(
	(CudaOptimize::TERMINATION_FLAGS)(
	    CudaOptimize::TERMINATE_GENS | CudaOptimize::TERMINATE_FIT
	    )
	);
    opt.setGenerations(1000); 
    opt.setStoppingFitness(10.0);
    opt.setMutation(CudaOptimize::DE_RANDOM);
    opt.setCrossover(CudaOptimize::DE_BINOMIAL);
    opt.setHostFitnessEvaluation(false);

    // Constrain the parameters - unfortunately we can't constrain x1 < x2 :(
    // We give corresponding parameters in the second pulse the same
    // constraints as the first pulse:
    //  - Position allowed past the ends of the trace.
    //  - Baseline <= 25% of the full ADC range.

    opt.setBounds(0, A1, make_float2(saturation*10, 0.0));
    opt.setBounds(0, A2, make_float2(saturation*10, 0.0));
    opt.setBounds(0, K1, make_float2(2.0, 0.0));
    opt.setBounds(0, K3, make_float2(2.0, 0.0));
    opt.setBounds(0, K2, make_float2(0.1, 0.0));
    opt.setBounds(0, K4, make_float2(0.1, 0.0));
    opt.setBounds(0, X1, make_float2(nPoints+50, -50));    
    opt.setBounds(0, X2, make_float2(nPoints+50, -50)); 
    opt.setBounds(0, C,  make_float2(saturation/4.0, 0.0));

    opt.optimize();

    freeTrace(); // Always!!
  
    // We only allowed one case so pull the best fitness and best solution
    // from it:
    pResult->fitStatus = 0;
    pResult->iterations= opt.getCurrentEvals();
    float * pParams    = opt.getBestSolution(0);
    pResult->offset    = pParams[C];

    pResult->pulses[0].position = pParams[X1];
    pResult->pulses[0].amplitude= pParams[A1];
    pResult->pulses[0].steepness= pParams[K1];
    pResult->pulses[0].decayTime = pParams[K2];

    pResult->pulses[1].position = pParams[X2];
    pResult->pulses[1].amplitude= pParams[A2];
    pResult->pulses[1].steepness= pParams[K3];
    pResult->pulses[1].decayTime = pParams[K4];
    pResult->chiSquare = DDAS::AnalyticFit::chiSquare2(
	pParams[A1], pParams[K1], pParams[K2], pParams[X1],
	pParams[A2], pParams[K3], pParams[K4], pParams[X2],
	pParams[C], trace, limits.first, limits.second
	);
}
